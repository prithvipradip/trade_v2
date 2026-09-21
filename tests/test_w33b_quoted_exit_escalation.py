"""W33b: a QUOTED exit that does not fill must not repeat itself either.

W33 taught the NO-QUOTE exit to step its price each attempt. The quoted path
kept `ask + cross`, on the theory that a limit through a live ask fills. The
first live exit on W33 showed where that theory ends (2026-09-21, QQQ condor
T-20260917-095601-aa4fa8, touch stop, this account on DELAYED data — IBKR
warning 10167 on the exit BAG itself):

  10:25:04  BAG 8.46/8.57 -> BUY LMT 8.67      worked 11 minutes, unfilled
  10:36:08  cancelled (the 300s rule; scan cycles delayed it to 664s)
  10:37:10  BAG 9.50/9.66 -> BUY LMT 9.76      working, unfilled, QQQ still rising

Every price was built from a quote ~15 minutes old, so in a trending tape each
"marketable" limit was already behind the market when it was sent, and the next
one started from the next stale quote with no memory that the last one failed.
A stop that chases a rally one stale quote at a time never catches it.

The fix is memory: each consecutive unfilled placement for the same trade raises
the limit by `exit_unfilled_escalation_pct` of the marketable price, under the
same caps as before (wing width, the mark-anchored ceiling, the broker's band).
It is a LIMIT, so a higher one fills at the market's offer, not at our number.

Drives the real `_close_multi_leg` on the W33 rig.
"""
from __future__ import annotations

import pytest

from ait.config.settings import load_settings
from tests.test_w33_exit_pricing_corrective import _placed, _rig


def _quoted(bag, mark=8.50, **kw):
    return _rig(leg_quotes={}, bag=bag, mark_cost=mark, entry=4.99, **kw)


class TestAQuotedExitRemembersItDidNotFill:
    def test_the_same_stale_quote_is_not_answered_with_the_same_price(self):
        """The quote is frozen (delayed feed) and nothing fills: 8.67, 8.67, 8.67
        at HEAD."""
        o, ibkr, trade = _quoted((8.46, 8.57))
        prices = [_placed(o, ibkr, trade) for _ in range(3)]
        assert prices[0] == pytest.approx(8.67)              # first attempt unchanged
        assert prices == sorted(prices) and len(set(prices)) == 3, prices

    def test_this_mornings_second_attempt_reaches_past_the_stale_ask(self):
        o, ibkr, trade = _quoted((8.46, 8.57))
        assert _placed(o, ibkr, trade) == pytest.approx(8.67)
        o._ibkr.ib.ticker = lambda c: __import__("types").SimpleNamespace(bid=9.50, ask=9.66)
        second = _placed(o, ibkr, trade)
        assert second > 9.76, (
            f"second attempt placed {second}: `ask + 0.10` again, off a quote "
            f"15 minutes old, after the first had already failed to fill")
        pct = load_settings().exit.exit_unfilled_escalation_pct
        assert second == pytest.approx(round(9.76 * (1 + pct), 2))

    def test_escalation_is_config_homed(self):
        assert 0.0 <= load_settings().exit.exit_unfilled_escalation_pct <= 0.25

    def test_the_caps_still_bind(self):
        """Many failed attempts: never past the mark-anchored ceiling."""
        o, ibkr, trade = _quoted((8.46, 8.57), mark=8.50)
        last = None
        for _ in range(40):
            last = _placed(o, ibkr, trade)
        assert last == pytest.approx(o._exit_price_ceiling(8.50, 39.0, 0.10))

    def test_the_broker_band_still_wins(self):
        o, ibkr, trade = _quoted((8.46, 8.57), mark=8.50, band=9.00, market=8.60)
        for _ in range(6):
            last = _placed(o, ibkr, trade)
        assert last < 9.00

    def test_a_new_exit_episode_starts_clean(self):
        o, ibkr, trade = _quoted((8.46, 8.57))
        _placed(o, ibkr, trade)
        _placed(o, ibkr, trade)
        o._exit_quoted_attempts["T-1"]["at"] -= 2 * 3600
        assert _placed(o, ibkr, trade) == pytest.approx(8.67)

    def test_a_debit_close_is_never_escalated_upward(self):
        """Closing a LONG structure sells it: the as-defined quote is negative
        and 'more aggressive' means LOWER. The existing debit cap (never pay
        more than the cross to leave a long) must still hold after retries."""
        o, ibkr, trade = _rig(leg_quotes={}, bag=(-1.20, -1.00), mark_cost=None)
        trade.strategy = "long_straddle"
        legs = [{"strike": 500, "right": "C", "action": "BUY", "expiry": "20261016"},
                {"strike": 500, "right": "P", "action": "BUY", "expiry": "20261016"}]
        prices = [_placed(o, ibkr, trade, legs) for _ in range(3)]
        assert all(p <= o._settings.exit.exit_cross_amount for p in prices), prices
