"""W56 (tastytrade-style tested/untested-side condor defense, decision
layer): execution/condor_defense.py.

classify_tested_delta is pure-number tested against the table's exact
edges. evaluate_condor_defense is driven end-to-end through the REAL W55
solve (synthetic_vertical_deltas -> solve_iv_from_price) with marks priced
by py_vollib's own black_scholes -- strikes are chosen with a brentq
root-find against py_vollib's own delta() so each fixture's solved |delta|
provably lands in its intended band (asserted, not assumed). Real
TradeRecord + legs-JSON fixtures, same convention test_w53_roll_target_
selection.py and test_w55_synthetic_delta.py use -- never MagicMock for
option_marks.

No inspect.getsource.
"""
from __future__ import annotations

import json
from datetime import timedelta

import pytest

pytest.importorskip("py_vollib")
pytest.importorskip("scipy")
from py_vollib.black_scholes import black_scholes  # noqa: E402
from py_vollib.black_scholes.greeks.analytical import delta as bs_delta  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

from ait.bot.state import TradeDirection, TradeRecord, TradeStatus  # noqa: E402
from ait.config.settings import CondorDefenseConfig, Settings, load_settings  # noqa: E402
from ait.execution.condor_defense import (  # noqa: E402
    DefenseAction,
    DefenseDecision,
    classify_tested_delta,
    evaluate_condor_defense,
)
from ait.execution.synthetic_delta import _expiry_key  # noqa: E402
from ait.utils.time import today_et  # noqa: E402

SYMBOL = "SPY"
SPOT = 500.0
SIGMA = 0.22
R = 0.05
# synthetic_leg_greeks (synthetic_delta.py) recomputes DTE itself from
# (expiry_date - today_et()).days -- it does NOT trust a passed-in dte. Every
# fixture below must therefore pick an expiry that is really `dte_days` from
# today, or the price synthesized at one t and the t re-derived inside the
# solve diverge and the "solved delta lands near the target" assertions fail.
EXPIRY_STR = str(today_et() + timedelta(days=90))  # only used by tests that
                                                    # don't care about the DTE


# ---------------------------------------------------------------------------
# fixtures: strike-for-target-delta (brentq against py_vollib's own delta),
# real black_scholes pricing, legs-JSON / TradeRecord / option_marks builders
# ---------------------------------------------------------------------------

def _expiry_for_dte(dte_days: int) -> str:
    """The expiry string synthetic_leg_greeks will re-derive `dte_days`
    calendar days from -- see the module docstring note above."""
    return str(today_et() + timedelta(days=dte_days))


def _strike_for_delta(target_abs_delta: float, right: str, dte_days: int,
                       spot: float = SPOT, sigma: float = SIGMA, r: float = R) -> float:
    t = dte_days / 365.0
    flag = "c" if right == "C" else "p"
    target = target_abs_delta if right == "C" else -target_abs_delta

    def f(k: float) -> float:
        return bs_delta(flag, spot, k, t, r, sigma) - target

    return brentq(f, spot * 0.3, spot * 1.7)


def _price(strike: float, right: str, dte_days: int, spot: float = SPOT,
           sigma: float = SIGMA) -> float:
    flag = "c" if right == "C" else "p"
    return round(black_scholes(flag, spot, strike, dte_days / 365.0, R, sigma), 4)


def _condor_legs(short_put: float, long_put: float, short_call: float,
                  long_call: float, expiry: str) -> str:
    return json.dumps([
        {"strike": long_put, "right": "P", "action": "BUY", "expiry": expiry},
        {"strike": short_put, "right": "P", "action": "SELL", "expiry": expiry},
        {"strike": short_call, "right": "C", "action": "SELL", "expiry": expiry},
        {"strike": long_call, "right": "C", "action": "BUY", "expiry": expiry},
    ])


def _trade(legs: str, strategy: str = "iron_condor") -> TradeRecord:
    return TradeRecord(
        trade_id="T-W56", symbol=SYMBOL, strategy=strategy,
        direction=TradeDirection.SHORT, status=TradeStatus.FILLED,
        entry_time="2026-09-27T10:00:00", entry_price=1.50, quantity=1,
        contract_type=strategy, legs=legs)


def _marks(*entries: tuple[float, str, float], expiry: str) -> dict[tuple, float]:
    key = _expiry_key(expiry)
    return {(SYMBOL, key, strike, right): price for strike, right, price in entries}


def _put_tested_condor(target_abs_delta: float, dte_days: int,
                        width: float = 10.0) -> tuple[TradeRecord, dict, int]:
    """A condor whose TESTED side is the put (short put close to spot),
    with the short put's own real py_vollib delta landing at
    target_abs_delta, and the call side comfortably OTM/untested."""
    expiry = _expiry_for_dte(dte_days)
    short_put = _strike_for_delta(target_abs_delta, "P", dte_days)
    long_put = short_put - width
    short_call = SPOT * 1.20
    long_call = short_call + width
    legs = _condor_legs(short_put, long_put, short_call, long_call, expiry)
    marks = _marks(
        (short_put, "P", _price(short_put, "P", dte_days)),
        (long_put, "P", _price(long_put, "P", dte_days)),
        (short_call, "C", _price(short_call, "C", dte_days)),
        (long_call, "C", _price(long_call, "C", dte_days)),
        expiry=expiry,
    )
    return _trade(legs), marks, dte_days


# ---------------------------------------------------------------------------
# classify_tested_delta -- pure table lookup, exact numbers
# ---------------------------------------------------------------------------

class TestClassifyTestedDelta:
    C = CondorDefenseConfig()  # 0.25 / 0.25-0.30 / 14-21 / 0.30 / 22 / 0.40

    @pytest.mark.parametrize("abs_delta,dte,expected", [
        (0.10, 18, DefenseAction.HOLD),          # row 1: below monitor
        (0.24, 18, DefenseAction.HOLD),
        (0.27, 18, DefenseAction.ROLL_UNTESTED),  # row 2: untested band+window
        (0.32, 25, DefenseAction.ROLL_TESTED),    # row 3: tested band, >=22 DTE
        (0.45, 5, DefenseAction.CLOSE),           # row 4: close, any DTE
        (0.99, 100, DefenseAction.CLOSE),
    ])
    def test_each_table_row(self, abs_delta, dte, expected):
        action, _ = classify_tested_delta(abs_delta, dte, self.C)
        assert action == expected

    @pytest.mark.parametrize("abs_delta,dte", [
        (0.35, 18),   # tested-band delta, still inside the untested DTE window
        (0.27, 25),   # untested-band delta, past the untested DTE window
        (0.30, 10),   # tested-band delta, well before roll_tested_dte_min
    ])
    def test_gap_cells_hold_with_no_rule_matched(self, abs_delta, dte):
        action, reason = classify_tested_delta(abs_delta, dte, self.C)
        assert action == DefenseAction.HOLD
        assert reason == "no_rule_matched"

    def test_monitor_floor_edge_is_inclusive_of_the_band_above(self):
        # exactly monitor_delta_max (0.25) is the untested band's own floor
        action, _ = classify_tested_delta(0.25, 18, self.C)
        assert action == DefenseAction.ROLL_UNTESTED

    def test_untested_upper_bound_is_exclusive(self):
        # exactly roll_untested_delta_max (0.30) belongs to the tested band,
        # but DTE=18 is below roll_tested_dte_min (22) -- no rule matches
        action, reason = classify_tested_delta(0.30, 18, self.C)
        assert action == DefenseAction.HOLD
        assert reason == "no_rule_matched"

    def test_untested_dte_upper_edge_is_inclusive(self):
        action, _ = classify_tested_delta(0.27, 21, self.C)
        assert action == DefenseAction.ROLL_UNTESTED

    def test_tested_dte_floor_edge_is_inclusive(self):
        action, _ = classify_tested_delta(0.32, 22, self.C)
        assert action == DefenseAction.ROLL_TESTED

    def test_close_threshold_is_inclusive_and_beats_everything(self):
        action, reason = classify_tested_delta(0.40, 1000, self.C)
        assert action == DefenseAction.CLOSE
        assert reason == "close_delta_threshold"

    def test_just_below_monitor_floor_holds(self):
        action, reason = classify_tested_delta(0.2499, 18, self.C)
        assert action == DefenseAction.HOLD
        assert reason == "below_monitor_band"


# ---------------------------------------------------------------------------
# evaluate_condor_defense -- end to end, real W55 solve
# ---------------------------------------------------------------------------

class TestEvaluateCondorDefenseDecisions:
    def test_close_when_tested_delta_crosses_the_threshold(self):
        config = CondorDefenseConfig(enabled=True)
        trade, marks, dte = _put_tested_condor(0.45, 18)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.CLOSE
        assert decision.tested_side == "P"
        assert decision.tested_short_delta == pytest.approx(0.45, abs=0.02)

    def test_roll_tested_when_delta_and_dte_qualify(self):
        config = CondorDefenseConfig(enabled=True, side="both")
        trade, marks, dte = _put_tested_condor(0.35, 25)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.ROLL_TESTED
        assert decision.tested_short_delta == pytest.approx(0.35, abs=0.02)

    def test_roll_untested_when_delta_and_dte_qualify(self):
        config = CondorDefenseConfig(enabled=True, side="both")
        trade, marks, dte = _put_tested_condor(0.27, 18)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.ROLL_UNTESTED
        assert decision.tested_short_delta == pytest.approx(0.27, abs=0.02)

    def test_hold_below_monitor_band(self):
        config = CondorDefenseConfig(enabled=True)
        trade, marks, dte = _put_tested_condor(0.15, 18)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "below_monitor_band"


class TestSideFilter:
    def test_untested_only_blocks_a_roll_tested_verdict(self):
        config = CondorDefenseConfig(enabled=True, side="untested_only")
        trade, marks, dte = _put_tested_condor(0.35, 25)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "roll_tested_blocked_by_side"

    def test_tested_only_blocks_a_roll_untested_verdict(self):
        config = CondorDefenseConfig(enabled=True, side="tested_only")
        trade, marks, dte = _put_tested_condor(0.27, 18)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "roll_untested_blocked_by_side"

    def test_both_allows_roll_tested(self):
        config = CondorDefenseConfig(enabled=True, side="both")
        trade, marks, dte = _put_tested_condor(0.35, 25)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.ROLL_TESTED

    def test_both_allows_roll_untested(self):
        config = CondorDefenseConfig(enabled=True, side="both")
        trade, marks, dte = _put_tested_condor(0.27, 18)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.ROLL_UNTESTED

    @pytest.mark.parametrize("side", ["tested_only", "untested_only"])
    def test_close_is_never_side_filtered(self, side):
        config = CondorDefenseConfig(enabled=True, side=side)
        trade, marks, dte = _put_tested_condor(0.45, 18)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.CLOSE


class TestTestedSideSelection:
    def test_call_side_tested_when_spot_is_close_to_the_short_call(self):
        # short call near the money (tested), short put far OTM (untested)
        short_call = SPOT * 1.02
        long_call = short_call + 10.0
        short_put = SPOT * 0.80
        long_put = short_put - 10.0
        dte = 18
        expiry = _expiry_for_dte(dte)
        legs = _condor_legs(short_put, long_put, short_call, long_call, expiry)
        marks = _marks(
            (short_call, "C", _price(short_call, "C", dte)),
            (long_call, "C", _price(long_call, "C", dte)),
            (short_put, "P", _price(short_put, "P", dte)),
            (long_put, "P", _price(long_put, "P", dte)),
            expiry=expiry,
        )
        config = CondorDefenseConfig(enabled=True)
        decision = evaluate_condor_defense(_trade(legs), marks, SPOT, dte, config)
        assert decision.tested_side == "C"
        assert decision.reason != "tested_delta_unavailable"

    def test_itm_short_put_is_tested_over_an_otm_short_call(self):
        # short put ITM (spot below strike), short call comfortably OTM
        short_put = SPOT * 1.02
        long_put = short_put - 10.0
        short_call = SPOT * 1.06
        long_call = short_call + 10.0
        dte = 18
        expiry = _expiry_for_dte(dte)
        legs = _condor_legs(short_put, long_put, short_call, long_call, expiry)
        marks = _marks(
            (short_put, "P", _price(short_put, "P", dte)),
            (long_put, "P", _price(long_put, "P", dte)),
            (short_call, "C", _price(short_call, "C", dte)),
            (long_call, "C", _price(long_call, "C", dte)),
            expiry=expiry,
        )
        config = CondorDefenseConfig(enabled=True)
        decision = evaluate_condor_defense(_trade(legs), marks, SPOT, dte, config)
        assert decision.tested_side == "P"
        assert decision.reason != "tested_delta_unavailable"


class TestFailureModes:
    def test_disabled_holds_before_touching_marks_or_solving(self):
        config = CondorDefenseConfig(enabled=False)
        trade, _marks_unused, dte = _put_tested_condor(0.45, 18)
        decision = evaluate_condor_defense(trade, {}, SPOT, dte, config)
        assert decision == DefenseDecision(DefenseAction.HOLD, "disabled")

    def test_non_iron_condor_strategy_holds(self):
        config = CondorDefenseConfig(enabled=True)
        trade, marks, dte = _put_tested_condor(0.45, 18)
        trade.strategy = "short_strangle"
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "not_an_iron_condor"

    def test_malformed_legs_json_holds_no_clean_condor(self):
        config = CondorDefenseConfig(enabled=True)
        trade = _trade(legs="not json")
        decision = evaluate_condor_defense(trade, {}, SPOT, 18, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "no_clean_condor"

    def test_one_side_not_a_clean_pair_holds_no_clean_condor(self):
        # two short puts on the put side -- not a clean single short+long pair
        legs = json.dumps([
            {"strike": 470.0, "right": "P", "action": "SELL", "expiry": EXPIRY_STR},
            {"strike": 460.0, "right": "P", "action": "SELL", "expiry": EXPIRY_STR},
            {"strike": 520.0, "right": "C", "action": "SELL", "expiry": EXPIRY_STR},
            {"strike": 530.0, "right": "C", "action": "BUY", "expiry": EXPIRY_STR},
        ])
        config = CondorDefenseConfig(enabled=True)
        decision = evaluate_condor_defense(_trade(legs), {}, SPOT, 18, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "no_clean_condor"

    def test_missing_tested_short_mark_holds_tested_delta_unavailable(self):
        trade, marks, dte = _put_tested_condor(0.35, 25)
        # strip the tested (put) side's marks entirely
        marks = {k: v for k, v in marks.items() if k[3] != "P"}
        config = CondorDefenseConfig(enabled=True)
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "tested_delta_unavailable"
        assert decision.tested_side == "P"

    def test_dte_none_holds_dte_unavailable(self):
        trade, marks, _dte = _put_tested_condor(0.35, 25)
        config = CondorDefenseConfig(enabled=True)
        decision = evaluate_condor_defense(trade, marks, SPOT, None, config)
        assert decision.action == DefenseAction.HOLD
        assert decision.reason == "dte_unavailable"

    def test_untested_side_marks_missing_still_decides(self):
        """evaluate_condor_defense only ever reads the TESTED short's own
        solve -- the untested side's marks (and even the tested side's own
        paired long) being absent must not block a decision."""
        trade, marks, dte = _put_tested_condor(0.35, 25)
        tested_short_key = next(k for k in marks if k[3] == "P" and
                                 marks[k] == max(v for kk, v in marks.items() if kk[3] == "P"))
        marks = {tested_short_key: marks[tested_short_key]}
        config = CondorDefenseConfig(enabled=True, side="both")
        decision = evaluate_condor_defense(trade, marks, SPOT, dte, config)
        assert decision.action == DefenseAction.ROLL_TESTED


# ---------------------------------------------------------------------------
# CondorDefenseConfig -- defaults, validation, real config.yaml
# ---------------------------------------------------------------------------

class TestCondorDefenseConfig:
    def test_defaults_match_the_playbook_table(self):
        c = CondorDefenseConfig()
        assert c.enabled is False
        assert c.side == "untested_only"
        assert c.monitor_delta_max == 0.25
        assert (c.roll_untested_delta_min, c.roll_untested_delta_max) == (0.25, 0.30)
        assert (c.roll_untested_dte_min, c.roll_untested_dte_max) == (14, 21)
        assert c.roll_tested_delta_min == 0.30
        assert c.roll_tested_dte_min == 22
        assert c.close_delta_threshold == 0.40

    @pytest.mark.parametrize("kwargs", [
        dict(monitor_delta_max=0.30, roll_untested_delta_min=0.25),   # inverted
        dict(roll_untested_delta_min=0.30, roll_untested_delta_max=0.25),
        dict(roll_untested_delta_max=0.35, roll_tested_delta_min=0.30),
        dict(roll_tested_delta_min=0.45, close_delta_threshold=0.40),
        dict(roll_untested_dte_min=25, roll_untested_dte_max=21),
        dict(roll_untested_dte_max=25, roll_tested_dte_min=22),
    ])
    def test_rejects_inverted_or_overlapping_bands(self, kwargs):
        with pytest.raises(ValueError):
            CondorDefenseConfig(**kwargs)

    def test_extra_key_is_rejected(self):
        with pytest.raises(ValueError):
            CondorDefenseConfig(untested_roll_target_delta=0.20)

    def test_real_config_yaml_loads_condor_defense_disabled(self):
        settings: Settings = load_settings("config.yaml")
        assert settings.condor_defense.enabled is False
        assert isinstance(settings.condor_defense, CondorDefenseConfig)
