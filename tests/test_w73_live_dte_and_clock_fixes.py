"""W73: two live bugs found while preparing short-DTE research, fixed at ZERO
change for the live dte_range [14, 30].

1. reconciler: the "expired?" test read the HOST date (`datetime.now().date()`),
   which is tomorrow's date from 18:00 ET on a CEST host, so a position that
   vanished the evening before expiry was booked "expired" at intrinsic.
2. options_chain.get_chain: `min_dte or cfg[0]` replaced an explicit 0 with the
   config floor, so a 1-DTE lookup (get_chain_for_expiry's `days - 1`) never
   found its expiry.

House rules: real code paths, a pinned non-ET host zone for clock behaviour, a
negative control, and a class guard that catches the NEXT instance.
"""
from __future__ import annotations

import ast
import itertools
import os
import pathlib
import time
from datetime import date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from zoneinfo import ZoneInfo

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "ait"
ET = ZoneInfo("US/Eastern")
BERLIN = ZoneInfo("Europe/Berlin")
HAS_TZSET = hasattr(time, "tzset")
posix_only = pytest.mark.skipif(not HAS_TZSET, reason="TZ + tzset is POSIX only")


@pytest.fixture
def host_tz():
    """Pin the process TZ (POSIX only), restoring it afterwards (W20/W28 pattern)."""
    original = os.environ.get("TZ")

    def _set(name: str) -> None:
        os.environ["TZ"] = name
        if HAS_TZSET:
            time.tzset()

    yield _set
    if original is None:
        os.environ.pop("TZ", None)
    else:
        os.environ["TZ"] = original
    if HAS_TZSET:
        time.tzset()


def _freeze(monkeypatch, instant_et: datetime) -> None:
    """Make BOTH clocks read one real instant, each the way its caller asks.

    `datetime.now(tz)` returns the instant in `tz`; the NO-ARGUMENT `now()` (the
    defect) returns the pinned HOST's naive local time (Berlin here), exactly
    what a CEST trading box reads.
    """
    class _Frozen(datetime):
        @classmethod
        def now(cls, tz=None):
            if tz is None:
                local = instant_et.astimezone(BERLIN).replace(tzinfo=None)
                return cls(local.year, local.month, local.day, local.hour,
                           local.minute, local.second)
            got = instant_et.astimezone(tz)
            return cls(got.year, got.month, got.day, got.hour, got.minute,
                       got.second, tzinfo=got.tzinfo)

    import ait.execution.reconciler as rec_mod
    import ait.utils.time as time_mod
    monkeypatch.setattr(rec_mod, "datetime", _Frozen)
    monkeypatch.setattr(time_mod, "datetime", _Frozen)


# --------------------------------------------------------------------------
# 1. reconciler: expired-test uses the ET date
# --------------------------------------------------------------------------

def _trade(expiry: str):
    return SimpleNamespace(
        trade_id="T-W73", symbol="SPY", strategy="iron_condor", contract_type="spread",
        entry_price=2.0, quantity=1, expiry=expiry, direction=None, strike=None,
        legs='[{"strike": 600.0, "right": "P", "action": "SELL", "expiry": "%s"},'
             ' {"strike": 590.0, "right": "P", "action": "BUY", "expiry": "%s"},'
             ' {"strike": 640.0, "right": "C", "action": "SELL", "expiry": "%s"},'
             ' {"strike": 650.0, "right": "C", "action": "BUY", "expiry": "%s"}]'
             % (expiry, expiry, expiry, expiry))


def _reconciler(settle: float | None = 620.0):
    from ait.execution.reconciler import PositionReconciler
    rec = PositionReconciler.__new__(PositionReconciler)
    rec._ibkr = MagicMock()
    rec._ibkr.get_portfolio.return_value = []
    rec._state = MagicMock()
    rec._underlying_last = AsyncMock(return_value=settle)
    return rec


async def _book(rec, trade):
    ok = await rec.book_vanished_trade(trade)
    assert ok is True
    return rec._state.close_trade.call_args.kwargs


@posix_only
class TestReconcilerExpiryUsesTheEtDate:
    @pytest.mark.asyncio
    async def test_negative_control_the_pin_makes_host_and_et_dates_disagree(self, host_tz):
        host_tz("Europe/Berlin")
        evening = datetime(2026, 9, 17, 19, 30, tzinfo=ET)
        assert evening.astimezone(BERLIN).date() != evening.date()

    @pytest.mark.asyncio
    async def test_a_position_vanishing_the_evening_before_expiry_is_not_booked_expired(
            self, host_tz, monkeypatch):
        """19:30 ET on 09-17 is already 09-18 on a Berlin host. Expiry 09-18 has NOT
        arrived in ET: the old code booked a fabricated intrinsic P&L."""
        host_tz("Europe/Berlin")
        _freeze(monkeypatch, datetime(2026, 9, 17, 19, 30, tzinfo=ET))
        rec = _reconciler()
        kw = await _book(rec, _trade("2026-09-18"))
        assert kw["exit_reason_detailed"] != "reconciler_expired_intrinsic"
        from ait.execution.reconciler import UNKNOWN_EXIT_REASON
        assert kw["exit_reason_detailed"] == UNKNOWN_EXIT_REASON
        assert kw["realized_pnl"] == 0.0              # flagged for review, nothing invented
        rec._underlying_last.assert_not_called()      # no settle price was even fetched

    @pytest.mark.asyncio
    async def test_expiry_date_in_et_is_still_booked_as_expired(self, host_tz, monkeypatch):
        host_tz("Europe/Berlin")
        _freeze(monkeypatch, datetime(2026, 9, 17, 19, 30, tzinfo=ET))
        rec = _reconciler(settle=620.0)               # inside both wings: full credit kept
        kw = await _book(rec, _trade("2026-09-17"))
        assert kw["exit_reason_detailed"] == "reconciler_expired_intrinsic"
        assert kw["realized_pnl"] == pytest.approx(200.0)

    @pytest.mark.asyncio
    async def test_past_expiry_is_still_expired(self, host_tz, monkeypatch):
        host_tz("Europe/Berlin")
        _freeze(monkeypatch, datetime(2026, 9, 17, 19, 30, tzinfo=ET))
        kw = await _book(_reconciler(), _trade("2026-09-10"))
        assert kw["exit_reason_detailed"] == "reconciler_expired_intrinsic"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("et_hour", [9, 10, 12, 15])
    async def test_inside_rth_host_and_et_dates_agree_so_nothing_changes(
            self, host_tz, monkeypatch, et_hour):
        """Zero change where the bot actually runs: across RTH both dates match, so
        the verdict is identical to the pre-W73 one for every expiry."""
        host_tz("Europe/Berlin")
        now = datetime(2026, 9, 17, et_hour, 30, tzinfo=ET)
        assert now.astimezone(BERLIN).date() == now.date()
        _freeze(monkeypatch, now)
        for expiry, want_expired in (("2026-09-16", True), ("2026-09-17", True), ("2026-09-18", False)):
            rec = _reconciler()
            kw = await _book(rec, _trade(expiry))
            assert (kw["exit_reason_detailed"] == "reconciler_expired_intrinsic") is want_expired


class TestNaiveNowDateRatchet:
    """CLASS GUARD for the shape W28's `.today()` guard missed: `<x>.now().date()`
    with NO timezone. W73 held reconciler.py and options_chain.py to zero and registered
    three orchestrator sites for the iv_rank work; W71b fixed those, so the WHOLE tree is
    now held to zero and any new site fails."""

    REGISTERED: set[tuple[str, str]] = set()

    @staticmethod
    def _sites(source: str) -> set[str]:
        """Enclosing function names of every naive `<x>.now().date()` in ``source``."""
        tree = ast.parse(source)
        enclosing: dict[int, str] = {}
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for n in ast.walk(fn):
                    enclosing[id(n)] = fn.name
        out: set[str] = set()
        for n in ast.walk(tree):
            if (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                    and n.func.attr == "date" and isinstance(n.func.value, ast.Call)
                    and isinstance(n.func.value.func, ast.Attribute)
                    and n.func.value.func.attr == "now"
                    and not n.func.value.args and not n.func.value.keywords):
                out.add(enclosing.get(id(n), "<module>"))
        return out

    def _offenders(self) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for path in sorted(SRC.rglob("*.py")):
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                names = self._sites(text)
            except SyntaxError:
                continue
            out |= {(path.relative_to(SRC).as_posix(), name) for name in names}
        return out

    def test_the_whole_tree_has_none(self):
        assert self._offenders() == self.REGISTERED, (
            "a naive `.now().date()` (host date, tomorrow from 18:00 ET on a CEST host) "
            f"is back: {sorted(self._offenders())}. Use ait.utils.time.today_et().")

    def test_the_guard_detects_the_shape(self):
        """The tree is clean, so prove the detector on synthetic source instead."""
        bad = ("from datetime import datetime\n"
               "def f():\n    return datetime.now().date()\n"
               "def g():\n    return datetime.now(tz).date()\n"
               "X = datetime.now().date()\n")
        assert self._sites(bad) == {"f", "<module>"}      # g passes a tz: not the naive shape
        assert self._sites("def h():\n    return today_et()\n") == set()

    def test_the_guard_actually_scans_the_tree(self):
        assert len(list(SRC.rglob("*.py"))) > 50


# --------------------------------------------------------------------------
# 2. options_chain: 0 is a bound, not "missing"
# --------------------------------------------------------------------------

def _service(band=(14, 30)):
    from ait.config.settings import OptionsConfig
    from ait.data.options_chain import OptionsChainService
    cfg = OptionsConfig(dte_range=list(band))
    md = MagicMock()
    md.get_current_price = AsyncMock(return_value=100.0)
    svc = OptionsChainService(MagicMock(), md, cfg)
    svc._calculate_greeks = lambda chain: None
    return svc


class _FakeChain(SimpleNamespace):
    pass


class TestGetChainBounds:
    @pytest.mark.asyncio
    async def test_no_arguments_use_the_config_band(self):
        svc = _service((14, 30))
        svc._get_ibkr_chain = AsyncMock(return_value=[])
        svc._get_yahoo_chain = AsyncMock(return_value=[])
        await svc.get_chain("SPY")
        assert svc._get_ibkr_chain.call_args.args[2:] == (14, 30)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lo,hi", [(0, 2), (-1, 0), (0, 0)])
    async def test_an_explicit_zero_is_honoured(self, lo, hi):
        svc = _service((14, 30))
        svc._get_ibkr_chain = AsyncMock(return_value=[])
        svc._get_yahoo_chain = AsyncMock(return_value=[])
        await svc.get_chain("SPY", min_dte=lo, max_dte=hi)
        assert svc._get_ibkr_chain.call_args.args[2:] == (lo, hi)

    @pytest.mark.asyncio
    async def test_one_sided_none_falls_back_only_for_that_side(self):
        svc = _service((14, 30))
        svc._get_ibkr_chain = AsyncMock(return_value=[])
        svc._get_yahoo_chain = AsyncMock(return_value=[])
        await svc.get_chain("SPY", min_dte=3)
        assert svc._get_ibkr_chain.call_args.args[2:] == (3, 30)
        svc._cache = type(svc._cache)(default_ttl=120)
        await svc.get_chain("SPY", max_dte=45)
        assert svc._get_ibkr_chain.call_args.args[2:] == (14, 45)

    @pytest.mark.asyncio
    async def test_zero_change_for_every_reachable_input(self):
        """Equivalence with the old `x or default` over the inputs live can produce
        (None, or a bound >= 1): the fix must not move a single one of them."""
        for lo, hi in itertools.product((None, 1, 5, 14, 30), (None, 2, 30, 45)):
            svc = _service((14, 30))
            svc._get_ibkr_chain = AsyncMock(return_value=[])
            svc._get_yahoo_chain = AsyncMock(return_value=[])
            await svc.get_chain("SPY", min_dte=lo, max_dte=hi)
            old = (lo or 14, hi or 30)
            assert svc._get_ibkr_chain.call_args.args[2:] == old

    @pytest.mark.asyncio
    async def test_get_chain_for_expiry_finds_a_one_dte_expiry(self):
        from ait.utils.time import today_et
        svc = _service((14, 30))
        tomorrow = today_et() + timedelta(days=1)
        far = today_et() + timedelta(days=20)
        fakes = [_FakeChain(expiry=tomorrow, source=""), _FakeChain(expiry=far, source="")]

        async def _ibkr(symbol, price, min_dte, max_dte):
            # the real fetch's inclusive DTE window over the listed expiries
            return [c for c in fakes if min_dte <= (c.expiry - today_et()).days <= max_dte]

        svc._get_ibkr_chain = _ibkr
        svc._get_yahoo_chain = AsyncMock(return_value=[])
        got = await svc.get_chain_for_expiry("SPY", tomorrow)
        assert got is not None and got.expiry == tomorrow
        # the long-dated lookup the live roll path relies on is unchanged
        far_got = await svc.get_chain_for_expiry("SPY", far)
        assert far_got is not None and far_got.expiry == far
