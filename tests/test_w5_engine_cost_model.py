"""W5 — the research-honesty cluster from reports/blindspot_composition_hunt_20260825.md.

Three CONFIRMED research_validity findings, all in the backtest engine's cost
and pricing model, all biased the SAME direction (optimistic). The audit's own
severity note: together they are "the scale of the whole per-condor
expectancy, so together they can flip the sign of the researched edge that
authorizes go-live".

  model-vs-reality-commission-constant
      The engine charged a flat $0.65/contract/leg as the ENTIRE broker cost.
      IBKR charges ~$0.65 commission PLUS regulatory/exchange/clearing fees on
      every option leg, entry AND exit. The finding's probe on the repo's own
      executions ledger (data/ait_state.db, 78 leg fills, BAG summary rows
      excluded) measured all-in per contract-leg mean 0.9168 / median 1.0284,
      i.e. $7.33 vs the modelled $5.20 per 4-leg condor round trip — the
      headline $2.13/contract understatement, ~41% of real friction.

  model-vs-reality-touch-gap-pricing
      A day whose session OPEN is already beyond a short strike was booked as
      a near-scratch touch AT the strike — a level that never traded, because
      options do not trade overnight and a resting stop would trigger at the
      open too. Measured cost: $324 / $720 / $1,405 per contract at 1% / 2% /
      3.5% gaps, on days that occur 9.6% (SPY) to 18.1% (QQQ) of the time in
      the exact 604-day sample every study runs on.

  model-vs-reality-skew-10x-flat
      Put-side skew was hardcoded at 0.10 absolute IV per unit
      |log-moneyness| against 0.916-1.141 regressed on the repo's own 3,159
      IBKR quotes — 9-11x flatter than reality, which made model wings
      near-free (long put $0.08 vs $0.71-0.92 real chain mid).

Every test here EXECUTES the real engine path — no inspect.getsource, no
source-string assertions. Consequence to state plainly: every backtest
absolute produced before this landed is stale.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from ait.backtesting.engine import Backtester
from ait.backtesting.pricing import OptionType, black_scholes_price
from ait.config.settings import BacktestConfig, load_settings


# ---------------------------------------------------------------------------
# Fixtures — real Backtester construction (empty-OHLCV ctor pattern from
# tests/test_r21b_pr7_review_round.py); pricing tests drive the internal
# helpers directly with synthetic inputs.
# ---------------------------------------------------------------------------

EMPTY_COLS = ["Open", "High", "Low", "Close", "Volume"]


def _bt(**overrides) -> Backtester:
    kw = dict(
        data=pd.DataFrame(columns=EMPTY_COLS),
        strategies=["iron_condor"],
        macro_event_gate=False,
        allow_live_model_fallback=False,
    )
    kw.update(overrides)
    return Backtester(**kw)


@pytest.fixture(scope="module")
def bt() -> Backtester:
    return _bt()


@pytest.fixture(scope="module")
def yaml_backtest():
    return load_settings("config.yaml").backtest


# ===========================================================================
# (a) model-vs-reality-commission-constant — commission + fees, per leg,
#     entry AND exit.
# ===========================================================================

class TestCommissionAndFees:
    def test_all_in_cost_is_commission_plus_fees(self, bt):
        # The engine no longer treats the commission LINE as the whole cost.
        assert bt._cost_per_contract_leg == pytest.approx(
            bt._commission + bt._regulatory_fees)
        assert bt._cost_per_contract_leg == pytest.approx(0.9168, abs=1e-9)

    def test_condor_round_trip_matches_measured_ledger(self, bt):
        # The finding's headline: 4 legs x 2 sides x 1 contract.
        # measured $7.33 all-in vs the old flat-0.65 model's $5.20.
        round_trip = bt._cost_per_contract_leg * 4 * 2
        assert round_trip == pytest.approx(7.33, abs=0.01)
        old_flat_model = 0.65 * 4 * 2
        assert round_trip - old_flat_model == pytest.approx(2.13, abs=0.01)

    def test_calc_pnl_charges_both_sides_of_a_condor(self, bt):
        # _calc_pnl is the single P&L authority: it must deduct the entry
        # charge stamped at open PLUS an identical exit charge.
        contracts, n_legs = 8, 4
        entry_price, exit_value = 4.31, 2.00
        entry_commission = bt._cost_per_contract_leg * contracts * n_legs
        pos = {
            "contracts": contracts, "n_legs": n_legs,
            "entry_price": entry_price, "trade_type": "credit",
            "entry_commission": entry_commission,
        }
        raw = (entry_price - exit_value) * 100 * contracts
        pnl = bt._calc_pnl(pos, exit_value)
        deducted = raw - pnl
        assert deducted == pytest.approx(
            bt._cost_per_contract_leg * n_legs * 2 * contracts, abs=0.01)
        # ... and per contract that is exactly the measured $7.33.
        assert deducted / contracts == pytest.approx(7.33, abs=0.01)
        # Pre-W5 the same trade was charged the flat-0.65 figure.
        assert deducted / contracts - (0.65 * n_legs * 2) == pytest.approx(
            2.13, abs=0.01)

    def test_entry_charge_stamped_by_a_real_run_is_all_in(self):
        # End-to-end: run() stamps pos["entry_commission"], and it must be the
        # all-in per-leg cost, not the commission line.
        from ait.ml.features import FeatureEngine

        idx = pd.bdate_range("2024-01-02", periods=160)
        rng = np.random.default_rng(42)
        close = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.008, len(idx))))
        df = pd.DataFrame(
            {"Open": close, "High": close * 1.004, "Low": close * 0.996,
             "Close": close, "Volume": 1e6}, index=idx)
        engine = _bt(
            data=df, initial_capital=100_000, features_cache=FeatureEngine().compute(df),
            hurst_hard_veto_multiplier=0.0, hurst_regime_threshold=0.0,
            iv_rank_rise_threshold=10.0, ic_min_credit=0.0,
            ic_min_credit_width=0.0, iv_floor=0.01,
        )
        result = engine.run()
        assert result.trades, "fixture must produce trades for this to mean anything"
        for t in result.trades:
            assert t["entry_commission"] == pytest.approx(
                engine._cost_per_contract_leg * t["contracts"] * t["n_legs"])
            # the pre-W5 charge, which this must NOT equal
            assert t["entry_commission"] > 0.65 * t["contracts"] * t["n_legs"]

    def test_zero_fees_reproduces_the_pre_w5_charge(self, bt):
        # Escape hatch for reproducing a disavowed study — and proof the
        # correction is entirely the fees term.
        legacy = _bt(regulatory_fees_per_contract=0.0)
        assert legacy._cost_per_contract_leg == pytest.approx(0.65)
        assert legacy._cost_per_contract_leg * 4 * 2 == pytest.approx(5.20)

    def test_walkforward_pinned_commission_still_pays_fees(self, yaml_backtest):
        # walkforward.py pins commission_per_contract=0.65 and passes it into
        # every window. Explicit wins for the commission LINE, but the fees
        # still resolve from config — a pinned research entry point can no
        # longer opt out of the measured friction.
        from ait.backtesting.walkforward import WalkForwardConfig

        pinned = float(WalkForwardConfig().commission_per_contract)
        engine = _bt(commission_per_contract=pinned)
        assert engine._commission == pytest.approx(pinned)
        assert engine._regulatory_fees == pytest.approx(
            yaml_backtest.regulatory_fees_per_contract)
        assert engine._cost_per_contract_leg == pytest.approx(
            pinned + yaml_backtest.regulatory_fees_per_contract)


# ===========================================================================
# (b) model-vs-reality-touch-gap-pricing — a gapped open is not a touch.
# ===========================================================================

SHORT_PUT, SHORT_CALL = 620.0, 666.0
LONG_PUT, LONG_CALL = 576.0, 710.0
ENTRY_DATE, EXIT_DATE = "2026-08-04", date(2026, 8, 11)   # 7 days held


def _condor(entry_price: float = 4.00) -> dict:
    """The finding's own probe geometry: SPY-like S=640, 620/666 shorts,
    576/710 wings, entry IV 0.18, 21 DTE, held 7 days."""
    return {
        "strategy": "iron_condor", "trade_type": "credit",
        "entry_date": ENTRY_DATE, "entry_price": entry_price,
        "contracts": 1, "n_legs": 4,
        "short_put_strike": SHORT_PUT, "short_call_strike": SHORT_CALL,
        "long_put_strike": LONG_PUT, "long_call_strike": LONG_CALL,
        "expiry_date": "2026-08-25", "entry_iv": 0.18,
        "underlying_at_entry": 640.0, "high_water_mark": 0.0,
        "option_type": "iron_condor",
    }


def _row(open_, low, high, close) -> pd.Series:
    return pd.Series({"Open": open_, "High": high, "Low": low, "Close": close})


@pytest.fixture(scope="module")
def gap_bt() -> Backtester:
    # entry_dte 21 = the probe's option; iv_floor low so the floor does not
    # mask the surface under test.
    return _bt(entry_dte=21, iv_floor=0.10)


class TestTouchGapPricing:
    def test_gap_through_put_prices_worse_than_the_same_intraday_touch(self, gap_bt):
        # IDENTICAL bar except the Open. Both days pierce 620 intraday, so
        # pre-W5 both booked the buyback AT 620 — the level that never traded
        # on the gap day.
        touch = gap_bt._check_exit(
            _condor(), _row(open_=636.0, low=613.0, high=638.0, close=618.0),
            EXIT_DATE)
        gapped = gap_bt._check_exit(
            _condor(), _row(open_=613.8, low=613.0, high=638.0, close=618.0),
            EXIT_DATE)
        assert touch is not None and touch["exit_reason"] == "touch_stop"
        assert gapped is not None and gapped["exit_reason"] == "touch_stop"
        # The intraday-touch day is still booked at the strike (unchanged).
        assert touch["exit_underlying"] == pytest.approx(SHORT_PUT)
        # The gap day is booked at the gapped open.
        assert gapped["exit_underlying"] == pytest.approx(613.8)
        # WORSE: costlier buyback, smaller P&L.
        assert gapped["exit_price"] > touch["exit_price"]
        assert gapped["pnl"] < touch["pnl"]

    def test_gap_loss_scales_with_gap_size(self, gap_bt):
        # The finding priced 1% / 2% / 3.5% gaps below the short put at
        # 10.99 / 14.95 / 21.79 per share against the engine's 7.75 at the
        # strike. Absolute levels move with the W5 skew fix; the ORDERING and
        # the sign of the correction are the invariant.
        at_strike = gap_bt._check_exit(
            _condor(), _row(open_=636.0, low=598.0, high=638.0, close=618.0),
            EXIT_DATE)["exit_price"]
        prev = at_strike
        for pct in (0.01, 0.02, 0.035):
            open_ = SHORT_PUT * (1 - pct)
            out = gap_bt._check_exit(
                _condor(), _row(open_=open_, low=min(598.0, open_ - 1),
                                high=638.0, close=618.0),
                EXIT_DATE)
            assert out["exit_price"] > prev, f"{pct:.1%} gap must cost more"
            prev = out["exit_price"]
        # A 1% gap alone is worth hundreds of dollars per contract of loss the
        # engine used to hide (the finding measured $324).
        one_pct = gap_bt._check_exit(
            _condor(), _row(open_=SHORT_PUT * 0.99, low=598.0, high=638.0,
                            close=618.0),
            EXIT_DATE)
        assert (at_strike - one_pct["exit_price"]) * 100 < -100.0

    def test_gap_through_call_prices_worse(self, gap_bt):
        touch = gap_bt._check_exit(
            _condor(), _row(open_=650.0, low=648.0, high=673.0, close=668.0),
            EXIT_DATE)
        gapped = gap_bt._check_exit(
            _condor(), _row(open_=672.6, low=648.0, high=673.0, close=668.0),
            EXIT_DATE)
        assert touch["exit_reason"] == gapped["exit_reason"] == "touch_stop"
        assert touch["exit_underlying"] == pytest.approx(SHORT_CALL)
        assert gapped["exit_underlying"] == pytest.approx(672.6)
        assert gapped["exit_price"] > touch["exit_price"]
        assert gapped["pnl"] < touch["pnl"]

    def test_open_exactly_at_the_strike_is_the_strike_price(self, gap_bt):
        # Boundary: the conservative floor ("never better than the
        # strike-touch price") must be an equality here, not a discount.
        touch = gap_bt._check_exit(
            _condor(), _row(open_=636.0, low=613.0, high=638.0, close=618.0),
            EXIT_DATE)
        at_strike_open = gap_bt._check_exit(
            _condor(), _row(open_=SHORT_PUT, low=613.0, high=638.0, close=618.0),
            EXIT_DATE)
        assert at_strike_open["exit_price"] == pytest.approx(touch["exit_price"])
        assert at_strike_open["pnl"] == pytest.approx(touch["pnl"])

    def test_non_gapped_touch_still_books_the_strike_touch_price(self, gap_bt):
        # Regression guard: R16's touch-at-the-strike behaviour is untouched
        # for days that did NOT open beyond the strike.
        out = gap_bt._check_exit(
            _condor(), _row(open_=636.0, low=613.0, high=638.0, close=618.0),
            EXIT_DATE)
        days_held = (EXIT_DATE - date.fromisoformat(ENTRY_DATE)).days
        expected = gap_bt._reprice_position(_condor(), SHORT_PUT, days_held, None)
        remaining_dte = (date.fromisoformat("2026-08-25") - EXIT_DATE).days
        expected *= (1 + gap_bt._options_half_spread(0.18, remaining_dte))
        assert out["exit_price"] == pytest.approx(round(expected, 4), abs=1e-4)

    def test_missing_open_column_is_safe(self, gap_bt):
        # Older/synthetic frames without Open must keep the R16 behaviour.
        out = gap_bt._check_exit_credit(
            _condor(), pnl_pct=0.20, current_date=EXIT_DATE,
            row=pd.Series({"Low": 613.0, "High": 638.0, "Close": 618.0}))
        assert out is not None and out["exit_reason"] == "touch_stop"
        assert out["touch_underlying"] == pytest.approx(SHORT_PUT)
        assert "touch_gap_underlying" not in out

    def test_gap_key_only_present_on_a_gap_day(self, gap_bt):
        inside_open = gap_bt._check_exit_credit(
            _condor(), pnl_pct=0.20, current_date=EXIT_DATE,
            row=_row(open_=636.0, low=613.0, high=638.0, close=618.0))
        assert "touch_gap_underlying" not in inside_open
        gapped = gap_bt._check_exit_credit(
            _condor(), pnl_pct=0.20, current_date=EXIT_DATE,
            row=_row(open_=613.8, low=613.0, high=638.0, close=618.0))
        assert gapped["touch_gap_underlying"] == pytest.approx(613.8)


# ===========================================================================
# (c) model-vs-reality-skew-10x-flat — wings are not free.
# ===========================================================================

# The finding's probe anchor: SPY S=770.6, ATM IV 0.1289, DTE 20.
ANCHOR_S, ANCHOR_IV, ANCHOR_DTE = 770.6, 0.1289, 20


def _put_strike(pct_otm: float) -> float:
    """Strike `pct_otm` percentage points OTM in log-moneyness."""
    return ANCHOR_S * float(np.exp(-pct_otm / 100.0))


@pytest.fixture(scope="module")
def skew_bt() -> Backtester:
    return _bt(iv_floor=0.01)          # floor off: the surface is under test


@pytest.fixture(scope="module")
def flat_bt() -> Backtester:
    # slope 0 AND skew_factor 0 = genuinely flat IV, the model the finding
    # says research has effectively been running.
    return _bt(iv_floor=0.01, skew_slope_per_pct_otm=0.0, skew_factor=0.0)


@pytest.fixture(scope="module")
def legacy_bt() -> Backtester:
    # slope 0 only = the exact pre-W5 engine (legacy 0.10 / 0.02 terms).
    return _bt(iv_floor=0.01, skew_slope_per_pct_otm=0.0)


class TestSkewSurface:
    def test_default_slope_lifts_otm_put_iv_above_flat(self, skew_bt, flat_bt):
        for pct in (2.0, 5.0, 8.0, 10.0):
            k = _put_strike(pct)
            skewed = skew_bt._get_leg_iv(ANCHOR_IV, k, ANCHOR_S, OptionType.PUT)
            flat = flat_bt._get_leg_iv(ANCHOR_IV, k, ANCHOR_S, OptionType.PUT)
            assert flat == pytest.approx(ANCHOR_IV)     # flat = the base IV
            assert skewed > flat

    def test_default_slope_reproduces_the_measured_regression(self, skew_bt):
        # dIV/d|ln(K/S)| must land on the sample-weighted measurement across
        # the repo's own 3,159 IBKR quotes: SPY 1.141 (n=140), QQQ 0.983
        # (n=81), IWM 0.916 (n=57) -> 1.049. Pre-W5 it was 0.10.
        for pct in (2.0, 5.0, 10.0):
            k = _put_strike(pct)
            iv = skew_bt._get_leg_iv(ANCHOR_IV, k, ANCHOR_S, OptionType.PUT)
            slope = (iv - ANCHOR_IV) / (pct / 100.0)
            assert slope == pytest.approx(1.049, abs=0.01)
            assert slope > 9 * 0.10          # 9-11x the hardcoded value

    def test_otm_put_wing_prices_higher_than_under_flat_iv(self, skew_bt, flat_bt, legacy_bt):
        # The consequence the finding measured at the leg level: long put wing
        # $0.08 model vs $0.71-0.92 real chain mid.
        k = _put_strike(8.0)
        t = ANCHOR_DTE / 365.0
        def _price(engine):
            return black_scholes_price(
                ANCHOR_S, k, t, 0.05,
                engine._get_leg_iv(ANCHOR_IV, k, ANCHOR_S, OptionType.PUT),
                OptionType.PUT)
        w5, flat, legacy = _price(skew_bt), _price(flat_bt), _price(legacy_bt)
        assert w5 > flat and w5 > legacy
        assert legacy < 0.15, "pre-W5 wings really were near-free"
        assert w5 > 10 * legacy
        # Order-of-magnitude reality check against the measured chain mid.
        assert 0.4 < w5 < 1.6

    def test_slope_zero_reproduces_pre_w5_pricing_exactly(self, legacy_bt, flat_bt):
        # Backward-compat escape hatch. Two senses, both pinned:
        #  1. slope 0 == the legacy 0.10 put / 0.02 call terms, to the bit.
        #  2. slope 0 + skew_factor 0 == genuinely flat IV.
        for pct in (0.0, 2.0, 5.0, 10.0):
            kp = _put_strike(pct)
            kc = ANCHOR_S * float(np.exp(pct / 100.0))
            legacy_put = ANCHOR_IV + (pct / 100.0) * 0.10
            legacy_call = ANCHOR_IV + (pct / 100.0) * 0.02
            assert legacy_bt._get_leg_iv(
                ANCHOR_IV, kp, ANCHOR_S, OptionType.PUT) == pytest.approx(
                    legacy_put, abs=1e-12)
            assert legacy_bt._get_leg_iv(
                ANCHOR_IV, kc, ANCHOR_S, OptionType.CALL) == pytest.approx(
                    legacy_call, abs=1e-12)
            assert flat_bt._get_leg_iv(
                ANCHOR_IV, kp, ANCHOR_S, OptionType.PUT) == pytest.approx(
                    ANCHOR_IV, abs=1e-12)
            assert flat_bt._get_leg_iv(
                ANCHOR_IV, kc, ANCHOR_S, OptionType.CALL) == pytest.approx(
                    ANCHOR_IV, abs=1e-12)

    def test_call_side_is_deliberately_unchanged(self, skew_bt, legacy_bt):
        # The same regression measured the CALL slope at 0.064 / 0.112 /
        # -0.089 — sign-unstable across symbols, so there is no calibrated
        # slope to ship. Documented in BacktestConfig.skew_slope_per_pct_otm.
        for pct in (2.0, 5.0, 10.0):
            kc = ANCHOR_S * float(np.exp(pct / 100.0))
            assert skew_bt._get_leg_iv(
                ANCHOR_IV, kc, ANCHOR_S, OptionType.CALL) == pytest.approx(
                    legacy_bt._get_leg_iv(ANCHOR_IV, kc, ANCHOR_S,
                                          OptionType.CALL), abs=1e-12)

    def test_itm_and_atm_strikes_are_untouched(self, skew_bt):
        # The term is one-sided: only OTM distance steepens.
        assert skew_bt._get_leg_iv(
            ANCHOR_IV, ANCHOR_S, ANCHOR_S, OptionType.PUT) == pytest.approx(
                ANCHOR_IV, abs=1e-12)
        itm = ANCHOR_S * 1.05
        assert skew_bt._get_leg_iv(
            ANCHOR_IV, itm, ANCHOR_S, OptionType.PUT) == pytest.approx(
                ANCHOR_IV, abs=1e-12)

    def test_skew_reaches_the_condor_reprice_path(self, gap_bt):
        # _get_leg_iv is the single per-leg IV input point; prove the slope
        # actually moves a whole structure, not just a leg helper.
        flat = _bt(entry_dte=21, iv_floor=0.10, skew_slope_per_pct_otm=0.0)
        pos = _condor()
        skewed_val = gap_bt._reprice_position(pos, 640.0, 7, None)
        flat_val = flat._reprice_position(pos, 640.0, 7, None)
        assert skewed_val != pytest.approx(flat_val)


# ===========================================================================
# (d) config resolution — r20b style: None -> load_settings() value.
# ===========================================================================

class TestConfigResolution:
    def test_bare_engine_resolves_from_config_yaml(self, bt, yaml_backtest):
        assert bt._commission == pytest.approx(
            yaml_backtest.commission_per_contract)
        assert bt._regulatory_fees == pytest.approx(
            yaml_backtest.regulatory_fees_per_contract)
        assert bt._skew_slope_per_pct_otm == pytest.approx(
            yaml_backtest.skew_slope_per_pct_otm)

    def test_config_yaml_matches_the_model_defaults(self, yaml_backtest):
        # No third fork: config.yaml and the pydantic model must agree, so a
        # module constructing BacktestConfig() bare researches the same costs.
        d = BacktestConfig()
        assert yaml_backtest.commission_per_contract == pytest.approx(
            d.commission_per_contract)
        assert yaml_backtest.regulatory_fees_per_contract == pytest.approx(
            d.regulatory_fees_per_contract)
        assert yaml_backtest.skew_slope_per_pct_otm == pytest.approx(
            d.skew_slope_per_pct_otm)

    def test_shipped_defaults_are_the_measured_values(self):
        d = BacktestConfig()
        assert d.commission_per_contract == pytest.approx(0.65)
        # 0.9168 measured all-in per contract-leg - 0.65 commission line
        assert d.regulatory_fees_per_contract == pytest.approx(0.2668)
        assert d.skew_slope_per_pct_otm == pytest.approx(0.0736)
        assert d.skew_slope_per_pct_otm > 0.0, (
            "the shipped default must be the measured value, not the "
            "backward-compat 0"
        )

    def test_explicit_values_always_win(self):
        e = _bt(commission_per_contract=1.11,
                regulatory_fees_per_contract=0.22,
                skew_slope_per_pct_otm=0.05)
        assert e._commission == pytest.approx(1.11)
        assert e._regulatory_fees == pytest.approx(0.22)
        assert e._cost_per_contract_leg == pytest.approx(1.33)
        assert e._skew_slope_per_pct_otm == pytest.approx(0.05)

    def test_explicit_zero_is_honoured_not_treated_as_unset(self):
        # 0.0 is the documented escape hatch; `if explicit is not None`
        # must not swallow it.
        e = _bt(regulatory_fees_per_contract=0.0, skew_slope_per_pct_otm=0.0,
                commission_per_contract=0.0)
        assert e._regulatory_fees == 0.0
        assert e._skew_slope_per_pct_otm == 0.0
        assert e._cost_per_contract_leg == 0.0

    def test_no_config_yaml_falls_back_to_model_defaults(self, monkeypatch):
        import ait.config.settings as settings_mod

        def _boom(*a, **k):
            raise FileNotFoundError("no config.yaml")

        monkeypatch.setattr(settings_mod, "load_settings", _boom)
        e = _bt()
        d = BacktestConfig()
        assert e._commission == pytest.approx(d.commission_per_contract)
        assert e._regulatory_fees == pytest.approx(d.regulatory_fees_per_contract)
        assert e._skew_slope_per_pct_otm == pytest.approx(d.skew_slope_per_pct_otm)

    def test_partial_settings_stub_degrades_to_model_defaults(self):
        from types import SimpleNamespace

        stub = SimpleNamespace(backtest=SimpleNamespace())
        e = _bt(settings=stub)
        d = BacktestConfig()
        assert e._commission == pytest.approx(d.commission_per_contract)
        assert e._regulatory_fees == pytest.approx(d.regulatory_fees_per_contract)
        assert e._skew_slope_per_pct_otm == pytest.approx(d.skew_slope_per_pct_otm)

    def test_settings_object_is_threaded_through(self):
        from types import SimpleNamespace

        stub = SimpleNamespace(backtest=SimpleNamespace(
            commission_per_contract=0.80,
            regulatory_fees_per_contract=0.40,
            skew_slope_per_pct_otm=0.12))
        e = _bt(settings=stub)
        assert e._commission == pytest.approx(0.80)
        assert e._regulatory_fees == pytest.approx(0.40)
        assert e._skew_slope_per_pct_otm == pytest.approx(0.12)
