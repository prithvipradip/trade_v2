"""W109: the narrow-wing and multi-lot study's pure logic (rule, statistics, wiring checks, rendering).

No data store and no walk-forward run: rows are fabricated. The decision rule is pre-registered in PLAN.md
("W109 AS PLANNED"); these tests pin that the script implements exactly that rule, so the verdict of the
real run cannot be bent after seeing it.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("narrow_wing_study", ROOT / "scripts" / "narrow_wing_study.py")
nw = importlib.util.module_from_spec(_spec)
sys.modules["narrow_wing_study"] = nw
_spec.loader.exec_module(nw)

POOL = 5000.0


def row(n=60, pf=1.0, net=100.0, npd=0.10, dd=200.0, fs=0.20, nets=None, gt1=0.0, cmax=1, wing=3.0):
    return {"n": n, "pf": pf, "net": net, "net_per_position_day": npd, "max_dd_dollars": dd, "pool": POOL,
            "friction_share": fs, "window_nets": nets if nets is not None else {}, "lots_gt1_pct": gt1,
            "contracts_max": cmax, "mean_wing": wing}


BASE_NETS = {f"w{i}": 10.0 for i in range(10)}


def better(**kw):
    """A candidate row that beats the baseline row on every criterion unless overridden."""
    args = dict(pf=1.30, net=300.0, npd=0.20, dd=220.0, fs=0.22,
                nets={k: 25.0 for k in BASE_NETS}, wing=2.0)
    args.update(kw)
    return row(**args)


class TestTheVariantTable:
    def test_the_baseline_is_the_shipped_configuration(self):
        v = nw.VARIANTS[nw.BASELINE]
        assert (v.budget, v.wing_k, v.lots, v.role) == (0.05, 1.6, 1, "base")

    def test_the_sanity_variant_differs_only_by_the_config_default_wing_k(self):
        b, s = nw.VARIANTS["base"], nw.VARIANTS["sanity_default"]
        assert (s.budget, s.lots, s.wing_k) == (b.budget, b.lots, None) and s.role == "sanity"

    def test_the_reference_is_never_a_candidate(self):
        assert "ref20_k160_1" not in nw.candidates() and nw.VARIANTS["ref20_k160_1"].budget == 0.20
        assert {nw.BASELINE, nw.SANITY} .isdisjoint(nw.candidates())

    def test_every_multi_lot_variant_has_a_one_lot_twin_with_the_same_budget_and_k(self):
        multi = [n for n, v in nw.VARIANTS.items() if v.lots > 1]
        assert multi == ["b5_k010_3", "b10_k015_3", "b10_k010_3"]
        for n in multi:
            twin = nw.twin_of(n)
            assert twin and nw.VARIANTS[twin].lots == 1
            assert (nw.VARIANTS[twin].budget, nw.VARIANTS[twin].wing_k) == (nw.VARIANTS[n].budget, nw.VARIANTS[n].wing_k)
        assert nw.twin_of("b10_k030_1") is None

    def test_the_script_implements_the_numbers_pre_registered_in_plan_md(self):
        plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
        entry = plan[plan.index("W109 AS PLANNED"):]
        for text in ("n >= 0.85 x base n", "PF >= base PF + 0.10", "<= 8% of the pool", "<= 1.25 x base's",
                     ">= 60% of the walk-forward windows", "<= base + 10 points", "`friction_stress_multiplier` 2.0",
                     ">= 25% of its trades", "n >= 30"):
            assert text in entry, text
        assert (nw.MIN_N_RATIO, nw.PF_MARGIN, nw.DD_POOL_CEILING_PCT, nw.DD_BASE_MULT, nw.MIN_WINDOW_SHARE,
                nw.FRICTION_MARGIN, nw.MIN_MULTILOT_SHARE_PCT, nw.STRESS_MULTIPLIER, nw.MIN_TRADES) == (
            0.85, 0.10, 8.0, 1.25, 0.60, 0.10, 25.0, 2.0, 30)


class TestStatistics:
    def test_trade_stats(self):
        trades = [{"contracts": 1, "wing_width": 3, "max_loss_per_contract": 200, "entry_price": 0.9,
                   "mid_credit": 1.0, "entry_commission": 2.6},
                  {"contracts": 3, "wing_width": 3, "max_loss_per_contract": 190, "entry_price": 0.95,
                   "mid_credit": 1.0, "entry_commission": 7.8}]
        s = nw.trade_stats(trades)
        assert s["mean_wing"] == 3 and s["contracts_max"] == 3 and s["lots_gt1_pct"] == 50.0
        assert s["mean_lots"] == 2.0
        gross = 1.0 * 100 * 1 + 1.0 * 100 * 3
        cost = 0.1 * 100 * 1 + 0.05 * 100 * 3 + 2 * (2.6 + 7.8)
        assert s["friction_share"] == pytest.approx(cost / gross, abs=1e-4)

    def test_no_trades(self):
        assert nw.trade_stats([])["contracts_max"] == 0 and nw.trade_stats([])["friction_share"] is None

    def test_window_nets_skip_windows_without_trades(self):
        got = nw.window_nets([("w1", [{"pnl": 10}, {"pnl": -4}]), ("w2", []), ("w3", [{"pnl": 5}])])
        assert got == {"w1": 6.0, "w3": 5.0}

    def test_the_paired_comparison_on_fixed_numbers(self):
        base = {"w1": 100.0, "w2": -50.0, "w3": 20.0}
        var = {"w1": 120.0, "w2": -10.0, "w3": 10.0}
        p = nw.paired_window_stats(base, var)
        assert p["windows"] == 3 and p["share_ge"] == pytest.approx(2 / 3, abs=1e-4) and p["point"] == 50.0
        assert p == nw.paired_window_stats(base, var)                      # seeded: reproducible
        same = nw.paired_window_stats(base, dict(base))
        assert (same["share_ge"], same["point"], same["ci_lo"], same["ci_hi"]) == (1.0, 0.0, 0.0, 0.0)

    def test_a_dominating_variant_has_a_positive_interval_and_a_dominated_one_a_negative_one(self):
        base = {f"w{i}": 10.0 for i in range(12)}
        up = nw.paired_window_stats(base, {k: 40.0 for k in base})
        down = nw.paired_window_stats(base, {k: -20.0 for k in base})
        assert up["ci_lo"] > 0 and down["ci_hi"] < 0 and up["share_ge"] == 1.0 and down["share_ge"] == 0.0

    def test_a_window_only_one_side_traded_counts(self):
        p = nw.paired_window_stats({"w1": 10.0}, {"w1": 10.0, "w2": 5.0})
        assert p["windows"] == 2 and p["point"] == 5.0


class TestTheCriteria:
    base = row(nets=BASE_NETS)

    def check(self, cand):
        paired = nw.paired_window_stats(self.base["window_nets"], cand["window_nets"])
        return nw.criteria(cand, self.base, paired)

    def test_a_better_candidate_passes_everything(self):
        assert all(self.check(better()).values())

    @pytest.mark.parametrize("override, failing", [
        ({"n": 29}, "1_n"),
        ({"n": 50}, "1_n"),                                   # 50 < 0.85 x 60
        ({"pf": 1.05}, "2_pf_and_rate"),                      # PF < base + 0.10
        ({"npd": 0.05}, "2_pf_and_rate"),
        ({"dd": 410.0}, "3_drawdown"),                        # 8.2% of the pool
        ({"dd": 260.0}, "3_drawdown"),                        # 5.2%: under 8% but > 1.25 x base's 4%
        ({"nets": {k: (25.0 if i < 5 else 1.0) for i, k in enumerate(BASE_NETS)}}, "4_windows"),
        ({"nets": {k: 10.0 for k in BASE_NETS}}, "4_windows"),   # ties but a zero point estimate
        ({"fs": 0.40}, "5_friction"),
        ({"fs": None}, "5_friction"),
    ])
    def test_each_criterion_can_fail_alone(self, override, failing):
        res = self.check(better(**override))
        assert not res[failing]
        assert [k for k, ok in res.items() if not ok] == [failing]

    def test_the_boundaries_are_inclusive(self):
        at = better(n=51, pf=1.10, npd=0.10, dd=250.0, fs=0.30)
        res = self.check(at)
        assert res["1_n"] and res["2_pf_and_rate"] and res["3_drawdown"] and res["5_friction"]

    def test_criteria_1_to_3_ignore_the_window_and_friction_criteria(self):
        c = nw.criteria_1_to_3(better(), self.base)
        assert set(c) == {"1_n", "2_pf_and_rate", "3_drawdown"} and all(c.values())


def sets(dte_rows: dict[int, dict[str, dict]]):
    return dte_rows


def full(rows_for_candidates: dict[str, dict], dtes=(18, 25)):
    out = {}
    for d in dtes:
        out[d] = {nw.BASELINE: row(nets=BASE_NETS), nw.SANITY: row(nets=BASE_NETS)}
        out[d].update(rows_for_candidates)
    return out


class TestWiring:
    def test_a_clean_run_has_no_void_reason(self):
        rows = full({"b5_k010_1": better(wing=2.0), "b10_k160_1": better(wing=6.0)})
        void, flags = nw.wiring(rows, True)
        assert void == []

    def test_the_sanity_variant_must_reproduce_the_baseline(self):
        rows = full({})
        rows[18][nw.SANITY] = row(net=101.0, nets=BASE_NETS)
        void, _ = nw.wiring(rows, True)
        assert any("does not reproduce" in v for v in void)

    def test_a_missing_friction_model_voids_the_study(self):
        void, _ = nw.wiring(full({}), False)
        assert any("friction" in v for v in void)

    def test_wing_k_must_narrow_and_the_budget_must_widen(self):
        rows = full({"b5_k010_1": better(wing=3.0), "b10_k160_1": better(wing=2.5)})
        void, _ = nw.wiring(rows, True)
        assert any("b5_k010_1" in v and "wing_k not wired" in v for v in void)
        assert any("b10_k160_1" in v and "budget is not wired" in v for v in void)

    def test_a_three_lot_variant_that_never_took_a_second_lot_is_flagged(self):
        rows = full({"b10_k010_3": better(cmax=1, wing=2.0)})
        _, flags = nw.wiring(rows, True)
        assert any("multi-lot never triggered" in f and "b10_k010_3" in f for f in flags)

    def test_identical_results_with_different_parameters_are_flagged_as_an_inert_knob(self):
        same = {"n": 60, "net": 100.0, "pf": 1.0, "dd": 200.0}
        rows = full({"b5_k015_1": better(**same), "b5_k010_1": better(**same, wing=2.0)})
        _, flags = nw.wiring(rows, True)
        assert any("b5_k010_1" in f and "b5_k015_1" in f and "inert knob" in f for f in flags)

    def test_the_baseline_sanity_pair_is_not_flagged_as_inert(self):
        _, flags = nw.wiring(full({}), True)
        assert not any("inert knob" in f for f in flags)


class TestTheVerdict:
    def run(self, cands, stress=None, friction_ok=True, dtes=(18, 25)):
        rows = full(cands, dtes)
        return nw.verdict(rows, stress if stress is not None else {}, friction_ok)

    def stressed(self, names, ok=True, dtes=(18, 25)):
        out = {}
        for d in dtes:
            out[d] = {nw.BASELINE: row(nets=BASE_NETS)}
            for n in names:
                out[d][n] = better() if ok else better(pf=0.5)
        return out

    def test_nothing_beating_the_baseline_is_a_valid_verdict(self):
        res = self.run({"b10_k030_1": row(nets=BASE_NETS, wing=6.0), "b5_k010_1": row(nets=BASE_NETS, wing=2.0),
                        "b10_k160_1": row(nets=BASE_NETS, wing=7.0)})
        assert res["verdict"] == "NOTHING BEATS BASELINE" and res["worth"] == []

    def test_a_one_lot_winner_that_survives_the_stress_is_narrow_only(self):
        res = self.run({"b5_k010_1": better(wing=2.0), "b10_k160_1": row(nets=BASE_NETS, wing=7.0)},
                       stress=self.stressed(["b5_k010_1"]))
        assert res["verdict"] == "NARROW-ONLY" and res["worth"] == ["b5_k010_1"] and res["phase2"] == []

    def test_a_winner_must_pass_at_both_dte_levels(self):
        rows = full({"b5_k010_1": better(wing=2.0)})
        rows[25]["b5_k010_1"] = row(nets=BASE_NETS, wing=2.0)           # no better at DTE 25
        res = nw.verdict(rows, self.stressed(["b5_k010_1"]), True)
        assert res["verdict"] == "NOTHING BEATS BASELINE"

    def test_a_survivor_that_fails_the_stressed_repeat_is_not_worth_pursuing(self):
        res = self.run({"b5_k010_1": better(wing=2.0)}, stress=self.stressed(["b5_k010_1"], ok=False))
        assert res["verdict"] == "NOTHING BEATS BASELINE"
        assert any("criterion 6" in n for n in res["notes"])

    def test_skipping_the_stress_repeat_never_yields_a_positive_verdict(self):
        res = self.run({"b5_k010_1": better(wing=2.0)}, stress={})
        assert res["verdict"] == "NOTHING BEATS BASELINE"
        assert any("NOT stress-tested" in n for n in res["notes"])

    def test_phase_2_needs_the_multi_lot_variant_to_beat_its_one_lot_twin(self):
        multi = better(wing=2.0, net=500.0, dd=240.0, gt1=60.0, cmax=3)
        twin = better(wing=2.0, net=300.0, dd=220.0)
        res = self.run({"b10_k010_3": multi, "b10_k010_1": twin, "b5_k010_1": row(nets=BASE_NETS, wing=2.0)},
                       stress=self.stressed(["b10_k010_3", "b10_k010_1"]))
        assert res["verdict"] == "PHASE 2 JUSTIFIED" and res["phase2"] == ["b10_k010_3"]

    def test_a_multi_lot_winner_that_does_not_beat_its_twin_is_narrow_only(self):
        multi = better(wing=2.0, net=250.0, dd=240.0, gt1=60.0, cmax=3)
        twin = better(wing=2.0, net=300.0, dd=220.0)
        res = self.run({"b10_k010_3": multi, "b10_k010_1": twin},
                       stress=self.stressed(["b10_k010_3", "b10_k010_1"]))
        assert res["verdict"] == "NARROW-ONLY" and res["phase2"] == []
        assert any("does not beat the 1-lot twin" in n for n in res["notes"])

    def test_a_tie_with_the_one_lot_twin_is_not_a_win(self):
        multi = better(wing=2.0, net=300.0, dd=220.0, gt1=60.0, cmax=3)
        twin = better(wing=2.0, net=300.0, dd=220.0)
        res = self.run({"b10_k010_3": multi, "b10_k010_1": twin},
                       stress=self.stressed(["b10_k010_3", "b10_k010_1"]))
        assert res["phase2"] == [] and any("does not beat the 1-lot twin" in n for n in res["notes"])

    def test_no_stress_rows_means_no_survivor_passes_criterion_6(self):
        assert nw.stage_b({}, ["b5_k010_1"]) == {"b5_k010_1": False}

    def test_a_multi_lot_winner_that_rarely_used_the_extra_lots_is_not_phase_2(self):
        multi = better(wing=2.0, net=500.0, dd=240.0, gt1=10.0, cmax=3)
        twin = better(wing=2.0, net=300.0, dd=220.0)
        res = self.run({"b10_k010_3": multi, "b10_k010_1": twin},
                       stress=self.stressed(["b10_k010_3", "b10_k010_1"]))
        assert res["phase2"] == [] and any("took more than 1 lot" in n for n in res["notes"])

    def test_the_reference_arm_can_never_win(self):
        rows = full({"ref20_k160_1": better(wing=11.0)})
        assert "ref20_k160_1" not in nw.stage_a(rows)

    def test_a_void_wiring_check_overrides_everything(self):
        res = self.run({"b5_k010_1": better(wing=2.0)}, stress=self.stressed(["b5_k010_1"]), friction_ok=False)
        assert res["verdict"] == "VOID" and res["worth"] == []


class TestRendering:
    def test_the_report_carries_the_verdict_the_limits_and_every_variant(self):
        rows = full({"b5_k010_1": better(wing=2.0)})
        for d in rows:
            for n, r in rows[d].items():
                r.update({"variant": n, "budget": nw.VARIANTS[n].budget, "wing_k": nw.VARIANTS[n].wing_k,
                          "lots": nw.VARIANTS[n].lots, "win_rate_pct": 50.0, "dd_pool_pct": 4.0, "mean_max_loss": 200,
                          "mean_credit": 0.9, "mean_lots": 1.0})
        res = nw.verdict(rows, {}, True)
        md = nw.render(rows, {}, res)
        assert "**Verdict:" in md and "Entry DTE 18" in md and "Entry DTE 25" in md
        assert "b5_k010_1" in md and "read RELATIVE effects only" in md
