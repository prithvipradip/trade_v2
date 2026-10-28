"""W109b: the budget-ladder study's pure logic (rule, dose-response, wiring checks, verdict, rendering).

Fabricated rows only: no data store, no walk-forward run. The rule is pre-registered in PLAN.md
("W109b AS PLANNED"); these tests pin that the script implements exactly that rule.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
_spec = importlib.util.spec_from_file_location("budget_ladder_study", ROOT / "scripts" / "budget_ladder_study.py")
bl = importlib.util.module_from_spec(_spec)
sys.modules["budget_ladder_study"] = bl
_spec.loader.exec_module(bl)

POOL = 5000.0
BASE_NETS = {f"w{i}": 10.0 for i in range(10)}


def row(n=60, pf=0.5, net=-100.0, npd=-0.10, dd=1600.0, fs=0.20, nets=None, wing=3.0, loss=215.0, single=215.0):
    return {"n": n, "pf": pf, "net": net, "net_per_position_day": npd, "max_dd_dollars": dd, "pool": POOL,
            "friction_share": fs, "window_nets": nets if nets is not None else dict(BASE_NETS),
            "mean_wing": wing, "mean_max_loss": loss, "max_single_loss": single}


def better(**kw):
    """A candidate that beats the baseline row on every criterion unless overridden."""
    args = dict(pf=1.30, net=300.0, npd=0.20, dd=600.0, fs=0.12, nets={k: 25.0 for k in BASE_NETS},
                wing=6.0, loss=430.0, single=480.0)
    args.update(kw)
    return row(**args)


BASE = row()


class TestTheVariantTable:
    def test_roles(self):
        v = bl.VARIANTS
        assert (v["base"].budget, v["base"].wing_k, v["base"].lots) == (0.05, 1.6, 1)
        assert bl.candidates() == ["b075", "b10"]
        assert (v["b15"].role, v["b20"].role, v["ref20_k055"].role) == ("above_cap", "above_cap", "reference")
        assert (v["ref20_k055"].budget, v["ref20_k055"].wing_k) == (0.20, 0.55)
        assert all(x.lots == 1 for x in v.values())
        assert (v["sanity_default"].budget, v["sanity_default"].wing_k) == (0.05, None)

    def test_the_ladder_is_the_five_budgets_in_order(self):
        assert [bl.VARIANTS[n].budget for n in bl.LADDER] == [0.05, 0.075, 0.10, 0.15, 0.20]
        assert all(bl.VARIANTS[n].wing_k == 1.6 for n in bl.LADDER)

    def test_the_script_implements_the_numbers_pre_registered_in_plan_md(self):
        plan = (ROOT / "PLAN.md").read_text(encoding="utf-8")
        entry = plan[plan.index("W109b AS PLANNED"):]
        for text in ("n >= 0.85 x base n", "PF >= base PF + 0.10", "<= 15% of the pool", "<= 1.25 x base's",
                     "<= 10% of the pool", ">= 60% of traded windows", "<= base + 10 points", "net > 0 and PF >= 1.0",
                     "`friction_stress_multiplier` 2.0", ">= 3 of the 4 adjacent pairs", "per-trade loss cap of **10%**"):
            assert text in entry, text
        assert (bl.MIN_N_RATIO, bl.PF_MARGIN, bl.DD_POOL_CEILING_PCT, bl.DD_BASE_MULT, bl.MAX_SINGLE_LOSS_POOL_PCT,
                bl.MIN_WINDOW_SHARE, bl.FRICTION_MARGIN, bl.MIN_ABSOLUTE_PF, bl.MONOTONE_MIN_PAIRS,
                bl.STRESS_MULTIPLIER, bl.MIN_TRADES) == (0.85, 0.10, 15.0, 1.25, 10.0, 0.60, 0.10, 1.0, 3, 2.0, 30)


class TestTheCriteria:
    def check(self, cand):
        paired = bl.paired_window_stats(BASE["window_nets"], cand["window_nets"])
        return bl.criteria(cand, BASE, paired)

    def test_a_better_profitable_candidate_passes_everything(self):
        assert all(self.check(better()).values())

    @pytest.mark.parametrize("override, failing", [
        ({"n": 29}, "1_n"),
        ({"n": 50}, "1_n"),                                    # 50 < 0.85 x 60
        ({"pf": 0.55}, "2_pf_and_rate"),                       # < base + 0.10 (and not >= 1.0 either)
        ({"npd": -0.50}, "2_pf_and_rate"),
        ({"dd": 800.0}, "3_risk"),                             # 16% of the pool
        ({"single": 520.0}, "3_risk"),                         # a $520 largest loss: 10.4% of the pool
        ({"nets": {k: (25.0 if i < 5 else 1.0) for i, k in enumerate(BASE_NETS)}}, "4_windows"),
        ({"nets": {k: 10.0 for k in BASE_NETS}}, "4_windows"),
        ({"fs": 0.40}, "5_friction"),
        ({"fs": None}, "5_friction"),
    ])
    def test_each_criterion_can_fail_alone(self, override, failing):
        res = self.check(better(**override))
        assert not res[failing]
        # criteria 2 and 6 interact on PF; check the named one failed and nothing unrelated did
        others = [k for k, ok in res.items() if not ok and k != failing]
        assert not others or (failing == "2_pf_and_rate" and others == ["6_absolute"])

    def test_the_drawdown_cannot_exceed_the_base_multiple_even_under_the_ceiling(self):
        base = row(dd=300.0)                                   # 6% of the pool
        cand = better(dd=500.0)                                # 10%: under 15% but > 1.25 x 6%
        paired = bl.paired_window_stats(base["window_nets"], cand["window_nets"])
        assert not bl.criteria(cand, base, paired)["3_risk"]

    def test_a_candidate_that_beats_a_worse_base_but_loses_money_is_refused(self):
        """The absolute criterion: PF 0.73 beats PF 0.41 and still loses money."""
        res = self.check(better(pf=0.73, net=-848.0, npd=-0.10, dd=700.0))
        assert res["2_pf_and_rate"] and not res["6_absolute"]

    def test_pf_exactly_one_with_positive_net_is_absolute_ok_but_a_zero_net_is_not(self):
        assert self.check(better(pf=1.0, net=1.0))["6_absolute"]
        assert not self.check(better(pf=1.2, net=0.0))["6_absolute"]

    def test_the_profit_factor_clause_stands_on_its_own(self):
        """net > 0 implies PF > 1 on real trades; the clause still refuses an inconsistent row."""
        assert not self.check(better(pf=0.9, net=50.0))["6_absolute"]

    def test_the_boundaries_are_inclusive(self):
        at = better(n=51, pf=1.30, dd=750.0, single=500.0, fs=0.30)
        res = self.check(at)
        assert res["1_n"] and res["3_risk"] and res["5_friction"]

    def test_the_stress_criteria_are_1_2_3_and_6_only(self):
        c = bl.criteria_stress(better(), BASE)
        assert set(c) == {"1_n", "2_pf_and_rate", "3_risk", "6_absolute"} and all(c.values())


def ladder(pfs=(0.4, 0.7, 0.9, 1.3, 1.6), wings=(3.0, 4.0, 6.0, 8.0, 11.0), losses=(215, 300, 430, 600, 850), **per):
    rows = {}
    for name, pf, w, ml in zip(bl.LADDER, pfs, wings, losses):
        rows[name] = row(pf=pf, wing=w, loss=float(ml), **per.get(name, {}))
    rows["sanity_default"] = dict(rows["base"])
    return rows


def both(rows):
    return {18: {k: dict(v) for k, v in rows.items()}, 25: {k: dict(v) for k, v in rows.items()}}


class TestDoseResponse:
    def test_a_rising_ladder_is_monotone(self):
        assert bl.monotone(ladder()) == (True, 4, 4)

    def test_one_dip_is_still_monotone_two_are_not(self):
        assert bl.monotone(ladder(pfs=(0.4, 0.7, 0.6, 1.3, 1.6)))[0]          # 3 of 4 up
        assert not bl.monotone(ladder(pfs=(0.4, 0.7, 0.6, 0.5, 1.6)))[0]      # 2 of 4 up

    def test_a_missing_rung_cannot_be_monotone(self):
        rows = ladder()
        del rows["b15"]
        assert bl.monotone(rows) == (False, 0, 0)


class TestWiring:
    def test_a_clean_ladder_has_no_void_reason(self):
        void, flags = bl.wiring(both(ladder()), True)
        assert void == [] and flags == []

    def test_a_flat_ladder_is_void_the_budget_is_not_wired(self):
        void, _ = bl.wiring(both(ladder(wings=(3.0,) * 5)), True)
        assert any("budget is not wired" in v for v in void)

    def test_a_ladder_whose_wing_dips_is_void(self):
        void, _ = bl.wiring(both(ladder(wings=(3.0, 4.0, 3.5, 8.0, 11.0))), True)
        assert any("non-decreasing" in v for v in void)

    def test_a_ladder_whose_max_loss_dips_is_void(self):
        void, _ = bl.wiring(both(ladder(losses=(215, 300, 250, 600, 850))), True)
        assert any("max loss" in v for v in void)

    def test_the_sanity_variant_must_reproduce_the_baseline(self):
        rows = both(ladder())
        rows[18]["sanity_default"]["net"] = 5.0
        void, _ = bl.wiring(rows, True)
        assert any("does not reproduce" in v for v in void)

    def test_a_missing_friction_model_voids_the_study(self):
        void, _ = bl.wiring(both(ladder()), False)
        assert any("friction" in v for v in void)

    def test_identical_rungs_are_flagged_as_inert(self):
        rows = both(ladder())
        for d in rows:
            rows[d]["b075"].update({k: rows[d]["b10"][k] for k in ("n", "net", "pf", "max_dd_dollars")})
        _, flags = bl.wiring(rows, True)
        assert any("inert rung" in f and "b075" in f and "b10" in f for f in flags)

    def test_the_reference_arm_is_outside_the_ladder_checks(self):
        rows = both(ladder())
        for d in rows:
            rows[d]["ref20_k055"] = row(wing=7.0, loss=500.0)
        void, _ = bl.wiring(rows, True)
        assert void == []


def stressed(names, ok=True):
    out = {}
    for d in (18, 25):
        out[d] = {"base": row()}
        for n in names:
            out[d][n] = better() if ok else better(pf=0.4, net=-50.0)
    return out


def with_candidates(**cands):
    rows = both(ladder())
    for d in rows:
        for n, r in cands.items():
            rows[d][n] = dict(r)
    return rows


class TestTheVerdict:
    def test_the_expected_outcome_nothing_within_the_cap_beats_5pct(self):
        rows = with_candidates(b075=row(pf=0.6, net=-500.0, npd=-0.2, wing=4.0, loss=300.0, single=300.0),
                               b10=row(pf=0.73, net=-848.0, npd=-0.12, wing=6.0, loss=430.0, single=430.0, dd=1580.0))
        res = bl.verdict(rows, {}, True)
        assert res["verdict"] == "NO BUDGET WITHIN THE CAP BEATS 5%" and res["supported"] == []

    def test_a_monotone_ladder_with_a_supported_candidate(self):
        rows = with_candidates(b10=better())
        res = bl.verdict(rows, stressed(["b10"]), True)
        assert res["verdict"].startswith("BUDGET RAISE SUPPORTED to 10.0%") and res["supported"] == ["b10"]

    def test_the_highest_rate_wins_and_a_tie_goes_to_the_lower_budget(self):
        both_ok = with_candidates(b075=better(npd=0.30, wing=4.0, loss=300.0, single=300.0), b10=better(npd=0.30))
        res = bl.verdict(both_ok, stressed(["b075", "b10"]), True)
        assert "7.5%" in res["verdict"]
        faster = with_candidates(b075=better(npd=0.25, wing=4.0, loss=300.0, single=300.0), b10=better(npd=0.35))
        assert "10.0%" in bl.verdict(faster, stressed(["b075", "b10"]), True)["verdict"]

    def test_a_pass_on_a_non_monotone_ladder_is_single_point(self):
        rows = both(ladder(pfs=(0.4, 1.2, 1.3, 0.5, 0.45)))      # up, up, down, down: 2 of 4
        for d in rows:
            rows[d]["b10"] = better()
        res = bl.verdict(rows, stressed(["b10"]), True)
        assert res["verdict"] == "SINGLE-POINT" and res["supported"] == ["b10"]

    def test_a_survivor_that_fails_the_stressed_repeat_is_not_supported(self):
        res = bl.verdict(with_candidates(b10=better()), stressed(["b10"], ok=False), True)
        assert res["verdict"] == "NO BUDGET WITHIN THE CAP BEATS 5%" and any("criterion 7" in n for n in res["notes"])

    def test_skipping_the_stress_repeat_never_yields_a_positive_verdict(self):
        res = bl.verdict(with_candidates(b10=better()), {}, True)
        assert res["verdict"] == "NO BUDGET WITHIN THE CAP BEATS 5%" and any("NOT stress-tested" in n for n in res["notes"])

    def test_no_stress_rows_means_no_survivor_passes_criterion_7(self):
        assert bl.stage_b({}, ["b10"]) == {"b10": False}

    def test_a_candidate_must_pass_at_both_dte(self):
        rows = with_candidates(b10=better())
        rows[25]["b10"] = row(pf=0.9, net=-5.0, npd=-0.1, wing=6.0, loss=430.0, single=430.0)
        assert bl.verdict(rows, stressed(["b10"]), True)["verdict"] == "NO BUDGET WITHIN THE CAP BEATS 5%"

    def test_rungs_above_the_cap_and_the_reference_can_never_be_supported(self):
        rows = with_candidates(b15=better(wing=8.0, loss=600.0, single=600.0), b20=better(wing=11.0, loss=850.0, single=850.0),
                               ref20_k055=better())
        stage = bl.stage_a(rows)
        assert set(stage) == {"b075", "b10"}
        assert bl.verdict(rows, {}, True)["supported"] == []

    def test_a_void_wiring_check_overrides_everything(self):
        res = bl.verdict(with_candidates(b10=better()), stressed(["b10"]), False)
        assert res["verdict"] == "VOID" and res["supported"] == []


class TestRendering:
    def test_the_report_marks_the_rows_outside_the_cap_and_carries_the_verdict(self):
        rows = both(ladder())
        for d in rows:
            for n, r in rows[d].items():
                v = bl.VARIANTS[n]
                r.update({"variant": n, "role": v.role, "budget": v.budget, "wing_k": v.wing_k, "win_rate_pct": 50.0,
                          "dd_pool_pct": 4.0, "worst_trade": -100.0, "mean_credit": 0.9})
        res = bl.verdict(rows, {}, True)
        md = bl.render(rows, {}, res)
        assert "**Verdict:" in md and "Entry DTE 18" in md and "Entry DTE 25" in md
        assert "above_cap (outside the operator's cap: informational)" in md and "read RELATIVE effects only" in md
