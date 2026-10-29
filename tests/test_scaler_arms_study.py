"""W108 phase 3c: the pre-registered rule of scripts/scaler_arms_study.py, and that each arm runs under
the settings it names (the walk-forward hands the frozen settings to its workers)."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import scaler_arms_study as sa  # noqa: E402


def _row(n=60, net=1000.0, pf=1.3, dd=2000.0, lots=2.0):
    return {"n": n, "net": net, "pf": pf, "max_dd_dollars": dd, "mean_lots": lots}


def _rows(base=None, cand=None, ref=None, sanity=None):
    ref = ref or _row(lots=1.0, net=400.0, dd=900.0)
    return {sa.BASELINE: base or _row(), sa.CANDIDATE: cand or _row(net=900.0, dd=1400.0, lots=1.4),
            sa.REFERENCE: ref, sa.SANITY: sanity or dict(ref)}


GOOD = {18: _rows(), 25: _rows()}


class TestWiring:
    def test_clean(self):
        assert sa.wiring(GOOD, True) == []

    @pytest.mark.parametrize("bad,why", [
        ({"sanity": _row(lots=1.0, net=401.0, dd=900.0)}, "differs from"),
        ({"base": _row(lots=1.0)}, "multi-lot never exercised"),
        ({"cand": _row(lots=2.0, net=900.0, dd=1400.0)}, "not below"),
        ({"cand": _row(n=59, net=900.0, dd=1400.0, lots=1.4)}, "scalers changed viability"),
    ])
    def test_each_check_voids(self, bad, why):
        v = sa.wiring({18: _rows(**bad), 25: _rows()}, True)
        assert v and why in v[0]

    def test_missing_friction_or_arm_voids(self):
        assert sa.wiring(GOOD, False)
        rows = _rows()
        del rows[sa.SANITY]
        assert "did not run" in sa.wiring({18: rows}, True)[0]


class TestRule:
    def test_risk_adjusted_margin_when_the_base_makes_money(self):
        base = _row(net=1000.0, dd=2000.0)                                # ratio 0.50
        assert sa.criteria(_row(net=600.0, dd=1090.0), base)["risk-adjusted better"]       # 0.550
        assert not sa.criteria(_row(net=600.0, dd=1100.0), base)["risk-adjusted better"]   # 0.545 < 0.55

    def test_when_the_base_loses_the_candidate_must_lose_less(self):
        base = _row(net=-500.0, dd=3000.0)
        assert sa.criteria(_row(net=-100.0, dd=1000.0), base)["risk-adjusted better"]
        assert not sa.criteria(_row(net=-600.0, dd=1000.0), base)["risk-adjusted better"]

    def test_drawdown_net_and_count(self):
        c = sa.criteria(_row(n=29, net=-1.0, dd=2500.0), _row())
        assert not c["n>=30"] and not c["dd<=base"] and not c["net>0"]

    def test_supported_needs_both_dte_and_the_stress_repeat(self):
        stress = {d: {sa.BASELINE: _row(), sa.CANDIDATE: _row(net=900.0, dd=1400.0)} for d in (18, 25)}
        assert sa.verdict(GOOD, stress, True)["verdict"] == "SCALERS SUPPORTED FOR MULTI-LOT"
        assert sa.verdict(GOOD, {}, True)["verdict"] == "NOT SUPPORTED"          # stress skipped
        bad_stress = {**stress, 25: {sa.BASELINE: _row(), sa.CANDIDATE: _row(net=-5.0, dd=1400.0)}}
        assert sa.verdict(GOOD, bad_stress, True)["verdict"] == "NOT SUPPORTED"
        one_dte_fails = {18: _rows(), 25: _rows(cand=_row(net=900.0, dd=2500.0, lots=1.4))}
        assert sa.verdict(one_dte_fails, stress, True)["verdict"] == "NOT SUPPORTED"
        assert sa.verdict({18: _rows(base=_row(lots=1.0))}, stress, True)["verdict"] == "VOID"


class TestArms:
    def test_arm_settings_set_the_pool_and_the_scalers_without_touching_the_base(self):
        from ait.config.settings import load_settings
        base = load_settings()
        s_on = sa.arm_settings(base, sa.ARMS[sa.CANDIDATE])
        s_off = sa.arm_settings(base, sa.ARMS[sa.BASELINE])
        assert s_on.position_sizing.scaled_strategies == ["iron_condor"]
        assert s_off.position_sizing.scaled_strategies == []
        assert s_on.account.capital == s_off.account.capital == sa.POOL
        assert base.account.capital != sa.POOL and base.position_sizing.scaled_strategies == []

    @pytest.mark.parametrize("name", list(sa.ARMS))
    def test_each_arm_runs_under_its_own_frozen_settings(self, name, monkeypatch, tmp_path):
        from ait.config.settings import load_settings
        seen = {}

        async def _fake_run_variant(nm, v, dte, bc, data, n_jobs, out, db_path, pool, stress=None):
            s = load_settings()                       # what the walk-forward (and its workers) will read
            seen.update(scaled=s.position_sizing.scaled_strategies, capital=s.account.capital,
                        lots=v.lots, budget=v.budget, pool=pool)
            return {"variant": nm}

        monkeypatch.setattr(sa, "run_variant", _fake_run_variant)
        asyncio.run(sa.run_arm(name, 18, load_settings(), {}, 1, tmp_path, tmp_path / "h.db"))
        arm = sa.ARMS[name]
        assert seen == {"scaled": ["iron_condor"] if arm.scaled else [], "capital": sa.POOL,
                        "lots": arm.lots, "budget": sa.BUDGET, "pool": sa.POOL}
