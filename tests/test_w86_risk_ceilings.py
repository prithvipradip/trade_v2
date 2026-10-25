"""W86a: the risk-family ceilings are raised, consistent, and linked.

Operator decision 2026-10-21: risk.max_position_risk_pct 0.10 -> 0.25, risk.max_daily_loss_pct 0.10 -> 0.20,
positions.max_position_pct 0.25 -> 0.30 (config VALUES unchanged). The same wave:

  * backtest.position_size_pct (research per-trade size) no longer outranges live's per-trade ceiling
  * the per-trade budget must also fit under the aggregate cap (gate 6b-2) and the single-symbol cap
    (gate 6c); above either, every entry is refused at runtime, silently, so it is a startup error
  * the GUIDE table of ranges is the model's (a drift test)
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from pydantic import ValidationError

from ait.config.settings import (
    BacktestConfig, PositionConfig, RiskConfig, Settings, load_settings, resolve_position_size_pct,
)

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "GUIDE.md"          # the table of ranges lives there (generated: scripts/gen_docs.py)


@pytest.fixture(scope="module")
def real_settings():
    return load_settings(str(ROOT / "config.yaml"), ignore_frozen=True)


def _bounds(model, field):
    fi = model.model_fields[field]
    ge = le = None
    for m in fi.metadata:
        ge = getattr(m, "ge", ge)
        le = getattr(m, "le", le)
    return ge, le


class TestCeilings:
    @pytest.mark.parametrize("model, field, top, above", [
        (RiskConfig, "max_position_risk_pct", 0.25, 0.26),
        (RiskConfig, "max_daily_loss_pct", 0.20, 0.21),
        (PositionConfig, "max_position_pct", 0.30, 0.31),
        (BacktestConfig, "position_size_pct", 0.25, 0.26),
    ])
    def test_the_new_top_is_accepted_and_just_above_is_not(self, model, field, top, above):
        assert getattr(model(**{field: top}), field) == top
        with pytest.raises(ValidationError):
            model(**{field: above})
        assert _bounds(model, field)[1] == top

    def test_the_floors_did_not_move(self):
        assert _bounds(RiskConfig, "max_position_risk_pct")[0] == 0.005
        assert _bounds(RiskConfig, "max_daily_loss_pct")[0] == 0.005
        assert _bounds(PositionConfig, "max_position_pct")[0] == 0.01
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss_pct=0.001)

    def test_the_ranges_that_stayed_still_reject(self):
        """tests/test_config.py pins 0.50 for both; they must still be out of range."""
        with pytest.raises(ValidationError):
            PositionConfig(max_position_pct=0.50)
        with pytest.raises(ValidationError):
            RiskConfig(max_daily_loss_pct=0.50)

    def test_the_shipped_values_are_unchanged(self, real_settings):
        s = real_settings
        assert (s.risk.max_position_risk_pct, s.risk.max_daily_loss_pct,
                s.positions.max_position_pct) == (0.05, 0.045, 0.05)
        assert (s.positions.max_portfolio_risk_pct, s.risk.max_symbol_concentration_pct) == (0.20, 0.20)
        again = load_settings("config.yaml", ignore_frozen=True)       # the file itself loads
        assert again.risk.max_position_risk_pct == 0.05

    def test_an_explicit_research_size_is_not_bounded_by_the_yaml_ceiling(self):
        """The ceiling bounds the config VALUE; a deliberate constructor argument (a sizing study) is
        not validated against it (resolve_config_value returns an explicit argument as given)."""
        assert resolve_position_size_pct(0.50, None) == 0.50


class TestLinks:
    @staticmethod
    def _with(real_settings, **over):
        d = real_settings.model_dump(exclude={"ibkr", "api_keys"})
        for k, v in over.items():
            section, field = k.split("__")
            d[section][field] = v
        return Settings(**d)

    def test_a_raise_that_clears_every_cap_is_accepted(self, real_settings):
        s = self._with(real_settings, risk__max_position_risk_pct=0.25, positions__max_position_pct=0.30,
                       positions__max_portfolio_risk_pct=0.50, risk__max_symbol_concentration_pct=0.50)
        assert s.risk.max_position_risk_pct == 0.25

    def test_equal_caps_are_allowed(self, real_settings):
        self._with(real_settings, risk__max_position_risk_pct=0.20, positions__max_position_pct=0.20)

    def test_above_the_aggregate_cap_is_a_startup_error(self, real_settings):
        with pytest.raises(ValueError, match="max_portfolio_risk_pct.*gate 6b-2"):
            self._with(real_settings, risk__max_position_risk_pct=0.25, positions__max_position_pct=0.30,
                       risk__max_symbol_concentration_pct=0.50)

    def test_above_the_symbol_concentration_cap_is_a_startup_error(self, real_settings):
        with pytest.raises(ValueError, match="max_symbol_concentration_pct.*gate 6c"):
            self._with(real_settings, risk__max_position_risk_pct=0.25, positions__max_position_pct=0.30,
                       positions__max_portfolio_risk_pct=0.50)

    def test_above_the_sizer_cap_is_still_a_startup_error(self, real_settings):
        with pytest.raises(ValueError, match="max_position_pct"):
            self._with(real_settings, risk__max_position_risk_pct=0.25, positions__max_portfolio_risk_pct=0.50,
                       risk__max_symbol_concentration_pct=0.50)

    def test_lowering_the_aggregate_cap_below_the_shipped_budget_is_refused(self, real_settings):
        with pytest.raises(ValueError, match="gate 6b-2"):
            self._with(real_settings, positions__max_portfolio_risk_pct=0.04)

    def test_the_breaker_below_one_max_loss_is_not_a_startup_error(self, real_settings):
        """Alert-only by design (orchestrator A2): a measurement of the real builder."""
        self._with(real_settings, risk__max_position_risk_pct=0.20, positions__max_position_pct=0.30,
                   risk__max_daily_loss_pct=0.05)


def _ceilings_block() -> str:
    m = re.search(r"<!-- BEGIN GENERATED: risk_ceilings -->\n(.*?)<!-- END GENERATED: risk_ceilings -->", GUIDE.read_text(), re.S)
    assert m, "docs/GUIDE.md has no generated block 'risk_ceilings'"
    return m.group(1)


class TestGuideIsTheModel:
    def test_the_ranges_table_matches_the_schema(self):
        sec = _ceilings_block()
        rows = re.findall(r"^\|\s*`([a-z_.]+)`\s*\|[^|]*\|\s*([0-9.]+) - ([0-9.]+)\s*\|", sec, re.M)
        models = {"risk": RiskConfig, "positions": PositionConfig, "backtest": BacktestConfig}
        assert {n for n, _, _ in rows} == {
            "risk.max_position_risk_pct", "risk.max_daily_loss_pct", "positions.max_position_pct",
            "positions.max_portfolio_risk_pct", "risk.max_symbol_concentration_pct",
            "backtest.position_size_pct"}
        for name, lo, hi in rows:
            section, field = name.split(".")
            ge, le = _bounds(models.get(section, RiskConfig), field)
            assert (float(lo), float(hi)) == (ge, le), f"GUIDE says {name} {lo}-{hi}, model {ge}-{le}"

    def test_the_shipped_values_column_is_the_config(self, real_settings):
        sec = _ceilings_block()
        for name, value in (("risk.max_position_risk_pct", real_settings.risk.max_position_risk_pct),
                            ("risk.max_daily_loss_pct", real_settings.risk.max_daily_loss_pct),
                            ("positions.max_position_pct", real_settings.positions.max_position_pct)):
            m = re.search(rf"`{re.escape(name)}`\s*\|\s*([0-9.]+)\s*\|", sec)
            assert m and float(m.group(1)) == value, name
