"""W66: per-leg option half-spread model in DOLLARS (friction model v2).

Why this exists. The legacy engine model charges ``half_spread_pct x leg_mid`` per leg,
fit on every strike of the nearest expiry (cheap illiquid wings inflate the percentage).
Measured on IB bid/ask bars of listed IWM options (2026-10-09), a leg's half-spread is
roughly constant in DOLLARS (~$0.012-0.02), not a fixed % of its price, so the % model
overcharged a $0.95-credit QQQ condor $0.669 against a measured ~$0.085 (W65 CORRECTION).

Three candidate forms are fit per symbol and compared by session-grouped cross-validation
(`fit_leg_spread_models`): constant $/leg, affine ``a + b*mid``, and ``pct`` (``b*mid``,
the legacy shape, refit on the relevant contracts only). The winner is stored in
``option_leg_spread_params`` and read by `Backtester` when ``friction_model="per_leg_usd"``.
Pure functions only: no IO, so the sampler script, the engine and the tests share one
definition.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FORMS = ("const_usd", "pct", "affine_usd")  # ordered simplest -> most flexible
MIN_HALF_SPREAD_USD = 0.005  # half of the $0.01 option tick: nothing quotes tighter
# A more flexible form must beat a simpler one's CV error by this fraction to be chosen
# (Occam guard: with ~20 sessions an affine fit can "win" on noise).
PARSIMONY_MARGIN = 0.05


@dataclass(frozen=True)
class LegSpreadModel:
    """Half-spread of ONE option leg in dollars per share as a function of its mid."""

    form: str
    a: float = 0.0
    b: float = 0.0
    min_usd: float = MIN_HALF_SPREAD_USD

    def __post_init__(self) -> None:
        if self.form not in FORMS:
            raise ValueError(f"unknown friction form {self.form!r}; expected one of {FORMS}")

    def half_spread_usd(self, mid: float) -> float:
        mid = max(float(mid), 0.0)
        if self.form == "const_usd":
            v = self.a
        elif self.form == "pct":
            v = self.b * mid
        else:
            v = self.a + self.b * mid
        return max(float(v), self.min_usd)

    def to_row(self) -> dict:
        return {"form": self.form, "a": float(self.a), "b": float(self.b)}

    @classmethod
    def from_row(cls, row: dict) -> LegSpreadModel:
        return cls(form=str(row["form"]), a=float(row.get("a") or 0.0),
                   b=float(row.get("b") or 0.0))


def _fit_one(form: str, mid: np.ndarray, hs: np.ndarray) -> LegSpreadModel:
    """Fit one form. Median-based where the loss is MAE (robust to quote flickers)."""
    if form == "const_usd":
        return LegSpreadModel("const_usd", a=float(np.median(hs)))
    if form == "pct":
        return LegSpreadModel("pct", b=float(np.median(hs / np.maximum(mid, 1e-9))))
    if len(mid) < 3 or float(np.ptp(mid)) < 1e-9:
        return LegSpreadModel("affine_usd", a=float(np.median(hs)), b=0.0)
    b, a = np.polyfit(mid, hs, 1)
    # Physical constraints: a spread cannot shrink as the option gets dearer, nor go negative.
    b = max(float(b), 0.0)
    a = max(float(np.median(hs - b * mid)) if b == 0.0 else float(a), 0.0)
    return LegSpreadModel("affine_usd", a=a, b=b)


def _model_mae(model: LegSpreadModel, mid: np.ndarray, hs: np.ndarray) -> float:
    pred = np.array([model.half_spread_usd(m) for m in mid])
    return float(np.mean(np.abs(pred - hs)))


def fit_leg_spread_models(mid, half_spread_usd, session) -> dict:
    """Fit every form and pick one by leave-one-session-out CV.

    ``session`` groups samples (the trading day): hourly bars of one day are strongly
    correlated, so a random K-fold would leak a day into its own validation set.
    Returns ``{"best": LegSpreadModel, "cv_mae": {form: float}, "n": int, "n_sessions":
    int, "cv": "grouped"|"insample"}``.
    """
    mid = np.asarray(mid, dtype=float)
    hs = np.asarray(half_spread_usd, dtype=float)
    sess = np.asarray(session)
    ok = np.isfinite(mid) & np.isfinite(hs) & (hs > 0) & (mid >= 0)
    mid, hs, sess = mid[ok], hs[ok], sess[ok]
    if len(mid) < 5:
        raise ValueError(f"need >= 5 usable samples to fit a friction model, got {len(mid)}")

    groups = sorted(set(sess.tolist()))
    grouped = len(groups) >= 3
    cv: dict[str, float] = {}
    for form in FORMS:
        if grouped:
            errs, weights = [], []
            for g in groups:
                te = sess == g
                tr = ~te
                if tr.sum() < 3:
                    continue
                m = _fit_one(form, mid[tr], hs[tr])
                errs.append(_model_mae(m, mid[te], hs[te]))
                weights.append(int(te.sum()))
            cv[form] = float(np.average(errs, weights=weights))
        else:
            cv[form] = _model_mae(_fit_one(form, mid, hs), mid, hs)

    best_form = FORMS[0]
    for form in FORMS[1:]:
        if cv[form] < cv[best_form] * (1.0 - PARSIMONY_MARGIN):
            best_form = form
    return {"best": _fit_one(best_form, mid, hs), "cv_mae": cv, "n": int(len(mid)),
            "n_sessions": len(groups), "cv": "grouped" if grouped else "insample"}


def condor_leg_cost_usd(model: LegSpreadModel, leg_mids, capture: float = 1.0,
                        stress: float = 1.0) -> float:
    """Total dollars/share to cross the spread on every leg of a structure once."""
    return float(capture) * float(stress) * sum(model.half_spread_usd(m) for m in leg_mids)
