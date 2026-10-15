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


# ---------------------------------------------------------------------------
# W76: friction by DTE band (0-6 DTE)
# ---------------------------------------------------------------------------
#
# The v2 fit above has no sample below 11 DTE (the sampler mask was DTE 7-45), so every
# short-DTE leg was charged an EXTRAPOLATED half-spread. At 0-2 DTE the mids are cents,
# the tick is a large share of them, and credits are $0.4-1.3 over four legs: friction is
# most of the answer there. Until a band has collected enough sessions of real bid/ask
# bars, the model below refuses to pretend: it uses the all-DTE model, floors every leg
# at a conservative per-leg tick, and reports that the number was extrapolated.

# W82: bands 0-2, 3-6, 7-13, 14-29, 30-45 (46+ is open). Every band is gated on
# friction_min_sessions_per_band collected sessions like 0-6 was, but only 0-6 DTE is EXTRAPOLATED
# (floored at the tick) while unfitted: a 7+ band without a fit keeps being answered by the
# all-DTE model exactly as before (see BandedFriction.floor_max_dte).
SHORT_BAND_EDGES = (2, 6, 13, 29, 45)
FLOOR_MAX_DTE = 6          # the all-DTE sampler's mask started at DTE 7: below it, extrapolation


def friction_band(dte: int, edges: tuple[int, ...] = SHORT_BAND_EDGES) -> str:
    """'0-2', '3-6', '7-13', '14-29', '30-45' or '46+' for a (non-negative) calendar DTE."""
    lo = 0
    for e in edges:
        if dte <= e:
            return f"{lo}-{e}"
        lo = e + 1
    return f"{lo}+"


def band_dte_range(band: str, edges: tuple[int, ...] = SHORT_BAND_EDGES) -> tuple[int, int]:
    """Inclusive DTE range of a band label produced by ``friction_band``."""
    lo = 0
    for e in edges:
        if band == f"{lo}-{e}":
            return lo, e
        lo = e + 1
    if band == f"{lo}+":
        return lo, 10_000
    raise ValueError(f"unknown friction band {band!r} for edges {edges}")


@dataclass(frozen=True)
class BandedFriction:
    """All-DTE model + fitted short-DTE band models + the extrapolation rule.

    ``bands`` holds ONLY models whose band reached ``min_sessions`` (the loader decides);
    a short-DTE query with no such model is answered by ``default`` floored at
    ``floor_usd`` per leg and flagged ``extrapolated=True``.
    """

    default: LegSpreadModel
    bands: dict
    floor_usd: float
    edges: tuple[int, ...] = SHORT_BAND_EDGES
    floor_max_dte: int = FLOOR_MAX_DTE     # bands starting above this are the all-DTE fit's home

    def half_spread_usd(self, mid: float, dte: int) -> tuple[float, bool]:
        """(dollars per leg per share, extrapolated?)."""
        band = friction_band(max(int(dte), 0), self.edges)
        fitted = self.bands.get(band)
        if fitted is not None:
            return fitted.half_spread_usd(mid), False
        v = self.default.half_spread_usd(mid)
        lo, _hi = band_dte_range(band, self.edges)
        if lo > self.floor_max_dte:        # W82: 7+ without a fit: the all-DTE model, unfloored
            return v, False
        return max(v, self.floor_usd), True


def banded_condor_leg_cost_usd(model: BandedFriction, leg_mids, dte: int, capture: float = 1.0,
                               stress: float = 1.0) -> tuple[float, bool]:
    """Dollars/share to cross the spread on every leg once, and whether any leg's number was
    extrapolated (a short-DTE band without enough fitted sessions)."""
    total, extrapolated = 0.0, False
    for m in leg_mids:
        v, ex = model.half_spread_usd(m, dte)
        total += v
        extrapolated = extrapolated or ex
    return float(capture) * float(stress) * total, extrapolated
