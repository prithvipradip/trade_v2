"""W83a-2: the math of the strike-aware range study (pure; no IO).

The study asks whether a model of the holding-period EXCURSION predicts "no touch of the short
strikes" better than option prices and realized vol already do (PLAN.md W83 + amendments 1-6).
The excursion of day D over h sessions, measured from the close of D-1 with the daily High / Low:

    up = max(High[D..D+h-1]) / S - 1,   down = 1 - min(Low[D..D+h-1]) / S,   S = Close[D-1]

and M = max(up, down) in units of the implied move sigma_h = IV[D-1] * sqrt(h / 252). A symmetric
condor whose short strikes sit z implied deviations away is not touched iff M < z, with
z(delta) = Phi^-1(1 - delta). Every function here is shared by the study script and the tests.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd
from scipy.stats import norm

TAUS = tuple(round(0.05 * k, 2) for k in range(1, 20))   # 0.05 .. 0.95 (19 quantiles)
DELTAS = (0.10, 0.15, 0.20, 0.25, 0.30)
CDF_CLIP = (0.025, 0.975)
TRADING_DAYS = 252


# ---------------------------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------------------------

def excursions(df: pd.DataFrame, h: int) -> pd.DataFrame:
    """Per row D: ``up`` / ``down`` / ``m_frac`` (fractions of the D-1 close) over sessions
    D..D+h-1, and ``window_end`` (the date of D+h-1). NaN / NaT where D-1 or the window is missing."""
    hi, lo, cl = (df[c].to_numpy(dtype=float) for c in ("High", "Low", "Close"))
    n = len(df)
    up = np.full(n, np.nan)
    down = np.full(n, np.nan)
    end = np.full(n, np.datetime64("NaT"), dtype="datetime64[ns]")
    idx = df.index.to_numpy(dtype="datetime64[ns]")
    for i in range(1, n - h + 1):
        s = cl[i - 1]
        up[i] = hi[i:i + h].max() / s - 1.0
        down[i] = 1.0 - lo[i:i + h].min() / s
        end[i] = idx[i + h - 1]
    out = pd.DataFrame({"up": up, "down": down}, index=df.index)
    out["m_frac"] = out[["up", "down"]].max(axis=1, skipna=False)
    out["window_end"] = end
    return out


def implied_move(iv_prev: np.ndarray | pd.Series, h: int) -> np.ndarray:
    """sigma_h as a fraction of spot: IV (decimal, of D-1) * sqrt(h / 252)."""
    return np.asarray(iv_prev, dtype=float) * math.sqrt(h / TRADING_DAYS)


def z_of_delta(delta: float) -> float:
    """Strike distance in implied deviations for a short strike at |delta| (flat-IV map)."""
    return float(norm.ppf(1.0 - delta))


def barrier_no_touch(z: float | np.ndarray, terms: int = 50) -> np.ndarray:
    """Baseline A: P(a driftless Brownian path over unit time stays inside (-z, z)), the two-sided
    no-exit series (4/pi) sum_k (-1)^k / (2k+1) exp(-(2k+1)^2 pi^2 / (8 z^2))."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    k = np.arange(terms)[:, None]
    with np.errstate(divide="ignore", over="ignore"):
        s = ((-1.0) ** k / (2 * k + 1) * np.exp(-((2 * k + 1) ** 2) * math.pi ** 2 / (8 * z ** 2))).sum(0)
    return np.clip(4.0 / math.pi * s, 0.0, 1.0)


# ---------------------------------------------------------------------------------------------
# realized variance and HAR (baseline B)
# ---------------------------------------------------------------------------------------------

def garman_klass(df: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass variance 0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2 (floored at a tiny
    positive value, so its log is finite on a flat bar)."""
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    v = 0.5 * hl ** 2 - (2 * math.log(2) - 1) * co ** 2
    return v.clip(lower=1e-10)


def har_lags(gk: pd.Series) -> pd.DataFrame:
    """log mean GK variance over the 1, 5 and 22 sessions ending at D-1 (row D)."""
    g = gk.shift(1)
    return pd.DataFrame({"lag1": np.log(g), "lag5": np.log(g.rolling(5).mean()),
                         "lag22": np.log(g.rolling(22).mean())}, index=gk.index)


def har_target(gk: pd.Series, h: int) -> pd.Series:
    """log of the summed GK variance over sessions D..D+h-1 (row D; NaN at the end)."""
    # rolling(h) is NaN until h values exist, so an incomplete window at the end stays NaN (an explicit
    # count filter here was redundant: W83a-2 mutation check)
    return np.log(gk[::-1].rolling(h).sum()[::-1])


def har_fit(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, float]:
    """OLS with an intercept; returns (beta, Duan smearing factor mean(exp(residual)))."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return beta, float(np.mean(np.exp(resid)))


def har_predict_sigma(beta: np.ndarray, smear: float, X: np.ndarray) -> np.ndarray:
    """Predicted h-session sigma (a fraction of spot): sqrt(exp(X beta) * smear)."""
    A = np.column_stack([np.ones(len(X)), X])
    return np.sqrt(np.exp(A @ beta) * smear)


# ---------------------------------------------------------------------------------------------
# probabilities
# ---------------------------------------------------------------------------------------------

def empirical_cdf(train: np.ndarray):
    """F(x) = share of training values <= x (so P(M < z) for a continuous M)."""
    s = np.sort(np.asarray(train, dtype=float)[~np.isnan(train)])
    n = len(s)

    def f(x):
        return np.searchsorted(s, np.asarray(x, dtype=float), side="right") / n
    return f


def cdf_from_quantiles(q: np.ndarray, z: float | np.ndarray, taus: Sequence[float] = TAUS) -> np.ndarray:
    """P(M < z) per row from predicted quantiles of log M (rows x taus): the quantiles are sorted
    (monotone rearrangement), F is linear in log M between them and clipped to CDF_CLIP outside."""
    qs = np.sort(np.asarray(q, dtype=float), axis=1)
    lz = np.log(np.broadcast_to(np.asarray(z, dtype=float), (qs.shape[0],)))
    t = np.asarray(taus, dtype=float)
    out = np.empty(qs.shape[0])
    for i in range(qs.shape[0]):
        out[i] = np.interp(lz[i], qs[i], t, left=CDF_CLIP[0], right=CDF_CLIP[1])
    return np.clip(out, *CDF_CLIP)


# ---------------------------------------------------------------------------------------------
# scores
# ---------------------------------------------------------------------------------------------

def brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def bss(p: np.ndarray, p_ref: np.ndarray, y: np.ndarray) -> float:
    """Brier skill score of ``p`` against the reference: 1 - Brier(p) / Brier(ref)."""
    b_ref = brier(p_ref, y)
    return float("nan") if b_ref == 0 else 1.0 - brier(p, y) / b_ref


def crps_pinball(q: np.ndarray, y_log: np.ndarray, taus: Sequence[float] = TAUS) -> float:
    """CRPS of predicted log-M quantiles approximated by 2 x the mean pinball loss over the grid."""
    qs = np.sort(np.asarray(q, float), axis=1)
    d = np.asarray(y_log, float)[:, None] - qs
    t = np.asarray(taus, float)[None, :]
    return float(2.0 * np.mean(np.maximum(t * d, (t - 1.0) * d)))


def calibration_slope(p: np.ndarray, y: np.ndarray, iters: int = 50) -> float:
    """Slope of an unpenalized logistic regression of y on logit(p) (p clipped to [1e-4, 1-1e-4]);
    1 = calibrated, < 1 = overconfident. Newton-Raphson on two parameters."""
    pc = np.clip(np.asarray(p, float), 1e-4, 1 - 1e-4)
    x = np.log(pc / (1 - pc))
    yy = np.asarray(y, float)
    A = np.column_stack([np.ones_like(x), x])
    b = np.array([0.0, 1.0])
    for _ in range(iters):
        mu = 1.0 / (1.0 + np.exp(-(A @ b)))
        w = mu * (1 - mu)
        H = A.T @ (A * w[:, None]) + 1e-12 * np.eye(2)
        step = np.linalg.solve(H, A.T @ (yy - mu))
        b = b + step
        if np.max(np.abs(step)) < 1e-10:
            break
    return float(b[1])


def stationary_bootstrap_indices(n: int, mean_block: float = 20.0, n_boot: int = 2000,
                                 seed: int = 83) -> np.ndarray:
    """(n_boot, n) indices of the Politis-Romano stationary bootstrap: each next index is the
    following one (wrapping) with probability 1 - 1/mean_block, else a fresh uniform start. The
    same indices are applied to every predictor (paired)."""
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty((n_boot, n), dtype=np.int64)
    starts = rng.integers(0, n, size=(n_boot, n))
    jump = rng.random((n_boot, n)) < p
    out[:, 0] = starts[:, 0]
    for t in range(1, n):
        out[:, t] = np.where(jump[:, t], starts[:, t], (out[:, t - 1] + 1) % n)
    return out


# ---------------------------------------------------------------------------------------------
# folds and the verdict
# ---------------------------------------------------------------------------------------------

def yearly_folds(row_dates: Sequence[date], window_end: Sequence, sessions: Sequence[date],
                 test_years: Sequence[int], purge: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """[(year, train_mask, test_mask)]: test = rows dated in the year with a complete window;
    train = rows whose window ENDS before the session ``purge`` sessions ahead of the year's first
    session (so no training label overlaps the test year or the purge gap)."""
    rd = pd.to_datetime(pd.Series(list(row_dates))).dt.date.to_numpy()
    we = pd.to_datetime(pd.Series(list(window_end)))
    complete = we.notna().to_numpy()
    we_d = np.array([x.date() if pd.notna(x) else None for x in we], dtype=object)
    ses = list(sessions)
    out = []
    for y in test_years:
        first = next((i for i, s in enumerate(ses) if s.year == y), None)
        if first is None:
            continue
        cutoff = ses[max(0, first - purge)]
        test = np.array([d.year == y for d in rd]) & complete
        train = np.array([c and e < cutoff for c, e in zip(complete, we_d)])
        out.append((y, train, test))
    return out


PASS_BSS = 0.02
SLOPE_BAND = (0.8, 1.2)
CLIN_TIE = 0.01


def verdict(m: dict) -> dict:
    """The registered rule for ONE symbol (W83 + amendments 3 and 6).

    ``m`` = {"C": {"bss": mean-over-deltas BSS, "lo": bootstrap 2.5%, "slope": at delta 0.20},
             "C-lin": {...}}. The forwarded model is C unless C-lin is within CLIN_TIE of C. PASS iff
    the forwarded model has bss >= PASS_BSS, lo > 0 and slope inside SLOPE_BAND."""
    fwd = "C-lin" if m["C-lin"]["bss"] >= m["C"]["bss"] - CLIN_TIE else "C"
    r = m[fwd]
    checks = {"bss>=0.02": r["bss"] >= PASS_BSS, "lo>0": r["lo"] > 0,
              "slope_in_band": SLOPE_BAND[0] <= r["slope"] <= SLOPE_BAND[1]}
    return {"forwarded": fwd, "checks": checks, "pass": all(checks.values())}


def outcome(verdicts: dict[str, dict]) -> str:
    """The programme outcome across symbols (amendment 3: QQQ primary)."""
    q = verdicts.get("QQQ", {}).get("pass")
    others = [s for s, v in verdicts.items() if s != "QQQ" and v.get("pass")]
    if q:
        return "PASS: W83b may wire QQQ" + (f" and {', '.join(sorted(others))}" if others else " only")
    if others:
        return f"MIXED: QQQ fails; {', '.join(sorted(others))} pass (reported; does not start W83b)"
    return "FAIL: no symbol passes; next = the best baseline as a strike-aware probability"
