#!/usr/bin/env python
"""W83a-2: the strike-aware range study, run under the PRE-REGISTERED rule (PLAN.md W83 + amendments 1-6).

Question: does a model of the holding-period excursion (max up / down move from the D-1 close over
sessions D..D+h-1, in units of the IV-implied move) predict "no touch of the short strikes" better than
option prices (A, A') and realized vol (B, HAR) already do? QQQ is judged on its own rows.

    python scripts/excursion_range_study.py --workers 3 [--h 9] [--sensitivities] [--symbols QQQ SPY IWM]

Predictors, per test year 2019..2026 (expanding training, purge = h):
  A      the two-sided Brownian barrier probability at z (pure IV)
  A'     the training rows' empirical CDF of the excursion (IV units)
  B      HAR-RV: the excursion in HAR-sigma units, its training CDF evaluated at z * sigma_IV / sigma_HAR
  C      LightGBM quantile regression of log M (19 quantiles), pooled or own-symbol by a validation year
  C-lin  linear quantile regression of log M, same candidate rule
BSS of C / C-lin against the better of A' and B (chosen on training), averaged over delta 0.10..0.30, with a
paired stationary bootstrap; the verdict per symbol is `ait.ml.excursion.verdict` (QQQ primary).

Read-only: every database is opened read-only (`mode=ro`); nothing is written except the JSON report
(reports/w83a_excursion_<date>.json, not committed) and the fractal-feature cache
(data/research_cache/w83a/, a pure function of the stored prices).
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()  # before any trading-subsystem import (CLAUDE.md)

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import sqlite3  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from datetime import date  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "historical.db"
CACHE = ROOT / "data" / "research_cache" / "w83a"
SYMBOLS = ("QQQ", "SPY", "IWM")
TEST_YEARS = tuple(range(2019, 2027))
SEED = 83
LGB_PARAMS = dict(objective="quantile", n_estimators=300, learning_rate=0.03, num_leaves=15,
                  min_child_samples=50, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                  random_state=SEED, deterministic=True, force_row_wise=True, verbose=-1, n_jobs=1)

IV_BLOCK = ["iv", "iv_rank", "vrp", "gk1", "gk5", "gk22"]
VIX_BLOCK = ["vix", "vix_vix3m", "vix9d_vix"]
PRICE_BLOCK = ["gap1", "ret20", "atr14"]
FRACTAL_BLOCK = ["mf_width", "hurst_spread"]
MACRO_BLOCK = ["fomc_in", "cpi_in", "nfp_in", "to_next_macro"]
EARN_BLOCK = ["earn_w_in", "earn_top5_in", "to_next_top5"]
DIV_BLOCK = ["exdiv_in", "prev_div_yield"]
FEATURES = IV_BLOCK + VIX_BLOCK + PRICE_BLOCK + FRACTAL_BLOCK + ["dow"] + MACRO_BLOCK + EARN_BLOCK + DIV_BLOCK


def _ro(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{db}?mode=ro", uri=True)


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------

def load_inputs(db: Path, symbols) -> dict:
    """Prices + IV per symbol, VIX family, the three W83a-1 calendars and the dividends. Asserts each
    loaded and prints its size (a silent empty join must not read as a result)."""
    from ait.backtesting.replay import ResearchData

    # W91: the research_* tables were merged into the shared tables (scripts/unify_history_tables.py; rows verified
    # identical), so the calendars, holdings and dividends are read un-prefixed.
    out = {"prices": {}, "dividends": {}}
    for s in symbols:
        f = ResearchData(str(db), s).daily_frame()
        assert len(f) > 3000 and f["implied_vol"].notna().sum() > 3000, f"{s}: price/IV history did not load"
        out["prices"][s] = f
    with _ro(db) as c:
        vix = pd.read_sql_query("SELECT symbol, date, iv_close FROM daily_iv WHERE symbol IN "
                                "('VIX','VIX3M','VIX9D')", c)
        out["econ"] = pd.read_sql_query("SELECT event_date, event FROM econ_schedule", c)
        out["holdings"] = pd.read_sql_query("SELECT filed_date, company_key, weight FROM qqq_holdings "
                                            "WHERE company_key IS NOT NULL", c)
        out["earnings"] = pd.read_sql_query("SELECT company_key, event_session FROM earnings_schedule", c)
        for s in symbols:
            out["dividends"][s] = pd.read_sql_query(
                "SELECT ex_date, amount FROM dividends WHERE symbol=? ORDER BY ex_date", c, params=(s,))
    v = vix.pivot(index="date", columns="symbol", values="iv_close")
    v.index = pd.to_datetime(v.index)
    out["vix"] = v
    assert len(v) > 1500 and set(v.columns) == {"VIX", "VIX3M", "VIX9D"}, "VIX family did not load"
    assert set(out["econ"]["event"]) == {"FOMC", "CPI", "NFP"} and len(out["econ"]) > 500, "econ schedule"
    assert len(out["holdings"]) > 400, "QQQ holdings did not load"
    assert len(out["earnings"]) > 1500, "earnings schedule did not load"
    for s in symbols:
        assert len(out["dividends"][s]) > 60, f"{s}: dividends did not load"
    print("Inputs: " + ", ".join(f"{s} {len(out['prices'][s])} sessions / {int(out['prices'][s]['implied_vol'].notna().sum())} IV"
                                 for s in symbols)
          + f"; VIX {len(v)}; econ {len(out['econ'])}; holdings {len(out['holdings'])}; "
            f"earnings {len(out['earnings'])}; dividends "
          + ", ".join(f"{s} {len(out['dividends'][s])}" for s in symbols), flush=True)
    return out


def fractal_columns(frame: pd.DataFrame, symbol: str, cache: Path | None = CACHE) -> pd.DataFrame:
    """As-of-date fractal features (row j uses closes through j): MFDFA width over the trailing 504
    log returns, |h60 - h180| over the trailing 180 log prices. Cached by a hash of the closes."""
    from ait.ml.features import FeatureEngine

    closes = frame["Close"].to_numpy(dtype=float)
    key = hashlib.sha256(closes.tobytes() + frame.index.to_numpy().tobytes()).hexdigest()[:16]
    path = cache / f"fractal_{symbol}_{key}.pkl" if cache else None
    if path is not None and path.exists():
        return pickle.loads(path.read_bytes())
    fe = FeatureEngine()
    lp = np.log(closes)
    lr = np.diff(lp, prepend=np.nan)
    mf = np.full(len(lp), np.nan)
    hs = np.full(len(lp), np.nan)
    for j in range(len(lp)):
        if j >= 504:
            mf[j] = fe._mfdfa_features(lr[j - 503:j + 1])[0]
        if j >= 179:
            h60, h180 = fe._multiscale_hurst(lp[j - 179:j + 1])
            hs[j] = abs(h60 - h180)
    out = pd.DataFrame({"mf_width": mf, "hurst_spread": hs}, index=frame.index)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(out))
    return out


def _event_positions(sessions: np.ndarray, dates) -> np.ndarray:
    """Session positions of event dates (an event on a non-session date maps to the next session)."""
    d = np.sort(pd.to_datetime(pd.Series(list(dates))).to_numpy(dtype="datetime64[ns]"))
    pos = np.searchsorted(sessions, d, side="left")
    return np.unique(pos[pos < len(sessions)])


def _in_window(pos: np.ndarray, n: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """(event in [i, i+h-1], sessions from i to the next event at or after i) for every row i."""
    i = np.arange(n)
    if len(pos) == 0:
        return np.zeros(n), np.full(n, np.nan)
    nxt = np.searchsorted(pos, i, side="left")
    nxt_pos = np.where(nxt < len(pos), pos[np.minimum(nxt, len(pos) - 1)], np.iinfo(np.int64).max)
    inw = nxt_pos <= i + h - 1
    dist = np.where(nxt < len(pos), nxt_pos - i, np.nan).astype(float)
    return inw.astype(float), dist


def build_rows(symbol: str, frame: pd.DataFrame, inputs: dict, h: int, fractal: pd.DataFrame | None = None
               ) -> pd.DataFrame:
    """One row per session D: the 24 features (all from information through the close of D-1, or the
    published schedules), the target and its parts, the HAR inputs and the live-eligible flag."""
    from ait.backtesting.ex_div import ex_div_blocks
    from ait.backtesting.expiry_calendar import listed_expiry
    from ait.data.iv_rank import iv_rank_series
    from ait.data.research_calendars import weights_as_of
    from ait.ml import excursion as ex

    f = frame
    n = len(f)
    sessions = f.index.to_numpy(dtype="datetime64[ns]")
    prev = lambda s: s.shift(1)  # noqa: E731 - the D-1 value at row D
    gk = ex.garman_klass(f)
    iv = f["implied_vol"].astype(float)
    r = pd.DataFrame(index=f.index)
    r["iv"] = prev(iv)
    r["iv_rank"] = prev(iv_rank_series(iv).reindex(f.index))
    gk22 = gk.rolling(22).mean()
    r["vrp"] = np.log(prev(iv) / np.sqrt(prev(gk22) * ex.TRADING_DAYS))
    for k in (1, 5, 22):
        r[f"gk{k}"] = np.log(np.sqrt(prev(gk.rolling(k).mean()) * ex.TRADING_DAYS))
    vx = inputs["vix"].reindex(f.index)
    r["vix"] = prev(vx["VIX"])
    r["vix_vix3m"] = prev(vx["VIX"] / vx["VIX3M"])
    r["vix9d_vix"] = prev(vx["VIX9D"] / vx["VIX"])
    r["gap1"] = prev(f["Open"] / f["Close"].shift(1) - 1)
    r["ret20"] = prev(f["Close"] / f["Close"].shift(20) - 1)
    tr = pd.concat([f["High"] - f["Low"], (f["High"] - f["Close"].shift(1)).abs(),
                    (f["Low"] - f["Close"].shift(1)).abs()], axis=1).max(axis=1)
    r["atr14"] = prev(tr.rolling(14).mean() / f["Close"])
    fr = fractal if fractal is not None else fractal_columns(f, symbol)
    r["mf_width"] = prev(fr["mf_width"])
    r["hurst_spread"] = prev(fr["hurst_spread"])
    r["dow"] = f.index.dayofweek.astype(float)
    econ = inputs["econ"]
    macro_pos = []
    for name, ev in (("fomc_in", "FOMC"), ("cpi_in", "CPI"), ("nfp_in", "NFP")):
        pos = _event_positions(sessions, econ.loc[econ["event"] == ev, "event_date"])
        r[name], _d = _in_window(pos, n, h)
        macro_pos.append(pos)
    _w, r["to_next_macro"] = _in_window(np.unique(np.concatenate(macro_pos)), n, h)
    # earnings: QQQ top-10 weights FILED before D; releases by their event session
    hold = inputs["holdings"]
    reports = [(date.fromisoformat(fd), dict(zip(g["company_key"], g["weight"])))
               for fd, g in hold.groupby("filed_date")]
    earn = inputs["earnings"]
    epos = {k: _event_positions(sessions, g["event_session"]) for k, g in earn.groupby("company_key")}
    w_in = np.full(n, np.nan)
    top5_in = np.full(n, np.nan)
    to_next5 = np.full(n, np.nan)
    for i, d in enumerate(f.index.date):
        w = weights_as_of(reports, d)
        if not w:
            continue
        top5 = sorted(w, key=lambda k: -w[k])[:5]
        acc, cnt, nxt = 0.0, 0, np.inf
        for k, wk in w.items():
            p = epos.get(k)
            if p is None or not len(p):
                continue
            j = np.searchsorted(p, i, side="left")
            if j < len(p):
                if p[j] <= i + h - 1:
                    acc += wk
                    cnt += k in top5
                if k in top5:
                    nxt = min(nxt, p[j] - i)
        w_in[i], top5_in[i] = acc, cnt
        to_next5[i] = nxt if np.isfinite(nxt) else np.nan
    r["earn_w_in"], r["earn_top5_in"], r["to_next_top5"] = w_in, top5_in, to_next5
    divs = inputs["dividends"][symbol]
    ex_dates = sorted(date.fromisoformat(x) for x in divs["ex_date"])
    dpos = _event_positions(sessions, divs["ex_date"])
    r["exdiv_in"], _d = _in_window(dpos, n, h)
    amt = pd.Series(divs["amount"].to_numpy(dtype=float), index=pd.to_datetime(divs["ex_date"]))
    last_amt = amt.reindex(f.index.union(amt.index)).ffill().shift(1).reindex(f.index)  # strictly before D
    r["prev_div_yield"] = last_amt / f["Close"].shift(1)
    # target
    e = ex.excursions(f, h)
    r["m_frac"] = e["m_frac"]
    r["window_end"] = e["window_end"]
    r["sigma_iv"] = ex.implied_move(prev(iv), h)
    r["M"] = r["m_frac"] / r["sigma_iv"]
    r["logM"] = np.log(r["M"].clip(lower=1e-6))
    lags = ex.har_lags(gk)
    r[["lag1", "lag5", "lag22"]] = lags
    r["har_y"] = ex.har_target(gk, h)
    r["live_ok"] = [not ex_div_blocks("live", ex_dates, d, listed_expiry(d, 14))[0] for d in f.index.date]
    r["symbol"] = symbol
    usable = r["mf_width"].notna() & r["hurst_spread"].notna() & r["iv"].gt(0) & r["gk22"].notna()
    return r[usable]


# ---------------------------------------------------------------------------------------------
# models (run in worker processes)
# ---------------------------------------------------------------------------------------------

_TABLE: pd.DataFrame | None = None


def _init(table: pd.DataFrame) -> None:
    global _TABLE
    _TABLE = table
    warnings.filterwarnings("ignore")


def _x_lgb(df: pd.DataFrame, feats: list[str], pooled: bool) -> pd.DataFrame:
    X = df[feats].copy()
    if pooled:
        X["symbol"] = pd.Categorical(df["symbol"], categories=list(SYMBOLS))
    return X


def _fit_quantiles(kind: str, train: pd.DataFrame, pred: pd.DataFrame, feats: list[str], pooled: bool):
    from ait.ml.excursion import TAUS

    y = train["logM"].to_numpy()
    q = np.empty((len(pred), len(TAUS)))
    if kind == "C":
        import lightgbm as lgb
        Xt, Xp = _x_lgb(train, feats, pooled), _x_lgb(pred, feats, pooled)
        for j, t in enumerate(TAUS):
            m = lgb.LGBMRegressor(alpha=t, **LGB_PARAMS)
            m.fit(Xt, y)
            q[:, j] = m.predict(Xp)
        return q
    from sklearn.linear_model import QuantileRegressor
    # amendment 6 item 3: a missing value -> the training median. A feature with NO training value at
    # all (the VIX block before 2018-08 in the earliest windows) has no median: it becomes the constant
    # 0 after standardization, i.e. it carries no information in that window.
    med = train[feats].median()
    mu = train[feats].fillna(med).mean().fillna(0.0)
    sd = train[feats].fillna(med).std().fillna(1.0).replace(0, 1.0)

    def prep(df):
        X = ((df[feats].fillna(med) - mu) / sd).fillna(0.0)
        X["vix_missing"] = df["vix"].isna().astype(float)
        if pooled:
            for s in SYMBOLS[1:]:
                X[f"is_{s}"] = (df["symbol"] == s).astype(float)
        return X.to_numpy(dtype=float)
    Xt, Xp = prep(train), prep(pred)
    for j, t in enumerate(TAUS):
        m = QuantileRegressor(quantile=t, alpha=0.0, solver="highs")
        m.fit(Xt, y)
        q[:, j] = m.predict(Xp)
    return q


def _baselines(train: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """{name: (n_pred x deltas) probabilities} for A, A', B, the reference choice and its name."""
    from ait.ml import excursion as ex

    zs = np.array([ex.z_of_delta(d) for d in ex.DELTAS])
    out = {"A": np.tile(ex.barrier_no_touch(zs), (len(pred), 1))}
    f_a = ex.empirical_cdf(train["M"].to_numpy())
    out["A'"] = np.tile(f_a(zs), (len(pred), 1))
    ok = train[["har_y", "lag1", "lag5", "lag22"]].notna().all(axis=1) & np.isfinite(train["har_y"])
    beta, smear = ex.har_fit(train.loc[ok, "har_y"].to_numpy(), train.loc[ok, ["lag1", "lag5", "lag22"]].to_numpy())

    def p_b(df):
        sig = ex.har_predict_sigma(beta, smear, df[["lag1", "lag5", "lag22"]].fillna(df[["lag1", "lag5", "lag22"]].median()).to_numpy())
        return sig
    sig_tr = p_b(train)
    f_b = ex.empirical_cdf((train["m_frac"] / sig_tr).to_numpy())
    sig_pr = p_b(pred)
    out["B"] = np.column_stack([f_b(z * pred["sigma_iv"].to_numpy() / sig_pr) for z in zs])
    # the reference: the better of A' and B by its TRAINING Brier, averaged over the deltas
    y_tr = (train["M"].to_numpy()[:, None] < zs[None, :]).astype(float)
    pa_tr = np.tile(f_a(zs), (len(train), 1))
    pb_tr = np.column_stack([f_b(z * train["sigma_iv"].to_numpy() / sig_tr) for z in zs])
    b_a = float(np.mean((pa_tr - y_tr) ** 2))
    b_b = float(np.mean((pb_tr - y_tr) ** 2))
    out["ref_name"] = "A'" if b_a <= b_b else "B"
    out["ref"] = out[out["ref_name"]]
    return out


def _bss_mean(p: np.ndarray, ref: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean([1 - np.mean((p[:, k] - y[:, k]) ** 2) / np.mean((ref[:, k] - y[:, k]) ** 2)
                          for k in range(y.shape[1])]))


def run_fold(task: dict) -> dict:
    """One test year: baselines, the validation-year candidate choice, the final fits, predictions."""
    from ait.ml import excursion as ex

    t = _TABLE
    y_test, h, feats, symbols = task["year"], task["h"], task["features"], task["symbols"]
    sessions = sorted(set(pd.to_datetime(t["date"]).dt.date))
    zs = np.array([ex.z_of_delta(d) for d in ex.DELTAS])
    out = {"year": y_test, "symbols": {}}

    def masks(year):
        f = ex.yearly_folds(t["date"].tolist(), t["window_end"].tolist(), sessions, [year], purge=h)
        return f[0][1], f[0][2]
    tr, te = masks(y_test)
    trv, va = masks(y_test - 1)
    va = va & tr                                    # the validation year lies inside the training rows
    train, test = t[tr], t[te]
    train_v, val = t[trv], t[va]
    fits = {}
    for kind in ("C", "C-lin"):
        fits[(kind, "pooled", "val")] = _fit_quantiles(kind, train_v, val, feats, True)
        fits[(kind, "pooled", "test")] = _fit_quantiles(kind, train, test, feats, True)
    for s in symbols:
        s_tr, s_te = train[train["symbol"] == s], test[test["symbol"] == s]
        s_trv, s_va = train_v[train_v["symbol"] == s], val[val["symbol"] == s]
        if s_te.empty:
            continue
        base = _baselines(s_tr, s_te)
        base_v = _baselines(s_trv, s_va)
        y_va = (s_va["M"].to_numpy()[:, None] < zs[None, :]).astype(float)
        rec = {"date": s_te["date"].astype(str).tolist(), "live_ok": s_te["live_ok"].astype(bool).tolist(),
               "y": (s_te["M"].to_numpy()[:, None] < zs[None, :]).astype(int).tolist(),
               "logM": s_te["logM"].tolist(), "ref_name": base["ref_name"]}
        for k in ("A", "A'", "B", "ref"):
            rec[k] = base[k].tolist()
        for kind in ("C", "C-lin"):
            own_v = _fit_quantiles(kind, s_trv, s_va, feats, False)
            pool_v = fits[(kind, "pooled", "val")][(val["symbol"] == s).to_numpy()]
            p_own_v = np.column_stack([ex.cdf_from_quantiles(own_v, z) for z in zs])
            p_pool_v = np.column_stack([ex.cdf_from_quantiles(pool_v, z) for z in zs])
            choice = "pooled" if _bss_mean(p_pool_v, base_v["ref"], y_va) >= _bss_mean(p_own_v, base_v["ref"], y_va) else "own"
            q_own = _fit_quantiles(kind, s_tr, s_te, feats, False)
            q_pool = fits[(kind, "pooled", "test")][(test["symbol"] == s).to_numpy()]
            for name, q in (("own", q_own), ("pooled", q_pool)):
                rec[f"{kind}|{name}"] = np.column_stack([ex.cdf_from_quantiles(q, z) for z in zs]).tolist()
                rec[f"{kind}|{name}|q"] = q.tolist()
            rec[f"{kind}|choice"] = choice
            rec[kind] = rec[f"{kind}|{choice}"]
            rec[f"{kind}|q"] = rec[f"{kind}|{choice}|q"]
        out["symbols"][s] = rec
    return out


# ---------------------------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------------------------

def score(rows: dict, only_live: bool = False, n_boot: int = 2000) -> dict:
    """Metrics of one symbol's out-of-sample rows (concatenated over the folds, date order)."""
    from sklearn.metrics import roc_auc_score

    from ait.ml import excursion as ex

    keep = np.array(rows["live_ok"]) if only_live else np.ones(len(rows["date"]), bool)
    order = np.argsort(np.array(rows["date"])[keep], kind="stable")
    y = np.array(rows["y"], float)[keep][order]
    P = {k: np.array(rows[k], float)[keep][order] for k in ("A", "A'", "B", "ref", "C", "C-lin",
                                                           "C|own", "C|pooled", "C-lin|own", "C-lin|pooled")}
    logm = np.array(rows["logM"], float)[keep][order]
    idx = ex.stationary_bootstrap_indices(len(y), 20.0, n_boot, SEED)
    se_ref = (P["ref"] - y) ** 2
    res = {"n": int(len(y)), "base_rate_no_touch": y.mean(axis=0).round(4).tolist(),
           "brier": {k: [round(ex.brier(P[k][:, j], y[:, j]), 5) for j in range(y.shape[1])] for k in P}}
    for k in ("A", "A'", "B", "C", "C-lin", "C|own", "C|pooled", "C-lin|own", "C-lin|pooled"):
        se = (P[k] - y) ** 2
        per = 1 - se.mean(axis=0) / se_ref.mean(axis=0)
        boot = np.array([np.mean(1 - se[b].mean(axis=0) / se_ref[b].mean(axis=0)) for b in idx])
        j20 = list(ex.DELTAS).index(0.20)
        res[k] = {"bss": float(per.mean()), "bss_per_delta": per.round(4).tolist(),
                  "lo": float(np.percentile(boot, 2.5)), "hi": float(np.percentile(boot, 97.5)),
                  "slope": ex.calibration_slope(P[k][:, j20], y[:, j20]),
                  "auc": [float(roc_auc_score(y[:, j], P[k][:, j])) if 0 < y[:, j].mean() < 1 else float("nan")
                          for j in range(y.shape[1])]}
    for kind in ("C", "C-lin"):
        q = np.array(rows[f"{kind}|q"], float)[keep][order]
        res[kind]["crps"] = ex.crps_pinball(q, logm)
    res["verdict"] = ex.verdict({"C": res["C"], "C-lin": res["C-lin"]})
    return res


def run_study(inputs: dict, symbols, h: int, features: list[str], workers: int) -> dict:
    tables = []
    for s in symbols:
        fr = fractal_columns(inputs["prices"][s], s)
        rows = build_rows(s, inputs["prices"][s], inputs, h, fractal=fr)
        rows = rows.reset_index().rename(columns={rows.index.name or "index": "date"})
        tables.append(rows)
    table = pd.concat(tables, ignore_index=True)
    table["date"] = pd.to_datetime(table["date"]).dt.date
    table = table[np.isfinite(table["logM"])]
    print(f"h={h}: {len(table)} usable rows ({table['date'].min()}..{table['date'].max()}); "
          f"{len(features)} features", flush=True)
    tasks = [{"year": y, "h": h, "features": features, "symbols": list(symbols)} for y in TEST_YEARS]
    with ProcessPoolExecutor(max_workers=max(1, workers), initializer=_init, initargs=(table,)) as pool:
        folds = list(pool.map(run_fold, tasks))
    # the fits are the expensive part: keep the raw fold predictions before scoring (W83a-2: the first
    # run crashed in scoring and lost them)
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"folds_h{h}_{len(features)}f.pkl").write_bytes(pickle.dumps(folds))
    merged: dict = {}
    for f in folds:
        for s, rec in f["symbols"].items():
            m = merged.setdefault(s, {k: [] for k in rec if not k.endswith("choice") and k != "ref_name"})
            for k, v in rec.items():
                if k.endswith("choice") or k == "ref_name":
                    merged[s].setdefault("choices", []).append((f["year"], k, v))
                else:
                    m[k] += v
    out = {"h": h, "features": features, "symbols": {}}
    for s in symbols:
        if s not in merged:
            continue
        out["symbols"][s] = {"all_days": score(merged[s]), "live_eligible": score(merged[s], only_live=True),
                             "choices": merged[s].get("choices", [])}
    out["outcome"] = __import__("ait.ml.excursion", fromlist=["outcome"]).outcome(
        {s: v["all_days"]["verdict"] for s, v in out["symbols"].items()})
    return out


def print_report(res: dict, title: str) -> None:
    from ait.ml.excursion import DELTAS

    print(f"\n## {title} (h = {res['h']}, {len(res['features'])} features)\n")
    print("| symbol | subset | n | ref | model | BSS mean | 95% CI | slope @0.20 | CRPS | verdict |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for s, v in res["symbols"].items():
        refs = sorted({c[2] for c in v["choices"] if c[1] == "ref_name"})
        for sub in ("all_days", "live_eligible"):
            m = v[sub]
            for k in ("C", "C-lin"):
                fwd = m["verdict"]["forwarded"] == k
                print(f"| {s} | {sub} | {m['n']} | {'/'.join(refs)} | {k}{' (fwd)' if fwd else ''} | "
                      f"{m[k]['bss']:+.4f} | [{m[k]['lo']:+.4f}, {m[k]['hi']:+.4f}] | {m[k]['slope']:.2f} | "
                      f"{m[k]['crps']:.4f} | {('PASS' if m['verdict']['pass'] else 'FAIL') if fwd else ''} |")
    print("\n| symbol | subset | predictor | " + " | ".join(f"BSS d{d:.2f}" for d in DELTAS) + " | AUC d0.20 |")
    print("|---|---|---|" + "---|" * (len(DELTAS) + 1))
    for s, v in res["symbols"].items():
        for sub in ("all_days",):
            m = v[sub]
            for k in ("A", "A'", "B", "C", "C-lin", "C|own", "C|pooled", "C-lin|own", "C-lin|pooled"):
                print(f"| {s} | {sub} | {k} | " + " | ".join(f"{x:+.4f}" for x in m[k]["bss_per_delta"])
                      + f" | {m[k]['auc'][2]:.3f} |")
    print(f"\nOUTCOME: {res['outcome']}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS))
    ap.add_argument("--h", type=int, default=None, help="horizon in sessions (default: the live spec, 9)")
    ap.add_argument("--sensitivities", action="store_true",
                    help="also run h = 5 and 14, and without the earnings / dividend features")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--db-path", default=str(DB))
    ap.add_argument("--report-dir", default=str(ROOT / "reports"))
    args = ap.parse_args(argv)
    warnings.filterwarnings("ignore")

    from ait.config.research import use_research_config
    use_research_config()
    from ait.ml.range_spec import horizon_for

    h = args.h or horizon_for(14)
    inputs = load_inputs(Path(args.db_path), args.symbols)
    report = {"primary": run_study(inputs, args.symbols, h, FEATURES, args.workers)}
    print_report(report["primary"], "PRIMARY (the registered verdict)")
    if args.sensitivities:
        for name, hh, feats in (("h5", 5, FEATURES), ("h14", 14, FEATURES),
                                ("no_earnings", h, [x for x in FEATURES if x not in EARN_BLOCK]),
                                ("no_dividends", h, [x for x in FEATURES if x not in DIV_BLOCK])):
            report[name] = run_study(inputs, args.symbols, hh, feats, args.workers)
            print_report(report[name], f"SENSITIVITY {name} (reported only)")
    from ait.utils.time import now_et
    out = Path(args.report_dir) / f"w83a_excursion_{now_et().strftime('%Y%m%d_%H%M')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, default=str))
    print(f"\nReport written: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
