/* ===================================================================
   AIT — Walk-Forward Analysis : mock data generator
   Deterministic (seeded) so it is stable across reloads.
   Shapes match the real exports:
     - backtest_results.json  (experiment + results + trades + equity)
     - window_NNN.json        (per-window metrics + best_params + trades_detail)
     - optimization_result.json (optuna study)
   Exposes window.AIT
   =================================================================== */
(function () {
  "use strict";

  // ---- tiny seeded PRNG (mulberry32) -------------------------------
  function mulberry32(a) {
    return function () {
      a |= 0; a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }
  const rnd = mulberry32(20240617);
  const gauss = () => {
    // Box-Muller
    let u = 0, v = 0;
    while (u === 0) u = rnd();
    while (v === 0) v = rnd();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
  const round = (x, n = 2) => { const f = Math.pow(10, n); return Math.round(x * f) / f; };
  const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

  // ---- trading-day calendar ----------------------------------------
  function tradingDays(startISO, n) {
    const days = [];
    let d = new Date(startISO + "T00:00:00Z");
    while (days.length < n) {
      const dow = d.getUTCDay();
      if (dow !== 0 && dow !== 6) days.push(d.toISOString().slice(0, 10));
      d = new Date(d.getTime() + 86400000);
    }
    return days;
  }

  const N = 320;                       // trading days (~15 months)
  const dates = tradingDays("2024-03-01", N);

  // ---- price path: regime-switching GBM for QQQ --------------------
  const bars = [];
  let price = 428.0;
  // volatility regimes (annualized) over the path
  const regimes = [0.12, 0.10, 0.18, 0.34, 0.16, 0.11, 0.22, 0.14];
  for (let i = 0; i < N; i++) {
    const regIdx = Math.min(regimes.length - 1, Math.floor(i / (N / regimes.length)));
    const annVol = regimes[regIdx] * (0.9 + 0.2 * rnd());
    const dailyVol = annVol / Math.sqrt(252);
    const drift = 0.085 / 252;          // mild upward drift
    const ret = drift + dailyVol * gauss();
    const prevClose = price;
    const close = prevClose * (1 + ret);
    const intra = close * dailyVol * 1.4;
    const open = prevClose * (1 + dailyVol * 0.3 * gauss());
    const high = Math.max(open, close) + Math.abs(intra) * (0.4 + 0.6 * rnd());
    const low = Math.min(open, close) - Math.abs(intra) * (0.4 + 0.6 * rnd());
    const volume = Math.round((38 + 22 * rnd() + (annVol > 0.25 ? 30 * rnd() : 0)) * 1e6);
    bars.push({
      time: dates[i],
      open: round(open), high: round(high), low: round(low), close: round(close),
      volume,
      _ret: ret, _annVol: annVol,
    });
    price = close;
  }

  // ---- indicators ---------------------------------------------------
  const closes = bars.map(b => b.close);
  function sma(arr, p, i) {
    if (i < p - 1) return null;
    let s = 0; for (let k = i - p + 1; k <= i; k++) s += arr[k];
    return s / p;
  }
  function emaSeries(arr, p) {
    const k = 2 / (p + 1); const out = []; let prev = arr[0];
    for (let i = 0; i < arr.length; i++) { prev = i === 0 ? arr[0] : arr[i] * k + prev * (1 - k); out.push(prev); }
    return out;
  }
  function rsiSeries(arr, p) {
    const out = new Array(arr.length).fill(null);
    let gain = 0, loss = 0;
    for (let i = 1; i < arr.length; i++) {
      const ch = arr[i] - arr[i - 1];
      const g = Math.max(ch, 0), l = Math.max(-ch, 0);
      if (i <= p) { gain += g; loss += l; if (i === p) { gain /= p; loss /= p; out[i] = 100 - 100 / (1 + gain / (loss || 1e-9)); } }
      else { gain = (gain * (p - 1) + g) / p; loss = (loss * (p - 1) + l) / p; out[i] = 100 - 100 / (1 + gain / (loss || 1e-9)); }
    }
    return out;
  }
  const ema12 = emaSeries(closes, 12), ema26 = emaSeries(closes, 26);
  const macdLine = closes.map((c, i) => (ema12[i] - ema26[i]) / c);
  const macdSignal = emaSeries(macdLine, 9);
  const rsi14 = rsiSeries(closes, 14);

  // realized vol (20d), bollinger, atr%
  const features = [];
  for (let i = 0; i < N; i++) {
    const b = bars[i];
    // realized vol 20d
    let rv = null;
    if (i >= 20) {
      let s = 0, m = 0; const win = [];
      for (let k = i - 19; k <= i; k++) { win.push(Math.log(closes[k] / closes[k - 1])); }
      m = win.reduce((a, x) => a + x, 0) / win.length;
      s = Math.sqrt(win.reduce((a, x) => a + (x - m) ** 2, 0) / (win.length - 1)) * Math.sqrt(252);
      rv = s;
    }
    const mid = sma(closes, 20, i);
    let bbStd = null;
    if (mid != null) {
      let s = 0; for (let k = i - 19; k <= i; k++) s += (closes[k] - mid) ** 2;
      bbStd = Math.sqrt(s / 20);
    }
    const bbUpper = mid != null ? mid + 2 * bbStd : null;
    const bbLower = mid != null ? mid - 2 * bbStd : null;
    const bbPos = bbUpper != null ? clamp((b.close - bbLower) / (bbUpper - bbLower), 0, 1) : null;
    // atr%
    let atr = null;
    if (i >= 14) {
      let s = 0;
      for (let k = i - 13; k <= i; k++) {
        const tr = Math.max(bars[k].high - bars[k].low, Math.abs(bars[k].high - bars[k - 1].close), Math.abs(bars[k].low - bars[k - 1].close));
        s += tr;
      }
      atr = (s / 14) / b.close;
    }
    const vix = clamp(11 + (rv != null ? rv * 55 : 14) + 3 * gauss(), 9, 48);
    const ivRank = clamp((vix - 10) / 30 + 0.06 * gauss(), 0, 1);
    // fractal hurst: <0.5 mean-reverting, >0.5 trending
    const hurst = clamp(0.5 + (macdLine[i] * 6) + 0.05 * gauss(), 0.25, 0.78);
    features.push({
      time: b.time,
      rsi_14: rsi14[i] != null ? round(rsi14[i], 1) : null,
      macd: round(macdLine[i], 5),
      macd_signal: round(macdSignal[i], 5),
      macd_hist: round(macdLine[i] - macdSignal[i], 5),
      sma_20: mid != null ? round(mid) : null,
      sma_50: sma(closes, 50, i) != null ? round(sma(closes, 50, i)) : null,
      bb_upper: bbUpper != null ? round(bbUpper) : null,
      bb_lower: bbLower != null ? round(bbLower) : null,
      bb_position: bbPos != null ? round(bbPos, 3) : null,
      atr_pct: atr != null ? round(atr, 4) : null,
      realized_vol_20: rv != null ? round(rv, 4) : null,
      iv_rank: round(ivRank, 3),
      vix_level: round(vix, 2),
      hurst_wavelet: round(hurst, 3),
      sentiment_composite: round(clamp(0.5 + macdLine[i] * 4 + 0.12 * gauss(), 0, 1), 3),
      put_call_ratio: round(clamp(0.9 + (vix - 16) * 0.02 + 0.08 * gauss(), 0.5, 1.6), 3),
      volume_ratio: round(clamp(b.volume / 55e6, 0.4, 2.2), 2),
    });
  }

  // ---- ML predictions per bar --------------------------------------
  // direction (3-class) + range model (P stays in range) + vol magnitude
  const predictions = [];
  for (let i = 0; i < N; i++) {
    const f = features[i];
    const rv = f.realized_vol_20 ?? 0.16;
    const trend = macdLine[i];
    let pUp = clamp(0.5 + trend * 9 + 0.05 * gauss(), 0.02, 0.96);
    let pDown = clamp(0.5 - trend * 9 + 0.05 * gauss(), 0.02, 0.96);
    let pNeutral = clamp(1.1 - (pUp + pDown), 0.02, 0.9);
    const sum = pUp + pDown + pNeutral; pUp /= sum; pDown /= sum; pNeutral /= sum;
    const conf = Math.max(pUp, pDown, pNeutral);
    const cls = conf === pUp ? "bullish" : conf === pDown ? "bearish" : "neutral";
    // range prob: high when low realized vol & mid bb position (good for iron condor)
    const rangeProb = clamp(0.9 - (rv - 0.12) * 1.8 - Math.abs((f.bb_position ?? 0.5) - 0.5) * 0.5 + 0.05 * gauss(), 0.15, 0.95);
    const volMag = clamp(rv * (0.9 + 0.3 * rnd()), 0.05, 0.6);
    const metaTake = rangeProb > 0.55 && rv < 0.22 ? (rnd() > 0.18) : (rnd() > 0.72);
    predictions.push({
      time: f.time,
      dir_class: cls,
      dir_conf: round(conf, 3),
      p_up: round(pUp, 3), p_down: round(pDown, 3), p_neutral: round(pNeutral, 3),
      range_prob: round(rangeProb, 3),
      vol_magnitude: round(volMag, 3),
      meta_take: metaTake,
    });
  }

  // ---- walk-forward windows ----------------------------------------
  const cfg = {
    train_days: 365, test_days: 63, step_days: 21, gap_days: 5,
    initial_capital: 50000, optimize_per_window: true, optimize_n_trials: 50,
    objective: "composite", min_confidence: 0.55, range_min_confidence: 0.55,
    max_concurrent_positions: 3, optimize_patience: 10, optimize_seed: 42,
  };
  // tile test windows across the visible bars (test windows are the OOS portion)
  const windows = [];
  const firstTestIdx = 110;            // leave room for "training"
  const testLen = 22, step = 21;
  let wid = 0;
  for (let start = firstTestIdx; start + testLen <= N; start += step) {
    wid++;
    const tStart = dates[start], tEnd = dates[Math.min(N - 1, start + testLen - 1)];
    windows.push({
      window_id: wid,
      train_start: dates[Math.max(0, start - 90)],
      train_end: dates[Math.max(0, start - 6)],
      test_start: tStart, test_end: tEnd,
      _startIdx: start, _endIdx: Math.min(N - 1, start + testLen - 1),
      model_accuracy: round(0.52 + 0.08 * rnd(), 3),
    });
  }

  // ---- iron-condor trades ------------------------------------------
  // entered on bars where range model is confident; resolved over hold window
  const EXIT_REASONS = ["profit_target", "stop_loss", "expiry", "trailing_stop", "thesis_invalidated"];
  const trades = [];
  let tid = 0;
  for (const w of windows) {
    // best params per window (from IRON_CONDOR_SPACE)
    const bp = {
      iron_condor__stop_loss_pct: round(0.30 + 0.4 * rnd(), 3),
      iron_condor__profit_target_pct: round(0.35 + 0.3 * rnd(), 3),
      iron_condor__trailing_stop_fraction: round(0.4 + 0.45 * rnd(), 3),
      iron_condor__delta_short: round(0.15 + 0.13 * rnd(), 3),
      iron_condor__max_hold_days: 10 + Math.floor(11 * rnd()),
      iron_condor__wing_k: round(0.4 + 1.4 * rnd(), 3),
      iron_condor__hurst_regime_threshold: round(0.08 + 0.22 * rnd(), 3),
      iron_condor__hurst_regime_penalty: round(0.25 * rnd(), 3),
      iron_condor__multifractal_max_width: round(0.30 + 0.35 * rnd(), 3),
    };
    w.best_params = bp;

    let entered = 0;
    for (let i = w._startIdx; i <= w._endIdx - 4 && entered < 4; i++) {
      const p = predictions[i], f = features[i], b = bars[i];
      if (p.range_prob < 0.6 || !p.meta_take) continue;
      if ((f.realized_vol_20 ?? 0.2) > 0.24) continue;
      if (rnd() > 0.62) continue;
      entered++;
      tid++;
      const holdMax = bp.iron_condor__max_hold_days;
      const exitIdx = Math.min(w._endIdx, i + 4 + Math.floor(rnd() * (holdMax - 4)));
      const entryPx = b.close;
      const exitPx = bars[exitIdx].close;
      const moved = Math.abs(exitPx - entryPx) / entryPx;
      // iron condor short strikes
      const wing = round(entryPx * (0.02 + 0.01 * bp.iron_condor__wing_k), 0);
      const shortPutK = round(entryPx * (1 - 0.035), 0);
      const shortCallK = round(entryPx * (1 + 0.035), 0);
      const credit = round(entryPx * (0.012 + 0.006 * f.iv_rank) * 100, 0); // $ per condor
      const contracts = 2 + Math.floor(rnd() * 4);
      const maxLoss = round((wing - credit / 100) * 100 * contracts, 0);
      // outcome: condor profits if price stays inside short strikes
      const stayed = exitPx > shortPutK && exitPx < shortCallK;
      let exitReason, pnl;
      if (moved > 0.05 && !stayed) {
        exitReason = rnd() > 0.5 ? "stop_loss" : "thesis_invalidated";
        pnl = -round(maxLoss * (0.45 + 0.4 * rnd()), 0);
      } else if (stayed && rnd() > 0.35) {
        exitReason = rnd() > 0.4 ? "profit_target" : "trailing_stop";
        pnl = round(credit * contracts * (0.45 + 0.45 * rnd()), 0);
      } else if (stayed) {
        exitReason = "expiry";
        pnl = round(credit * contracts * (0.6 + 0.35 * rnd()), 0);
      } else {
        exitReason = "stop_loss";
        pnl = -round(maxLoss * (0.3 + 0.3 * rnd()), 0);
      }
      const dirConf = p.dir_conf, rangeProb = p.range_prob;
      const vol10 = round((f.realized_vol_20 ?? 0.16) * (0.85 + 0.2 * rnd()), 3);
      const maxEntryVol = 0.80;
      const metaProb = round(clamp(rangeProb * 0.8 + 0.1 + 0.08 * gauss(), 0.2, 0.95), 3);
      trades.push({
        id: "T" + String(tid).padStart(3, "0"),
        window_id: w.window_id,
        symbol: "QQQ",
        strategy: "iron_condor",
        direction: "neutral",
        entry_date: b.time,
        exit_date: bars[exitIdx].time,
        entry_time: b.time + " 10:30:00",
        exit_time: bars[exitIdx].time + " 15:30:00",
        entry_idx: i, exit_idx: exitIdx,
        entry_price: entryPx, exit_price: exitPx,
        exit_reason: exitReason,
        pnl: pnl,
        return_pct: round(pnl / maxLoss, 4),
        contracts, n_legs: 4,
        credit, max_loss: maxLoss,
        hold_days: exitIdx - i,
        entry_confidence: rangeProb,
        range_prob: rangeProb,
        entry_regime: f.realized_vol_20 > 0.2 ? "high_volatility" : "range_bound",
        entry_iv_rank: f.iv_rank,
        entry_vix_level: f.vix_level,
        legs: [
          { type: "short_put", strike: shortPutK, premium: round(credit * 0.0042, 2) },
          { type: "long_put", strike: round(shortPutK - wing, 0), premium: round(credit * 0.0022, 2) },
          { type: "short_call", strike: shortCallK, premium: round(credit * 0.0040, 2) },
          { type: "long_call", strike: round(shortCallK + wing, 0), premium: round(credit * 0.0020, 2) },
        ],
        decision: {
          direction_class: p.dir_class,
          direction_conf: dirConf,
          range_gate: { prob: rangeProb, threshold: cfg.range_min_confidence, pass: rangeProb >= cfg.range_min_confidence },
          vol_gate: { vol_10d: vol10, max: maxEntryVol, pass: vol10 <= maxEntryVol },
          meta_label: { take: true, prob: metaProb, threshold: 0.5 },
          fractal_gate: { hurst_spread: round(0.12 + 0.18 * rnd(), 3), threshold: w.best_params.iron_condor__hurst_regime_threshold, pass: true },
          regime: f.realized_vol_20 > 0.2 ? "high_volatility" : "range_bound",
          earnings_skip: false,
        },
        features_at_entry: {
          rsi_14: f.rsi_14, macd_hist: f.macd_hist, bb_position: f.bb_position,
          atr_pct: f.atr_pct, realized_vol_20: f.realized_vol_20, iv_rank: f.iv_rank,
          vix_level: f.vix_level, hurst_wavelet: f.hurst_wavelet,
          sentiment_composite: f.sentiment_composite, put_call_ratio: f.put_call_ratio,
        },
      });
    }
  }

  // ---- aggregate per-window metrics from trades --------------------
  let equity = cfg.initial_capital;
  const equityCurve = [];
  for (const w of windows) {
    const wt = trades.filter(t => t.window_id === w.window_id);
    const pnl = wt.reduce((a, t) => a + t.pnl, 0);
    const wins = wt.filter(t => t.pnl > 0).length;
    w.trades = wt.length;
    w.pnl = round(pnl, 2);
    w.return_pct = round((pnl / cfg.initial_capital) * 100, 4);
    w.win_rate = wt.length ? round(wins / wt.length, 4) : 0;
    const pnls = wt.map(t => t.pnl);
    const mean = pnls.length ? pnls.reduce((a, x) => a + x, 0) / pnls.length : 0;
    const sd = pnls.length > 1 ? Math.sqrt(pnls.reduce((a, x) => a + (x - mean) ** 2, 0) / (pnls.length - 1)) : 0;
    w.sharpe = sd ? round((mean / sd) * Math.sqrt(252), 4) : 0;
    // window drawdown
    let eq = 0, peak = 0, mdd = 0;
    for (const t of wt) { eq += t.pnl; peak = Math.max(peak, eq); mdd = Math.max(mdd, peak - eq); }
    w.max_drawdown = round(peak > 0 ? mdd / (cfg.initial_capital) * 100 : 0, 4);
    w.strategies = { iron_condor: wt.length };
  }
  // equity curve over trades chronologically
  const sortedTrades = [...trades].sort((a, b) => a.exit_idx - b.exit_idx);
  for (const t of sortedTrades) {
    equity += t.pnl;
    equityCurve.push({ date: t.exit_date, equity: round(equity, 2), pnl: t.pnl, strategy: t.strategy, symbol: t.symbol, window: t.window_id });
  }

  // ---- top-level results -------------------------------------------
  const allPnls = trades.map(t => t.pnl);
  const totalPnl = allPnls.reduce((a, x) => a + x, 0);
  const wins = trades.filter(t => t.pnl > 0);
  const losses = trades.filter(t => t.pnl <= 0);
  const grossWin = wins.reduce((a, t) => a + t.pnl, 0);
  const grossLoss = Math.abs(losses.reduce((a, t) => a + t.pnl, 0));
  const meanPnl = totalPnl / (allPnls.length || 1);
  // Sharpe/Sortino on per-window returns, annualized for ~4 windows/yr (test_days≈63)
  const winRets = windows.map(w => w.return_pct / 100);
  const wMean = winRets.reduce((a, x) => a + x, 0) / (winRets.length || 1);
  const wStd = winRets.length > 1 ? Math.sqrt(winRets.reduce((a, x) => a + (x - wMean) ** 2, 0) / (winRets.length - 1)) : 0;
  const wDown = winRets.filter(x => x < wMean);
  const wDownStd = wDown.length > 1 ? Math.sqrt(wDown.reduce((a, x) => a + (x - wMean) ** 2, 0) / (wDown.length - 1)) : (wDown.length === 1 ? Math.abs(wDown[0]) : 0);
  const ANN = Math.sqrt(4);
  let eq = cfg.initial_capital, peak = cfg.initial_capital, mdd = 0;
  for (const t of sortedTrades) { eq += t.pnl; peak = Math.max(peak, eq); mdd = Math.max(mdd, (peak - eq) / peak); }
  const profitableWindows = windows.filter(w => w.pnl > 0).length;

  const results = {
    total_return: round(totalPnl / cfg.initial_capital, 4),
    cash_drag_adjusted_return: round(totalPnl / cfg.initial_capital + 0.031, 4),
    win_rate: round(wins.length / (trades.length || 1), 4),
    sharpe_ratio: wStd ? round((wMean / wStd) * ANN, 2) : 0,
    sortino_ratio: wDownStd ? round((wMean / wDownStd) * ANN, 2) : 0,
    max_drawdown: round(mdd, 4),
    profit_factor: grossLoss ? round(grossWin / grossLoss, 2) : 0,
    consistency: round(profitableWindows / windows.length, 4),
    total_trades: trades.length,
    windows: windows.length,
    avg_window_return: round(windows.reduce((a, w) => a + w.return_pct, 0) / windows.length / 100, 4),
    expectancy: round(meanPnl, 2),
    avg_win: round(wins.length ? grossWin / wins.length : 0, 2),
    avg_loss: round(losses.length ? grossLoss / losses.length : 0, 2),
    best_trade: round(Math.max(...allPnls), 2),
    worst_trade: round(Math.min(...allPnls), 2),
    capital_utilization: round(0.18 + 0.05 * rnd(), 3),
    raroc: round((totalPnl / cfg.initial_capital) / 0.2, 3),
    final_capital: round(cfg.initial_capital + totalPnl, 2),
    avg_hold_days: round(trades.reduce((a, t) => a + t.hold_days, 0) / (trades.length || 1), 1),
  };

  const experiment = {
    id: "WFO-2025-0617",
    name: "QQQ Iron Condor · Walk-Forward + Per-Window Optuna",
    strategy: "iron_condor",
    strategy_label: "Iron Condor",
    symbols: ["QQQ"],
    status: "completed",
    run_at: "2025-06-17T02:14:33",
    duration_sec: 4187,
    config: cfg,
    results,
    date_range: { start: dates[0], end: dates[N - 1] },
    git_sha: "a3f9c21",
    data_source: "IB historical (daily) + 5-min intraday",
  };

  // ---- Optuna studies (one per window) -----------------------------
  // composite objective = 0.4*sharpe + 0.4*win_rate - 0.2*|max_dd|
  const optuna_studies = {};
  windows.forEach((w, idx) => {
    const nReq = cfg.optimize_n_trials;
    // one window early-stopped via patience; one pruned heavily
    let stopReason, nRun, status;
    if (idx === 3) { nRun = 31; status = "early_stopped"; stopReason = `Early-stopped: ${cfg.optimize_patience} consecutive non-improving trials (patience reached at trial 31).`; }
    else if (idx === 6) { nRun = 50; status = "completed"; stopReason = "Completed all 50 trials. 14 trials pruned by MedianPruner (insufficient intermediate value)."; }
    else { nRun = nReq; status = "completed"; stopReason = "Completed all 50 trials."; }

    const space = {
      stop_loss_pct: [0.30, 0.70], profit_target_pct: [0.30, 0.70],
      trailing_stop_fraction: [0.30, 0.90], delta_short: [0.15, 0.30],
      max_hold_days: [10, 21], wing_k: [0.30, 2.00],
      hurst_regime_threshold: [0.08, 0.30], hurst_regime_penalty: [0.0, 0.25],
      multifractal_max_width: [0.30, 0.65],
    };
    const trials = [];
    let best = -Infinity, bestParams = null, bestTrial = 0;
    let noImprove = 0;
    for (let n = 0; n < nRun; n++) {
      const params = {};
      for (const k in space) {
        const [lo, hi] = space[k];
        let v = lo + (hi - lo) * rnd();
        if (k === "max_hold_days") v = Math.round(v);
        else v = round(v, 3);
        // TPE convergence: later trials cluster near best
        if (bestParams && rnd() > 0.4) {
          const bv = bestParams[k];
          v = k === "max_hold_days" ? Math.round(clamp(bv + (rnd() - 0.5) * 4, lo, hi)) : round(clamp(bv + (rnd() - 0.5) * (hi - lo) * 0.25, lo, hi), 3);
        }
        params[k] = v;
      }
      // pruning: ~20% pruned early
      const pruned = rnd() < (idx === 6 ? 0.28 : 0.14) && n > 4;
      let value, sharpe, winRate, maxDD, nTrades;
      if (pruned) {
        value = null; sharpe = null; winRate = null; maxDD = null; nTrades = 2 + Math.floor(rnd() * 4);
      } else {
        nTrades = 4 + Math.floor(rnd() * 16);
        // objective improves over trials with noise; depends a bit on params
        const base = 0.35 + 0.55 * (n / nRun) + 0.18 * gauss() * (1 - n / (nRun * 1.5));
        sharpe = round(clamp(base * 2.0 + 0.3 * gauss(), -0.5, 3.2), 3);
        winRate = round(clamp(0.45 + base * 0.4 + 0.05 * gauss(), 0.2, 0.92), 3);
        maxDD = round(clamp(0.18 - base * 0.08 + 0.04 * Math.abs(gauss()), 0.02, 0.4), 3);
        // trade-count penalty (quadratic below min_trades=10)
        let composite = 0.4 * sharpe + 0.4 * winRate - 0.2 * maxDD;
        if (nTrades < 3) composite = -100;
        else if (nTrades < 10) composite *= (nTrades / 10) ** 2;
        value = round(composite, 4);
      }
      const improved = value != null && value > best;
      if (improved) { best = value; bestParams = { ...params }; bestTrial = n; noImprove = 0; }
      else if (value != null) noImprove++;
      trials.push({
        number: n,
        value,
        state: pruned ? "PRUNED" : "COMPLETE",
        params,
        n_trades: nTrades,
        sharpe, win_rate: winRate, max_drawdown: maxDD,
        duration_s: round(6 + 50 * rnd(), 1),
        intermediate: value,
      });
    }
    const completed = trials.filter(t => t.state === "COMPLETE");
    optuna_studies[w.window_id] = {
      window_id: w.window_id,
      study_name: `wf_w${w.window_id}_QQQ_iron_condor`,
      objective: "composite",
      objective_formula: "0.4·Sharpe + 0.4·WinRate − 0.2·|MaxDD|",
      n_trials_requested: nReq,
      n_trials_run: nRun,
      n_pruned: trials.filter(t => t.state === "PRUNED").length,
      n_complete: completed.length,
      status, stop_reason: stopReason,
      patience: cfg.optimize_patience,
      sampler: "TPESampler(seed=42)",
      pruner: "MedianPruner(n_warmup_steps=1)",
      best_value: best === -Infinity ? null : round(best, 4),
      best_trial: bestTrial,
      best_params: bestParams,
      trials,
      test_start: w.test_start, test_end: w.test_end,
    };
  });

  // ===================================================================
  // Per-window predictor-model performance  (Predictor Models view)
  // Mirrors real model outputs:
  //   DirectionPredictor (ensemble.py): members xgboost + lightgbm, AUROC CV,
  //     fitted weights from edge over the 0.5 baseline.
  //   RangePredictor (range_predictor.py): members xgboost, lightgbm, garch,
  //     msgarch, oujump; balanced-accuracy CV; in_range_rate base rate;
  //     weights from edge over 0.5; gated when avg balanced-acc edge <
  //     MIN_EDGE_OVER_BASELINE (0.10).
  // ===================================================================
  const RANGE_MIN_EDGE = 0.10;          // RangePredictor.MIN_EDGE_OVER_BASELINE
  const DIR_NO_EDGE = 0.02;             // below this AUROC edge → no usable skill
  const GARCH_VARIANTS = ["GARCH(1,1)", "EGARCH(1,1)", "GJR-GARCH(1,1)"];

  function weightsFromEdges(cv, baseline) {
    const e = {}; let tot = 0;
    for (const k in cv) { if (cv[k] == null) continue; const v = Math.max(0, cv[k] - baseline); e[k] = v; tot += v; }
    const ks = Object.keys(e);
    const o = {};
    if (tot <= 1e-9) { const w = 1 / ks.length; ks.forEach(k => o[k] = round(w, 3)); }
    else ks.forEach(k => o[k] = round(e[k] / tot, 3));
    return o;
  }

  function reliability(skill, lo, hi, nbins) {
    const overconf = 0.06 + 0.42 * (1 - skill);
    const bins = [];
    for (let b = 0; b < nbins; b++) {
      const p = lo + (hi - lo) * (b + 0.5) / nbins;
      const gap = overconf * (p - 0.5);
      const actual = clamp(p - gap + 0.035 * gauss() * (1 - 0.6 * skill), 0.02, 0.98);
      const n = Math.round(7 + 30 * Math.exp(-Math.pow((p - 0.6) / 0.27, 2)) + 5 * rnd());
      bins.push({ p: round(p, 3), actual: round(actual, 3), n });
    }
    return bins;
  }

  function memberSkill(cv, baseline, span) { return clamp((cv - baseline) / span, 0, 1); }
  function calibSet(cvObj, members, ensembleSkill, baseline, span, lo, hi, nbins) {
    const out = { ensemble: reliability(ensembleSkill, lo, hi, nbins) };
    members.forEach(m => { if (cvObj[m] == null) return; out[m] = reliability(memberSkill(cvObj[m], baseline, span), lo, hi, nbins); });
    return out;
  }

  const model_perf = {};
  windows.forEach((w) => {
    const s = w._startIdx, e = w._endIdx;
    let vsum = 0, vn = 0;
    for (let i = s; i <= e; i++) { const rv = features[i].realized_vol_20; if (rv != null) { vsum += rv; vn++; } }
    const avgVol = vn ? vsum / vn : 0.16;
    const calm = clamp(1 - (avgVol - 0.10) / 0.24, 0, 1);
    const absMove = Math.abs(bars[e].close / bars[s].close - 1);
    const trend = clamp(absMove / 0.06, 0, 1);
    const moveSign = bars[e].close >= bars[s].close ? 1 : -1;

    const dBase = 0.515 + 0.16 * trend + 0.03 * gauss();
    const dCv = {
      xgboost: round(clamp(dBase + 0.015 + 0.02 * gauss(), 0.45, 0.82), 3),
      lightgbm: round(clamp(dBase - 0.012 + 0.02 * gauss(), 0.45, 0.80), 3),
    };
    const dWeights = weightsFromEdges(dCv, 0.5);
    const dAvg = (dCv.xgboost + dCv.lightgbm) / 2;
    const dEdge = dAvg - 0.5;
    const dGated = dEdge < DIR_NO_EDGE;
    const dSkill = clamp(dEdge / 0.25, 0, 1);

    const inRangeRate = round(clamp(0.55 + 0.30 * calm + 0.04 * gauss(), 0.35, 0.93), 3);
    const rCv = {
      xgboost: round(clamp(0.55 + 0.17 * calm + 0.025 * gauss(), 0.46, 0.83), 3),
      lightgbm: round(clamp(0.54 + 0.16 * calm + 0.025 * gauss(), 0.46, 0.81), 3),
      garch: round(clamp(0.525 + 0.09 * calm + 0.05 * (1 - calm) + 0.03 * gauss(), 0.45, 0.74), 3),
      msgarch: round(clamp(0.515 + 0.11 * calm + 0.03 * gauss(), 0.45, 0.73), 3),
      oujump: round(clamp(0.505 + 0.09 * calm + 0.035 * (1 - calm) + 0.03 * gauss(), 0.44, 0.71), 3),
    };
    if (calm < 0.32 && rnd() > 0.5) rCv.oujump = null;
    const rWeights = weightsFromEdges(rCv, 0.5);
    const rActive = Object.keys(rCv).filter(k => rCv[k] != null);
    const rAvg = rActive.reduce((a, k) => a + rCv[k], 0) / rActive.length;
    const rEdge = rAvg - 0.5;
    const rGated = rEdge < RANGE_MIN_EDGE;
    const rSkill = clamp(rEdge / 0.22, 0, 1);

    const variant = GARCH_VARIANTS[Math.min(GARCH_VARIANTS.length - 1, Math.floor(GARCH_VARIANTS.length * rnd()))];
    const dist = calm > 0.5 ? (rnd() > 0.5 ? "studentst" : "skewt") : "skewt";
    const reg0 = round(clamp(0.08 + 0.05 * (1 - calm) + 0.01 * gauss(), 0.05, 0.22), 3);
    const reg1 = round(clamp(0.24 + 0.16 * (1 - calm) + 0.02 * gauss(), 0.20, 0.62), 3);
    const ouDir = trend < 0.25 ? "neutral" : (moveSign > 0 ? "up" : "down");

    model_perf[w.window_id] = {
      window_id: w.window_id,
      test_start: w.test_start, test_end: w.test_end,
      avg_vol: round(avgVol, 3),
      regime: avgVol > 0.22 ? "high_volatility" : avgVol > 0.16 ? "elevated" : "range_bound",
      directional: {
        predictor: "directional", label: "Directional Predictor",
        task: "3-class · bullish / neutral / bearish",
        purpose: "Trend-continuation signal for credit-spread entries.",
        metric: "AUROC", metric_label: "CV AUROC · one-vs-rest", baseline: 0.5,
        members: ["xgboost", "lightgbm"],
        cv: dCv, fitted_weights: dWeights,
        avg_cv: round(dAvg, 3), avg_edge: round(dEdge, 3),
        gated: dGated,
        gate_reason: dGated ? "Mean OOS edge below the confidence floor — directional signals suppressed for this window." : null,
        confidence_floor: 0.70,
        n_signals: dGated ? 0 : Math.round(6 + 24 * trend + 6 * rnd()),
        version: `dir-2025${String(10 + w.window_id).padStart(2, "0")}`,
        calibration: calibSet(dCv, ["xgboost", "lightgbm"], dSkill, 0.5, 0.25, 0.40, 0.92, 8),
      },
      range: {
        predictor: "range", label: "Range Predictor",
        task: "binary · stays in ±5% over 30d",
        purpose: "P(in-range) confidence gate for iron-condor entries.",
        metric: "Balanced acc", metric_label: "CV balanced accuracy", baseline: 0.5,
        members: ["xgboost", "lightgbm", "garch", "msgarch", "oujump"],
        cv: rCv, fitted_weights: rWeights,
        avg_cv: round(rAvg, 3), avg_edge: round(rEdge, 3),
        in_range_rate: inRangeRate, min_edge: RANGE_MIN_EDGE,
        gated: rGated,
        gate_reason: rGated ? `Mean balanced-accuracy edge ${round(rEdge, 3)} < ${RANGE_MIN_EDGE} floor — model has no skill this window, so it refuses to predict.` : null,
        n_signals: rGated ? 0 : Math.round(5 + 22 * calm + 5 * rnd()),
        version: `range-v-2025${String(10 + w.window_id).padStart(2, "0")}`,
        calibration: calibSet(rCv, ["xgboost", "lightgbm", "garch", "msgarch", "oujump"], rSkill, 0.5, 0.22, 0.20, 0.92, 9),
        garch_meta: {
          variant, dist,
          bic: round(1680 + 520 * (1 - calm) + 120 * gauss(), 1),
          fallback_used: rnd() > 0.88,
          jb_pvalue: round(clamp(0.02 + 0.35 * calm + 0.05 * gauss(), 0.001, 0.6), 3),
          resid_skewness: round(-0.15 - 0.4 * (1 - calm) + 0.1 * gauss(), 3),
        },
        msgarch_meta: {
          converged: rnd() > 0.12,
          bic: round(1720 + 560 * (1 - calm) + 130 * gauss(), 1),
          regime0_vol: reg0, regime1_vol: reg1,
          transition: [
            [round(clamp(0.93 + 0.03 * gauss(), 0.80, 0.985), 3), 0],
            [0, round(clamp(0.90 + 0.04 * gauss(), 0.78, 0.985), 3)],
          ],
        },
        oujump_meta: rCv.oujump == null ? null : {
          converged: rnd() > 0.1,
          bic: round(1760 + 540 * (1 - calm) + 140 * gauss(), 1),
          direction: ouDir,
          confidence: round(clamp(0.5 + 0.35 * trend + 0.05 * gauss(), 0.3, 0.95), 3),
          kappa: round(clamp(0.6 + 1.8 * (1 - calm) + 0.2 * gauss(), 0.2, 4.0), 3),
          theta: round(clamp(0.12 + 0.18 * (1 - calm), 0.08, 0.45), 3),
          sigma: round(clamp(0.18 + 0.3 * (1 - calm) + 0.03 * gauss(), 0.1, 0.6), 3),
          jump_intensity: round(clamp(0.05 + 0.22 * (1 - calm) + 0.02 * gauss(), 0.01, 0.4), 3),
          jump_mean: round(-0.01 - 0.03 * (1 - calm) + 0.01 * gauss(), 3),
        },
      },
    };
    const tm = model_perf[w.window_id].range.msgarch_meta.transition;
    tm[0][1] = round(1 - tm[0][0], 3);
    tm[1][0] = round(1 - tm[1][1], 3);
  });

  const MODEL_META = {
    directional: {
      label: "Directional Predictor", module: "ait.ml.ensemble.DirectionPredictor",
      task: "3-class · bullish / neutral / bearish",
      metric: "AUROC", metric_label: "CV AUROC · one-vs-rest", baseline: 0.5,
      members: ["xgboost", "lightgbm"],
      objective: "Trend continuation for credit-spread entries (confidence floor 0.70).",
    },
    range: {
      label: "Range Predictor", module: "ait.ml.range_predictor.RangePredictor",
      task: "binary · P(stays in ±5% over 30d)",
      metric: "Balanced acc", metric_label: "CV balanced accuracy", baseline: 0.5,
      members: ["xgboost", "lightgbm", "garch", "msgarch", "oujump"],
      objective: "Iron-condor confidence gate (skips when edge < 0.10 over baseline).",
    },
    member_labels: { xgboost: "XGBoost", lightgbm: "LightGBM", garch: "GARCH", msgarch: "MS-GARCH", oujump: "OU-Kou-GARCH" },
    member_family: { xgboost: "ml", lightgbm: "ml", garch: "stat", msgarch: "stat", oujump: "stat" },
    member_kind: {
      xgboost: "Gradient-boosted trees", lightgbm: "Gradient-boosted trees",
      garch: "Conditional-volatility model", msgarch: "Regime-switching volatility",
      oujump: "Mean-reversion + jump-diffusion",
    },
  };

  // strip internal helper fields from bars before exposing
  const cleanBars = bars.map(({ _ret, _annVol, ...b }) => b);

  window.AIT = {
    experiment,
    bars: cleanBars,
    features,
    predictions,
    windows,
    trades,
    equityCurve,
    optuna_studies,
    model_perf,
    MODEL_META,
    // convenience: list of selectable experiments (only one fully populated)
    experiments: [
      { id: experiment.id, name: experiment.name, strategy: "iron_condor", symbols: ["QQQ"], status: "completed", run_at: experiment.run_at, total_return: results.total_return, sharpe: results.sharpe_ratio, win_rate: results.win_rate, trades: results.total_trades },
    ],
    FEATURE_LIBRARY: [
      { key: "rsi_14", label: "RSI (14)", group: "Momentum", pane: "rsi", scale: [0, 100] },
      { key: "macd_hist", label: "MACD Histogram", group: "Momentum", pane: "macd" },
      { key: "realized_vol_20", label: "Realized Vol (20d)", group: "Volatility", pane: "vol" },
      { key: "atr_pct", label: "ATR %", group: "Volatility", pane: "vol" },
      { key: "iv_rank", label: "IV Rank", group: "Volatility", pane: "iv" },
      { key: "vix_level", label: "VIX", group: "Cross-Asset", pane: "iv" },
      { key: "bb_position", label: "Bollinger %B", group: "Volatility", pane: "bb" },
      { key: "hurst_wavelet", label: "Hurst (wavelet)", group: "Fractal", pane: "fractal" },
      { key: "sentiment_composite", label: "Sentiment", group: "Sentiment", pane: "sent" },
      { key: "put_call_ratio", label: "Put/Call Ratio", group: "Sentiment", pane: "sent" },
    ],
  };
})();
