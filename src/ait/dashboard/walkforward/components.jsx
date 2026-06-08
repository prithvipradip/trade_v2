/* ============================================================
   Experiment Analysis components (Babel/JSX)
   Exposes shared components on window.
   ============================================================ */
const { useState, useRef, useEffect, useMemo, useCallback } = React;

/* ---------- formatters ---------- */
const fmtPct = (x, d = 1) => (x == null ? "–" : (x * 100).toFixed(d) + "%");
const fmtMoney = (x, d = 0) => (x == null ? "–" : (x < 0 ? "-$" : "$") + Math.abs(x).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d }));
const fmtNum = (x, d = 2) => (x == null ? "–" : (+x).toFixed(d));
const cx = (...a) => a.filter(Boolean).join(" ");

/* ---------- small atoms ---------- */
function Chip({ children, tone }) {
  return <span className={cx("chip", tone && "chip-" + tone)}>{children}</span>;
}

function Metric({ label, value, tone, sub }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className={cx("metric-value", tone)}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

/* ---------- experiment header ---------- */
function ExperimentHeader({ exp, windowMetrics }) {
  const r = exp.results;
  const w = windowMetrics;

  // Metric strip: window-level when a window is selected, overall otherwise.
  const strip = w ? (
    <div className="metric-strip">
      <div className="window-metrics-badge">
        Window {w._window.window_id} · OOS {w._window.test_start} → {w._window.test_end}
      </div>
      <Metric label="Return" value={fmtPct(w.total_return)} tone={w.total_return >= 0 ? "up" : "down"} sub="this window" />
      <Metric label="Sharpe" value={fmtNum(w.sharpe_ratio)} />
      <Metric label="Sortino" value={w.sortino_ratio != null ? fmtNum(w.sortino_ratio) : "–"} />
      <Metric label="Win Rate" value={fmtPct(w.win_rate)} />
      <Metric label="Profit Factor" value={w.profit_factor != null ? fmtNum(w.profit_factor) : "–"} tone={w.profit_factor != null ? (w.profit_factor >= 1 ? "up" : "down") : null} />
      <Metric label="Max Drawdown" value={fmtPct(w.max_drawdown)} tone="down" />
      <Metric label="P&L" value={fmtMoney(w.pnl)} tone={w.pnl >= 0 ? "up" : "down"} sub="window total" />
      <Metric label="Expectancy" value={w.expectancy != null ? fmtMoney(w.expectancy) : "–"} tone={w.expectancy != null ? (w.expectancy >= 0 ? "up" : "down") : null} sub="per trade" />
      <Metric label="Trades" value={w.total_trades} sub={w.avg_hold_days != null ? `avg hold ${w.avg_hold_days}d` : ""} />
      <Metric label="Avg Win / Loss" value={w.avg_win != null ? fmtMoney(w.avg_win) : "–"} sub={w.avg_loss != null ? `loss ${fmtMoney(-w.avg_loss)}` : ""} />
    </div>
  ) : (
    <div className="metric-strip">
      <Metric label="Total Return" value={fmtPct(r.total_return)} tone={r.total_return >= 0 ? "up" : "down"} />
      <Metric label="Sharpe" value={fmtNum(r.sharpe_ratio)} />
      <Metric label="Sortino" value={r.sortino_ratio != null ? fmtNum(r.sortino_ratio) : "–"} />
      <Metric label="Win Rate" value={fmtPct(r.win_rate)} />
      <Metric label="Profit Factor" value={fmtNum(r.profit_factor)} tone={r.profit_factor >= 1 ? "up" : "down"} />
      <Metric label="Max Drawdown" value={fmtPct(r.max_drawdown)} tone="down" />
      <Metric label="Consistency" value={fmtPct(r.consistency, 0)} sub={`${Math.round(r.consistency * r.windows)}/${r.windows} windows +`} />
      <Metric label="Expectancy" value={fmtMoney(r.expectancy)} tone={r.expectancy >= 0 ? "up" : "down"} sub="per trade" />
      <Metric label="Trades" value={r.total_trades} sub={`avg hold ${r.avg_hold_days}d`} />
      <Metric label="Final Capital" value={fmtMoney(r.final_capital)} sub={`from ${fmtMoney(exp.config.initial_capital)}`} />
    </div>
  );

  return (
    <div className="exp-header">
      <div className="exp-id-row">
        <div>
          <div className="exp-eyebrow">
            <span className="exp-id">{exp.id}</span>
            <Chip tone="ok">{exp.status}</Chip>
          </div>
          <h1 className="exp-name">{exp.name}</h1>
          <div className="exp-meta">
            <span><b>Strategy</b> {exp.strategy_label}</span>
            <span><b>Symbols</b> {exp.symbols.join(", ")}</span>
            <span><b>Range</b> {exp.date_range.start} → {exp.date_range.end}</span>
            <span><b>Objective</b> {exp.config.objective}</span>
            <span><b>Windows</b> {r.windows} · train {exp.config.train_days}d / test {exp.config.test_days}d / step {exp.config.step_days}d</span>
            <span><b>Run</b> {exp.run_at.replace("T", " ")} · sha {exp.git_sha}</span>
          </div>
        </div>
      </div>
      {strip}
    </div>
  );
}

/* ---------- series / feature toggle panel ---------- */
function SeriesPanel({ active, onToggle, windows, selectedWindow, onWindow, hasPredictions }) {
  const overlays = [
    { key: "ov:sma20", label: "SMA 20", swatch: "var(--c-sma20)" },
    { key: "ov:sma50", label: "SMA 50", swatch: "var(--c-sma50)" },
    { key: "ov:bb", label: "Bollinger Bands", swatch: "var(--c-bb)" },
  ];
  const panes = [
    { key: "pane:ml", label: "ML Predictions", desc: hasPredictions ? "P(in-range) · dir conf" : "run needed for real data", swatch: "var(--c-data)", unavailable: !hasPredictions },
    { key: "pane:rsi", label: "RSI (14)", desc: "momentum", swatch: "var(--c-rsi)" },
    { key: "pane:macd", label: "MACD", desc: "momentum", swatch: "var(--c-data)" },
    { key: "pane:vol", label: "Realized Vol / ATR", desc: "volatility", swatch: "var(--c-down)" },
    { key: "pane:iv", label: "IV Rank / VIX", desc: "vol regime", swatch: "var(--c-data)" },
    { key: "pane:fractal", label: "Hurst", desc: "fractal regime", swatch: "var(--c-up)" },
    { key: "pane:sent", label: "Sentiment / P-C", desc: "flow", swatch: "var(--c-data)" },
  ];
  const Row = ({ item }) => (
    <button
      className={cx("toggle-row", active.has(item.key) && "on", item.unavailable && "unavailable")}
      onClick={() => !item.unavailable && onToggle(item.key)}
      style={item.unavailable ? { opacity: 0.45, cursor: "default" } : undefined}
    >
      <span className="toggle-swatch" style={{ background: item.swatch, opacity: active.has(item.key) ? 1 : 0.25 }} />
      <span className="toggle-text">
        <span className="toggle-label">{item.label}</span>
        {item.desc && <span className="toggle-desc">{item.desc}</span>}
      </span>
      <span className={cx("toggle-switch", active.has(item.key) && !item.unavailable && "on")} />
    </button>
  );
  return (
    <div className="series-panel">
      <div className="panel-section">
        <div className="panel-heading">Window</div>
        <select className="window-select" value={selectedWindow} onChange={e => onWindow(e.target.value === "all" ? "all" : +e.target.value)}>
          <option value="all">All windows ({windows.length})</option>
          {windows.map(w => <option key={w.window_id} value={w.window_id}>Window {w.window_id} · {w.test_start} → {w.test_end}</option>)}
        </select>
      </div>
      <div className="panel-section">
        <div className="panel-heading">Price overlays</div>
        {overlays.map(o => <Row key={o.key} item={o} />)}
      </div>
      <div className="panel-section">
        <div className="panel-heading">Indicator panes · own axis</div>
        {panes.map(p => <Row key={p.key} item={p} />)}
      </div>
    </div>
  );
}

/* ---------- chart bridge ---------- */
function ChartArea({ data, activeSeries, theme, onHover, onPin, pinned, extHoverTime, windowRange }) {
  const ref = useRef();
  const chartRef = useRef();
  const onHoverRef = useRef(onHover); onHoverRef.current = onHover;
  const onPinRef = useRef(onPin); onPinRef.current = onPin;

  useEffect(() => {
    chartRef.current = new WFChart(ref.current);
    return () => chartRef.current && chartRef.current.destroy();
  }, []);

  useEffect(() => {
    if (!chartRef.current) return;
    try {
      // windowRange is passed into render() so the zoom is applied atomically
      // inside chart construction — no separate effect needed (avoids race condition
      // where fitContent() in render() would clobber a subsequent setWindowRange()).
      chartRef.current.render({
        data, activeSeries, theme, windowRange,
        onHover: t => onHoverRef.current && onHoverRef.current(t),
        onMarkerClick: tr => onPinRef.current && onPinRef.current(tr),
      });
    } catch (e) { console.error("chart render error:", e); }
  }, [activeSeries, theme, data, windowRange]);

  useEffect(() => {
    if (!chartRef.current) return;
    try {
      if (pinned) chartRef.current.pinTrade(pinned, true); else chartRef.current.unpin();
    } catch (e) { console.error("pinTrade error:", e); }
  }, [pinned]);

  useEffect(() => {
    if (!chartRef.current) return;
    try {
      if (extHoverTime) chartRef.current.setHoverTime(extHoverTime); else chartRef.current.clearHover();
    } catch (e) {}
  }, [extHoverTime]);

  return <div className="chart-host" ref={ref} />;
}

/* ---------- exit reason styling ---------- */
const EXIT_TONE = {
  profit_target: "up", trailing_stop: "up", expiry: "neutral",
  stop_loss: "down", thesis_invalidated: "down", backtest_end: "neutral",
};

/* ---------- trades table ---------- */
function TradesTable({ trades, onHoverRow, onSelect, selectedId, hoverTime }) {
  const [sort, setSort] = useState({ key: "entry_date", dir: 1 });
  const [q, setQ] = useState("");
  const [exitFilter, setExitFilter] = useState("all");
  const [pnlFilter, setPnlFilter] = useState("all");

  const filtered = useMemo(() => {
    let rows = trades.filter(t => {
      if (exitFilter !== "all" && t.exit_reason !== exitFilter) return false;
      if (pnlFilter === "win" && t.pnl <= 0) return false;
      if (pnlFilter === "loss" && t.pnl > 0) return false;
      if (q) {
        const s = (t.id + " " + t.exit_reason + " " + t.entry_regime + " " + t.entry_date).toLowerCase();
        if (!s.includes(q.toLowerCase())) return false;
      }
      return true;
    });
    rows = [...rows].sort((a, b) => {
      const k = sort.key; let av = a[k], bv = b[k];
      if (typeof av === "string") return av.localeCompare(bv) * sort.dir;
      return ((av || 0) - (bv || 0)) * sort.dir;
    });
    return rows;
  }, [trades, sort, q, exitFilter, pnlFilter]);

  const exitReasons = useMemo(() => [...new Set(trades.map(t => t.exit_reason))], [trades]);
  const Th = ({ k, children, num }) => (
    <th className={num ? "num" : ""} onClick={() => setSort(s => ({ key: k, dir: s.key === k ? -s.dir : 1 }))}>
      {children}{sort.key === k && <span className="sort-caret">{sort.dir > 0 ? "▲" : "▼"}</span>}
    </th>
  );

  return (
    <div className="data-panel">
      <div className="filter-bar">
        <input className="search" placeholder="Search trades…" value={q} onChange={e => setQ(e.target.value)} />
        <select value={exitFilter} onChange={e => setExitFilter(e.target.value)}>
          <option value="all">All exits</option>
          {exitReasons.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
        <div className="seg">
          {["all", "win", "loss"].map(v => (
            <button key={v} className={cx(pnlFilter === v && "on")} onClick={() => setPnlFilter(v)}>{v}</button>
          ))}
        </div>
        <span className="filter-count">{filtered.length} of {trades.length}</span>
      </div>
      <div className="table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              <Th k="id">Trade</Th>
              <Th k="entry_date">Entry</Th>
              <Th k="exit_date">Exit</Th>
              <Th k="hold_days" num>Hold</Th>
              <Th k="entry_confidence" num>P(range)</Th>
              <Th k="entry_iv_rank" num>IV rank</Th>
              <Th k="entry_vix_level" num>VIX</Th>
              <Th k="entry_regime">Regime</Th>
              <Th k="exit_reason">Exit reason</Th>
              <Th k="contracts" num>Qty</Th>
              <Th k="pnl" num>P&L</Th>
              <Th k="return_pct" num>Return</Th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(t => (
              <tr key={t.id}
                className={cx(selectedId === t.id && "sel", hoverTime && (t.entry_date === hoverTime || t.exit_date === hoverTime) && "hl")}
                onMouseEnter={() => onHoverRow(t.entry_date)}
                onMouseLeave={() => onHoverRow(null)}
                onClick={() => onSelect(t)}>
                <td className="mono">{t.id}</td>
                <td className="mono dim">{t.entry_date}</td>
                <td className="mono dim">{t.exit_date}</td>
                <td className="num mono">{t.hold_days}d</td>
                <td className="num mono">{fmtNum(t.entry_confidence)}</td>
                <td className="num mono">{fmtNum(t.entry_iv_rank)}</td>
                <td className="num mono">{fmtNum(t.entry_vix_level, 1)}</td>
                <td><Chip tone={t.entry_regime === "range_bound" ? "ok" : "warn"}>{t.entry_regime}</Chip></td>
                <td><span className={cx("exit-tag", EXIT_TONE[t.exit_reason])}>{t.exit_reason}</span></td>
                <td className="num mono">{t.contracts}</td>
                <td className={cx("num mono bold", t.pnl >= 0 ? "up" : "down")}>{fmtMoney(t.pnl)}</td>
                <td className={cx("num mono", t.return_pct >= 0 ? "up" : "down")}>{fmtPct(t.return_pct)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ---------- time-series (column) data table ---------- */
function TimeSeriesTable({ data, windowRange }) {
  const ALL_COLS = [
    { k: "time", label: "Date", fmt: v => v },
    { k: "close", label: "Close", src: "bars", fmt: v => fmtNum(v) },
    { k: "volume", label: "Volume", src: "bars", fmt: v => (v / 1e6).toFixed(1) + "M" },
    { k: "rsi_14", label: "RSI", fmt: v => fmtNum(v, 1) },
    { k: "macd_hist", label: "MACD hist", fmt: v => fmtNum(v, 4) },
    { k: "realized_vol_20", label: "RVol20", fmt: v => fmtPct(v) },
    { k: "atr_pct", label: "ATR%", fmt: v => fmtPct(v, 2) },
    { k: "iv_rank", label: "IV rank", fmt: v => fmtNum(v) },
    { k: "vix_level", label: "VIX", fmt: v => fmtNum(v, 1) },
    { k: "bb_position", label: "%B", fmt: v => fmtNum(v) },
    { k: "hurst_wavelet", label: "Hurst", fmt: v => fmtNum(v, 3) },
    { k: "sentiment_composite", label: "Sentiment", fmt: v => fmtNum(v) },
    { k: "put_call_ratio", label: "P/C", fmt: v => fmtNum(v) },
    { k: "range_prob", label: "P(range)", src: "pred", fmt: v => fmtNum(v) },
    { k: "dir_conf", label: "Dir conf", src: "pred", fmt: v => fmtNum(v) },
    { k: "dir_class", label: "Dir class", src: "pred", fmt: v => v },
    { k: "meta_take", label: "Meta", src: "pred", fmt: v => (v ? "✓ take" : "skip") },
  ];
  const [cols, setCols] = useState(new Set(ALL_COLS.map(c => c.k)));
  const [q, setQ] = useState("");
  const [colMenu, setColMenu] = useState(false);

  const rows = useMemo(() => {
    const byTimeF = Object.fromEntries(data.features.map(f => [f.time, f]));
    const byTimeB = Object.fromEntries(data.bars.map(b => [b.time, b]));
    const byTimeP = Object.fromEntries(data.predictions.map(p => [p.time, p]));
    let out = data.bars.map(b => ({ bar: b, f: byTimeF[b.time], p: byTimeP[b.time] }));
    if (windowRange) out = out.filter(r => r.bar.time >= windowRange[0] && r.bar.time <= windowRange[1]);
    if (q) out = out.filter(r => r.bar.time.includes(q));
    return out.reverse();
  }, [data, q, windowRange]);

  const valueOf = (row, c) => {
    if (c.src === "bars") return row.bar[c.k];
    if (c.src === "pred") return row.p ? row.p[c.k] : null;
    if (c.k === "time") return row.bar.time;
    return row.f ? row.f[c.k] : null;
  };
  const visible = ALL_COLS.filter(c => cols.has(c.k));

  return (
    <div className="data-panel">
      <div className="filter-bar">
        <input className="search" placeholder="Filter by date (e.g. 2024-09)…" value={q} onChange={e => setQ(e.target.value)} />
        <div className="col-menu-wrap">
          <button className="btn-ghost" onClick={() => setColMenu(m => !m)}>Columns ({visible.length})</button>
          {colMenu && (
            <div className="col-menu" onMouseLeave={() => setColMenu(false)}>
              {ALL_COLS.map(c => (
                <label key={c.k} className="col-menu-item">
                  <input type="checkbox" checked={cols.has(c.k)} disabled={c.k === "time"}
                    onChange={() => setCols(s => { const n = new Set(s); n.has(c.k) ? n.delete(c.k) : n.add(c.k); return n; })} />
                  {c.label}
                </label>
              ))}
            </div>
          )}
        </div>
        <span className="filter-count">{rows.length} bars</span>
      </div>
      <div className="table-scroll">
        <table className="data-table compact">
          <thead><tr>{visible.map(c => <th key={c.k} className={c.k === "time" || c.k === "dir_class" ? "" : "num"}>{c.label}</th>)}</tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.bar.time}>
                {visible.map(c => {
                  const v = valueOf(r, c);
                  const isClass = c.k === "dir_class";
                  return <td key={c.k} className={cx("mono", c.k !== "time" && !isClass && "num", c.k === "time" && "dim")}>
                    {isClass ? <Chip tone={v === "bullish" ? "ok" : v === "bearish" ? "down" : "neutral"}>{v}</Chip> : (v == null ? "–" : c.fmt(v))}
                  </td>;
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

Object.assign(window, {
  fmtPct, fmtMoney, fmtNum, cx, Chip, Metric,
  ExperimentHeader, SeriesPanel, ChartArea, TradesTable, TimeSeriesTable, EXIT_TONE,
});
