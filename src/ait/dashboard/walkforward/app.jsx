/* ============================================================
   App shell — tabs, theme, state wiring
   ============================================================ */
const LS = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch (e) {} },
};

function ThemeToggle({ theme, onChange }) {
  return (
    <div className="theme-toggle" role="group" aria-label="theme">
      {["light", "dark"].map(m => (
        <button key={m} className={cx(theme === m && "on")} onClick={() => onChange(m)}>
          {m === "light" ? "☀" : "☾"} {m}
        </button>
      ))}
    </div>
  );
}

function App() {
  const root = window.AIT;
  // Experiment switching: use _experiment_data map when multiple experiments loaded
  const [activeExpId, setActiveExpId] = useState(() => root.experiment.id);
  const data = useMemo(() => {
    if (root._experiment_data && root._experiment_data[activeExpId]) {
      // Merge active experiment data with shared lists (experiments, FEATURE_LIBRARY)
      return { ...root._experiment_data[activeExpId], experiments: root.experiments, FEATURE_LIBRARY: root.FEATURE_LIBRARY };
    }
    return root;
  }, [activeExpId]);

  const [theme, setTheme] = useState(() => LS.get("wf_theme", "dark"));
  const [tab, setTab] = useState("analysis");
  const [activeSeries, setActiveSeries] = useState(() => new Set(LS.get("wf_series", ["ov:sma20", "ov:bb", "pane:ml", "pane:rsi"])));
  const [selectedWindow, setSelectedWindow] = useState("all");
  const [selectedTrade, setSelectedTrade] = useState(null);
  const [hoverTime, setHoverTime] = useState(null);
  const [tableHoverTime, setTableHoverTime] = useState(null);
  const [dataView, setDataView] = useState("trades");
  const [viewMode, setViewMode] = useState("both");
  const [tw, setTweak] = useTweaks({ splitPct: 63, subPaneHeight: 40, density: "regular" });

  // Reset window/trade selection when experiment changes
  useEffect(() => { setSelectedWindow("all"); setSelectedTrade(null); }, [activeExpId]);
  useEffect(() => { document.documentElement.dataset.theme = theme; LS.set("wf_theme", theme); }, [theme]);
  useEffect(() => { LS.set("wf_series", [...activeSeries]); }, [activeSeries]);

  const toggleSeries = useCallback(k => setActiveSeries(s => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; }), []);

  const selectedWindowObj = selectedWindow === "all" ? null : data.windows.find(w => w.window_id === selectedWindow) || null;
  const windowRange = selectedWindowObj ? [selectedWindowObj.test_start, selectedWindowObj.test_end] : null;

  // trades and viewData must be declared before windowMetrics (which uses trades)
  const trades = useMemo(() => selectedWindow === "all" ? data.trades : data.trades.filter(t => t.window_id === selectedWindow), [selectedWindow, data]);
  const viewData = useMemo(() => ({ ...data, trades }), [trades, data]);

  // Per-window metric strip — computed from the already-filtered trades + window object.
  const windowMetrics = useMemo(() => {
    if (!selectedWindowObj) return null;
    const w = selectedWindowObj;
    const pnls = trades.map(t => t.pnl).filter(p => p != null);
    const wins = trades.filter(t => t.pnl > 0);
    const losses = trades.filter(t => t.pnl <= 0);
    const grossWin  = wins.reduce((a, t)   => a + t.pnl, 0);
    const grossLoss = Math.abs(losses.reduce((a, t) => a + t.pnl, 0));
    const mean = pnls.length ? pnls.reduce((a, x) => a + x, 0) / pnls.length : 0;
    const downside = pnls.filter(p => p < mean);
    let sortino = null;
    if (downside.length >= 2) {
      const dsStd = Math.sqrt(downside.reduce((a, p) => a + (p - mean) ** 2, 0) / (downside.length - 1));
      if (dsStd > 0) sortino = (mean / dsStd) * Math.sqrt(252);
    }
    const holdDays = trades.map(t => t.hold_days).filter(d => d != null && d > 0);
    return {
      total_return:   w.return_pct,
      sharpe_ratio:   w.sharpe,
      sortino_ratio:  sortino != null ? Math.round(sortino * 100) / 100 : null,
      win_rate:       w.win_rate,
      profit_factor:  grossLoss > 0 ? Math.round((grossWin / grossLoss) * 100) / 100 : null,
      max_drawdown:   w.max_drawdown,
      pnl:            w.pnl,
      expectancy:     pnls.length ? Math.round(mean * 100) / 100 : null,
      total_trades:   w.trades,
      avg_hold_days:  holdDays.length ? Math.round(holdDays.reduce((a, d) => a + d, 0) / holdDays.length * 10) / 10 : null,
      avg_win:        wins.length  ? Math.round(grossWin  / wins.length  * 100) / 100 : null,
      avg_loss:       losses.length ? Math.round(grossLoss / losses.length * 100) / 100 : null,
      _window:        w,
    };
  }, [selectedWindowObj, trades]);

  const onSelectTrade = useCallback(t => setSelectedTrade(t), []);
  const onHoverChart = useCallback(t => setHoverTime(t), []);
  const onHoverRow = useCallback(t => setTableHoverTime(t), []);

  const showChart = viewMode !== "table";
  const showTable = viewMode !== "plot";

  const handleExpChange = (id) => {
    setActiveExpId(id);
    setTab("analysis");
  };

  return (
    <div className="app" style={{ "--split-pct": tw.splitPct + "%", "--sub-pane-h": tw.subPaneHeight + "px", "--row-pad": tw.density === "compact" ? "4px" : tw.density === "comfy" ? "10px" : "7px" }}>
      <header className="topbar">
        <div className="brand">
          <span className="brand-mark">◆</span>
          <span className="brand-name">AIT <span className="dim">·</span> Walk-Forward Lab</span>
        </div>
        <div className="exp-switch">
          <select value={activeExpId} onChange={e => handleExpChange(e.target.value)}>
            {root.experiments.map(e => <option key={e.id} value={e.id}>{e.id} — {e.strategy} · {e.symbols.join(",")}</option>)}
          </select>
        </div>
        <nav className="tabs">
          <button className={cx(tab === "analysis" && "on")} onClick={() => setTab("analysis")}>Experiment Analysis</button>
          <button className={cx(tab === "optuna" && "on")} onClick={() => setTab("optuna")}>Optuna Optimization</button>
          <button className={cx(tab === "models" && "on")} onClick={() => setTab("models")}>Predictor Models</button>
        </nav>
        <ThemeToggle theme={theme} onChange={setTheme} />
      </header>

      {tab === "analysis" && (
        <main className="analysis">
          <ExperimentHeader exp={data.experiment} windowMetrics={windowMetrics} />
          <div className="workspace">
            <SeriesPanel active={activeSeries} onToggle={toggleSeries} windows={data.windows} selectedWindow={selectedWindow} onWindow={setSelectedWindow} hasPredictions={!!data._has_predictions} />
            <div className="work-main">
              <div className="work-toolbar">
                <div className="seg view-seg">
                  {[["both", "Both"], ["plot", "Plot"], ["table", "Table"]].map(([v, l]) => (
                    <button key={v} className={cx(viewMode === v && "on")} onClick={() => setViewMode(v)}>{l}</button>
                  ))}
                </div>
                {showTable && (
                  <div className="seg data-seg">
                    {[["trades", "Trades"], ["timeseries", "Time Series"]].map(([v, l]) => (
                      <button key={v} className={cx(dataView === v && "on")} onClick={() => setDataView(v)}>{l}</button>
                    ))}
                  </div>
                )}
                <div className="toolbar-spacer" />
                {selectedTrade && <button className="btn-ghost" onClick={() => { setSelectedTrade(null); }}>Clear pinned trade</button>}
                <span className="hint">Hover chart ⇄ table · click a trade to pin & inspect</span>
              </div>

              <div className={cx("work-body", "vm-" + viewMode)}>
                {showChart && (
                  <div className="chart-wrap">
                    <ChartArea data={viewData} activeSeries={activeSeries} theme={theme}
                      onHover={onHoverChart} onPin={onSelectTrade} pinned={selectedTrade} extHoverTime={tableHoverTime}
                      windowRange={windowRange} />
                  </div>
                )}
                {showTable && (
                  <div className="table-wrap">
                    {dataView === "trades"
                      ? <TradesTable trades={trades} onHoverRow={onHoverRow} onSelect={onSelectTrade} selectedId={selectedTrade && selectedTrade.id} hoverTime={hoverTime} />
                      : <TimeSeriesTable data={data} windowRange={windowRange} />}
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      )}

      {tab === "optuna" && (
        <main className="optuna-main">
          <OptunaTab data={data} />
        </main>
      )}

      {tab === "models" && (
        <main className="pm-main">
          {data.model_perf
            ? <PredictorModelsTab data={data} />
            : <div style={{padding:"40px",color:"var(--text-dim)",textAlign:"center"}}>Model performance data not available for this experiment.<br/>Re-export with a post-Layer-2 run to populate this tab.</div>
          }
        </main>
      )}

      {selectedTrade && <DecisionPanel trade={selectedTrade} onClose={() => setSelectedTrade(null)} />}

      <TweaksPanel title="Tweaks">
        <TweakSection label="Layout" />
        <TweakSlider label="Plot / table split" value={tw.splitPct} min={30} max={82} step={1} unit="% plot"
          onChange={v => setTweak("splitPct", v)} />
        <TweakSlider label="Indicator pane height" value={tw.subPaneHeight} min={28} max={120} step={2} unit="px"
          onChange={v => setTweak("subPaneHeight", v)} />
        <TweakSection label="Density" />
        <TweakRadio label="Table rows" value={tw.density} options={["compact", "regular", "comfy"]}
          onChange={v => setTweak("density", v)} />
      </TweaksPanel>
    </div>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
