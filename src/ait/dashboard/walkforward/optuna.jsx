/* ============================================================
   Optuna optimization tab
   ============================================================ */

/* ---- tiny SVG plotting helpers ---- */
function useSize() {
  const ref = useRef();
  const [size, setSize] = useState({ w: 600, h: 300 });
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(es => { const r = es[0].contentRect; setSize({ w: r.width, h: r.height }); });
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, size];
}

function niceTicks(min, max, n = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min;
  const step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(8));
  return ticks;
}

/* ---- objective-over-trials chart ---- */
function TrialsChart({ study, onHoverTrial }) {
  const [ref, { w, h }] = useSize();
  const [hover, setHover] = useState(null);
  const pad = { l: 52, r: 16, t: 16, b: 34 };
  const trials = study.trials;
  const complete = trials.filter(t => t.value != null);
  const vals = complete.map(t => t.value);
  const yMin = Math.min(...vals, 0), yMax = Math.max(...vals, 0.1);
  const xMax = trials.length - 1;
  const X = i => pad.l + (i / Math.max(1, xMax)) * (w - pad.l - pad.r);
  const Y = v => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (h - pad.t - pad.b);

  // best-so-far
  let best = -Infinity; const bestLine = [];
  trials.forEach(t => { if (t.value != null && t.value > best) best = t.value; bestLine.push({ i: t.number, v: best === -Infinity ? null : best }); });
  const bestPath = bestLine.filter(p => p.v != null).map((p, idx) => `${idx === 0 ? "M" : "L"}${X(p.i)},${Y(p.v)}`).join(" ");

  const yticks = niceTicks(yMin, yMax, 5);
  const xticks = niceTicks(0, xMax, 6).filter(t => t >= 0 && t <= xMax);

  return (
    <div className="plot-host" ref={ref}>
      <svg width={w} height={h} onMouseLeave={() => { setHover(null); onHoverTrial && onHoverTrial(null); }}>
        {yticks.map(t => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} className="plot-grid" />
            <text x={pad.l - 8} y={Y(t) + 3} className="plot-axis num" textAnchor="end">{t.toFixed(2)}</text>
          </g>
        ))}
        {xticks.map(t => <text key={t} x={X(t)} y={h - pad.b + 18} className="plot-axis" textAnchor="middle">{t}</text>)}
        <text x={(w) / 2} y={h - 4} className="plot-axis-title" textAnchor="middle">trial number</text>
        {/* best-so-far line */}
        <path d={bestPath} className="plot-best-line" />
        {/* points */}
        {trials.map(t => {
          const pruned = t.value == null;
          const cy = pruned ? h - pad.b - 6 : Y(t.value);
          const isBest = t.number === study.best_trial;
          return (
            <circle key={t.number} cx={X(t.number)} cy={cy} r={isBest ? 6 : pruned ? 2.5 : 4}
              className={cx("plot-pt", pruned ? "pruned" : "complete", isBest && "best")}
              onMouseEnter={() => { setHover(t); onHoverTrial && onHoverTrial(t); }} />
          );
        })}
        {hover && (
          <g>
            <line x1={X(hover.number)} x2={X(hover.number)} y1={pad.t} y2={h - pad.b} className="plot-cross" />
          </g>
        )}
      </svg>
      {hover && (
        <div className="plot-tip" style={{ left: Math.min(w - 150, X(hover.number) + 10), top: 20 }}>
          <div className="tip-row"><b>Trial {hover.number}</b> {hover.number === study.best_trial && <span className="tip-best">★ best</span>}</div>
          {hover.value == null
            ? <div className="tip-row down">PRUNED · {hover.n_trades} trades</div>
            : <>
              <div className="tip-row">objective <b className="num">{fmtNum(hover.value, 4)}</b></div>
              <div className="tip-row dim num">Sharpe {fmtNum(hover.sharpe)} · WR {fmtPct(hover.win_rate)} · DD {fmtPct(hover.max_drawdown)}</div>
              <div className="tip-row dim num">{hover.n_trades} trades · {hover.duration_s}s</div>
            </>}
        </div>
      )}
    </div>
  );
}

/* ---- param vs objective scatter ---- */
function ParamScatter({ study, param }) {
  const [ref, { w, h }] = useSize();
  const pad = { l: 52, r: 16, t: 16, b: 34 };
  const pts = study.trials.filter(t => t.value != null).map(t => ({ x: t.params[param], y: t.value, n: t.number, best: t.number === study.best_trial }));
  if (!pts.length) return <div className="plot-host" ref={ref} />;
  const xs = pts.map(p => p.x), ys = pts.map(p => p.y);
  const xMin = Math.min(...xs), xMax = Math.max(...xs), yMin = Math.min(...ys), yMax = Math.max(...ys);
  const X = v => pad.l + ((v - xMin) / (xMax - xMin || 1)) * (w - pad.l - pad.r);
  const Y = v => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (h - pad.t - pad.b);
  const [hover, setHover] = useState(null);
  return (
    <div className="plot-host" ref={ref}>
      <svg width={w} height={h} onMouseLeave={() => setHover(null)}>
        {niceTicks(yMin, yMax, 4).map(t => (
          <g key={t}><line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} className="plot-grid" />
            <text x={pad.l - 8} y={Y(t) + 3} className="plot-axis num" textAnchor="end">{t.toFixed(2)}</text></g>
        ))}
        {niceTicks(xMin, xMax, 5).map(t => <text key={t} x={X(t)} y={h - pad.b + 18} className="plot-axis num" textAnchor="middle">{(+t).toFixed(2)}</text>)}
        <text x={w / 2} y={h - 4} className="plot-axis-title" textAnchor="middle">{param}</text>
        {pts.map(p => (
          <circle key={p.n} cx={X(p.x)} cy={Y(p.y)} r={p.best ? 6 : 4}
            className={cx("plot-pt", p.best ? "best" : "complete")}
            onMouseEnter={() => setHover(p)} />
        ))}
      </svg>
      {hover && <div className="plot-tip" style={{ left: Math.min(w - 140, X(hover.x) + 10), top: Math.max(8, Y(hover.y) - 40) }}>
        <div className="tip-row"><b>Trial {hover.n}</b></div>
        <div className="tip-row num">{param} = {fmtNum(hover.x, 3)}</div>
        <div className="tip-row num">obj {fmtNum(hover.y, 4)}</div>
      </div>}
    </div>
  );
}

function OptunaTab({ data }) {
  const wids = Object.keys(data.optuna_studies).map(Number);
  const [wid, setWid] = useState(wids[0]);
  const study = data.optuna_studies[wid];
  const paramKeys = Object.keys(study.best_params || {});
  const [param, setParam] = useState(paramKeys[0]);
  const [q, setQ] = useState("");
  const [hoverTrial, setHoverTrial] = useState(null);

  useEffect(() => { if (!paramKeys.includes(param)) setParam(paramKeys[0]); }, [wid]);

  const filteredTrials = useMemo(() => {
    let rows = study.trials;
    if (q === "pruned") rows = rows.filter(t => t.value == null);
    else if (q === "complete") rows = rows.filter(t => t.value != null);
    return [...rows].sort((a, b) => (b.value ?? -Infinity) - (a.value ?? -Infinity));
  }, [study, q]);

  const win = data.windows.find(w => w.window_id === wid);
  const completion = study.n_trials_run / study.n_trials_requested;

  return (
    <div className="optuna-tab">
      <div className="optuna-head">
        <div className="optuna-window-pick">
          <span className="panel-heading">Per-window study</span>
          <select className="window-select" value={wid} onChange={e => setWid(+e.target.value)}>
            {wids.map(id => {
              const s = data.optuna_studies[id];
              return <option key={id} value={id}>Window {id} · {s.status === "completed" ? "✓" : "⚠"} {s.n_trials_run}/{s.n_trials_requested} trials · best {fmtNum(s.best_value, 3)}</option>;
            })}
          </select>
        </div>
        <div className="optuna-meta">
          <span><b>Study</b> <span className="mono">{study.study_name}</span></span>
          <span><b>Sampler</b> {study.sampler}</span>
          <span><b>Pruner</b> {study.pruner}</span>
          <span><b>Objective</b> {study.objective} = <span className="mono">{study.objective_formula}</span></span>
        </div>
      </div>

      {/* status banner */}
      <div className={cx("status-banner", study.status === "completed" ? "ok" : "warn")}>
        <span className="status-dot" />
        <div>
          <div className="status-title">
            {study.status === "completed" ? "Study completed" : "Study stopped early"}
            <span className="status-prog"> · {study.n_trials_run} of {study.n_trials_requested} trials run</span>
          </div>
          <div className="status-reason">{study.stop_reason}</div>
        </div>
        <div className="status-bars">
          <div className="sb"><span>{study.n_complete}</span> complete</div>
          <div className="sb"><span>{study.n_pruned}</span> pruned</div>
          <div className="sb"><span>{study.n_trials_requested - study.n_trials_run}</span> not run</div>
        </div>
      </div>

      <div className="optuna-cards">
        <Metric label="Best objective" value={fmtNum(study.best_value, 4)} tone="up" sub={`trial #${study.best_trial}`} />
        <Metric label="Completion" value={fmtPct(completion, 0)} sub={`${study.n_trials_run}/${study.n_trials_requested}`} />
        <Metric label="Pruned" value={study.n_pruned} sub="by MedianPruner" />
        <Metric label="OOS window return" value={win ? fmtPct(win.return_pct / 100) : "–"} tone={win && win.return_pct >= 0 ? "up" : "down"} sub={`win ${win ? fmtolike(win) : ""}`} />
      </div>

      {study._has_trial_history ? (
        <div className="optuna-grid">
          <div className="plot-card wide">
            <div className="plot-card-head">
              <span>Objective value over trials</span>
              <span className="legend-inline">
                <i className="lg-dot complete" /> complete
                <i className="lg-dot pruned" /> pruned
                <i className="lg-line" /> best so far
              </span>
            </div>
            <TrialsChart study={study} onHoverTrial={setHoverTrial} />
          </div>
          <div className="plot-card">
            <div className="plot-card-head">
              <span>Parameter vs objective</span>
              <select className="mini-select" value={param} onChange={e => setParam(e.target.value)}>
                {paramKeys.map(k => <option key={k} value={k}>{k.replace("iron_condor__", "")}</option>)}
              </select>
            </div>
            <ParamScatter study={study} param={param} />
          </div>
        </div>
      ) : (
        <div className="no-trial-notice">
          <div className="no-trial-icon">◈</div>
          <div className="no-trial-text">
            <b>Trial history not available for this experiment.</b>
            <span>Per-trial data (objective values, param sensitivity) is captured for experiments run after the Layer 2 upgrade. The optimal parameters above are from the completed Optuna study.</span>
          </div>
        </div>
      )}

      <div className="optuna-bottom">
        <div className="best-params-card">
          <div className="plot-card-head"><span>Optimal parameters · trial #{study.best_trial}</span></div>
          <div className="bp-list">
            {Object.entries(study.best_params || {}).map(([k, v]) => (
              <div className="bp-row" key={k}>
                <span className="bp-k">{k.replace("iron_condor__", "")}</span>
                <span className="bp-v mono">{typeof v === "number" ? fmtNum(v, 3) : v}</span>
              </div>
            ))}
          </div>
        </div>

        {study._has_trial_history ? (
          <div className="data-panel trials-panel">
            <div className="filter-bar">
              <span className="panel-heading">All {study.trials.length} candidate trials</span>
              <div className="seg">
                {["all", "complete", "pruned"].map(v => <button key={v} className={cx(q === v && "on")} onClick={() => setQ(v)}>{v}</button>)}
              </div>
            </div>
            <div className="table-scroll short">
              <table className="data-table compact">
                <thead><tr><th className="num">#</th><th>State</th><th className="num">Objective</th><th className="num">Sharpe</th><th className="num">Win rate</th><th className="num">Max DD</th><th className="num">Trades</th><th className="num">Time</th></tr></thead>
                <tbody>
                  {filteredTrials.map(t => (
                    <tr key={t.number} className={cx(t.number === study.best_trial && "sel", hoverTrial && hoverTrial.number === t.number && "hl")}>
                      <td className="num mono">{t.number}{t.number === study.best_trial && " ★"}</td>
                      <td><span className={cx("exit-tag", t.value == null ? "down" : "up")}>{t.state}</span></td>
                      <td className="num mono bold">{t.value == null ? "–" : fmtNum(t.value, 4)}</td>
                      <td className="num mono">{t.sharpe == null ? "–" : fmtNum(t.sharpe)}</td>
                      <td className="num mono">{t.win_rate == null ? "–" : fmtPct(t.win_rate)}</td>
                      <td className="num mono">{t.max_drawdown == null ? "–" : fmtPct(t.max_drawdown)}</td>
                      <td className="num mono">{t.n_trades}</td>
                      <td className="num mono dim">{t.duration_s}s</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}
function fmtolike(win) { return fmtPct(win.win_rate); }

Object.assign(window, { OptunaTab });
