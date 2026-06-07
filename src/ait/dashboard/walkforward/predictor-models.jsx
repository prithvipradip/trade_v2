/* ============================================================
   Predictor Models tab
   Visualises per-window skill of each ensemble member for the
   Directional and Range predictors: CV skill across windows,
   fitted-weight evolution, per-window member bars, reliability
   curve, and a compact-expandable GARCH-family detail card.
   ============================================================ */

/* member colors (CSS vars so they track the theme) */
const PM_COLOR = {
  xgboost: "var(--accent)",
  lightgbm: "var(--c-up)",
  garch: "var(--c-rsi)",
  msgarch: "var(--c-data)",
  oujump: "var(--c-warn)",
};
const PM_LABEL = {
  xgboost: "XGBoost", lightgbm: "LightGBM",
  garch: "GARCH", msgarch: "MS-GARCH", oujump: "OU-Kou-GARCH",
};

/* ---- size + tick helpers (local, unique names) ---- */
function pmUseSize() {
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
function pmTicks(min, max, n = 5) {
  if (min === max) { min -= 1; max += 1; }
  const span = max - min, step0 = span / n;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  const step = (norm < 1.5 ? 1 : norm < 3 ? 2 : norm < 7 ? 5 : 10) * mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let v = start; v <= max + 1e-9; v += step) ticks.push(+v.toFixed(8));
  return ticks;
}

/* ---- shared legend for member colors ---- */
function MemberLegend({ members }) {
  return (
    <span className="pm-legend">
      {members.map(m => (
        <span className="pm-lg" key={m}>
          <i className="pm-lg-dot" style={{ background: PM_COLOR[m] }} /> {PM_LABEL[m]}
        </span>
      ))}
    </span>
  );
}

/* ---- hover info tooltip — reads window.PM_INFO[id]; popup is
       viewport-anchored (position:fixed) so overflow:hidden cards
       can't clip it, and it flips near edges. ---- */
function InfoPop({ info, anchor, wide }) {
  const W = wide ? 320 : 264;
  const vw = window.innerWidth, vh = window.innerHeight;
  const left = Math.max(10, Math.min(vw - W - 10, anchor.cx - W / 2));
  const flipUp = anchor.bottom > vh - 190;
  const vstyle = flipUp ? { bottom: vh - anchor.top + 8 } : { top: anchor.bottom + 8 };
  return (
    <div className="info-pop2" style={{ left, width: W, ...vstyle }}>
      <div className="info-pop-title">{info.title}</div>
      <div className="info-pop-row"><b>What</b><span>{info.what}</span></div>
      <div className="info-pop-row"><b>Why</b><span>{info.why}</span></div>
    </div>
  );
}
function InfoTip({ id, wide }) {
  const info = (window.PM_INFO && window.PM_INFO[id]) || null;
  const ref = useRef();
  const [anchor, setAnchor] = useState(null);
  if (!info) return null;
  const show = () => {
    const r = ref.current.getBoundingClientRect();
    setAnchor({ cx: r.left + r.width / 2, top: r.top, bottom: r.bottom });
  };
  return (
    <span className="info-tip" ref={ref} onMouseEnter={show} onMouseLeave={() => setAnchor(null)}>
      <span className="info-dot">i</span>
      {anchor && <InfoPop info={info} anchor={anchor} wide={wide} />}
    </span>
  );
}

/* metric card with an info icon (top summary row) */
function PMMetric({ label, value, tone, sub, info }) {
  return (
    <div className="metric">
      <div className="metric-label">{label}{info && <InfoTip id={info} />}</div>
      <div className={cx("metric-value", tone)}>{value}</div>
      {sub && <div className="metric-sub">{sub}</div>}
    </div>
  );
}

/* =========================================================
   Skill-across-windows grouped BAR chart
   - one bar per member within each window group
   - clickable legend to show/hide members
   - x-axis: window number + OOS start→end dates
   ========================================================= */
function SkillBars({ rows, members, baseline }) {
  const [ref, { w, h }] = pmUseSize();
  const [hover, setHover] = useState(null);
  const [hidden, setHidden] = useState(() => new Set());
  useEffect(() => { setHidden(new Set()); }, [members]);
  const toggle = m => setHidden(s => { const n = new Set(s); n.has(m) ? n.delete(m) : n.add(m); return n; });
  const visible = members.filter(m => !hidden.has(m));

  const pad = { l: 46, r: 14, t: 14, b: 46 };
  const md = s => (s ? s.slice(5) : "");

  const allVals = [];
  rows.forEach(r => visible.forEach(m => { const v = r.pp.cv[m]; if (v != null) allVals.push(v); }));
  const yMin = Math.min(baseline - 0.02, ...(allVals.length ? allVals : [baseline])) - 0.01;
  const yMax = Math.max(...(allVals.length ? allVals : [baseline]), baseline + 0.05) + 0.02;
  const n = rows.length;
  const Y = v => pad.t + (1 - (v - yMin) / (yMax - yMin || 1)) * (h - pad.t - pad.b);
  const plotW = w - pad.l - pad.r;
  const groupW = plotW / n;
  const innerPad = Math.min(14, groupW * 0.22);
  const barAreaW = groupW - innerPad;
  const barW = barAreaW / Math.max(1, visible.length);
  const y0 = Y(yMin);
  const yticks = pmTicks(yMin, yMax, 5);

  return (
    <>
      <div className="skill-legend">
        {members.map(m => (
          <button key={m} className={cx("calib-chip", !hidden.has(m) && "on")} onClick={() => toggle(m)}>
            <i className="pm-lg-dot" style={{ background: PM_COLOR[m], opacity: hidden.has(m) ? 0.3 : 1 }} />{PM_LABEL[m]}
          </button>
        ))}
      </div>
      <div className="plot-host" ref={ref}>
        <svg width={w} height={h} onMouseLeave={() => setHover(null)}>
          {/* gated-window shading */}
          {rows.map((r, i) => r.pp.gated && (
            <rect key={"g" + i} x={pad.l + groupW * i} y={pad.t} width={groupW} height={h - pad.t - pad.b} className="pm-gated-band" />
          ))}
          {yticks.map(t => (
            <g key={t}>
              <line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} className="plot-grid" />
              <text x={pad.l - 7} y={Y(t) + 3} className="plot-axis num" textAnchor="end">{t.toFixed(2)}</text>
            </g>
          ))}
          {/* grouped bars */}
          {rows.map((r, i) => {
            const gx = pad.l + groupW * i + innerPad / 2;
            return (
              <g key={"b" + i}>
                {visible.map((m, j) => {
                  const v = r.pp.cv[m];
                  if (v == null) return null;
                  const bx = gx + barW * j;
                  const by = Y(v);
                  return <rect key={m} x={bx + 0.5} y={by} width={Math.max(1, barW - 1)} height={Math.max(0, y0 - by)}
                    className="pm-skill-bar" style={{ fill: PM_COLOR[m], opacity: hover == null || hover === i ? 1 : 0.45 }} />;
                })}
              </g>
            );
          })}
          {/* baseline */}
          <line x1={pad.l} x2={w - pad.r} y1={Y(baseline)} y2={Y(baseline)} className="pm-baseline" />
          <text x={w - pad.r} y={Y(baseline) - 4} className="pm-baseline-lbl" textAnchor="end">baseline {baseline.toFixed(2)}</text>
          {/* x labels: window number + date range */}
          {rows.map((r, i) => {
            const cx0 = pad.l + groupW * (i + 0.5);
            return (
              <g key={"x" + i}>
                <text x={cx0} y={h - pad.b + 16} className="plot-axis num" textAnchor="middle" style={{ fontWeight: 700 }}>W{r.id}</text>
                <text x={cx0} y={h - pad.b + 28} className="pm-xdate" textAnchor="middle">{md(r.meta.test_start)}→{md(r.meta.test_end)}</text>
              </g>
            );
          })}
          {/* hover columns */}
          {rows.map((r, i) => (
            <rect key={"h" + i} x={pad.l + groupW * i} y={pad.t} width={groupW} height={h - pad.t - pad.b}
              fill="transparent" onMouseEnter={() => setHover(i)} />
          ))}
        </svg>
        {hover != null && (
          <div className="plot-tip" style={{ left: Math.min(w - 160, Math.max(8, pad.l + groupW * (hover + 0.5) + 10)), top: 16 }}>
            <div className="tip-row"><b>Window {rows[hover].id}</b>{rows[hover].pp.gated && <span className="tip-best down"> gated</span>}</div>
            <div className="tip-row dim num">OOS {rows[hover].meta.test_start} → {rows[hover].meta.test_end}</div>
            {visible.map(m => rows[hover].pp.cv[m] != null && (
              <div className="tip-row dim num" key={m}>
                <i className="pm-lg-dot" style={{ background: PM_COLOR[m], marginRight: 5 }} />{PM_LABEL[m]} <b>{fmtNum(rows[hover].pp.cv[m], 3)}</b>
              </div>
            ))}
          </div>
        )}
      </div>
    </>
  );
}

/* =========================================================
   Fitted-weight stacked bars across windows
   ========================================================= */
function WeightStackChart({ rows, members }) {
  const [ref, { w, h }] = pmUseSize();
  const [hover, setHover] = useState(null);
  const pad = { l: 30, r: 14, t: 14, b: 30 };
  if (!rows.length) return <div className="plot-host" ref={ref} />;
  const n = rows.length;
  const bw = (w - pad.l - pad.r) / n;
  const barW = Math.min(46, bw * 0.62);
  const H = h - pad.t - pad.b;

  return (
    <div className="plot-host" ref={ref}>
      <svg width={w} height={h} onMouseLeave={() => setHover(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={pad.t + (1 - t) * H} y2={pad.t + (1 - t) * H} className="plot-grid" />
            <text x={pad.l - 6} y={pad.t + (1 - t) * H + 3} className="plot-axis num" textAnchor="end">{t === 0 || t === 1 ? t : ""}</text>
          </g>
        ))}
        {rows.map((r, i) => {
          const cx0 = pad.l + bw * (i + 0.5);
          let acc = 0;
          return (
            <g key={i} onMouseEnter={() => setHover(i)}>
              {members.map(m => {
                const wv = r.pp.fitted_weights[m] || 0;
                if (wv <= 0) return null;
                const y0 = pad.t + (1 - acc - wv) * H;
                const hh = wv * H;
                acc += wv;
                return <rect key={m} x={cx0 - barW / 2} y={y0} width={barW} height={Math.max(0, hh - 0.6)}
                  className="pm-wbar" style={{ fill: PM_COLOR[m], opacity: hover == null || hover === i ? 1 : 0.4 }} />;
              })}
              <text x={cx0} y={h - pad.b + 16} className="plot-axis num" textAnchor="middle">{r.id}</text>
            </g>
          );
        })}
        <text x={w / 2} y={h - 3} className="plot-axis-title" textAnchor="middle">window · fitted weight</text>
      </svg>
      {hover != null && (
        <div className="plot-tip" style={{ left: Math.min(w - 150, Math.max(8, pad.l + bw * (hover + 0.5) + 8)), top: 16 }}>
          <div className="tip-row"><b>Window {rows[hover].id}</b></div>
          {members.map(m => (rows[hover].pp.fitted_weights[m] > 0) && (
            <div className="tip-row dim num" key={m}>
              <i className="pm-lg-dot" style={{ background: PM_COLOR[m], marginRight: 5 }} />{PM_LABEL[m]} <b>{fmtPct(rows[hover].pp.fitted_weights[m], 0)}</b>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* =========================================================
   Per-window member CV bars with edge over baseline
   ========================================================= */
function CVBars({ pp }) {
  const baseline = pp.baseline;
  const members = pp.members.filter(m => pp.cv[m] != null);
  const naMembers = pp.members.filter(m => pp.cv[m] == null);
  const lo = 0.44, hi = Math.max(0.85, ...members.map(m => pp.cv[m])) + 0.02;
  const pct = v => ((v - lo) / (hi - lo)) * 100;
  return (
    <div className="cv-bars">
      {members.map(m => {
        const v = pp.cv[m], edge = v - baseline, wgt = pp.fitted_weights[m] || 0;
        return (
          <div className="cv-row" key={m}>
            <div className="cv-name"><i className="pm-lg-dot" style={{ background: PM_COLOR[m] }} />{PM_LABEL[m]}</div>
            <div className="cv-track">
              <div className="cv-baseline" style={{ left: pct(baseline) + "%" }} />
              <div className="cv-fill" style={{ width: pct(v) + "%", background: PM_COLOR[m] }} />
            </div>
            <div className="cv-val mono">{fmtNum(v, 3)}</div>
            <div className={cx("cv-edge mono", edge >= 0 ? "up" : "down")}>{edge >= 0 ? "+" : ""}{fmtNum(edge, 3)}</div>
            <div className="cv-weight mono dim">{wgt > 0 ? fmtPct(wgt, 0) : "—"}</div>
          </div>
        );
      })}
      {naMembers.map(m => (
        <div className="cv-row na" key={m}>
          <div className="cv-name"><i className="pm-lg-dot" style={{ background: PM_COLOR[m], opacity: .3 }} />{PM_LABEL[m]}</div>
          <div className="cv-track"><div className="cv-baseline" style={{ left: pct(baseline) + "%" }} /></div>
          <div className="cv-val mono dim">n/a</div>
          <div className="cv-edge mono dim">—</div>
          <div className="cv-weight mono dim">—</div>
        </div>
      ))}
      <div className="cv-head">
        <div className="cv-name" />
        <div className="cv-track-lbl">{pp.metric_label} · marker = baseline {baseline.toFixed(2)}</div>
        <div className="cv-val-lbl">cv</div>
        <div className="cv-val-lbl">edge</div>
        <div className="cv-val-lbl">wt</div>
      </div>
    </div>
  );
}

/* =========================================================
   Reliability / calibration curve (multi-series)
   ========================================================= */
function ReliabilityChart({ series }) {
  const [ref, { w, h }] = pmUseSize();
  const [hover, setHover] = useState(null);
  const pad = { l: 42, r: 14, t: 14, b: 34 };
  const X = v => pad.l + v * (w - pad.l - pad.r);
  const Y = v => pad.t + (1 - v) * (h - pad.t - pad.b);
  const allN = series.flatMap(s => s.bins.map(b => b.n));
  const maxN = Math.max(...allN, 1);
  return (
    <div className="plot-host" ref={ref}>
      <svg width={w} height={h} onMouseLeave={() => setHover(null)}>
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <g key={t}>
            <line x1={pad.l} x2={w - pad.r} y1={Y(t)} y2={Y(t)} className="plot-grid" />
            <text x={pad.l - 6} y={Y(t) + 3} className="plot-axis num" textAnchor="end">{t.toFixed(2)}</text>
            <text x={X(t)} y={h - pad.b + 16} className="plot-axis num" textAnchor="middle">{t.toFixed(2)}</text>
          </g>
        ))}
        {/* perfect-calibration diagonal */}
        <line x1={X(0)} y1={Y(0)} x2={X(1)} y2={Y(1)} className="pm-diag" />
        <text x={w / 2} y={h - 2} className="plot-axis-title" textAnchor="middle">predicted probability</text>
        <text transform={`translate(11,${(h) / 2}) rotate(-90)`} className="plot-axis-title" textAnchor="middle">empirical frequency</text>
        {series.map(s => {
          const d = s.bins.map((b, i) => `${i === 0 ? "M" : "L"}${X(b.p)},${Y(b.actual)}`).join(" ");
          return (
            <g key={s.key}>
              <path d={d} className={cx("pm-line", s.emphasis && "pm-line-em")} style={{ stroke: s.color, opacity: s.emphasis ? 1 : 0.85 }} />
              {s.bins.map((b, i) => (
                <circle key={i} cx={X(b.p)} cy={Y(b.actual)} r={(s.emphasis ? 3.5 : 2.5) + 4.5 * (b.n / maxN)}
                  className="pm-calib-dot" style={{ fill: s.color }}
                  onMouseEnter={() => setHover({ s: s.key, i, label: s.label, color: s.color, b })} />
              ))}
            </g>
          );
        })}
      </svg>
      {hover && (
        <div className="plot-tip" style={{ left: Math.min(w - 150, X(hover.b.p) + 10), top: Math.max(8, Y(hover.b.actual) - 52) }}>
          <div className="tip-row"><i className="pm-lg-dot" style={{ background: hover.color, marginRight: 5 }} /><b>{hover.label}</b></div>
          <div className="tip-row num">predicted <b>{fmtNum(hover.b.p, 2)}</b></div>
          <div className="tip-row num">actual <b>{fmtNum(hover.b.actual, 2)}</b></div>
          <div className="tip-row dim num">{hover.b.n} samples</div>
        </div>
      )}
    </div>
  );
}

/* reliability card: ensemble + per-member toggles */
function ReliabilityCard({ pp, ensembleColor }) {
  const present = pp.members.filter(m => pp.calibration[m]);
  const [shown, setShown] = useState(() => new Set(["ensemble", ...present]));
  // reset selection when the window/predictor changes (calibration identity changes)
  useEffect(() => { setShown(new Set(["ensemble", ...pp.members.filter(m => pp.calibration[m])])); }, [pp]);
  const toggle = k => setShown(s => { const n = new Set(s); n.has(k) ? n.delete(k) : n.add(k); return n; });

  const series = [];
  if (shown.has("ensemble")) series.push({ key: "ensemble", label: "Ensemble (blended)", color: ensembleColor, bins: pp.calibration.ensemble, emphasis: true });
  present.forEach(m => { if (shown.has(m)) series.push({ key: m, label: PM_LABEL[m], color: PM_COLOR[m], bins: pp.calibration[m] }); });

  const Chip = ({ k, label, color, em }) => (
    <button className={cx("calib-chip", shown.has(k) && "on", em && "em")} onClick={() => toggle(k)}>
      <i className="pm-lg-dot" style={{ background: color, opacity: shown.has(k) ? 1 : 0.3 }} />{label}
    </button>
  );

  return (
    <div className="plot-card">
      <div className="plot-card-head">
        <span>Reliability · predicted vs actual
          <InfoTip wide id="reliability" />
        </span>
        <span className="legend-inline"><i className="pm-diag-key" /> perfect calibration</span>
      </div>
      <div className="calib-chips">
        <Chip k="ensemble" label="Ensemble" color={ensembleColor} em />
        {present.map(m => <Chip key={m} k={m} label={PM_LABEL[m]} color={PM_COLOR[m]} />)}
      </div>
      <ReliabilityChart series={series} />
    </div>
  );
}

/* =========================================================
   GARCH-family compact-expandable detail (range only)
   ========================================================= */
function GarchRow({ name, color, cv, weight, converged, bic, summary, children }) {
  const [open, setOpen] = useState(false);
  const na = cv == null;
  return (
    <div className={cx("garch-item", na && "na")}>
      <button className="garch-bar" onClick={() => !na && setOpen(o => !o)} disabled={na}>
        <i className="pm-lg-dot" style={{ background: color, opacity: na ? .3 : 1 }} />
        <span className="garch-name">{name}</span>
        {na ? <span className="garch-na">unevaluable this window</span> : (
          <>
            <span className="garch-tag mono">{summary}</span>
            <span className={cx("garch-conv", converged ? "ok" : "warn")}>{converged ? "converged" : "no-converge"}</span>
            <span className="garch-stat mono dim">BIC {fmtNum(bic, 0)}</span>
            <span className="garch-stat mono">cv {fmtNum(cv, 3)}</span>
            <span className="garch-stat mono dim">wt {weight > 0 ? fmtPct(weight, 0) : "—"}</span>
            <span className="garch-caret">{open ? "▾" : "▸"}</span>
          </>
        )}
      </button>
      {open && !na && <div className="garch-detail">{children}</div>}
    </div>
  );
}

function GarchFamily({ pp }) {
  const g = pp.garch_meta, ms = pp.msgarch_meta, ou = pp.oujump_meta;
  const KV = ({ k, v }) => <div className="garch-kv"><span>{k}</span><b className="mono">{v}</b></div>;
  return (
    <div className="garch-family">
      <GarchRow name="GARCH" color={PM_COLOR.garch} cv={pp.cv.garch} weight={pp.fitted_weights.garch}
        converged={true} bic={g.bic} summary={`${g.variant} · ${g.dist}`}>
        <div className="garch-kv-grid">
          <KV k="variant" v={g.variant} />
          <KV k="distribution" v={g.dist} />
          <KV k="BIC" v={fmtNum(g.bic, 1)} />
          <KV k="JB p-value" v={fmtNum(g.jb_pvalue, 3)} />
          <KV k="resid skew" v={fmtNum(g.resid_skewness, 3)} />
          <KV k="fallback" v={g.fallback_used ? "yes" : "no"} />
        </div>
      </GarchRow>
      <GarchRow name="MS-GARCH" color={PM_COLOR.msgarch} cv={pp.cv.msgarch} weight={pp.fitted_weights.msgarch}
        converged={ms.converged} bic={ms.bic} summary={`2-regime · σ ${fmtPct(ms.regime0_vol, 0)}/${fmtPct(ms.regime1_vol, 0)}`}>
        <div className="garch-kv-grid">
          <KV k="regime 0 σ" v={fmtPct(ms.regime0_vol, 1)} />
          <KV k="regime 1 σ" v={fmtPct(ms.regime1_vol, 1)} />
          <KV k="BIC" v={fmtNum(ms.bic, 1)} />
          <KV k="converged" v={ms.converged ? "yes" : "no"} />
        </div>
        <div className="garch-sub">transition matrix</div>
        <table className="garch-trans mono">
          <tbody>
            <tr><td className="th">from \ to</td><td className="th">R0</td><td className="th">R1</td></tr>
            <tr><td className="th">R0</td><td>{fmtNum(ms.transition[0][0], 2)}</td><td>{fmtNum(ms.transition[0][1], 2)}</td></tr>
            <tr><td className="th">R1</td><td>{fmtNum(ms.transition[1][0], 2)}</td><td>{fmtNum(ms.transition[1][1], 2)}</td></tr>
          </tbody>
        </table>
      </GarchRow>
      {ou ? (
        <GarchRow name="OU-Kou-GARCH" color={PM_COLOR.oujump} cv={pp.cv.oujump} weight={pp.fitted_weights.oujump}
          converged={ou.converged} bic={ou.bic} summary={`drift ${ou.direction} · conf ${fmtPct(ou.confidence, 0)}`}>
          <div className="garch-kv-grid">
            <KV k="direction" v={ou.direction} />
            <KV k="confidence" v={fmtPct(ou.confidence, 0)} />
            <KV k="BIC" v={fmtNum(ou.bic, 1)} />
            <KV k="κ (mean-rev)" v={fmtNum(ou.kappa, 3)} />
            <KV k="θ (level)" v={fmtNum(ou.theta, 3)} />
            <KV k="σ (diffusion)" v={fmtNum(ou.sigma, 3)} />
            <KV k="jump λ" v={fmtNum(ou.jump_intensity, 3)} />
            <KV k="jump mean" v={fmtNum(ou.jump_mean, 3)} />
          </div>
        </GarchRow>
      ) : (
        <GarchRow name="OU-Kou-GARCH" color={PM_COLOR.oujump} cv={null} />
      )}
    </div>
  );
}

/* =========================================================
   Fitted-weight 100% bar for one window
   ========================================================= */
function WeightBar({ pp }) {
  const members = pp.members.filter(m => (pp.fitted_weights[m] || 0) > 0);
  return (
    <div className="wbar-wrap">
      <div className="wbar-track">
        {members.map(m => (
          <div key={m} className="wbar-seg" style={{ width: (pp.fitted_weights[m] * 100) + "%", background: PM_COLOR[m] }}
            title={`${PM_LABEL[m]} ${fmtPct(pp.fitted_weights[m], 0)}`} />
        ))}
      </div>
      <div className="wbar-legend">
        {members.map(m => (
          <span key={m} className="wbar-lg"><i className="pm-lg-dot" style={{ background: PM_COLOR[m] }} />{PM_LABEL[m]} <b className="mono">{fmtPct(pp.fitted_weights[m], 0)}</b></span>
        ))}
      </div>
    </div>
  );
}

/* =========================================================
   Main tab
   ========================================================= */
function PredictorModelsTab({ data }) {
  const meta = data.MODEL_META;
  const wids = Object.keys(data.model_perf).map(Number).sort((a, b) => a - b);
  const [predictor, setPredictor] = useState("directional");
  const [wid, setWid] = useState(wids[0]);

  const rows = useMemo(() => wids.map(id => ({ id, pp: data.model_perf[id][predictor], meta: data.model_perf[id] })), [predictor]);
  const m = meta[predictor];
  const members = m.members;
  const pp = data.model_perf[wid][predictor];
  const winMeta = data.model_perf[wid];

  // ---- summary stats across windows ----
  const summary = useMemo(() => {
    const avgCvs = rows.map(r => r.pp.avg_cv);
    const avgCv = avgCvs.reduce((a, x) => a + x, 0) / avgCvs.length;
    const best = rows.reduce((a, r) => r.pp.avg_cv > a.pp.avg_cv ? r : a, rows[0]);
    const meanEdge = rows.reduce((a, r) => a + r.pp.avg_edge, 0) / rows.length;
    const gated = rows.filter(r => r.pp.gated).length;
    // dominant member by mean weight
    const wsum = {}; members.forEach(k => wsum[k] = 0);
    rows.forEach(r => members.forEach(k => wsum[k] += (r.pp.fitted_weights[k] || 0)));
    const dominant = members.reduce((a, k) => wsum[k] > wsum[a] ? k : a, members[0]);
    const avgInRange = predictor === "range"
      ? rows.reduce((a, r) => a + r.pp.in_range_rate, 0) / rows.length : null;
    return { avgCv, best, meanEdge, gated, dominant, avgInRange };
  }, [rows, predictor]);

  const accentColor = predictor === "range" ? "var(--c-data)" : "var(--accent)";

  // top-weighted member for the selected window
  const topWeightMember = members.reduce((a, k) =>
    (pp.fitted_weights[k] || 0) > (pp.fitted_weights[a] || 0) ? k : a, members[0]);
  const nBlended = members.filter(k => (pp.fitted_weights[k] || 0) > 0).length;

  return (
    <div className="pm-tab">
      {/* head + predictor toggle */}
      <div className="pm-head">
        <div className="pm-head-left">
          <div className="pm-toggle seg">
            {["directional", "range"].map(p => (
              <button key={p} className={cx(predictor === p && "on")} onClick={() => setPredictor(p)}>
                {meta[p].label}
              </button>
            ))}
          </div>
          <div className="pm-sub">
            <span className="mono">{m.module}</span>
          </div>
        </div>
        <div className="pm-meta">
          <span><b>Task</b> {m.task}</span>
          <span><b>Metric</b> {m.metric_label}
            <InfoTip wide id={predictor === "range" ? "metric_range" : "metric_directional"} />
          </span>
          <span><b>Members</b> {members.map(k => PM_LABEL[k]).join(" · ")}</span>
        </div>
      </div>
      <p className="pm-objective">{m.objective}</p>

      {/* summary cards */}
      <div className="optuna-cards pm-cards">
        <PMMetric label={`Avg ${m.metric}`} value={fmtNum(summary.avgCv, 3)} tone={summary.avgCv > m.baseline ? "up" : "down"}
          sub={`baseline ${m.baseline.toFixed(2)}`} info="avg_cv" />
        <PMMetric label="Mean edge / window" value={`${summary.meanEdge >= 0 ? "+" : ""}${fmtNum(summary.meanEdge, 3)}`}
          tone={summary.meanEdge >= 0 ? "up" : "down"} sub="over baseline" info="mean_edge" />
        <PMMetric label="Best window" value={`#${summary.best.id}`} tone="up" sub={`${m.metric} ${fmtNum(summary.best.pp.avg_cv, 3)}`} info="best_window" />
        {predictor === "range"
          ? <PMMetric label="Windows gated" value={`${summary.gated}/${rows.length}`} tone={summary.gated ? "down" : "up"} sub={`edge < ${data.model_perf[wid].range.min_edge}`} info="gating" />
          : <PMMetric label="No-edge windows" value={`${summary.gated}/${rows.length}`} tone={summary.gated ? "down" : "up"} sub="below conf. floor" info="gating" />}
        <PMMetric label="Dominant member" value={PM_LABEL[summary.dominant]} sub="by mean fitted weight" info="dominant" />
        {predictor === "range" && <PMMetric label="Avg in-range rate" value={fmtPct(summary.avgInRange, 0)} sub="base rate · ±5% / 30d" info="in_range_rate" />}
      </div>

      {/* window-trend overview */}
      <div className="pm-section-label">Across windows · {rows.length} walk-forward windows</div>
      <div className="optuna-grid pm-grid">
        <div className="plot-card">
          <div className="plot-card-head">
            <span>Member skill across windows
              <InfoTip wide id="skill_trend" />
            </span>
            <span className="legend-inline pm-shade-key"><i className="pm-gated-swatch" /> shaded = gated window</span>
          </div>
          <SkillBars rows={rows} members={members} baseline={m.baseline} />
        </div>
        <div className="plot-card">
          <div className="plot-card-head">
            <span>Fitted ensemble weight
              <InfoTip wide id="weight_stack" />
            </span>
            <span className="legend-inline pm-shade-key"><i className="pm-gated-swatch" /> shaded = gated window</span>
          </div>
          <WeightStackChart rows={rows} members={members} />
        </div>
      </div>

      {/* single-window detail */}
      <div className="pm-section-label pm-detail-label">
        Single-window detail
        <select className="window-select pm-window-select" value={wid} onChange={e => setWid(+e.target.value)}>
          {wids.map(id => {
            const p = data.model_perf[id][predictor];
            const wm = data.model_perf[id];
            return <option key={id} value={id}>Window {id} · OOS {wm.test_start} → {wm.test_end} · {p.gated ? "⚠ gated" : `${m.metric} ${fmtNum(p.avg_cv, 3)}`}</option>;
          })}
        </select>
      </div>

      {/* status banner */}
      <div className={cx("status-banner", pp.gated ? "warn" : "ok")}>
        <span className="status-dot" />
        <div>
          <div className="status-title">
            Window {wid} · OOS {winMeta.test_start} → {winMeta.test_end}
            <span className="status-prog"> · {m.label} {pp.gated ? "gated" : "active"} · regime {winMeta.regime.replace("_", " ")}</span>
          </div>
          <div className="status-reason">
            {pp.gated ? pp.gate_reason
              : `${nBlended} members blended · top weight ${PM_LABEL[topWeightMember]} ${fmtPct(pp.fitted_weights[topWeightMember], 0)} · ${pp.n_signals} signals emitted.`}
          </div>
        </div>
        <div className="status-bars">
          <div className="sb"><span>{fmtNum(pp.avg_cv, 3)}</span> avg {m.metric.toLowerCase()}</div>
          <div className="sb"><span>{pp.avg_edge >= 0 ? "+" : ""}{fmtNum(pp.avg_edge, 3)}</span> edge</div>
          {predictor === "range"
            ? <div className="sb"><span>{fmtPct(pp.in_range_rate, 0)}</span> in-range</div>
            : <div className="sb"><span>{pp.n_signals}</span> signals</div>}
        </div>
      </div>

      <div className="pm-detail-grid">
        <div className="plot-card">
          <div className="plot-card-head"><span>Member CV vs baseline · window {wid}
            <InfoTip id="cv_bars" />
          </span></div>
          <div className="pm-card-body"><CVBars pp={pp} /></div>
        </div>
        <ReliabilityCard pp={pp} ensembleColor={accentColor} />
      </div>

      <div className="pm-detail-grid pm-detail-grid-2">
        <div className="plot-card">
          <div className="plot-card-head"><span>Fitted ensemble weights · window {wid}</span></div>
          <div className="pm-card-body"><WeightBar pp={pp} /></div>
        </div>
        {predictor === "range" ? (
          <div className="plot-card">
            <div className="plot-card-head">
              <span>GARCH-family members</span>
              <span className="legend-inline dim">click a row to expand</span>
            </div>
            <div className="pm-card-body"><GarchFamily pp={pp} /></div>
          </div>
        ) : (
          <div className="plot-card">
            <div className="plot-card-head"><span>Directional gate</span></div>
            <div className="pm-card-body">
              <div className="dir-gate">
                <div className="dir-gate-row"><span>Confidence floor</span><b className="mono">{fmtNum(pp.confidence_floor, 2)}</b></div>
                <div className="dir-gate-row"><span>Signals emitted</span><b className="mono">{pp.n_signals}</b></div>
                <div className="dir-gate-row"><span>Window regime</span><Chip tone={winMeta.regime === "range_bound" ? "ok" : winMeta.regime === "elevated" ? "warn" : "down"}>{winMeta.regime.replace("_", " ")}</Chip></div>
                <div className="dir-gate-row"><span>Avg realized vol</span><b className="mono">{fmtPct(winMeta.avg_vol, 1)}</b></div>
                <div className="dir-gate-note">{pp.purpose}</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

Object.assign(window, { PredictorModelsTab });
