/* ============================================================
   DecisionPanel — per-trade decision-chain drill-down drawer
   ============================================================ */
function StepIcon({ state }) {
  if (state === "pass") return <span className="step-icon pass">✓</span>;
  if (state === "fail") return <span className="step-icon fail">✕</span>;
  return <span className="step-icon info">•</span>;
}

function Gauge({ value, threshold, label, color }) {
  const pct = Math.max(0, Math.min(1, value));
  const tpct = Math.max(0, Math.min(1, threshold));
  return (
    <div className="gauge">
      <div className="gauge-track">
        <div className="gauge-fill" style={{ width: pct * 100 + "%", background: color }} />
        <div className="gauge-thresh" style={{ left: tpct * 100 + "%" }} title={"threshold " + threshold} />
      </div>
      <div className="gauge-labels">
        <span className="mono">{label}</span>
        <span className="mono dim">thr {fmtNum(threshold)}</span>
      </div>
    </div>
  );
}

const fmtDatetime = (s) => {
  if (!s) return "–";
  const clean = String(s).replace("T", " ").replace(/([+-]\d{2}:\d{2}|Z)$/, "").trim();
  const [date, time] = clean.split(" ");
  if (!time) return date;
  return `${date}  ${time.slice(0, 5)}`;
};

function DecisionPanel({ trade, onClose }) {
  if (!trade) return null;

  // Safe destructure — these fields are absent on archive-era trades (pre-Layer-2).
  const d = trade.decision || {};
  const rg = d.range_gate   || {};
  const vg = d.vol_gate     || {};
  const ml = d.meta_label   || {};
  const fg = d.fractal_gate || {};
  const hasDecision = Object.keys(d).length > 0;
  const legs = Array.isArray(trade.legs) ? trade.legs : [];
  const featEntries = Object.entries(trade.features_at_entry || {});

  const steps = hasDecision ? [
    {
      title: "Direction model", state: "info",
      detail: <>3-class ensemble → <b>{d.direction_class}</b> @ conf <b className="mono">{fmtNum(d.direction_conf)}</b>. Iron condor is market-neutral, so the directional gate is bypassed — the range model is the entry filter.</>,
    },
    {
      title: "Range model gate",
      state: rg.pass === true ? "pass" : rg.pass === false ? "fail" : "info",
      detail: <>P(price stays in range) = <b className="mono">{fmtNum(rg.prob)}</b> vs threshold <span className="mono">{fmtNum(rg.threshold)}</span>.
        <Gauge value={rg.prob ?? 0} threshold={rg.threshold || 0.55} label={"P(range) " + fmtNum(rg.prob)} color="var(--c-data)" /></>,
    },
    {
      title: "Volatility entry gate", state: vg.pass !== false ? "pass" : "fail",
      detail: <>10-day realized vol <b className="mono">{fmtPct(vg.vol_10d)}</b> ≤ max entry vol <span className="mono">{fmtPct(vg.max)}</span>. Condors fail in high-vol regimes, so a breach skips the entry.</>,
    },
    {
      title: "Meta-label filter", state: ml.take !== false ? "pass" : "fail",
      detail: <>Meta-classifier P(profitable) = <b className="mono">{fmtNum(ml.prob)}</b> vs <span className="mono">{fmtNum(ml.threshold)}</span> → <b>{ml.take !== false ? "take trade" : "veto"}</b>.
        <Gauge value={ml.prob || 0} threshold={ml.threshold || 0.5} label={"P(profit) " + fmtNum(ml.prob)} color="var(--c-up)" /></>,
    },
    {
      title: "Fractal regime gate", state: fg.pass !== false ? "pass" : "fail",
      detail: <>Hurst scale-spread <b className="mono">{fmtNum(fg.hurst_spread)}</b> vs threshold <span className="mono">{fmtNum(fg.threshold)}</span>. Wide spread ⇒ confidence penalty (trending/chaotic regime).</>,
    },
    {
      title: "Regime classification", state: "info",
      detail: <>Entry regime tagged <Chip tone={d.regime === "range_bound" ? "ok" : "warn"}>{d.regime || "–"}</Chip></>,
    },
  ] : [];

  return (
    <>
      <div className="drawer-scrim" onClick={onClose} />
      <aside className="drawer">
        <div className="drawer-head">
          <div>
            <div className="drawer-eyebrow">{trade.symbol} · {trade.strategy} · {trade.id}</div>
            <div className="drawer-title">{trade.entry_date} → {trade.exit_date} <span className="dim">({trade.hold_days}d)</span></div>
          </div>
          <button className="drawer-close" onClick={onClose}>✕</button>
        </div>

        <div className="drawer-pnl">
          <div className={cx("pnl-big", trade.pnl >= 0 ? "up" : "down")}>{fmtMoney(trade.pnl)}</div>
          <div className="pnl-sub">
            {trade.return_pct != null
              ? <span className={cx(trade.return_pct >= 0 ? "up" : "down")}>{fmtPct(trade.return_pct)} on max loss</span>
              : <span className="dim">return % available after next run</span>}
            <span className="dim">·</span>
            <span className={cx("exit-tag", EXIT_TONE[trade.exit_reason] || "neutral")}>{trade.exit_reason}</span>
          </div>
        </div>

        <div className="drawer-section">
          <div className="drawer-heading">Execution</div>
          <div className="kv-grid">
            <div className="kv"><span>Order placed</span><b className="mono">{fmtDatetime(trade.entry_time)}</b></div>
            <div className="kv"><span>Underlying at scan</span><b className="mono">{trade.limit_price != null ? fmtNum(trade.limit_price) : "–"}</b></div>
            {trade.fill_time && <div className="kv"><span>Fill confirmed</span><b className="mono">{fmtDatetime(trade.fill_time)}</b></div>}
            <div className="kv"><span>Spread credit (open)</span><b className="mono">{trade.entry_price != null ? fmtMoney(trade.entry_price) : "–"}</b></div>
            <div className="kv"><span>Spread cost (close)</span><b className="mono">{trade.exit_price != null ? fmtMoney(trade.exit_price) : "–"}<span className="dim" style={{fontSize:"10px",marginLeft:"4px"}}>per share · ×100×contracts−comm = P&L</span></b></div>
            <div className="kv"><span>Exit time</span><b className="mono">{fmtDatetime(trade.exit_time)}</b></div>
            <div className="kv"><span>Underlying at exit</span><b className="mono">{trade.exit_underlying != null ? fmtNum(trade.exit_underlying) : "–"}</b></div>
          </div>
        </div>

        <div className="drawer-section">
          <div className="drawer-heading">Decision chain</div>
          {hasDecision ? (
            <div className="stepper">
              {steps.map((s, i) => (
                <div className={cx("step", "step-" + s.state)} key={i}>
                  <div className="step-rail"><StepIcon state={s.state} />{i < steps.length - 1 && <span className="step-line" />}</div>
                  <div className="step-body">
                    <div className="step-title">{s.title}</div>
                    <div className="step-detail">{s.detail}</div>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="no-trial-notice" style={{ margin: 0 }}>
              <div className="no-trial-icon">◈</div>
              <div className="no-trial-text">
                <b>Decision chain not available for this experiment.</b>
                <span>Per-gate data (range model, vol gate, meta-label, fractal gate) is captured for experiments run after the Layer 2 upgrade.</span>
              </div>
            </div>
          )}
        </div>

        <div className="drawer-section">
          <div className="drawer-heading">Iron condor structure</div>
          {legs.length > 0 ? (
            <>
              <table className="legs-table">
                <thead><tr><th>Leg</th><th className="num">Strike</th><th className="num">Premium</th></tr></thead>
                <tbody>
                  {legs.map((l, i) => (
                    <tr key={i}>
                      <td><span className={cx("leg-tag", l.type && l.type.startsWith("short") ? "short" : "long")}>{(l.type || "").replace("_", " ")}</span></td>
                      <td className="num mono">{l.strike}</td>
                      <td className="num mono">{l.premium}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          ) : (
            <p className="dim" style={{ fontSize: "12px", margin: "4px 0 8px" }}>Leg structure available after next run.</p>
          )}
          <div className="kv-grid">
            <div className="kv"><span>Net credit</span><b className="mono">{fmtMoney(trade.credit)}</b></div>
            <div className="kv"><span>Max loss</span><b className="mono">{fmtMoney(trade.max_loss)}</b></div>
            <div className="kv"><span>Contracts</span><b className="mono">{trade.contracts ?? "–"}</b></div>
          </div>
        </div>

        <div className="drawer-section">
          <div className="drawer-heading">Features at entry</div>
          {featEntries.length > 0 ? (
            <div className="feat-grid">
              {featEntries.map(([k, v]) => (
                <div className="feat-cell" key={k}>
                  <span className="feat-k">{k}</span>
                  <span className="feat-v mono">{v == null ? "–" : (Math.abs(v) < 1 && v !== 0 ? fmtNum(v, 3) : fmtNum(v, 2))}</span>
                </div>
              ))}
            </div>
          ) : (
            <p className="dim" style={{ fontSize: "12px", margin: "4px 0 8px" }}>Feature snapshot available after next run.</p>
          )}
        </div>
      </aside>
    </>
  );
}

Object.assign(window, { DecisionPanel, Gauge });
