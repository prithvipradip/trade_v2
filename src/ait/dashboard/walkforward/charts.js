/* ===================================================================
   WFChart — synced multi-pane TradingView-style chart manager
   Built on TradingView Lightweight Charts v4.
   - Price pane: candlesticks + toggleable overlays (SMA, Bollinger) + trade markers
   - Stacked sub-panes (each own y-axis): ML predictions, RSI, MACD, Vol, IV/VIX, Hurst, Sentiment
   - All panes time-synced + crosshair-synced
   - Linked-hover callback (onHover(time)) and trade pinning
   Exposes window.WFChart
   =================================================================== */
(function () {
  "use strict";
  const LWC = window.LightweightCharts;

  const THEMES = {
    dark: {
      bg: "transparent", text: "#9aa4b2", grid: "rgba(255,255,255,0.045)",
      border: "rgba(255,255,255,0.08)", crosshair: "#6b7686",
      up: "#26a17b", down: "#e0556e", candleUp: "#26a17b", candleDown: "#e0556e",
      sma20: "#e8a13c", sma50: "#8a7df0", bb: "rgba(120,140,180,0.5)",
      rangeProb: "#34c0eb", dirConf: "#e8a13c", rsi: "#c98bdb",
      macd: "#34c0eb", macdSig: "#e8a13c", rvol: "#e0556e", atr: "#e8a13c",
      iv: "#34c0eb", vix: "#e8a13c", hurst: "#7fd6a6", sent: "#34c0eb", pcr: "#e8a13c",
      guide: "rgba(255,255,255,0.14)",
    },
    light: {
      bg: "transparent", text: "#5b6472", grid: "rgba(20,30,50,0.06)",
      border: "rgba(20,30,50,0.10)", crosshair: "#9aa4b2",
      up: "#0c9b6e", down: "#d83a55", candleUp: "#0c9b6e", candleDown: "#d83a55",
      sma20: "#c47d12", sma50: "#6a5bd0", bb: "rgba(90,110,150,0.45)",
      rangeProb: "#1191c2", dirConf: "#c47d12", rsi: "#9a4bb5",
      macd: "#1191c2", macdSig: "#c47d12", rvol: "#d83a55", atr: "#c47d12",
      iv: "#1191c2", vix: "#c47d12", hurst: "#1d9e6b", sent: "#1191c2", pcr: "#c47d12",
      guide: "rgba(20,30,50,0.16)",
    },
  };

  function el(cls, parent) {
    const d = document.createElement("div");
    if (cls) d.className = cls;
    if (parent) parent.appendChild(d);
    return d;
  }

  class WFChart {
    constructor(root) {
      this.root = root;
      this.charts = [];        // [{key, chart, container, legend, series:[], readout(fn)}]
      this.priceChart = null;
      this.candle = null;
      this.overlay = {};
      this._syncing = false;
      this.onHover = null;
      this.onMarkerClick = null;
      this.theme = "dark";
      this.data = null;
      this.activeSeries = new Set();
      this._ro = null;
      this._pinned = null;
    }

    destroy() {
      this.charts.forEach(c => { try { c.chart.remove(); } catch (e) {} });
      this.charts = [];
      this.root.innerHTML = "";
      this.priceChart = null; this.candle = null; this.overlay = {};
    }

    _baseOpts(t, isPrice) {
      return {
        autoSize: true,
        layout: { background: { type: "solid", color: "transparent" }, textColor: t.text, fontSize: 11,
          fontFamily: "'IBM Plex Mono','SF Mono',Menlo,monospace" },
        grid: { vertLines: { color: t.grid }, horzLines: { color: t.grid } },
        rightPriceScale: { borderColor: t.border, scaleMargins: { top: 0.12, bottom: 0.12 } },
        timeScale: { borderColor: t.border, visible: isPrice, rightOffset: 4, barSpacing: 8 },
        crosshair: {
          mode: LWC.CrosshairMode.Normal,
          vertLine: { color: t.crosshair, width: 1, style: 3, labelBackgroundColor: t.crosshair },
          horzLine: { color: t.crosshair, width: 1, style: 3, labelBackgroundColor: t.crosshair },
        },
        handleScale: { axisPressedMouseMove: { time: true, price: false } },
        localization: { priceFormatter: p => p?.toFixed ? p.toFixed(2) : p },
      };
    }

    render(opts) {
      const prevRange = this.priceChart ? this.priceChart.timeScale().getVisibleRange() : null;
      if (opts.data) this.data = opts.data;
      if (opts.activeSeries) this.activeSeries = opts.activeSeries;
      if (opts.theme) this.theme = opts.theme;
      if (opts.windowRange !== undefined) this._windowRange = opts.windowRange;
      this.onHover = opts.onHover || this.onHover;
      this.onMarkerClick = opts.onMarkerClick || this.onMarkerClick;
      this.destroy();
      const t = THEMES[this.theme];
      const data = this.data;
      const A = this.activeSeries;

      // ---------- PRICE PANE ----------
      const priceWrap = el("wf-pane wf-pane-price", this.root);
      const priceLegend = el("wf-legend", priceWrap);
      const priceContainer = el("wf-canvas", priceWrap);
      const pchart = LWC.createChart(priceContainer, this._baseOpts(t, false));
      this.priceChart = pchart;
      const candle = pchart.addCandlestickSeries({
        upColor: t.candleUp, downColor: t.candleDown,
        borderUpColor: t.candleUp, borderDownColor: t.candleDown,
        wickUpColor: t.candleUp, wickDownColor: t.candleDown,
        priceLineVisible: false, lastValueVisible: false,
      });
      candle.setData(data.bars);
      this.candle = candle;

      // overlays
      const ovSeries = {};
      if (A.has("ov:sma20")) {
        const s = pchart.addLineSeries({ color: t.sma20, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        s.setData(data.features.filter(f => f.sma_20 != null).map(f => ({ time: f.time, value: f.sma_20 })));
        ovSeries.sma20 = s;
      }
      if (A.has("ov:sma50")) {
        const s = pchart.addLineSeries({ color: t.sma50, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        s.setData(data.features.filter(f => f.sma_50 != null).map(f => ({ time: f.time, value: f.sma_50 })));
        ovSeries.sma50 = s;
      }
      if (A.has("ov:bb")) {
        const up = pchart.addLineSeries({ color: t.bb, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        const lo = pchart.addLineSeries({ color: t.bb, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        up.setData(data.features.filter(f => f.bb_upper != null).map(f => ({ time: f.time, value: f.bb_upper })));
        lo.setData(data.features.filter(f => f.bb_lower != null).map(f => ({ time: f.time, value: f.bb_lower })));
        ovSeries.bbUp = up; ovSeries.bbLo = lo;
      }
      this.overlay = ovSeries;

      // trade markers
      this._applyMarkers();

      const priceReadout = (time, seriesData) => {
        const b = seriesData.get(candle);
        const f = this._featAt(time);
        let html = time
          ? `<span class="wf-lg-title">${time} · QQQ · 1D</span>`
          : `<span class="wf-lg-title">QQQ · 1D</span>`;
        if (b) {
          const c = b.close >= b.open ? "wf-up" : "wf-down";
          html += ` <span class="wf-lg ${c}">O ${b.open?.toFixed(2)}</span><span class="wf-lg ${c}">H ${b.high?.toFixed(2)}</span><span class="wf-lg ${c}">L ${b.low?.toFixed(2)}</span><span class="wf-lg ${c}">C ${b.close?.toFixed(2)}</span>`;
        }
        if (A.has("ov:sma20") && f?.sma_20) html += `<span class="wf-lg" style="color:${t.sma20}">SMA20 ${f.sma_20.toFixed(2)}</span>`;
        if (A.has("ov:sma50") && f?.sma_50) html += `<span class="wf-lg" style="color:${t.sma50}">SMA50 ${f.sma_50.toFixed(2)}</span>`;
        priceLegend.innerHTML = html;
      };
      this.charts.push({ key: "price", chart: pchart, container: priceContainer, legend: priceLegend, readout: priceReadout });
      priceReadout(null, new Map());

      // ---------- SUB-PANES ----------
      const paneDefs = this._paneDefs(t, data, A);
      for (const def of paneDefs) {
        const wrap = el("wf-pane wf-pane-sub", this.root);
        const legend = el("wf-legend", wrap);
        const container = el("wf-canvas", wrap);
        const chart = LWC.createChart(container, this._baseOpts(t, false));
        const built = def.build(chart);
        this.charts.push({ key: def.key, chart, container, legend, readout: built.readout, title: def.title });
        built.readout(null, new Map());
      }

      // last pane shows the time axis
      if (this.charts.length) {
        const last = this.charts[this.charts.length - 1];
        last.chart.applyOptions({ timeScale: { visible: true, borderColor: t.border } });
      }

      this._wireSync();
      // Apply zoom: window range takes priority, then restore prev user zoom, else fit all.
      if (this._windowRange) {
        try { this.priceChart.timeScale().setVisibleRange({ from: this._windowRange[0], to: this._windowRange[1] }); } catch (e) {
          this.priceChart.timeScale().fitContent();
        }
      } else if (prevRange) {
        try { this.priceChart.timeScale().setVisibleRange(prevRange); } catch (e) {}
      } else {
        this.priceChart.timeScale().fitContent();
      }

      if (this._pinned) this.pinTrade(this._pinned, false);
    }

    _featAt(time) {
      if (!time) return null;
      return this.data.features.find(f => f.time === time) || null;
    }
    _predAt(time) {
      if (!time) return null;
      return this.data.predictions.find(p => p.time === time) || null;
    }

    _paneDefs(t, data, A) {
      const lineData = (key, src) => (src || data.features).filter(f => f[key] != null).map(f => ({ time: f.time, value: f[key] }));
      const defs = [];

      if (A.has("pane:ml")) defs.push({ key: "ml", title: "ML Predictions", build: (chart) => {
        chart.priceScale("right").applyOptions({ scaleMargins: { top: 0.15, bottom: 0.1 }, mode: 0 });
        const rp = chart.addLineSeries({ color: t.rangeProb, lineWidth: 2, priceLineVisible: false, lastValueVisible: false });
        rp.setData(data.predictions.map(p => ({ time: p.time, value: p.range_prob })));
        const dc = chart.addLineSeries({ color: t.dirConf, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        dc.setData(data.predictions.map(p => ({ time: p.time, value: p.dir_conf })));
        const guide = chart.addLineSeries({ color: t.guide, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        guide.setData(data.predictions.map(p => ({ time: p.time, value: 0.55 })));
        return { readout: (time) => {
          const p = this._predAt(time);
          this.charts.find(c => c.key === "ml").legend.innerHTML =
            `<span class="wf-lg-title">ML Predictions</span>` +
            `<span class="wf-lg" style="color:${t.rangeProb}">P(in-range) ${p ? p.range_prob.toFixed(2) : "–"}</span>` +
            `<span class="wf-lg" style="color:${t.dirConf}">Dir conf ${p ? p.dir_conf.toFixed(2) : "–"}</span>` +
            (p ? `<span class="wf-lg">${p.dir_class}</span>` : "");
        } };
      } });

      if (A.has("pane:rsi")) defs.push({ key: "rsi", title: "RSI (14)", build: (chart) => {
        const s = chart.addLineSeries({ color: t.rsi, lineWidth: 1.8, priceLineVisible: false, lastValueVisible: false });
        s.setData(lineData("rsi_14"));
        [30, 70].forEach(lv => { const g = chart.addLineSeries({ color: t.guide, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false }); g.setData(data.features.filter(f => f.rsi_14 != null).map(f => ({ time: f.time, value: lv }))); });
        return { readout: (time) => { const f = this._featAt(time); this.charts.find(c => c.key === "rsi").legend.innerHTML = `<span class="wf-lg-title">RSI (14)</span><span class="wf-lg" style="color:${t.rsi}">${f && f.rsi_14 != null ? f.rsi_14.toFixed(1) : "–"}</span>`; } };
      } });

      if (A.has("pane:macd")) defs.push({ key: "macd", title: "MACD", build: (chart) => {
        const h = chart.addHistogramSeries({ priceLineVisible: false, lastValueVisible: false });
        h.setData(data.features.filter(f => f.macd_hist != null).map(f => ({ time: f.time, value: f.macd_hist, color: f.macd_hist >= 0 ? t.up : t.down })));
        const m = chart.addLineSeries({ color: t.macd, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        m.setData(lineData("macd"));
        const sg = chart.addLineSeries({ color: t.macdSig, lineWidth: 1.5, priceLineVisible: false, lastValueVisible: false });
        sg.setData(lineData("macd_signal"));
        return { readout: (time) => { const f = this._featAt(time); this.charts.find(c => c.key === "macd").legend.innerHTML = `<span class="wf-lg-title">MACD</span><span class="wf-lg" style="color:${t.macd}">${f && f.macd != null ? f.macd.toFixed(4) : "–"}</span><span class="wf-lg" style="color:${t.macdSig}">sig ${f && f.macd_signal != null ? f.macd_signal.toFixed(4) : "–"}</span>`; } };
      } });

      if (A.has("pane:vol")) defs.push({ key: "vol", title: "Realized Vol / ATR", build: (chart) => {
        const rv = chart.addLineSeries({ color: t.rvol, lineWidth: 1.8, priceLineVisible: false, lastValueVisible: false });
        rv.setData(lineData("realized_vol_20"));
        const atr = chart.addLineSeries({ color: t.atr, lineWidth: 1.4, priceLineVisible: false, lastValueVisible: false, priceScaleId: "left" });
        atr.setData(lineData("atr_pct"));
        chart.priceScale("left").applyOptions({ visible: true, borderColor: t.border });
        return { readout: (time) => { const f = this._featAt(time); this.charts.find(c => c.key === "vol").legend.innerHTML = `<span class="wf-lg-title">Vol</span><span class="wf-lg" style="color:${t.rvol}">RV20 ${f && f.realized_vol_20 != null ? (f.realized_vol_20 * 100).toFixed(1) + "%" : "–"}</span><span class="wf-lg" style="color:${t.atr}">ATR ${f && f.atr_pct != null ? (f.atr_pct * 100).toFixed(2) + "%" : "–"}</span>`; } };
      } });

      if (A.has("pane:iv")) defs.push({ key: "iv", title: "IV Rank / VIX", build: (chart) => {
        const iv = chart.addLineSeries({ color: t.iv, lineWidth: 1.8, priceLineVisible: false, lastValueVisible: false });
        iv.setData(lineData("iv_rank"));
        const vix = chart.addLineSeries({ color: t.vix, lineWidth: 1.4, priceLineVisible: false, lastValueVisible: false, priceScaleId: "left" });
        vix.setData(lineData("vix_level"));
        chart.priceScale("left").applyOptions({ visible: true, borderColor: t.border });
        return { readout: (time) => { const f = this._featAt(time); this.charts.find(c => c.key === "iv").legend.innerHTML = `<span class="wf-lg-title">IV / VIX</span><span class="wf-lg" style="color:${t.iv}">IVR ${f && f.iv_rank != null ? f.iv_rank.toFixed(2) : "–"}</span><span class="wf-lg" style="color:${t.vix}">VIX ${f && f.vix_level != null ? f.vix_level.toFixed(1) : "–"}</span>`; } };
      } });

      if (A.has("pane:fractal")) defs.push({ key: "fractal", title: "Hurst", build: (chart) => {
        const s = chart.addLineSeries({ color: t.hurst, lineWidth: 1.8, priceLineVisible: false, lastValueVisible: false });
        s.setData(lineData("hurst_wavelet"));
        const g = chart.addLineSeries({ color: t.guide, lineWidth: 1, lineStyle: 2, priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false });
        g.setData(data.features.filter(f => f.hurst_wavelet != null).map(f => ({ time: f.time, value: 0.5 })));
        return { readout: (time) => { const f = this._featAt(time); this.charts.find(c => c.key === "fractal").legend.innerHTML = `<span class="wf-lg-title">Hurst (wavelet)</span><span class="wf-lg" style="color:${t.hurst}">${f && f.hurst_wavelet != null ? f.hurst_wavelet.toFixed(3) : "–"}</span><span class="wf-lg">${f && f.hurst_wavelet != null ? (f.hurst_wavelet < 0.5 ? "mean-revert" : "trending") : ""}</span>`; } };
      } });

      if (A.has("pane:sent")) defs.push({ key: "sent", title: "Sentiment / Put-Call", build: (chart) => {
        const s = chart.addLineSeries({ color: t.sent, lineWidth: 1.8, priceLineVisible: false, lastValueVisible: false });
        s.setData(lineData("sentiment_composite"));
        const pc = chart.addLineSeries({ color: t.pcr, lineWidth: 1.4, priceLineVisible: false, lastValueVisible: false, priceScaleId: "left" });
        pc.setData(lineData("put_call_ratio"));
        chart.priceScale("left").applyOptions({ visible: true, borderColor: t.border });
        return { readout: (time) => { const f = this._featAt(time); this.charts.find(c => c.key === "sent").legend.innerHTML = `<span class="wf-lg-title">Sentiment</span><span class="wf-lg" style="color:${t.sent}">sent ${f && f.sentiment_composite != null ? f.sentiment_composite.toFixed(2) : "–"}</span><span class="wf-lg" style="color:${t.pcr}">P/C ${f && f.put_call_ratio != null ? f.put_call_ratio.toFixed(2) : "–"}</span>`; } };
      } });

      return defs;
    }

    _applyMarkers() {
      if (!this.candle) return;
      const markers = [];
      const t = THEMES[this.theme];
      for (const tr of this.data.trades) {
        const pinned = this._pinned && this._pinned.id === tr.id;
        markers.push({ time: tr.entry_date, position: "belowBar", color: pinned ? "#f0c040" : t.dirConf, shape: "arrowUp", text: pinned ? "▶ " + tr.id : "" , size: pinned ? 2 : 1, id: "in:" + tr.id });
        markers.push({ time: tr.exit_date, position: "aboveBar", color: tr.pnl >= 0 ? t.up : t.down, shape: "arrowDown", text: pinned ? (tr.pnl >= 0 ? "+" : "") + "$" + tr.pnl : "", size: pinned ? 2 : 1, id: "out:" + tr.id });
      }
      markers.sort((a, b) => a.time < b.time ? -1 : 1);
      this.candle.setMarkers(markers);
    }

    _wireSync() {
      const charts = this.charts.map(c => c.chart);
      const syncRange = (src) => {
        if (this._syncing) return;
        this._syncing = true;
        const r = src.timeScale().getVisibleRange();
        if (r) charts.forEach(c => { if (c !== src) { try { c.timeScale().setVisibleRange(r); } catch (e) {} } });
        this._syncing = false;
      };
      charts.forEach(c => c.timeScale().subscribeVisibleTimeRangeChange(() => syncRange(c)));

      // crosshair sync + readouts + hover callback
      this.charts.forEach(entry => {
        entry.chart.subscribeCrosshairMove(param => {
          const time = param.time || null;
          // update all readouts
          this.charts.forEach(c => c.readout && c.readout(time, param.seriesData || new Map()));
          // sync crosshair to other charts
          if (!this._syncing) {
            this._syncing = true;
            this.charts.forEach(c => {
              if (c.chart === entry.chart) return;
              const anySeries = c.chart && c._firstSeries;
            });
            this._syncing = false;
          }
          if (this.onHover) this.onHover(time);
        });
        entry.chart.subscribeClick(param => {
          if (!param.time) return;
          // find nearest trade entry at/near this time
          const tr = this.data.trades.find(t => t.entry_date === param.time || t.exit_date === param.time);
          if (tr && this.onMarkerClick) this.onMarkerClick(tr);
        });
      });
    }

    // external hover (from table) → move crosshair on price chart
    setHoverTime(time) {
      if (!this.priceChart || !this.candle || !time) return;
      const b = this.data.bars.find(x => x.time === time);
      if (b) { try { this.priceChart.setCrosshairPosition(b.close, time, this.candle); } catch (e) {} }
    }
    clearHover() { try { this.priceChart && this.priceChart.clearCrosshairPosition(); } catch (e) {} }

    pinTrade(trade, recenter = true) {
      this._pinned = trade;
      this._applyMarkers();
      // draw strike price lines on price chart
      if (this._strikeLines) this._strikeLines.forEach(l => { try { this.candle.removePriceLine(l); } catch (e) {} });
      this._strikeLines = [];
      if (trade) {
        const t = THEMES[this.theme];
        trade.legs.forEach(leg => {
          const isShort = leg.type.startsWith("short");
          const pl = this.candle.createPriceLine({
            price: leg.strike, color: isShort ? t.down : t.bb, lineWidth: 1,
            lineStyle: isShort ? 0 : 2, axisLabelVisible: true,
            title: leg.type.replace("_", " ") + " " + leg.strike,
          });
          this._strikeLines.push(pl);
        });
        if (recenter) {
          const bars = this.data.bars;
          const fromDate = bars[Math.max(0, trade.entry_idx - 15)]?.time ?? trade.entry_date;
          const toDate   = bars[Math.min(bars.length - 1, trade.exit_idx + 15)]?.time ?? trade.exit_date;
          this.charts.forEach(c => { try { c.chart.timeScale().setVisibleRange({ from: fromDate, to: toDate }); } catch (e) {} });
        }
      }
    }
    unpin() { this.pinTrade(null, false); }

    // Zoom all panes to the given OOS window date range (YYYY-MM-DD strings).
    // Adds a small margin so the first and last candle aren't flush against the edge.
    setWindowRange(start, end) {
      if (!this.priceChart) return;
      try {
        const from = start, to = end;
        this.charts.forEach(c => {
          try { c.chart.timeScale().setVisibleRange({ from, to }); } catch (e) {}
        });
      } catch (e) {}
    }

    // Reset all panes to show the full dataset.
    fitAll() {
      if (!this.priceChart) return;
      try {
        this.priceChart.timeScale().fitContent();
        // Sub-panes sync via subscribeVisibleTimeRangeChange wiring.
      } catch (e) {}
    }
  }

  window.WFChart = WFChart;
})();
