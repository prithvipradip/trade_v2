/* ============================================================
   Predictor-Models — metric & plot dictionary
   ------------------------------------------------------------
   Single source of truth for every info-box definition shown in
   the Predictor Models view. Each entry is intentionally terse:
     title — the metric/plot name
     what  — one-line definition
     why   — why it matters / how to read it
   Edit here for wording changes; the UI reads window.PM_INFO[id].
   ============================================================ */
window.PM_INFO = {
  /* ---- top summary metrics ---- */
  avg_cv: {
    title: "Average skill",
    what: "Mean cross-validated skill score (AUROC for directional, balanced accuracy for range) averaged over every walk-forward window.",
    why: "Confirms the predictor has consistent skill through time — not a one-window fluke.",
  },
  mean_edge: {
    title: "Mean edge / window",
    what: "Average of (skill − 0.50 baseline) across windows.",
    why: "The size of the model's real edge over a coin-flip; near zero means no usable signal.",
  },
  best_window: {
    title: "Best window",
    what: "The walk-forward window with the highest average skill score.",
    why: "Pinpoints the regime the models handle best — a starting point for diagnosis.",
  },
  gating: {
    title: "Gated windows",
    what: "Windows where the predictor was switched off because its edge fell below the usable floor (0.10 over baseline for range; the confidence floor for directional).",
    why: "Gated windows emit no live signals — a high count means the strategy often sat out.",
  },
  dominant: {
    title: "Dominant member",
    what: "The ensemble member with the highest average fitted weight across windows.",
    why: "Identifies which model actually drives predictions — what to trust and maintain.",
  },
  in_range_rate: {
    title: "In-range base rate",
    what: "Average share of days that stayed within ±5% over the next 30 days.",
    why: "The naive prior the model must beat; high values make 'in-range' easy and inflate raw accuracy.",
  },

  /* ---- metric definitions ---- */
  metric_directional: {
    title: "AUROC",
    what: "Area under the ROC curve (one-vs-rest, macro-averaged) for the 3-class direction call.",
    why: "Measures how well the model ranks the correct class above the others; 0.50 = chance, and it is immune to class imbalance.",
  },
  metric_range: {
    title: "Balanced accuracy",
    what: "The average of sensitivity and specificity for the binary in-range / breakout call.",
    why: "Baselines at 0.50 no matter how lopsided in-range vs breakout days are — a fair skill measure.",
  },

  /* ---- plots ---- */
  skill_trend: {
    title: "Member skill across windows",
    what: "Each line is one ensemble member's out-of-sample skill score per window; the dashed line is the 0.50 no-skill baseline.",
    why: "Shows which members hold up across regimes and where skill decays. Amber bands mark gated windows.",
  },
  weight_stack: {
    title: "Fitted ensemble weight",
    what: "Per-window fitted weights, stacked to 100%.",
    why: "Reveals how the blend shifts over time; weight is proportional to edge, so zero-edge members disappear.",
  },
  cv_bars: {
    title: "Member CV vs baseline",
    what: "This window's skill score for each member; the black tick marks the 0.50 baseline.",
    why: "Compares members head-to-head for the selected window, with edge and fitted weight alongside.",
  },
  reliability: {
    title: "Reliability · predicted vs actual",
    what: "Predicted probability (x) vs actual observed frequency (y), binned; the dashed diagonal is perfect calibration.",
    why: "Below the diagonal = over-confident. Toggle members to see how each one's calibration feeds the blended ensemble.",
  },
};
