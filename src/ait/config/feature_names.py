"""W88d: feature column-name derivation -- a LEAF module in ``ait.config`` (stdlib only).

``settings.FeaturesConfig`` validates at import time (``Settings`` is built when ``ait.config.settings`` loads), so the
rule "derived column names are distinct and do not collide with a fixed column" must live where importing it cannot pull in
``ait.ml`` (whose package init imports the broker, which imports settings: a cycle). ``ait.ml.feature_spec`` builds on this
module and re-exports it; see there for the full description of the feature spec.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ait.config.settings import FeaturesConfig

# Bump when the MATHS of a feature group changes (not when a window does): it invalidates every recorded hash.
FEATURE_SET_VERSION = 1

# W98: the 26 VLMC / intraday column names (fixed; they do not follow a window). Moved here from ``FeatureEngine`` so the
# catalogue below, the config validator and the engine share one list (``FeatureEngine.VLMC_FEATURE_NAMES`` re-exports it).
VLMC_NAMES: tuple[str, ...] = (
    # Original 6 intraday features
    "intraday_vwap_position", "intraday_rsi", "intraday_momentum_1h",
    "intraday_atr_pct", "intraday_vol_ratio", "intraday_range_compression",
    # Phase 6a: intraday fractal (7 features)
    "hurst_wavelet_intraday", "hurst_scale_spread_intraday",
    "wavelet_L3_energy", "wavelet_L4_energy", "wavelet_L5_energy",
    "psd_beta_intraday", "mfdfa_width_intraday",
    # Phase 6b: VLMC session structure (13 features)
    "session_vwap_position", "session_vwap_q1", "session_vwap_q2", "session_vwap_q3",
    "session_high_timing", "session_low_timing",
    "session_volume_front_load", "session_volume_shape",
    "power_hour_momentum", "power_hour_vol_accel", "power_hour_vwap_cross",
    "closing_imbalance", "closing_range_position",
)

# W98: EVERY feature a model can use, by group, in the model's fixed column order. A group is the ``ml.features`` group
# whose windows the feature reads, and ``ml.features.<group>.use`` holds one on/off switch per name below. A name is a ROLE:
# for the 27 window-bearing columns it is the role key of ``derive_names`` (``rsi_long`` -> ``rsi_14`` at the defaults), so
# changing a window never renames a switch; for every other column it is the column name itself. The first six W88d groups
# and the W97 groups keep the order the model list has always had; ``intraday`` is the 26 VLMC names, appended last and only
# when an intraday store was used.
CATALOGUE: dict[str, tuple[str, ...]] = {
    "momentum": ("rsi_long", "rsi_short", "macd", "macd_signal", "macd_hist", "roc_short", "roc_mid", "roc_long"),
    "volatility": ("atr_pct", "bb_width", "bb_position", "bb_pct_above_upper", "bb_pct_below_lower",
                   "rv_long", "rv_short", "high_low_range"),
    "volume": ("volume_ratio", "obv_change", "volume_trend"),
    "trend": ("slope_fast", "slope_mid", "px_vs_mid", "px_vs_slow", "cross"),
    "market_structure": ("above_sma_long", "distance_sma_long",
                         # computed in the frame today but never a model input: switched OFF by default (W98)
                         "put_call_proxy", "vix_term_structure", "skew_proxy", "momentum_divergence", "range_compression"),
    "price_action": ("daily_return", "gap", "body_size", "upper_wick", "lower_wick", "consecutive_up", "consecutive_down"),
    "multi_timeframe": ("weekly_trend_aligned", "weekly_rsi", "weekly_momentum", "volume_confirmation"),
    "vol_regime": ("iv_rank", "vol_ratio", "vol_trend", "vol_of_vol", "vol_regime_expanding", "vol_mean_reversion_signal"),
    "cross_asset": ("vix_level", "vix_change_short", "vix_change_long", "vix_zscore", "vix_term_spread",
                    "rel_short", "rel_mid", "rel_long", "spy_momentum", "spy_rsi", "corr_spy"),
    "macro": ("us_2y_yield_level", "us_10y_yield_level", "yield_curve_spread", "yield_curve_inverted",
              "yield_curve_change", "dxy_level_norm", "dxy_change_short", "dxy_change_long", "us10y_change"),
    "seasonality": ("day_of_week", "month_of_year"),
    "fractal": ("hurst_wavelet", "hurst_fit_r2", "psd_beta", "psd_fit_r2", "hurst_psd_divergence",
                "hurst_short", "hurst_long", "hurst_scale_spread", "multifractal_width", "multifractal_asymmetry"),
    "intraday": VLMC_NAMES,
}

# W98: features the code computes but no model used before W98. They ship switched OFF, so the default model list (and
# every saved model's hash) is unchanged; a study switches one on. A NEW feature is added the same way: its name in
# CATALOGUE, off here, and its computation in ``ml/features.py``.
OFF_BY_DEFAULT: dict[str, frozenset[str]] = {
    "market_structure": frozenset({"put_call_proxy", "vix_term_structure", "skew_proxy",
                                   "momentum_divergence", "range_compression"}),
}

# group -> {feature: default switch}: what ``ml.features.<group>.use`` holds when config.yaml says nothing.
DEFAULT_USE: dict[str, dict[str, bool]] = {
    g: {r: r not in OFF_BY_DEFAULT.get(g, frozenset()) for r in roles} for g, roles in CATALOGUE.items()}

def use_switches(cfg: "FeaturesConfig") -> dict[str, dict[str, bool]]:
    """{group: its ``use`` switches} in catalogue order. The one place that reads every group's switches (an explicit
    attribute read per group, so that a group nothing else touches, like ``price_action``, still has a reader)."""
    return {"momentum": cfg.momentum.use, "volatility": cfg.volatility.use, "volume": cfg.volume.use,
            "trend": cfg.trend.use, "market_structure": cfg.market_structure.use, "price_action": cfg.price_action.use,
            "multi_timeframe": cfg.multi_timeframe.use, "vol_regime": cfg.vol_regime.use,
            "cross_asset": cfg.cross_asset.use, "macro": cfg.macro.use, "seasonality": cfg.seasonality.use,
            "fractal": cfg.fractal.use, "intraday": cfg.intraday.use}


# The catalogue names that follow a window (the role keys of ``derive_names`` that are model columns).
_WINDOW_ROLES = frozenset({
    "rsi_long", "rsi_short", "roc_short", "roc_mid", "roc_long", "rv_long", "rv_short", "volume_ratio",
    "slope_fast", "slope_mid", "px_vs_mid", "px_vs_slow", "cross",
    "above_sma_long", "distance_sma_long", "vix_change_short", "vix_change_long", "rel_short", "rel_mid", "rel_long",
    "spy_momentum", "spy_rsi", "corr_spy", "yield_curve_change", "us10y_change", "dxy_change_short", "dxy_change_long"})

# The DEFAULT model feature list, in its fixed order, derived from the catalogue (the intraday names are separate).
# ``{role}`` placeholders are the window-bearing names (thirteen from W88d, fourteen more from W97: the long-SMA pair, the
# VIX / relative-strength / SPY / correlation columns and the macro changes). A switch that differs from DEFAULT_USE is
# applied in ``FeatureSpec.feature_names`` (``ml/feature_spec.py``), not here.
_BASE_TEMPLATE: tuple[str, ...] = tuple(
    ("{%s}" % r) if r in _WINDOW_ROLES else r
    for g, roles in CATALOGUE.items() if g != "intraday" for r in roles if DEFAULT_USE[g][r])

# The meta-labeler's 20 inputs, in order (they are also dict keys built by the engine, the orchestrator and the
# meta-labeler's own builders, so the window-bearing ones follow the spec).
_META_TEMPLATE: tuple[str, ...] = (
    "primary_confidence", "regime_trending_up", "regime_trending_down", "regime_high_vol", "regime_range_bound",
    "vix", "iv_rank", "sentiment_score", "{rsi_long}", "{rsi_short}", "bb_position", "{volume_ratio}", "{rv_long}",
    "atr_pct", "weekly_trend_aligned", "volume_confirmation", "hour_of_day", "macd_hist", "{px_vs_mid}", "{cross}",
)

# Names the engine writes into frames that are NOT model features (raw levels): derived too, so they never collide.
_RAW_ROLES = ("sma_fast", "sma_mid", "sma_slow", "ema_fast", "ema_slow", "atr_raw")


def derive_names(cfg: "FeaturesConfig") -> dict[str, str]:
    """role -> column name for every window-bearing column (model features AND raw helper columns)."""
    m, v, vo, tr = cfg.momentum, cfg.volatility, cfg.volume, cfg.trend
    ms, ca, mc = cfg.market_structure, cfg.cross_asset, cfg.macro                       # W97
    return {
        "rsi_long": f"rsi_{m.rsi_long}", "rsi_short": f"rsi_{m.rsi_short}",
        "roc_short": f"roc_{m.roc_short}", "roc_mid": f"roc_{m.roc_mid}", "roc_long": f"roc_{m.roc_long}",
        "rv_long": f"realized_vol_{v.rv_long}", "rv_short": f"realized_vol_{v.rv_short}",
        "volume_ratio": f"volume_sma_{vo.volume_avg_window}_ratio",
        "slope_fast": f"sma_{tr.sma_fast}_slope", "slope_mid": f"sma_{tr.sma_mid}_slope",
        "px_vs_mid": f"price_vs_sma_{tr.sma_mid}", "px_vs_slow": f"price_vs_sma_{tr.sma_slow}",
        "cross": f"sma_{tr.sma_fast}_{tr.sma_mid}_cross",
        # raw helper columns (in the frame, not model inputs)
        "sma_fast": f"sma_{tr.sma_fast}", "sma_mid": f"sma_{tr.sma_mid}", "sma_slow": f"sma_{tr.sma_slow}",
        "ema_fast": f"ema_{m.macd_fast}", "ema_slow": f"ema_{m.macd_slow}", "atr_raw": f"atr_{v.atr_window}",
        # W97: the cross-asset / macro / long-SMA model columns whose name embeds a window
        "above_sma_long": f"above_sma{ms.long_sma_window}", "distance_sma_long": f"distance_sma{ms.long_sma_window}",
        "vix_change_short": f"vix_change_{ca.vix_change_short}d", "vix_change_long": f"vix_change_{ca.vix_change_long}d",
        "rel_short": f"rel_strength_{ca.rel_short}d", "rel_mid": f"rel_strength_{ca.rel_mid}d",
        "rel_long": f"rel_strength_{ca.rel_long}d",
        "spy_momentum": f"spy_momentum_{ca.spy_momentum_days}d", "spy_rsi": f"spy_rsi_{ca.spy_rsi_period}",
        "corr_spy": f"correlation_spy_{ca.corr_window}d",
        "yield_curve_change": f"yield_curve_change_{mc.change_days}d", "us10y_change": f"us_10y_change_{mc.change_days}d",
        "dxy_change_short": f"dxy_change_{mc.dxy_change_short}d", "dxy_change_long": f"dxy_change_{mc.dxy_change_long}d",
    }


def _fixed_names() -> set[str]:
    return {n for n in _BASE_TEMPLATE if "{" not in n} | {n for n in _META_TEMPLATE if "{" not in n} | {
        "Open", "High", "Low", "Close", "Volume", "atr_pct", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower",
        "bb_width", "bb_position"} | {n for roles in OFF_BY_DEFAULT.values() for n in roles}


def validate_names(names: dict[str, str]) -> None:
    """The derived names must be distinct from each other and from every fixed column name."""
    values = list(names.values())
    dup = sorted({n for n in values if values.count(n) > 1})
    if dup:
        raise ValueError(f"features: derived column names collide {dup}; give the windows distinct values")
    clash = sorted(set(values) & _fixed_names())
    if clash:
        raise ValueError(f"features: derived column names {clash} collide with fixed feature columns")
