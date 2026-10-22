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

# The model feature list, in its fixed order. ``{role}`` placeholders are the thirteen window-bearing names.
_BASE_TEMPLATE: tuple[str, ...] = (
    # Momentum
    "{rsi_long}", "{rsi_short}", "macd", "macd_signal", "macd_hist",
    "{roc_short}", "{roc_mid}", "{roc_long}",
    # Volatility (all normalized -- no raw price levels)
    "atr_pct", "bb_width", "bb_position",
    "bb_pct_above_upper", "bb_pct_below_lower",
    "{rv_long}", "{rv_short}",
    "high_low_range",
    # Volume
    "{volume_ratio}", "obv_change", "volume_trend",
    # Trend (slopes + ratios only -- raw SMA/EMA price levels excluded)
    "{slope_fast}", "{slope_mid}",
    "{px_vs_mid}", "{px_vs_slow}",
    "{cross}",
    "above_sma200", "distance_sma200",
    # Price action
    "daily_return", "gap", "body_size", "upper_wick", "lower_wick",
    "consecutive_up", "consecutive_down",
    # Multi-timeframe
    "weekly_trend_aligned", "weekly_rsi",
    "weekly_momentum", "volume_confirmation",
    # IV & Volatility Regime
    "iv_rank", "vol_ratio", "vol_trend", "vol_of_vol",
    "vol_regime_expanding", "vol_mean_reversion_signal",
    # Cross-Asset (VIX + relative strength)
    "vix_level", "vix_change_5d", "vix_change_20d",
    "vix_zscore", "vix_term_spread",
    "rel_strength_5d", "rel_strength_20d", "rel_strength_60d",
    "spy_momentum_10d", "spy_rsi_14", "correlation_spy_20d",
    # Macro (yield curve + DXY from FRED)
    "us_2y_yield_level", "us_10y_yield_level",
    "yield_curve_spread", "yield_curve_inverted", "yield_curve_change_20d",
    "dxy_level_norm", "dxy_change_5d", "dxy_change_20d",
    "us_10y_change_20d",
    # Seasonality
    "day_of_week", "month_of_year",
    # Fractal / multi-scale
    "hurst_wavelet", "hurst_fit_r2",
    "psd_beta", "psd_fit_r2", "hurst_psd_divergence",
    "hurst_short", "hurst_long", "hurst_scale_spread",
    "multifractal_width", "multifractal_asymmetry",
)

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
    }


def _fixed_names() -> set[str]:
    return {n for n in _BASE_TEMPLATE if "{" not in n} | {n for n in _META_TEMPLATE if "{" not in n} | {
        "Open", "High", "Low", "Close", "Volume", "atr_pct", "macd", "macd_signal", "macd_hist", "bb_upper", "bb_lower",
        "bb_width", "bb_position"}


def validate_names(names: dict[str, str]) -> None:
    """The derived names must be distinct from each other and from every fixed column name."""
    values = list(names.values())
    dup = sorted({n for n in values if values.count(n) > 1})
    if dup:
        raise ValueError(f"features: derived column names collide {dup}; give the windows distinct values")
    clash = sorted(set(values) & _fixed_names())
    if clash:
        raise ValueError(f"features: derived column names {clash} collide with fixed feature columns")
