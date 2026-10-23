"""W88d: the feature SPEC -- the indicator windows, the column names derived from them, and a hash of both.

Before W88d the windows of the core feature groups (RSI 14 / 7, SMA 10 / 20 / 50, realized vol 20 / 10, ...) were
literals inside ``ml/features.py``, and thirteen model columns carry a window in their NAME (``rsi_14``,
``price_vs_sma_20``). Making a window tunable therefore means three things move together: the maths, the column
name, and every consumer that reads the column by name (the meta-labeler's 20 inputs, the engine's and the live
orchestrator's meta-context dicts, the dashboard export). This module is their single source:

* ``derive_names(cfg)``: role -> column name (``rsi_long`` -> ``rsi_14`` at the defaults, ``rsi_10`` at 10);
* ``FeatureSpec``: the config plus its derived names, the ordered feature-name list and the meta-label feature list;
* ``spec_hash()``: a short hash of the windows (and ``FEATURE_SET_VERSION``) recorded in every saved model, so a model
  trained under other windows can be recognised and not served (``ml/hyperparams.model_spec`` carries it);
* ``default_feature_spec()``: the spec of the LOADED settings (the frozen research view inside a research process),
  cached so the 23 argument-free ``FeatureEngine()`` constructions do not re-read the config.

A model with no recorded hash predates W88d and was, by construction, trained under the default windows:
``DEFAULT_SPEC_HASH`` is the hash it is assumed to carry.

The name derivation and its validation live in ``ait.config.feature_names`` (a leaf that settings can import at load time).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from ait.config.settings import FeaturesConfig, MetaLabelNeutralsConfig

from ait.config.feature_names import (  # noqa: F401 - re-exported: the names are part of this module's API
    _BASE_TEMPLATE,
    _META_TEMPLATE,
    CATALOGUE,
    DEFAULT_USE,
    FEATURE_SET_VERSION,
    derive_names,
    use_switches,
    validate_names,
)

# W97: the groups added after W88d. They are hashed ONLY when a value differs from its default, so a spec at the default
# values canonicalises exactly as it did before W97 and every saved model keeps the hash it was recorded with (changing
# a W97 window changes the hash, and the usual refusal / retrain follows). Never remove a name from this tuple.
_W97_GROUPS = ("market_structure", "cross_asset", "macro", "fractal", "intraday")


# W98: the groups that hold only ``use`` switches (no window); never part of the window dump.
_W98_GROUPS = ("price_action", "seasonality")
_W88D_GROUPS = ("momentum", "volatility", "volume", "trend", "multi_timeframe", "vol_regime")


def _switch_changes(cfg: "FeaturesConfig") -> dict:
    """W98: {group: {feature: switch}} for the ``use`` switches that differ from their default (empty at the defaults)."""
    out: dict = {}
    switches = use_switches(cfg)
    for g, roles in CATALOGUE.items():
        use, default = switches[g], DEFAULT_USE[g]
        diff = {r: use[r] for r in roles if use.get(r, default[r]) != default[r]}
        if diff:
            out[g] = diff
    return out


def _hash(cfg: "FeaturesConfig") -> str:
    # W98: ``use`` is not part of the window dump (it would change the hash at the defaults); switches are added below,
    # and only when one differs from its default, exactly like the W97 windows.
    exclude: dict = {g: {"use"} for g in _W88D_GROUPS}
    exclude.update({g: True for g in _W97_GROUPS + _W98_GROUPS})
    features = cfg.model_dump(mode="json", exclude=exclude)
    payload: dict = {"version": FEATURE_SET_VERSION, "features": features}
    changed = {g: getattr(cfg, g).model_dump(mode="json", exclude_defaults=True, exclude={"use"}) for g in _W97_GROUPS}
    changed = {g: v for g, v in changed.items() if v}
    if changed:
        payload["w97"] = changed
    switches = _switch_changes(cfg)
    if switches:
        payload["w98"] = switches
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FeatureSpec:
    """The indicator windows plus everything derived from them."""

    cfg: "FeaturesConfig"
    names: dict

    @classmethod
    def from_config(cls, cfg: "FeaturesConfig") -> "FeatureSpec":
        return cls(cfg=cfg, names=derive_names(cfg))

    def feature_names(self, vlmc_names: list[str] | None = None) -> list[str]:
        """The ordered model feature list (``vlmc_names`` appended when given). W98: a feature whose ``ml.features.<group>.use``
        switch is false is left out; this is the ONE place the switches are applied (every model's training columns come
        from here). At the default switches the list is exactly ``_BASE_TEMPLATE`` filled with the window-bearing names."""
        base = []
        switches = use_switches(self.cfg)
        for g, roles in CATALOGUE.items():
            if g == "intraday":
                continue
            use, default = switches[g], DEFAULT_USE[g]
            base.extend(self.names.get(r, r) for r in roles if use.get(r, default[r]))
        if not vlmc_names:
            return base
        iuse = switches["intraday"]
        return base + [n for n in vlmc_names if iuse.get(n, True)]

    def meta_feature_names(self) -> list[str]:
        return [n.format_map(self.names) for n in _META_TEMPLATE]

    def meta_neutral_values(self, nz: "MetaLabelNeutralsConfig") -> dict:
        """W90c: the neutral value of each of the eleven TECHNICAL meta-label inputs, keyed by this spec's column names
        (``rsi_14`` at the default windows) from ``ml.meta_label_neutrals`` ``nz``. The engine's entry gate, both
        training-data builders and the live entry snapshot all fall back to these, so a missing column reads the
        same number everywhere."""
        n = self.names
        return {n["rsi_long"]: nz.rsi, n["rsi_short"]: nz.rsi, "bb_position": nz.bb_position,
                n["volume_ratio"]: nz.volume_ratio, n["rv_long"]: nz.realized_vol, "atr_pct": nz.atr_pct,
                "weekly_trend_aligned": nz.weekly_trend_aligned, "volume_confirmation": nz.volume_confirmation,
                "macd_hist": nz.macd_hist, n["px_vs_mid"]: nz.px_vs_mid, n["cross"]: nz.cross}

    def spec_hash(self) -> str:
        return _hash(self.cfg)

    def longest_lookback(self) -> int:
        """W97: the most history any windowed daily feature looks back over (bars). A walk-forward test window needs that
        much context in front of it so its features match the ones the model was trained on; the longest of the IV-rank
        range, the long moving average, the long Hurst window and the longest of the other daily windows (slow SMA, slow
        volatility, long relative strength, VIX z-score, vol term structure, momentum divergence). 252 at the defaults (the
        IV-rank range). The single definition: the walk-forward's context check reads it."""
        c = self.cfg
        return max(c.vol_regime.iv_rank_lookback, c.market_structure.long_sma_window, c.fractal.hurst_long_window,
                   c.trend.sma_slow, c.vol_regime.slow, c.cross_asset.rel_long, c.cross_asset.vix_z_window,
                   c.market_structure.term_slow, c.market_structure.divergence_long)


def spec_from_config(cfg: "FeaturesConfig | None") -> FeatureSpec:
    if cfg is None:
        from ait.config.settings import FeaturesConfig
        cfg = FeaturesConfig()
    return FeatureSpec.from_config(cfg)


def default_spec_hash() -> str:
    """Hash of the schema-default windows: what a model saved before W88d (no recorded hash) is assumed to carry."""
    from ait.config.settings import FeaturesConfig
    return _hash(FeaturesConfig())


_CACHE: dict = {"key": None, "spec": None}


def default_feature_spec() -> FeatureSpec:
    """The spec of the LOADED settings (``ml.features``), cached on the settings snapshot it was read from.

    In a research process the snapshot is the frozen overlay view, so a study's windows reach every argument-free
    ``FeatureEngine()``; in the live bot it is the frozen ``config.yaml``; unfrozen callers (tests, scripts) key on the
    config file's mtime. Falls back to the schema defaults when no config loads."""
    from pathlib import Path

    from ait.config import settings as st
    frozen = getattr(st, "_FROZEN", None)
    if frozen is not None:
        key: object = ("frozen", id(frozen))
    else:
        try:
            key = ("file", Path("config.yaml").stat().st_mtime_ns, id(st.load_settings))
        except OSError:
            key = ("nofile", id(st.load_settings))
    if _CACHE["key"] == key and _CACHE["spec"] is not None:
        return _CACHE["spec"]
    try:
        cfg = st.load_settings().ml.features
    except Exception:  # noqa: BLE001 - no config.yaml, or a stub settings object -> the schema defaults
        cfg = None
    spec = spec_from_config(cfg)
    _CACHE["key"], _CACHE["spec"] = key, spec
    return spec
