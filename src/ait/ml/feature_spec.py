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
    from ait.config.settings import FeaturesConfig

from ait.config.feature_names import (  # noqa: F401 - re-exported: the names are part of this module's API
    _BASE_TEMPLATE,
    _META_TEMPLATE,
    FEATURE_SET_VERSION,
    derive_names,
    validate_names,
)

def _hash(cfg: "FeaturesConfig") -> str:
    canon = json.dumps({"version": FEATURE_SET_VERSION, "features": cfg.model_dump(mode="json")}, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class FeatureSpec:
    """The indicator windows plus everything derived from them."""

    cfg: "FeaturesConfig"
    names: dict

    @classmethod
    def from_config(cls, cfg: "FeaturesConfig") -> "FeatureSpec":
        return cls(cfg=cfg, names=derive_names(cfg))

    def feature_names(self, vlmc_names: list[str] | None = None) -> list[str]:
        """The ordered model feature list (``vlmc_names`` appended when given)."""
        base = [n.format_map(self.names) for n in _BASE_TEMPLATE]
        return base + list(vlmc_names) if vlmc_names else base

    def meta_feature_names(self) -> list[str]:
        return [n.format_map(self.names) for n in _META_TEMPLATE]

    def spec_hash(self) -> str:
        return _hash(self.cfg)


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
