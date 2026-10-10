"""W68: the research configuration overlay (`config_research.yaml`).

Research needs values live must not have (gates enforced, IB-sourced research tables, friction
model v2, the $5k pool, ...). Rather than scatter them through `config.yaml`, they live in ONE
overlay file that is deep-merged over `config.yaml` -- only by RESEARCH entry points:

* the overlay holds only the keys that DIFFER, so a later live change in `config.yaml` flows into
  research automatically (parity by construction) and the file IS the research-vs-live diff;
* `use_research_config()` loads base + overlay once and `freeze_settings()`s the result, so the
  71 `load_settings()` calls inside research code see the merged config without being edited;
* the live entry points (`ait.main`, the supervisor) never call it -- a class guard in
  tests/test_w68_research_config.py fails if any live module references this module.

Call it first thing in a research entry point, before importing trading subsystems.
"""
from __future__ import annotations

from pathlib import Path

from ait.config.settings import Settings, freeze_settings, load_settings

RESEARCH_OVERLAY_NAME = "config_research.yaml"


def research_overlay_path(base_config: str | Path = "config.yaml") -> Path:
    """The overlay lives next to the base config file."""
    return Path(base_config).resolve().parent / RESEARCH_OVERLAY_NAME


def load_research_settings(base_config: str | Path = "config.yaml") -> Settings:
    """Base config + research overlay, validated (not frozen)."""
    return load_settings(base_config, overlay_path=research_overlay_path(base_config))


def use_research_config(base_config: str | Path = "config.yaml") -> Settings:
    """Make base+overlay this PROCESS's configuration (research entry points only)."""
    settings = load_research_settings(base_config)
    freeze_settings(settings)
    return settings
