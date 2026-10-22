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

W87b adds a THIRD layer for one study: ``studies/<name>.yaml`` (``load_study``), merged over the other two
by ``use_research_config(study=...)``. ``config_research.yaml`` stays the stable default for ordinary
research runs; a study file is committed, named and immutable once it has run, and its ``rule:`` says
whether the study is exploratory or pre-registered in PLAN.md (a test checks the PLAN entry exists).

Call it first thing in a research entry point, before importing trading subsystems.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ait.config.settings import Settings, freeze_settings, load_settings

RESEARCH_OVERLAY_NAME = "config_research.yaml"
STUDIES_DIR_NAME = "studies"
STUDY_KEYS = frozenset({"name", "question", "rule", "settings", "search_space"})
EXPLORATORY = "exploratory"
_NAME = re.compile(r"^[a-z0-9][a-z0-9_]*$")


@dataclass(frozen=True)
class Study:
    """One validated study file."""

    name: str
    question: str
    rule: str                                   # EXPLORATORY, or the text of a PLAN.md heading
    settings: dict = field(default_factory=dict)        # merged over config.yaml + config_research.yaml
    search_space: dict = field(default_factory=dict)    # {strategy: {dimension: [low, high]}}
    path: Path | None = None

    @property
    def exploratory(self) -> bool:
        return self.rule == EXPLORATORY


def _contract_conflicts(settings_block: dict) -> list[str]:
    """``section.key`` entries of a study's ``settings:`` whose home is read by the ENV CONTRACT.

    ``runtime_env.contract_float`` & co. read the RAW ``config.yaml`` and the environment and ignore both
    overlays, so a study value for one of them would be silently ignored by ``run_backtest``'s builders."""
    from ait.config.runtime_env import CONFIG_BACKED
    homes = {home: env for env, home in CONFIG_BACKED.items()}
    out = []
    for section, body in settings_block.items():
        for key in (body or {}) if isinstance(body, dict) else ():
            if (section, key) in homes:
                out.append(f"{section}.{key} ({homes[(section, key)]})")
    return sorted(out)


def load_study(path: str | Path) -> Study:
    """Read and validate a study file. Raises ``ValueError`` (naming the file and key) on anything wrong."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"study file not found: {p}")
    raw = yaml.safe_load(p.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: a study file is a YAML mapping")
    unknown = sorted(set(raw) - STUDY_KEYS)
    if unknown:
        raise ValueError(f"{p}: unknown key(s) {unknown}; allowed: {sorted(STUDY_KEYS)}")
    for key in ("name", "question", "rule"):
        if not isinstance(raw.get(key), str) or not raw[key].strip():
            raise ValueError(f"{p}: '{key}' is required (a non-empty string)")
    if not _NAME.match(raw["name"]):
        raise ValueError(f"{p}: name must be lowercase letters, digits and underscores, got {raw['name']!r}")
    settings_block = raw.get("settings") or {}
    if not isinstance(settings_block, dict):
        raise ValueError(f"{p}: 'settings' must be a mapping of section -> keys")
    for forbidden in ("ibkr", "api_keys"):
        if forbidden in settings_block:
            raise ValueError(f"{p}: a study may not set {forbidden!r} (secrets come from the environment only)")
    conflicts = _contract_conflicts(settings_block)
    if conflicts:
        raise ValueError(
            f"{p}: these settings are read by the env contract, which ignores the research overlay and "
            f"studies, so the value would be silently ignored: {conflicts}. Set the environment variable "
            "or pass the CLI flag instead.")
    space = raw.get("search_space") or {}
    if not isinstance(space, dict) or not all(isinstance(v, dict) for v in space.values()):
        raise ValueError(f"{p}: 'search_space' must map strategy -> {{dimension: [low, high]}}")
    from ait.optimization.param_spaces import effective_space      # validates every override now
    for strategy, dims in space.items():
        effective_space(strategy, dims)
    return Study(name=raw["name"], question=raw["question"].strip(), rule=raw["rule"].strip(),
                 settings=settings_block, search_space=space, path=p)


def rule_is_registered(rule: str, plan_text: str) -> bool:
    """True when ``rule`` is EXPLORATORY or names a PLAN.md heading: some ``## ...`` line contains it."""
    if rule == EXPLORATORY:
        return True
    return any(line.startswith("## ") and rule in line for line in plan_text.splitlines())


def assert_registered(study: "Study", plan_path: "str | Path | None" = None) -> None:
    """Refuse to RUN a pre-registered study whose ``rule:`` names no PLAN.md heading (the two-tier rule:
    exploratory studies need nothing; the others are registered BEFORE they run)."""
    path = Path(plan_path) if plan_path else Path(__file__).resolve().parents[3] / "PLAN.md"
    text = path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""
    if not rule_is_registered(study.rule, text):
        raise ValueError(f"study {study.name!r}: rule {study.rule!r} is neither 'exploratory' nor a heading in "
                         f"{path.name}; append the AS PLANNED entry first, or mark the study exploratory")


def resolve_universe(symbols: "list[str] | None", strategies: "list[str] | None",
                     settings: Settings) -> tuple[list[str], list[str]]:
    """(symbols, strategies) a research entry point runs: the explicit CLI lists, else the config.

    W87a: ``run_backtest.py`` defaulted to a hard-coded ten-name list that crashed on the first symbol
    without dividend rows, and to four strategies although the live selector is IC-only. Bare runs now
    use ``backtest.default_symbols`` (QQQ) and ``options.strategies`` (the live list); explicit lists win.
    """
    return (list(symbols) if symbols else list(settings.backtest.default_symbols),
            list(strategies) if strategies else list(settings.options.strategies))


def _leaves(d: dict, prefix: str = "") -> dict[str, object]:
    """Nested mapping -> {"a.b.c": leaf}; lists and scalars are leaves."""
    out: dict[str, object] = {}
    for k, v in (d or {}).items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.update(_leaves(v, key + "."))
        else:
            out[key] = v
    return out


def settings_origin(base_config: "str | Path" = "config.yaml",
                    study: "Study | str | Path | None" = None) -> dict[str, tuple[object, str]]:
    """{dotted.key: (value, origin)} for every non-secret setting of the research view (W88a).

    ``origin`` is the LAYER that decides the value: ``study`` > ``overlay`` (config_research.yaml) >
    ``config.yaml`` > ``default`` (the schema default). This is the one place a researcher can see every
    tunable with where it comes from, without any value being copied into a second file."""
    study_obj = None if study is None else (study if isinstance(study, Study) else load_study(study))
    effective = load_research_settings(base_config, study_obj).model_dump(mode="json")
    layers = (
        ("study", _leaves(study_obj.settings) if study_obj else {}),
        ("overlay", _leaves(yaml.safe_load(research_overlay_path(base_config).read_text()) or {})),
        ("config.yaml", _leaves(yaml.safe_load(Path(base_config).read_text()) or {})),
    )
    from ait.config.settings import _SECRET_SECTIONS, _is_secret
    out: dict[str, tuple[object, str]] = {}
    for dotted, value in _leaves(effective).items():
        section = dotted.split(".", 1)[0]
        if section in _SECRET_SECTIONS or _is_secret(section, dotted.rsplit(".", 1)[-1]):
            continue
        origin = next((name for name, leaves in layers if dotted in leaves), "default")
        out[dotted] = (value, origin)
    return out


def research_overlay_path(base_config: str | Path = "config.yaml") -> Path:
    """The overlay lives next to the base config file."""
    return Path(base_config).resolve().parent / RESEARCH_OVERLAY_NAME


def load_research_settings(base_config: str | Path = "config.yaml",
                           study: "Study | str | Path | None" = None) -> Settings:
    """Base config + research overlay (+ a study's ``settings:`` block), validated (not frozen)."""
    extra = []
    if study is not None:
        extra.append((study if isinstance(study, Study) else load_study(study)).settings)
    return load_settings(base_config, overlay_path=research_overlay_path(base_config),
                         extra_overlays=extra)


def use_research_config(base_config: str | Path = "config.yaml",
                        study: "Study | str | Path | None" = None) -> Settings:
    """Make base+overlay(+study) this PROCESS's configuration (research entry points only)."""
    settings = load_research_settings(base_config, study)
    freeze_settings(settings)
    from ait.config.invariants_research import log_research_invariants   # W88a: the fixed limits, once per process
    log_research_invariants()
    return settings
