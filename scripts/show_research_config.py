#!/usr/bin/env python
"""W88a: print the EFFECTIVE research configuration with the layer that decides each value.

    python scripts/show_research_config.py                       # every non-secret setting
    python scripts/show_research_config.py --section backtest    # one section
    python scripts/show_research_config.py --differs             # only values NOT at their schema default
    python scripts/show_research_config.py --study-file studies/<name>.yaml

Layers, strongest first: ``study`` (studies/<name>.yaml) > ``overlay`` (config_research.yaml) > ``config.yaml`` >
``default`` (the schema default in src/ait/config/settings.py). The fixed research limits (not configurable) are
listed after the settings. Read-only: loads the YAML files, touches no database and no broker.
"""
from __future__ import annotations

from ait.config.runtime_env import apply_runtime_env_defaults

apply_runtime_env_defaults()

import argparse  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--section", default=None, help="only keys under this section (e.g. backtest, ml)")
    ap.add_argument("--differs", action="store_true", help="only values not at their schema default")
    ap.add_argument("--study-file", default=None)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args(argv)

    from ait.config.research import _leaves, settings_origin
    from ait.config.settings import Settings
    schema_defaults = _leaves(Settings().model_dump(mode="json"))
    from ait.config.invariants_research import RESEARCH_INVARIANTS
    rows = settings_origin(args.config, args.study_file)
    shown = 0
    for dotted, (value, origin) in rows.items():
        if args.section and not dotted.startswith(args.section + "."):
            continue
        if args.differs and value == schema_defaults.get(dotted):
            continue
        print(f"{dotted:60s} {origin:12s} {value!r}")
        shown += 1
    print(f"\n{shown} setting(s) shown.")
    if not args.section:
        print("\nFixed research limits (NOT configurable; src/ait/config/invariants_research.py):")
        for inv in RESEARCH_INVARIANTS:
            print(f"  {inv.name:28s} {inv.value!r:10} {inv.unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
