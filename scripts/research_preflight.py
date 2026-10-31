"""W116: check, without running anything, whether the stored history holds every input a study needs.

    python scripts/research_preflight.py [--study-file studies/<name>.yaml] [--symbols SPY QQQ]
        [--db-path data/historical.db] [--history-start YYYY-MM-DD] [--history-end YYYY-MM-DD] [--json]

Research never substitutes or mixes data sources: a study whose inputs are not stored does not start. This is the
same check the walk-forward runs at the start of every study (``ait.backtesting.preflight`` builds the request from the
run's settings; ``ait.data.research_inputs`` checks it against the history database, read-only), run on its own so
the gaps can be fixed first. It lists EVERY gap at once, each with what was needed, what was found and the command
that closes it. Exit status: 0 = every input is stored, 2 = at least one gap (or the configuration could not be
loaded). It writes nothing and opens the database read-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

_root = Path(__file__).resolve().parent.parent
if str(_root / "src") not in sys.path:
    sys.path.insert(0, str(_root / "src"))


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--study-file", default=None, metavar="PATH",
                    help="a studies/<name>.yaml merged over config.yaml + config_research.yaml (as run_backtest.py does)")
    ap.add_argument("--symbols", nargs="+", default=None, help="default: the research universe")
    ap.add_argument("--db-path", default="data/historical.db", help="the history database (default data/historical.db)")
    ap.add_argument("--table-prefix", default=None, help="default: backtest.research_table_prefix")
    ap.add_argument("--history-start", default=None, metavar="YYYY-MM-DD", help="override backtest.history_start_date")
    ap.add_argument("--history-end", default=None, metavar="YYYY-MM-DD", help="override backtest.history_end_date")
    ap.add_argument("--json", action="store_true", help="print the report as JSON")
    args = ap.parse_args(argv)

    from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
    from ait.config.research import assert_registered, load_study, resolve_universe, use_research_config
    from ait.data.research_inputs import check_research_inputs, format_report

    try:
        study = load_study(args.study_file) if args.study_file else None
        if study is not None:
            assert_registered(study)
        cfg_all = use_research_config(study=study)
        symbols, strategies = resolve_universe(args.symbols, None, cfg_all)
        bt_cfg = cfg_all.backtest
        cfg = WalkForwardConfig(
            history_start_date=(date.fromisoformat(args.history_start) if args.history_start
                                else bt_cfg.history_start_date),
            history_end_date=(date.fromisoformat(args.history_end) if args.history_end else bt_cfg.history_end_date),
            history_days=bt_cfg.history_days, macro_event_types=tuple(bt_cfg.macro_event_types))
        prefix = args.table_prefix if args.table_prefix is not None else bt_cfg.research_table_prefix
        bt = WalkForwardBacktester(symbols=list(symbols), strategies=list(strategies), config=cfg,
                                   db_path=Path(args.db_path), table_prefix=prefix, strict_inputs=True)
        report = check_research_inputs(bt.research_request())
    except Exception as exc:  # noqa: BLE001 - a configuration that cannot be read is a refusal, said plainly
        print(f"REFUSED: the study's configuration could not be resolved: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report.as_dict(), indent=2) if args.json else format_report(report))
    return 0 if report.ok else 2


if __name__ == "__main__":
    sys.exit(main())
