"""CLI for generating fractal feature diagnostic reports.

Usage:
    python scripts/run_fractal_diagnostics.py \\
        --symbols SPY QQQ AAPL NVDA \\
        --start 2022-01-01 \\
        --end 2025-12-31 \\
        --output reports/fractal/ \\
        --format html
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure the src package is on the path when run directly (outside of an
# installed environment).
_src = Path(__file__).resolve().parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_fractal_diagnostics",
        description="Generate fractal feature diagnostic reports (HTML + IC CSV).",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        metavar="SYMBOL",
        help="One or more ticker symbols (e.g., SPY QQQ AAPL).",
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        metavar="YYYY-MM-DD",
        help="Start date for historical data fetch (default: 2022-01-01).",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        metavar="YYYY-MM-DD",
        help="End date for historical data fetch (default: 2025-12-31).",
    )
    parser.add_argument(
        "--output",
        default="reports/fractal",
        metavar="DIR",
        help="Output directory for reports (default: reports/fractal).",
    )
    parser.add_argument(
        "--format",
        dest="fmt",
        choices=["html", "png"],
        default="html",
        help="Output format: html (interactive) or png (static).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    from ait.diagnostics.report import generate_report

    generate_report(
        symbols=args.symbols,
        start=args.start,
        end=args.end,
        output_dir=args.output,
        fmt=args.fmt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
