"""W106 item 5: the IV-rank parity probe names the table set it compares.

It hardcoded `prefix="research_"`; after the W91 archive those tables are
`archive_w91_research_*`, so a bare run read an empty frame and failed deep in
pandas. It now takes `--research-prefix` (default the archive) and refuses
clearly when the table is not there.
"""
from __future__ import annotations

from tests.test_w71_iv_rank_parity import _load_probe, _make_db


def test_refuses_clearly_when_the_research_tables_are_gone(tmp_path, capsys):
    db = tmp_path / "h.db"
    _make_db(db)                                    # has research_daily_prices, no archive set
    assert _load_probe().main(["--db", str(db), "--symbols", "TST"]) == 2
    assert "archive_w91_research_daily_prices is not in" in capsys.readouterr().err


def test_the_prefix_selects_the_table_set(tmp_path, capsys):
    db = tmp_path / "h.db"
    _make_db(db)
    rc = _load_probe().main(["--db", str(db), "--symbols", "TST", "--since", "2023-09-01",
                             "--research-prefix", "research_"])
    assert rc == 0 and "| TST | full history |" in capsys.readouterr().out
