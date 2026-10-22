"""W87b: the per-run record of a research run (what produced this number?).

Since W65 nothing has saved what a research run used: ``run_backtest.py`` printed a parity manifest and wrote
nothing, the study scripts printed tables, and PLAN.md carried the provenance as prose. A result without its
inputs cannot be compared with the next study's. This module builds ONE record per run and writes it where the
run's output lives; it can also log it to MLflow when MLflow is installed.

The record holds everything that decides a result and is not in the code itself:

* the resolved settings (secrets excluded) and the RAW text of ``config.yaml``, the research overlay and the
  study file (with sha256), because the overlay holds only differences and ``config.yaml`` moves over time;
* the effective env-contract values (``AIT_*``), which are read from the raw ``config.yaml`` and from the
  environment and are NOT reached by either overlay;
* the git commit and whether the tree had uncommitted edits;
* data fingerprints (rows and date span per research table, and a hash of each symbol's close series) read
  through a READ-ONLY connection (``data/historical.db`` is 600 MB and changes daily, so it is never hashed whole);
* seeds, the command line, and the merged search space with its hash.

This module names no overlay or study loader: its callers pass the paths, so it can live under ``src/ait``
without breaking the W68 guard. It does IO by design (research tooling, never imported by the live bot).
MLflow is optional: a missing module or a failed call returns None and never fails a run.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from ait.config.invariants_research import research_invariant_values   # leaf; names no overlay

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA = "ait.run_record/1"

# Research table -> its date column (for the span in the fingerprint); None = count only.
_DATE_COLUMNS: dict[str, str | None] = {
    "daily_prices": "date", "daily_iv": "date", "econ_schedule": "event_date", "dividends": "ex_date",
    "earnings_schedule": "event_session", "qqq_holdings": "filed_date", "econ_events": "event_date",
    "option_leg_spread_params": None, "option_leg_spread_band_params": None,
}


def git_state(repo: Path = REPO_ROOT) -> dict:
    """{commit, branch, dirty, dirty_files}; ``"unknown"`` / None when git is unavailable."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=15)
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip() if out.returncode == 0 else None
    status = run("status", "--porcelain", "--untracked-files=no")
    return {"commit": run("rev-parse", "HEAD") or "unknown",
            "branch": run("rev-parse", "--abbrev-ref", "HEAD") or "unknown",
            "dirty": None if status is None else bool(status),
            "dirty_files": None if status is None else len([ln for ln in status.splitlines() if ln.strip()])}


def env_contract() -> dict[str, str]:
    """The effective ``AIT_*`` env-contract values a research process would use (env > raw config.yaml >
    contract default). Never includes secrets: the contract holds trading knobs only."""
    from ait.config.runtime_env import CONTRACT_DEFAULTS, contract_str
    return {k: str(contract_str(k)) for k in sorted(CONTRACT_DEFAULTS)}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_entry(path: str | Path | None) -> dict | None:
    """{path, sha256, text} of a text file, or None when there is no such file."""
    if path is None:
        return None
    p = Path(path)
    if not p.is_file():
        return None
    raw = p.read_bytes()
    return {"path": str(p), "sha256": _sha256(raw), "text": raw.decode("utf-8", "replace")}


def _scrub(settings_dump: dict) -> dict:
    """The settings dump without secrets: the env-only sections and any secret-looking field."""
    from ait.config.settings import _SECRET_SECTIONS, _is_secret
    out: dict = {}
    for section, body in settings_dump.items():
        if section in _SECRET_SECTIONS:
            continue
        if isinstance(body, dict):
            out[section] = {k: v for k, v in body.items() if not _is_secret(section, k)}
        else:
            out[section] = body
    return out


def data_fingerprint(db_path: str | Path, table_prefix: str, symbols: list[str]) -> dict:
    """Rows and date span per table, and a sha256 of each symbol's (date, close) series, through a
    read-only connection. Missing tables are reported, not raised."""
    db = Path(db_path)
    out: dict[str, Any] = {"db": str(db), "tables": {}, "closes": {}}
    if not db.is_file():
        out["error"] = "database not found"
        return out
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        for base, col in _DATE_COLUMNS.items():
            for prefix in sorted({table_prefix, ""}):
                name = f"{prefix}{base}"
                try:
                    rows = con.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                    span = (con.execute(f"SELECT MIN({col}), MAX({col}) FROM {name}").fetchone()
                            if col else (None, None))
                except sqlite3.Error:
                    continue
                out["tables"][name] = {"rows": rows, "first": span[0], "last": span[1]}
        for sym in symbols:
            try:
                rows = con.execute(f"SELECT date, close FROM {table_prefix}daily_prices WHERE symbol=? "
                                   f"ORDER BY date", (sym,)).fetchall()
            except sqlite3.Error:
                rows = []
            h = hashlib.sha256("\n".join(f"{d}|{c}" for d, c in rows).encode()).hexdigest()
            out["closes"][sym] = {"rows": len(rows), "sha256": h}
    finally:
        con.close()
    return out


def _feature_spec_hash(settings) -> str:
    from ait.ml.feature_spec import spec_from_config
    return spec_from_config(getattr(getattr(settings, "ml", None), "features", None)).spec_hash()


def seed_inventory(extra: dict | None = None, settings=None) -> dict:
    """The seeds that decide a research result. The ML estimators' seed is ``ml.hyperparams.random_state`` of the
    settings the run used (W88c; the schema default without settings); the Optuna seed is whatever the caller
    used (``extra``)."""
    from ait.ml.hyperparams import TREE_BASE
    seed = getattr(getattr(getattr(settings, "ml", None), "hyperparams", None), "random_state", TREE_BASE["random_state"])
    return {"ml_random_state": seed, **(extra or {})}


def build_run_record(*, kind: str, name: str, settings, config_path: str | Path,
                     overlay_path: str | Path | None, study_path: str | Path | None = None,
                     study: dict | None = None, cli_args: dict | None = None,
                     symbols: list[str] | None = None, db_path: str | Path | None = None,
                     search_space: dict | None = None, search_space_hash: str | None = None,
                     seeds: dict | None = None, metrics: dict | None = None,
                     settings_origin: dict | None = None,
                     now: datetime | None = None) -> dict:
    """Assemble the record. ``settings`` is the RESOLVED settings object the run used."""
    now = now or datetime.now()
    dump = settings.model_dump(mode="json") if settings is not None else {}
    prefix = getattr(getattr(settings, "backtest", None), "research_table_prefix", "research_")
    run_id = f"{kind}_{name}_{now.strftime('%Y%m%d_%H%M%S')}"
    return {
        "schema": SCHEMA, "run_id": run_id, "kind": kind, "name": name,
        "created": now.isoformat(timespec="seconds"),
        "study": study, "git": git_state(), "cli_args": cli_args or {},
        "symbols": list(symbols or []),
        "settings": _scrub(dump),
        "files": {"config": file_entry(config_path), "overlay": file_entry(overlay_path),
                  "study": file_entry(study_path)},
        "env_contract": env_contract(),
        "research_invariants": research_invariant_values(),     # W88a: fixed in code, not configurable
        # W88d: hash of the indicator windows this run computed features with (recorded in every model it trained)
        "feature_spec_hash": _feature_spec_hash(settings),
        # W88a: {dotted.key: [value, layer]} -- which layer decides each value (study > overlay > config.yaml > default)
        "settings_origin": ({k: list(v) for k, v in settings_origin.items()} if settings_origin else None),
        "data": data_fingerprint(db_path, prefix, list(symbols or [])) if db_path else None,
        "seeds": seed_inventory(seeds, settings),
        "search_space": search_space, "search_space_hash": search_space_hash,
        "metrics": metrics or {},
    }


def write_run_record(record: dict, run_dir: str | Path) -> Path:
    """``run_record.json`` plus the raw ``config.yaml`` / overlay / study text beside it."""
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    for key, entry in record["files"].items():
        if entry:
            (d / f"{key}.yaml").write_text(entry["text"])
    path = d / "run_record.json"
    path.write_text(json.dumps(record, indent=2, default=str))
    return path


def finish_run(*, runs_dir: str | Path = "reports/runs", mlflow: bool = True, extra: dict | None = None,
               **record_kwargs) -> tuple[Path | None, str | None]:
    """Build, write and (optionally) MLflow-log one run's record; returns (run_dir, mlflow_run_id).

    The entry points call this once, at the end of a run, inside their own try/except: a record that cannot be
    written is reported, never allowed to fail the research run that produced the number. ``extra`` keys are
    added to the record as given (the parity manifest, window summaries, ...)."""
    record = build_run_record(**record_kwargs)
    record.update(extra or {})
    run_dir = Path(runs_dir) / record["run_id"]
    write_run_record(record, run_dir)
    return run_dir, (log_to_mlflow(record, run_dir) if mlflow else None)


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    """Nested dict -> {"a.b.c": "value"} with values as short strings (MLflow params)."""
    out: dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = json.dumps(v, default=str)[:400]
    return out


def mlflow_params(record: dict) -> dict[str, str]:
    """What MLflow's compare view should show: what DIFFERS between studies, not the full 400-key settings
    (those stay in the ``run_record.json`` artifact): the study's settings and search-space overrides, the
    CLI arguments, the hashes, the commit, the seeds."""
    study = record.get("study") or {}
    p: dict[str, str] = {
        "git_commit": record["git"]["commit"], "git_dirty": str(record["git"]["dirty"]),
        "search_space_hash": str(record.get("search_space_hash")),
        "study_name": str(study.get("name", "adhoc")), "study_rule": str(study.get("rule", "")),
    }
    p.update({f"settings.{k}": v for k, v in flatten(study.get("settings") or {}).items()})
    p.update({f"search_space.{k}": v for k, v in flatten(study.get("search_space") or {}).items()})
    p.update({f"cli.{k}": v for k, v in flatten(record.get("cli_args") or {}).items()})
    p.update({f"seed.{k}": v for k, v in flatten(record.get("seeds") or {}).items()})
    p["symbols"] = ",".join(record.get("symbols") or [])
    return p


def log_to_mlflow(record: dict, run_dir: str | Path | None = None, *, experiment: str | None = None,
                  mlflow_module=None) -> str | None:
    """Log the record to MLflow when it is installed; returns the MLflow run id, or None (not installed, a
    duplicate ``run_id`` tag, or any error). The tracking URI is ``MLFLOW_TRACKING_URI`` or
    ``sqlite:///<repo>/data/mlflow.db``. A failure here never fails a run."""
    try:
        mlflow = mlflow_module or importlib.import_module("mlflow")
    except ImportError:
        return None
    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{REPO_ROOT / 'data' / 'mlflow.db'}"
        mlflow.set_tracking_uri(uri)
        exp = experiment or f"research_{(record.get('study') or {}).get('name') or 'adhoc'}"
        mlflow.set_experiment(exp)
        found = mlflow.search_runs(experiment_names=[exp],
                                   filter_string=f"tags.run_id = '{record['run_id']}'")
        if len(found):
            return None
        tags = {"run_id": record["run_id"], "kind": record["kind"], "name": record["name"],
                "git_commit": record["git"]["commit"], "rule": str((record.get("study") or {}).get("rule", ""))}
        with mlflow.start_run(run_name=record["run_id"], tags=tags) as run:
            mlflow.log_params(mlflow_params(record))
            numeric = {k: float(v) for k, v in (record.get("metrics") or {}).items()
                       if isinstance(v, (int, float)) and not isinstance(v, bool)}
            if numeric:
                mlflow.log_metrics(numeric)
            if run_dir and Path(run_dir).is_dir():
                mlflow.log_artifacts(str(run_dir))
            return str(run.info.run_id)
    except Exception:  # noqa: BLE001 - MLflow is optional tooling
        return None
