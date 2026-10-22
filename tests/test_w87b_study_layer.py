"""W87b: the study layer, the run record, and the tier rule.

  * ``studies/<name>.yaml`` loads and is validated (unknown key, secrets, env-contract keys, search-space errors)
  * it is a THIRD layer: study > research overlay > base config, applied to this process only
  * the run record holds what decides a result and never a secret; MLflow is optional and never fails a run
  * every committed study's ``rule:`` is ``exploratory`` or an existing PLAN.md heading
  * the entry points take ``--study-file``; pool workers receive the parent's settings (W88a)
"""
from __future__ import annotations

import ast
import json
import shutil
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from ait.backtesting import run_record as rr
from ait.config import research as rs
from ait.config.settings import Settings, load_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _write(tmp_path: Path, body: str, name: str = "study.yaml") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


GOOD = """
name: w88_wing_k_wider
question: Does a wider wing_k range change the wings Optuna picks?
rule: exploratory
settings:
  backtest:
    max_concurrent_positions: 2
search_space:
  iron_condor:
    wing_k: [1.0, 2.0]
"""


# --------------------------------------------------------------------------
# the study file
# --------------------------------------------------------------------------

class TestLoadStudy:
    def test_a_good_file_loads(self, tmp_path):
        s = rs.load_study(_write(tmp_path, GOOD))
        assert (s.name, s.exploratory) == ("w88_wing_k_wider", True)
        assert s.settings == {"backtest": {"max_concurrent_positions": 2}}
        assert s.search_space == {"iron_condor": {"wing_k": [1.0, 2.0]}}
        assert s.path.name == "study.yaml"

    def test_the_settings_and_search_space_are_optional(self, tmp_path):
        s = rs.load_study(_write(tmp_path, "name: a\nquestion: q\nrule: exploratory\n"))
        assert s.settings == {} and s.search_space == {}

    def test_a_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="study file not found"):
            rs.load_study(tmp_path / "nope.yaml")

    @pytest.mark.parametrize("body, match", [
        ("[1, 2]", "YAML mapping"),
        (GOOD + "extra: 1\n", "unknown key"),
        ("question: q\nrule: exploratory\n", "'name' is required"),
        ("name: a\nrule: exploratory\n", "'question' is required"),
        ("name: a\nquestion: q\n", "'rule' is required"),
        ("name: a\nquestion: '  '\nrule: exploratory\n", "'question' is required"),
        ("name: Bad-Name\nquestion: q\nrule: exploratory\n", "name must be lowercase"),
        ("name: a\nquestion: q\nrule: exploratory\nsettings: [1]\n", "'settings' must be a mapping"),
        ("name: a\nquestion: q\nrule: exploratory\nsettings: {ibkr: {host: x}}\n", "may not set 'ibkr'"),
        ("name: a\nquestion: q\nrule: exploratory\nsettings: {api_keys: {finnhub_api_key: x}}\n", "api_keys"),
        ("name: a\nquestion: q\nrule: exploratory\nsearch_space: [1]\n", "'search_space' must map"),
        ("name: a\nquestion: q\nrule: exploratory\nsearch_space: {iron_condor: {wing_k: [3, 1]}}\n",
         r"search_space\.iron_condor\.wing_k"),
        ("name: a\nquestion: q\nrule: exploratory\nsearch_space: {iron_condor: {entry_dte: [1, 5]}}\n",
         "dte_search"),
    ])
    def test_bad_files_fail_loudly(self, tmp_path, body, match):
        with pytest.raises(ValueError, match=match):
            rs.load_study(_write(tmp_path, body))

    @pytest.mark.parametrize("section, key, env", [("backtest", "wing_k", "AIT_IC_WING_K"),
                                                   ("backtest", "ic_min_credit", "AIT_IC_MIN_CREDIT"),
                                                   ("risk", "iv_rank_cap", "AIT_IV_RANK_CAP")])
    def test_env_contract_keys_are_refused_with_the_variable_named(self, tmp_path, section, key, env):
        body = f"name: a\nquestion: q\nrule: exploratory\nsettings: {{{section}: {{{key}: 1.0}}}}\n"
        with pytest.raises(ValueError, match=env):
            rs.load_study(_write(tmp_path, body))

    def test_the_committed_template_loads_and_is_exploratory(self):
        s = rs.load_study(ROOT / "studies" / "_template.yaml")
        assert s.exploratory and s.settings == {} and s.search_space == {}


class TestTierRule:
    PLAN = "# Plan\n\n## 2026-11-01 - W88 AS PLANNED: a pre-registered study\n\ntext mentioning W88 AS PLANNED\n"

    def test_exploratory_needs_nothing(self):
        assert rs.rule_is_registered("exploratory", "")

    def test_a_named_heading_is_found(self):
        assert rs.rule_is_registered("2026-11-01 - W88 AS PLANNED", self.PLAN)

    def test_a_body_mention_is_not_a_heading(self):
        assert not rs.rule_is_registered("text mentioning W88 AS PLANNED", self.PLAN)

    def test_a_missing_heading_fails(self):
        assert not rs.rule_is_registered("2026-12-01 - W99 AS PLANNED", self.PLAN)

    def test_every_committed_study_names_a_real_rule(self):
        plan = (ROOT / "PLAN.md").read_text()
        files = sorted((ROOT / "studies").glob("*.yaml"))
        assert files, "studies/ must hold at least the template"
        for f in files:
            s = rs.load_study(f)
            assert rs.rule_is_registered(s.rule, plan), (
                f"{f.name}: rule {s.rule!r} is neither 'exploratory' nor an existing PLAN.md heading")


# --------------------------------------------------------------------------
# three layers
# --------------------------------------------------------------------------

@pytest.fixture
def cfg_dir(tmp_path):
    """A private copy of config.yaml + config_research.yaml, so layering can be tested without touching the repo."""
    shutil.copy(ROOT / "config.yaml", tmp_path / "config.yaml")
    shutil.copy(ROOT / "config_research.yaml", tmp_path / "config_research.yaml")
    return tmp_path


class TestThreeLayers:
    def test_without_a_study_nothing_changes(self, cfg_dir):
        a = rs.load_research_settings(cfg_dir / "config.yaml")
        b = rs.load_research_settings(cfg_dir / "config.yaml", study=None)
        assert a.model_dump() == b.model_dump()

    def test_the_study_wins_over_the_overlay_which_wins_over_the_base(self, cfg_dir):
        base = load_settings(cfg_dir / "config.yaml", ignore_frozen=True)
        merged = rs.load_research_settings(cfg_dir / "config.yaml")
        assert base.backtest.gates_mode == "observe" and merged.backtest.gates_mode == "enforce"   # overlay > base
        study = _write(cfg_dir, "name: a\nquestion: q\nrule: exploratory\n"
                                "settings: {backtest: {gates_mode: observe, max_concurrent_positions: 2}}\n")
        s = rs.load_research_settings(cfg_dir / "config.yaml", study=study)
        assert s.backtest.gates_mode == "observe"                      # study > overlay
        assert s.backtest.max_concurrent_positions == 2                # study > base (3)
        assert s.backtest.friction_model == merged.backtest.friction_model   # everything else untouched

    def test_a_study_cannot_smuggle_an_unknown_key(self, cfg_dir):
        study = _write(cfg_dir, "name: a\nquestion: q\nrule: exploratory\nsettings: {backtest: {nonsense: 1}}\n")
        with pytest.raises(ValueError):
            rs.load_research_settings(cfg_dir / "config.yaml", study=study)

    def test_use_research_config_freezes_the_three_layers(self, cfg_dir):
        study = _write(cfg_dir, "name: a\nquestion: q\nrule: exploratory\n"
                                "settings: {backtest: {max_concurrent_positions: 2}}\n")
        rs.use_research_config(cfg_dir / "config.yaml", study=study)
        assert load_settings().backtest.max_concurrent_positions == 2        # the frozen view
        from ait.backtesting.walkforward import WalkForwardConfig
        assert WalkForwardConfig(train_window_models=False).max_concurrent_positions == 2


# --------------------------------------------------------------------------
# the run record
# --------------------------------------------------------------------------

def _db(tmp_path: Path, closes=(100.0, 101.0, 102.0)) -> Path:
    db = tmp_path / "h.db"
    c = sqlite3.connect(db)
    c.execute("CREATE TABLE research_daily_prices (symbol TEXT, date TEXT, close REAL)")
    c.executemany("INSERT INTO research_daily_prices VALUES (?,?,?)",
                  [("QQQ", f"2024-01-0{i + 2}", v) for i, v in enumerate(closes)])
    c.execute("CREATE TABLE research_dividends (symbol TEXT, ex_date TEXT, amount REAL)")
    c.execute("INSERT INTO research_dividends VALUES ('QQQ','2024-03-18',0.5)")
    c.commit()
    c.close()
    return db


def _settings_with_secrets() -> Settings:
    s = load_settings(str(ROOT / "config.yaml"), ignore_frozen=True)
    d = s.model_dump(exclude={"ibkr", "api_keys"})
    return Settings(**d, ibkr={"account": "DU9998887", "host": "10.9.9.9"},
                    api_keys={"finnhub_api_key": "SECRET-FINNHUB-KEY"})


class TestRunRecord:
    def test_the_record_has_what_decides_a_result_and_no_secret(self, tmp_path):
        db = _db(tmp_path)
        study = tmp_path / "study.yaml"
        study.write_text(GOOD)
        s = rs.load_study(study)
        rec = rr.build_run_record(
            kind="optimize", name=s.name, settings=_settings_with_secrets(), config_path=ROOT / "config.yaml",
            overlay_path=ROOT / "config_research.yaml", study_path=study, db_path=db, symbols=["QQQ"],
            study={"name": s.name, "rule": s.rule, "settings": s.settings, "search_space": s.search_space},
            cli_args={"symbols": ["QQQ"], "n_trials": 5}, search_space={"iron_condor": {"wing_k": ["float", 1, 2]}},
            search_space_hash="abc123", seeds={"optuna": 42})
        blob = json.dumps(rec, default=str)
        for secret in ("DU9998887", "SECRET-FINNHUB-KEY", "10.9.9.9"):
            assert secret not in blob
        assert rec["schema"] == rr.SCHEMA and rec["run_id"].startswith("optimize_w88_wing_k_wider_")
        assert rec["files"]["config"]["sha256"] == rr._sha256((ROOT / "config.yaml").read_bytes())
        assert "gates_mode: enforce" in rec["files"]["overlay"]["text"] and rec["files"]["study"]["text"] == GOOD
        assert rec["settings"]["backtest"]["max_concurrent_positions"] == 3
        assert rec["seeds"] == {"ml_random_state": 42, "optuna": 42}
        assert rec["search_space_hash"] == "abc123" and rec["cli_args"]["n_trials"] == 5
        assert "AIT_IC_WING_K" in rec["env_contract"]

    def test_git_state(self, tmp_path):
        g = rr.git_state()
        assert len(g["commit"]) == 40 and isinstance(g["dirty"], bool) and g["branch"]
        nowhere = rr.git_state(tmp_path)
        assert nowhere == {"commit": "unknown", "branch": "unknown", "dirty": None, "dirty_files": None}

    def test_data_fingerprint_is_read_only_and_sensitive(self, tmp_path):
        db = _db(tmp_path)
        before = db.stat().st_mtime_ns
        a = rr.data_fingerprint(db, "research_", ["QQQ", "SPY"])
        assert a["closes"]["QQQ"]["rows"] == 3 and a["closes"]["SPY"]["rows"] == 0
        assert a["tables"]["research_daily_prices"] == {"rows": 3, "first": "2024-01-02", "last": "2024-01-04"}
        assert a["tables"]["research_dividends"]["first"] == "2024-03-18"
        assert db.stat().st_mtime_ns == before
        con = sqlite3.connect(db)
        assert {r[0] for r in con.execute("SELECT name FROM sqlite_master")} == {
            "research_daily_prices", "research_dividends"}                   # nothing was created
        con.close()
        other = tmp_path / "x"
        other.mkdir()
        b = rr.data_fingerprint(_db(other, (100.0, 101.0, 102.5)), "research_", ["QQQ"])
        assert a["closes"]["QQQ"]["sha256"] != b["closes"]["QQQ"]["sha256"]
        assert rr.data_fingerprint(tmp_path / "missing.db", "research_", ["QQQ"])["error"] == "database not found"

    def test_write_and_round_trip(self, tmp_path):
        rec = rr.build_run_record(kind="backtest", name="adhoc", settings=None, config_path=ROOT / "config.yaml",
                                  overlay_path=None, symbols=[])
        path = rr.write_run_record(rec, tmp_path / "run")
        assert json.loads(path.read_text())["run_id"] == rec["run_id"]
        assert (tmp_path / "run" / "config.yaml").read_text() == (ROOT / "config.yaml").read_text()
        assert not (tmp_path / "run" / "overlay.yaml").exists() and not (tmp_path / "run" / "study.yaml").exists()

    def test_mlflow_params_show_what_differs_not_the_whole_config(self):
        rec = {"git": {"commit": "c0ffee", "dirty": False}, "search_space_hash": "h",
               "study": {"name": "s", "rule": "exploratory", "settings": {"backtest": {"max_concurrent_positions": 1}},
                         "search_space": {"iron_condor": {"wing_k": [1.0, 2.0]}}},
               "cli_args": {"n_trials": 5}, "seeds": {"optuna": 42}, "symbols": ["QQQ"], "settings": {"huge": 1}}
        p = rr.mlflow_params(rec)
        assert p["settings.backtest.max_concurrent_positions"] == "1"
        assert p["search_space.iron_condor.wing_k"] == "[1.0, 2.0]"
        assert p["git_commit"] == "c0ffee" and p["cli.n_trials"] == "5" and p["seed.optuna"] == "42"
        assert not any(k.startswith("settings.huge") for k in p)

    @staticmethod
    def _fake_mlflow(existing=0):
        calls = SimpleNamespace(uri=None, exp=None, params=None, tags=None, metrics=None, artifacts=None, started=0)

        class Run:
            info = SimpleNamespace(run_id="mlflow-run-1")

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        m = SimpleNamespace(
            set_tracking_uri=lambda u: setattr(calls, "uri", u),
            set_experiment=lambda e: setattr(calls, "exp", e),
            search_runs=lambda **kw: [0] * existing,
            start_run=lambda run_name=None, tags=None: (setattr(calls, "tags", tags),
                                                       setattr(calls, "started", calls.started + 1), Run())[2],
            log_params=lambda p: setattr(calls, "params", p),
            log_metrics=lambda p: setattr(calls, "metrics", p),
            log_artifacts=lambda d: setattr(calls, "artifacts", d))
        return m, calls

    def _record(self):
        return {"run_id": "r1", "kind": "backtest", "name": "s", "git": {"commit": "c", "dirty": False},
                "study": {"name": "s", "rule": "exploratory"}, "metrics": {"sharpe": 1.5, "label": "x"}}

    def test_mlflow_logging(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///elsewhere.db")
        m, calls = self._fake_mlflow()
        assert rr.log_to_mlflow(self._record(), tmp_path, mlflow_module=m) == "mlflow-run-1"
        assert calls.uri == "sqlite:///elsewhere.db" and calls.exp == "research_s"
        assert calls.tags["run_id"] == "r1" and calls.metrics == {"sharpe": 1.5}      # non-numeric dropped
        assert calls.artifacts == str(tmp_path)

    def test_mlflow_default_uri_is_the_repo_database(self, monkeypatch):
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
        m, calls = self._fake_mlflow()
        rr.log_to_mlflow(self._record(), mlflow_module=m)
        assert calls.uri == f"sqlite:///{rr.REPO_ROOT / 'data' / 'mlflow.db'}"

    def test_mlflow_is_idempotent_on_the_run_id(self):
        m, calls = self._fake_mlflow(existing=1)
        assert rr.log_to_mlflow(self._record(), mlflow_module=m) is None and calls.started == 0

    def test_mlflow_absent_or_failing_never_raises(self, monkeypatch):
        monkeypatch.setattr(rr.importlib, "import_module", lambda n: (_ for _ in ()).throw(ImportError(n)))
        assert rr.log_to_mlflow(self._record()) is None
        boom = SimpleNamespace(set_tracking_uri=lambda u: (_ for _ in ()).throw(RuntimeError("down")))
        assert rr.log_to_mlflow(self._record(), mlflow_module=boom) is None


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

class TestGuards:
    def test_the_modules_stay_out_of_the_live_tree(self):
        """Only research code may import run_record / study_harness / iv_model's research users."""
        offenders = []
        for p in (ROOT / "src" / "ait").rglob("*.py"):
            rel = p.relative_to(ROOT / "src" / "ait").as_posix()
            if rel.startswith(("backtesting/", "optimization/", "config/")):
                continue
            for n in ast.walk(ast.parse(p.read_text())):
                mods = ([a.name for a in n.names] if isinstance(n, ast.Import)
                        else [n.module or ""] if isinstance(n, ast.ImportFrom) else [])
                if any(("run_record" in m or "study_harness" in m) for m in mods):
                    offenders.append(f"{rel}:{n.lineno}")
        assert offenders == []

    def test_run_record_names_no_overlay_loader(self):
        text = (ROOT / "src" / "ait" / "backtesting" / "run_record.py").read_text()
        for banned in ("config_research", "use_research_config", "ait.config.research"):
            assert banned not in text

    @pytest.mark.parametrize("script", ["run_backtest.py", "run_optimizer.py"])
    def test_the_entry_points_take_a_study_file(self, script, monkeypatch):
        tree = ast.parse((ROOT / script).read_text())
        flags = {a.value for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "attr", "") == "add_argument"
                 for a in n.args if isinstance(a, ast.Constant)}
        assert "--study-file" in flags
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                 and getattr(n.func, "id", getattr(n.func, "attr", "")) == "use_research_config"]
        assert calls and all(any(k.arg == "study" for k in c.keywords) for c in calls)


# --------------------------------------------------------------------------
# the entry points apply a study (executed paths, not source strings)
# --------------------------------------------------------------------------

class TestAssertRegistered:
    def _study(self, rule):
        return rs.Study(name="s", question="q", rule=rule)

    def test_exploratory_always_passes(self, tmp_path):
        rs.assert_registered(self._study("exploratory"), tmp_path / "missing.md")

    def test_a_registered_heading_passes_and_an_unknown_one_fails(self, tmp_path):
        plan = tmp_path / "PLAN.md"
        plan.write_text("# Plan\n\n## 2026-10-30 - W90 AS PLANNED (wing study)\n\nbody mentions W91 only here\n")
        rs.assert_registered(self._study("W90 AS PLANNED"), plan)
        with pytest.raises(ValueError, match="neither 'exploratory' nor a heading"):
            rs.assert_registered(self._study("W91 AS PLANNED"), plan)

    def test_a_missing_plan_registers_nothing(self, tmp_path):
        with pytest.raises(ValueError):
            rs.assert_registered(self._study("W90 AS PLANNED"), tmp_path / "nope.md")


class TestFinishRun:
    def test_it_writes_the_record_and_logs_once(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(rr, "log_to_mlflow", lambda record, run_dir=None, **k: calls.append(
            (record["run_id"], Path(run_dir).name)) or "mlflow-id")
        run_dir, mid = rr.finish_run(
            runs_dir=tmp_path / "runs", kind="backtest", name="qqq_adhoc", settings=load_settings(),
            config_path=ROOT / "config.yaml", overlay_path=ROOT / "config_research.yaml",
            symbols=["QQQ"], extra={"parity_manifest": {"live": {}, "backtest": {}}},
            metrics={"total_return": 0.1})
        assert mid == "mlflow-id" and run_dir.parent == tmp_path / "runs"
        rec = json.loads((run_dir / "run_record.json").read_text())
        assert rec["parity_manifest"] == {"live": {}, "backtest": {}} and rec["metrics"] == {"total_return": 0.1}
        assert (run_dir / "config.yaml").is_file() and (run_dir / "overlay.yaml").is_file()
        assert not (run_dir / "study.yaml").exists()
        assert calls == [(rec["run_id"], run_dir.name)]

    def test_mlflow_can_be_switched_off(self, tmp_path, monkeypatch):
        monkeypatch.setattr(rr, "log_to_mlflow", lambda *a, **k: pytest.fail("MLflow must not be called"))
        _d, mid = rr.finish_run(runs_dir=tmp_path, mlflow=False, kind="backtest", name="x", settings=None,
                                config_path=ROOT / "config.yaml", overlay_path=None)
        assert mid is None


@pytest.fixture
def study_file(tmp_path):
    p = tmp_path / "w88.yaml"
    p.write_text(GOOD)
    return p


class _Stop(Exception):
    pass


class TestRunBacktestAppliesTheStudy:
    def _run(self, monkeypatch, argv):
        import asyncio

        import run_backtest as rb
        seen = {}
        orig = rs.use_research_config

        def spy_use(*a, **k):
            seen["study"] = k.get("study")
            return orig(*a, **k)

        class FakeWF:
            def __init__(self, *a, **k):
                seen["cfg"] = k["config"]
                raise _Stop

        monkeypatch.setattr(rs, "use_research_config", spy_use)
        monkeypatch.setattr(rb, "WalkForwardBacktester", FakeWF)
        monkeypatch.setattr(rb, "build_parity_manifest", lambda args: {"parity_warnings": [], "live": {}, "backtest": {}})
        monkeypatch.setattr(sys, "argv", ["run_backtest.py", *argv])
        with pytest.raises(_Stop):
            asyncio.run(rb.run_backtest(rb.parse_args()))
        return seen

    def test_the_study_reaches_the_config_the_walk_forward_gets(self, monkeypatch, study_file):
        seen = self._run(monkeypatch, ["--study-file", str(study_file)])
        assert seen["study"].name == "w88_wing_k_wider"
        assert seen["cfg"].search_space_overrides == {"iron_condor": {"wing_k": [1.0, 2.0]}}

    def test_without_a_study_nothing_is_overridden(self, monkeypatch):
        seen = self._run(monkeypatch, [])
        assert seen["study"] is None and seen["cfg"].search_space_overrides is None

    def test_an_unregistered_rule_is_refused_before_anything_runs(self, monkeypatch, tmp_path):
        p = tmp_path / "s.yaml"
        p.write_text(GOOD.replace("rule: exploratory", "rule: W999 AS PLANNED (no such heading)"))
        with pytest.raises(ValueError, match="neither 'exploratory' nor a heading"):
            self._run(monkeypatch, ["--study-file", str(p)])


class TestRunOptimizerAppliesTheStudy:
    def _main(self, monkeypatch, argv):
        import ait.optimization.optimizer as opt_mod
        import run_optimizer as ro
        seen = {}

        class Spy:
            def __init__(self, **kw):
                seen.update(kw)
                raise _Stop

        monkeypatch.setattr(opt_mod, "StrategyOptimizer", Spy)
        monkeypatch.setattr(sys, "argv", ["run_optimizer.py", "--n-trials", "1", *argv])
        with pytest.raises(_Stop):
            ro.main()
        return seen

    def test_the_search_space_override_reaches_the_optimizer(self, monkeypatch, study_file):
        seen = self._main(monkeypatch, ["--study-file", str(study_file)])
        assert seen["search_space_overrides"] == {"iron_condor": {"wing_k": [1.0, 2.0]}}

    def test_no_study_no_override(self, monkeypatch):
        assert self._main(monkeypatch, [])["search_space_overrides"] is None

    def test_apply_is_refused_with_a_study(self, monkeypatch, study_file):
        import run_optimizer as ro
        monkeypatch.setattr(sys, "argv", ["run_optimizer.py", "--study-file", str(study_file), "--apply"])
        with pytest.raises(SystemExit, match="REFUSING --apply"):
            ro.main()


class TestPoolWorkersKeepTheResearchLayers:
    """W88a: spawned walk-forward workers used to rebuild settings from plain config.yaml."""

    def test_the_initializer_makes_the_parents_settings_the_workers(self):
        from ait.backtesting.walkforward import _init_worker_settings
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update={"max_concurrent_positions": 7})})
        _init_worker_settings(moved)
        assert load_settings().backtest.max_concurrent_positions == 7
        _init_worker_settings(None)                                    # nothing to freeze: a no-op
        assert load_settings().backtest.max_concurrent_positions == 7

    def test_a_real_spawned_worker_sees_an_overlay_only_value(self):
        """The executed path: a spawn pool built the way the walk-forward builds it."""
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        from ait.backtesting.walkforward import _init_worker_settings
        s = load_settings()
        moved = s.model_copy(update={"backtest": s.backtest.model_copy(update={"max_concurrent_positions": 7})})
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn"),
                                 initializer=_init_worker_settings, initargs=(moved,)) as ex:
            got = ex.submit(load_settings).result(timeout=120)
        assert got.backtest.max_concurrent_positions == 7
        with ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:   # the old behaviour
            assert ex.submit(load_settings).result(timeout=120).backtest.max_concurrent_positions != 7

    def test_the_walk_forward_builds_its_pool_with_the_initializer(self, monkeypatch):
        """Run the parallel branch with a recording pool class and check how it is built."""
        import asyncio

        import concurrent.futures as cf

        from ait.backtesting import walkforward as wf
        built = {}

        class RecordingPool:
            def __init__(self, *a, **kw):
                built.update(kw)
                raise RuntimeError("stop")

        monkeypatch.setattr(cf, "ProcessPoolExecutor", RecordingPool)
        monkeypatch.setattr(wf.WalkForwardBacktester, "_pretrain_range_models", lambda self, *a, **k: [{} for _ in range(99)])
        import numpy as np
        import pandas as pd
        idx = pd.date_range("2022-01-03", periods=900, freq="B")
        close = 400 * np.cumprod(1 + np.full(900, 0.0003))
        df = pd.DataFrame({"Open": close, "High": close, "Low": close, "Close": close,
                           "Volume": np.full(900, 5e6)}, index=idx)
        cfg = wf.WalkForwardConfig(optimize_n_jobs=2, train_window_models=False, history_start_date=None)
        bt = wf.WalkForwardBacktester(symbols=["QQQ"], strategies=["iron_condor"], config=cfg)
        with pytest.raises(RuntimeError, match="stop"):
            asyncio.run(bt.run(data={"QQQ": df}))
        assert built["initializer"] is wf._init_worker_settings and built["initargs"] == (bt._settings,)
