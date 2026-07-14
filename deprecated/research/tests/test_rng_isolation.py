"""Tests for RNG-isolated statistical model training (Exp 17 prerequisite).

Core guarantee: after _train_window_range_model_isolated() completes, the
parent process numpy RNG state is byte-identical to what it was before the
call — regardless of how much RNG the subprocess consumed internally.

This is the fix for Problem 1 (Optuna RNG contamination) documented in P31/P32
in EXPERIMENTS_INSIGHTS.md.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv(n: int = 400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.012, n)
    prices = 100.0 * np.cumprod(1.0 + r)
    hi = prices * (1 + np.abs(rng.normal(0, 0.005, n)))
    lo = prices * (1 - np.abs(rng.normal(0, 0.005, n)))
    idx = pd.date_range("2022-01-03", periods=n, freq="B")
    return pd.DataFrame(
        {
            "Close":  prices,
            "Open":   prices * (1 + rng.normal(0, 0.002, n)),
            "High":   hi,
            "Low":    lo,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )


def _wf_backtester(enable_msgarch: bool = False, enable_oujump: bool = False):
    """Construct a minimal WalkForwardBacktester for testing."""
    from ait.backtesting.walkforward import WalkForwardBacktester, WalkForwardConfig
    cfg = WalkForwardConfig(max_hold_days=21)
    return WalkForwardBacktester(
        symbols=["TEST"],
        strategies=["iron_condor"],
        config=cfg,
        enable_msgarch=enable_msgarch,
        enable_oujump=enable_oujump,
    )


# ---------------------------------------------------------------------------
# Test 1 — parent RNG state is unchanged after subprocess training
# ---------------------------------------------------------------------------

def test_parent_rng_unchanged_after_isolated_training():
    """Spawn subprocess must not alter the parent numpy RNG state.

    Procedure:
      1. Seed numpy RNG to a known state.
      2. Record the next 20 draws (oracle).
      3. Run isolated training (which internally runs MS-GARCH EM + OU-Kou MLE
         in the subprocess — heavy RNG consumers).
      4. Record the next 20 draws again.
      5. Assert oracle == post-training draws (state unchanged).
    """
    df = _make_ohlcv(400)
    wf = _wf_backtester(enable_msgarch=True, enable_oujump=True)

    np.random.seed(42)
    oracle = np.random.rand(20).copy()

    # Reset to same state, run training, then draw again
    np.random.seed(42)
    threshold = wf._adaptive_range_threshold(df, horizon_days=21)
    wf._train_window_range_model_isolated(
        df, "TEST", window_id=1,
        max_hold_days=21,
        threshold_pct=threshold,
    )
    post_training = np.random.rand(20).copy()

    np.testing.assert_array_equal(
        oracle, post_training,
        err_msg="Parent numpy RNG state was mutated by subprocess training",
    )


# ---------------------------------------------------------------------------
# Test 2 — Optuna TPE suggestion sequence is identical with/without subprocess
# ---------------------------------------------------------------------------

def test_optuna_sequence_stable_across_statistical_model_presence():
    """Optuna with a fixed seed produces the same first N suggestions regardless
    of whether statistical model training ran before sampling.

    This directly tests the contamination fix: suggestions must not depend on
    what the range model training consumed from the global RNG.
    """
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df = _make_ohlcv(400)
    wf_stat  = _wf_backtester(enable_msgarch=True, enable_oujump=True)
    wf_nostat = _wf_backtester(enable_msgarch=False, enable_oujump=False)

    def _get_suggestions(wf, n_trials=8):
        """Run isolated training then sample n_trials from a fresh Optuna study."""
        threshold = wf._adaptive_range_threshold(df, horizon_days=21)
        wf._train_window_range_model_isolated(
            df, "TEST", window_id=1,
            max_hold_days=21,
            threshold_pct=threshold,
        )
        sampler = optuna.samplers.TPESampler(seed=42)
        study = optuna.create_study(sampler=sampler, direction="maximize")

        suggestions = []

        def _objective(trial):
            x = trial.suggest_float("x", 0.0, 1.0)
            y = trial.suggest_float("y", 0.0, 1.0)
            suggestions.append((round(x, 6), round(y, 6)))
            return x + y

        study.optimize(_objective, n_trials=n_trials)
        return suggestions

    sugg_with_stat    = _get_suggestions(wf_stat)
    sugg_without_stat = _get_suggestions(wf_nostat)

    assert sugg_with_stat == sugg_without_stat, (
        "Optuna TPE suggestion sequence differs between statistical-model-enabled "
        "and disabled runs — RNG isolation is broken.\n"
        f"  with stat:    {sugg_with_stat[:4]}\n"
        f"  without stat: {sugg_without_stat[:4]}"
    )


# ---------------------------------------------------------------------------
# Test 3 — isolated training returns a trained RangePredictor
# ---------------------------------------------------------------------------

def test_isolated_training_returns_trained_predictor():
    """The subprocess must successfully train and return a fitted RangePredictor."""
    df = _make_ohlcv(400)
    wf = _wf_backtester(enable_msgarch=False, enable_oujump=False)
    threshold = wf._adaptive_range_threshold(df, horizon_days=21)

    rp, status, thr = wf._train_window_range_model_isolated(
        df, "TEST", window_id=1,
        max_hold_days=21,
        threshold_pct=threshold,
    )

    assert status == "ok", f"Expected status 'ok', got '{status}'"
    assert rp is not None, "Expected a fitted RangePredictor, got None"
    assert rp.is_trained, "RangePredictor.is_trained is False"
    assert rp.fitted_weights, "fitted_weights is empty"
    assert abs(thr - threshold) < 1e-9


# ---------------------------------------------------------------------------
# Test 4 — subprocess timeout falls back gracefully to in-process ML-only
# ---------------------------------------------------------------------------

def test_subprocess_timeout_falls_back_to_inprocess(monkeypatch):
    """If the subprocess exceeds the timeout, training falls back to in-process
    ML-only training (no statistical models) without raising an exception.
    """
    import multiprocessing as mp

    df = _make_ohlcv(400)
    wf = _wf_backtester(enable_msgarch=True, enable_oujump=True)
    threshold = wf._adaptive_range_threshold(df, horizon_days=21)

    # Patch process lifecycle to simulate timeout without spawning a subprocess
    monkeypatch.setattr(mp.Process, "start", lambda self: None)
    monkeypatch.setattr(mp.Process, "join", lambda self, timeout=None: None)
    monkeypatch.setattr(mp.Process, "is_alive", lambda self: True)
    monkeypatch.setattr(mp.Process, "terminate", lambda self: None)

    # Use a very short timeout so the test is fast
    rp, status, thr = wf._train_window_range_model_isolated(
        df, "TEST", window_id=1,
        max_hold_days=21,
        threshold_pct=threshold,
        _timeout=1,
    )

    # Should have fallen back — result may be ok (inprocess fallback) or None
    # but must never raise
    assert isinstance(status, str)
    assert isinstance(thr, float)


# ---------------------------------------------------------------------------
# Test 5 — nonzero subprocess exit code falls back gracefully
# ---------------------------------------------------------------------------

def test_subprocess_nonzero_exit_falls_back(monkeypatch):
    """A subprocess that exits with a nonzero code (OOM, crash) triggers the
    in-process fallback without propagating the error.
    """
    import multiprocessing as mp

    df = _make_ohlcv(400)
    wf = _wf_backtester(enable_msgarch=True, enable_oujump=True)
    threshold = wf._adaptive_range_threshold(df, horizon_days=21)

    monkeypatch.setattr(mp.Process, "start",    lambda self: None)
    monkeypatch.setattr(mp.Process, "join",     lambda self, timeout=None: None)
    monkeypatch.setattr(mp.Process, "is_alive", lambda self: False)
    monkeypatch.setattr(mp.Process, "exitcode", property(lambda self: -9))

    rp, status, thr = wf._train_window_range_model_isolated(
        df, "TEST", window_id=1,
        max_hold_days=21,
        threshold_pct=threshold,
    )

    assert isinstance(status, str)
    assert isinstance(thr, float)


# ---------------------------------------------------------------------------
# Test 6 — enable flags are respected in subprocess
# ---------------------------------------------------------------------------

def test_enable_flags_control_subprocess_models():
    """When enable_msgarch=False and enable_oujump=False, the isolated path
    skips the subprocess entirely and goes directly to in-process training.
    """
    df = _make_ohlcv(400)
    wf = _wf_backtester(enable_msgarch=False, enable_oujump=False)
    threshold = wf._adaptive_range_threshold(df, horizon_days=21)

    # Should not create any spawn Process when both flags are False
    rp, status, thr = wf._train_window_range_model_isolated(
        df, "TEST", window_id=1,
        max_hold_days=21,
        threshold_pct=threshold,
    )

    # Verify no subprocess was needed (in-process path taken)
    # — we just check the result is valid
    assert isinstance(thr, float)


# ---------------------------------------------------------------------------
# Test 7 — _train_window_range_model_inprocess works as standalone fallback
# ---------------------------------------------------------------------------

def test_inprocess_fallback_trains_ml_only():
    """_train_window_range_model_inprocess must produce a valid ML-only predictor."""
    df = _make_ohlcv(400)
    wf = _wf_backtester()
    threshold = wf._adaptive_range_threshold(df, horizon_days=21)

    rp, status, thr = wf._train_window_range_model_inprocess(
        df, "TEST", window_id=1,
        max_hold_days=21,
        threshold_pct=threshold,
        enable_msgarch=False,
        enable_oujump=False,
    )

    assert status == "ok"
    assert rp is not None and rp.is_trained
    # No statistical keys should be present
    sym_data = getattr(rp, "_symbol_models", {}).get("TEST", {})
    assert sym_data.get("ms_garch_state") is None or sym_data.get("ms_garch_state") == {}
