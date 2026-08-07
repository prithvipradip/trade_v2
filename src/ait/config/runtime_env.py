"""Runtime environment defaults shared by EVERY bot entry point.

R16: these protective/economic defaults lived only in run_orchestrator.py —
a direct `python -m ait.main` launch silently ran different economics with
macro protections OFF (no AIT_SKIP_MACRO_EVENTS, k=1.0 wings, 0.20 floor,
undefined-risk gate open). Both entry points now call
apply_runtime_env_defaults() before any trading subsystem imports.

IMPORTANT: this module must stay import-light (no numpy/xgboost/pandas) —
the KMP/OMP guards below must be set before any OpenMP-bundling library
loads anywhere in the process.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def apply_runtime_env_defaults() -> None:
    """setdefault the protective/economic env contract, then load .env.

    setdefault semantics: an explicitly exported env var (or a scheduled
    task's environment) always wins; these are the safety net for bare
    launches. .env loads LAST and fills only still-unset keys.
    """
    # OpenMP conflict guard — MUST precede numpy/xgboost/lightgbm import
    # (c0000005 crash cluster, 2026-06-26).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    # Market data: delayed-frozen until U6 funds live entitlements
    # (2026-07-28 user decision; flip to "1" after funding + resubscribe).
    os.environ.setdefault("AIT_MARKET_DATA_TYPE", "4")

    # 2026-08-04 SHADOW R3 PROMOTION (pre-registered rule, PLAN): wide-wing
    # condor — k=1.6 wings with the ratio floor scaled 0.20 -> 0.10 so the
    # ABSOLUTE credit demand per structure is unchanged.
    os.environ.setdefault("AIT_IC_WING_K", "1.6")
    os.environ.setdefault("AIT_IC_MIN_CREDIT_WIDTH", "0.10")

    # R16: undefined-risk executor gate CLOSED. The =1 justification (paper
    # strangle edge-comparison) has been stale since the 07-22 IC-only
    # pivot; nothing live may emit an undefined-risk order. (INST-5.)
    os.environ.setdefault("AIT_ALLOW_UNDEFINED_RISK", "0")

    # Macro-event protection ON (2026-07-08 user decision). NOTE 2026-08-04:
    # defined-risk condors are EXEMPT from the rule-3d flatten in code;
    # this flag now governs undefined/assignment-risk strategies + the
    # engine parity mirror.
    os.environ.setdefault("AIT_SKIP_MACRO_EVENTS", "1")

    _load_dotenv(REPO_ROOT / ".env")


def _load_dotenv(env_file: Path) -> None:
    """Minimal .env loader (no dependency): fills only UNSET keys."""
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip()
        if val and val[0] in {'"', "'"}:
            quote = val[0]
            end = val.find(quote, 1)
            val = val[1:end] if end != -1 else val[1:]
        else:
            val = val.split("#", 1)[0].rstrip()
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = val
