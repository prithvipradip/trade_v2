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
    """Load .env, then setdefault the protective/economic env contract.

    Precedence (highest wins): an explicitly exported env var (or a
    scheduled task's environment) always wins, since it's already set
    before this function runs either step. Then .env. Then these hardcoded
    defaults, which exist purely as the safety net for a fully bare launch
    with no .env and no exported vars at all.

    R17: .env used to load LAST (after the setdefault calls below), so it
    could never override any of these 7 vars — including
    AIT_MARKET_DATA_TYPE, the delayed-vs-live data switch the code comment
    two lines down says is meant to be flipped via config. .env is the only
    documented user-facing config mechanism in this repo; it must actually
    take effect. Loading it first is safe: _load_dotenv only touches
    os/pathlib (no numpy/xgboost/pandas import), so it doesn't disturb the
    KMP/OMP-guards-before-heavy-import constraint below — those still land
    (via setdefault) before apply_runtime_env_defaults() returns, which is
    what both entry points require.
    """
    _load_dotenv(REPO_ROOT / ".env")

    # R19: driven by CONTRACT_DEFAULTS so the applier and every reader share
    # ONE table. The OpenMP guards must still land before numpy/xgboost import
    # anywhere — they are first in the dict and this runs before any heavy
    # import in both entry points.
    for _key, _val in CONTRACT_DEFAULTS.items():
        os.environ.setdefault(_key, _val)


# R19 (user audit: "have we hardcoded any config in code?"). ONE authority for
# every value in the env contract. Before this, 17 reader sites carried their
# OWN fallback literals and FOUR disagreed with the contract:
#   AIT_IC_WING_K            contract 1.6  -> readers defaulted 1.0  (x2)
#   AIT_IC_MIN_CREDIT_WIDTH  contract 0.10 -> readers defaulted 0.20 (x4)
#   AIT_SKIP_MACRO_EVENTS    contract "1"  -> readers defaulted "0"  (x3, FAIL-OPEN)
#   AIT_CREDIT_LOSS_LIMIT    absent        -> readers split 0 vs 1.25
# Live was protected only because both entry points call
# apply_runtime_env_defaults() first; ANY path that skips it (a test, a script,
# a notebook, a future entry point) silently ran pre-promotion economics with
# macro protection OFF. That is the wing_k four-disagreeing-sources incident
# recurring one layer down, so the fallback now lives in exactly one place:
# a promotion edits CONTRACT_DEFAULTS and nothing else.
CONTRACT_DEFAULTS: dict[str, str] = {
    # OpenMP guards MUST stay first — they have to land before numpy/xgboost
    # import anywhere in the process (c0000005 crash cluster, 2026-06-26).
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "AIT_MARKET_DATA_TYPE": "4",
    "AIT_IC_WING_K": "1.6",
    "AIT_IC_MIN_CREDIT_WIDTH": "0.10",
    "AIT_IC_MIN_CREDIT": "0.70",
    # R6/R16 evidence: flat credit stops underperform touch-close, so live
    # runs DISABLED. Absent from the contract until R19, which is how
    # portfolio.py (0) and the orchestrator coherence check (1.25) came to
    # reason about two different stops.
    "AIT_CREDIT_LOSS_LIMIT": "0",
    "AIT_ALLOW_UNDEFINED_RISK": "0",
    "AIT_SKIP_MACRO_EVENTS": "1",
}


def contract_str(key: str) -> str:
    """Env value for a contract key, falling back to the ONE declared default."""
    if key not in CONTRACT_DEFAULTS:
        raise KeyError(f"{key} is not part of the runtime env contract")
    return os.environ.get(key, CONTRACT_DEFAULTS[key])


def contract_float(key: str) -> float:
    """Numeric contract value. A malformed override falls back to the declared
    default rather than crashing a trading cycle."""
    try:
        return float(contract_str(key))
    except (TypeError, ValueError):
        return float(CONTRACT_DEFAULTS[key])


def contract_flag(key: str) -> bool:
    """Boolean contract value ('1' = on). Protective flags therefore default to
    their PROTECTIVE state at every reader, never to off."""
    return contract_str(key) == "1"


def capital_base(default: float = 196_000.0) -> float:
    """R16: single authority for the equity base used in return/DD percentages.

    The value was hardcoded '196000' in three files (dashboard, analytics,
    referee) and had ALREADY drifted from the live NLV (~198.1k), so every
    percentage the operator reads was computed off a stale denominator — and
    at go-live, off a base ~65x too large. Reads AIT_CAPITAL_BASE, else the
    live account snapshot cached by the bot, else the documented default.
    """
    return _capital_base_with_source(default)[0]


def capital_base_source(default: float = 196_000.0) -> str:
    """R17: which tier `capital_base()` actually resolved from.

    De-duplicating the three call sites (R16) didn't add any enforcement —
    if AIT_CAPITAL_BASE is unset AND the bot has never cached a live NLV
    (e.g. fresh go-live), capital_base() still silently returns the
    hardcoded default with no warning. Callers that care whether that
    happened in a LIVE (non-paper) context should check this.
    """
    return _capital_base_with_source(default)[1]


def _capital_base_with_source(default: float) -> tuple[float, str]:
    import os as _os
    raw = _os.environ.get("AIT_CAPITAL_BASE")
    if raw:
        try:
            return float(raw), "env"
        except (TypeError, ValueError):
            pass
    try:  # live NLV cached by the account manager (bot writes it each cycle)
        import sqlite3
        con = sqlite3.connect(
            f"file:{REPO_ROOT / 'data' / 'ait_state.db'}?mode=ro", uri=True)
        row = con.execute(
            "SELECT value FROM bot_state WHERE key='last_net_liquidation'"
        ).fetchone()
        con.close()
        if row and float(row[0]) > 0:
            return float(row[0]), "live_nlv"
    except Exception:  # noqa: BLE001
        pass
    return default, "default"


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
