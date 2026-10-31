"""ONE switch for whether the research engine may substitute for a missing input.

W116. The research engine and the walk-forward used to fill every hole themselves: a VIX-proxy or realized-vol IV
when no IV was stored, a hard-coded calendar when no calendar was stored, the legacy percentage friction when no
per-leg model was fitted, empty VIX / SPY frames when the context would not load, a silent daily-only run when the
intraday bars were short. With ``strict_inputs`` on (the default, and the only mode a study can run in) each of
those is a refusal that names the gap (``ResearchDataUnavailable``) or, where the policy declares a modelled
behaviour, a counted entry in the run record.

The module default exists for ONE reason: the test suite builds engines on synthetic frames that carry no stored IV,
no calendar and no friction model, and one autouse fixture in ``tests/conftest.py`` turns the default off for those.
There is deliberately no environment variable and no config key for it: a switch an operator can flip is a way to
mix sources again. A class guard (``tests/test_w116_strict_inputs.py``) pins that nothing outside the conftest sets it.
"""

from __future__ import annotations

STRICT_INPUTS_DEFAULT = True


def resolve_strict(flag: "bool | None") -> bool:
    """An explicit flag wins; ``None`` takes the module default."""
    return STRICT_INPUTS_DEFAULT if flag is None else bool(flag)
