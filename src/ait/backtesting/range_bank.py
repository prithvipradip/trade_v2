"""W78: per-horizon range models for ONE (window, symbol).

Why this exists. The research range gate used to be a single model trained for a single
horizon (``options.dte_range[0] - EXPIRY_APPROACHING_DTE`` bars), handed unchanged to every
Optuna trial. Once the entry DTE is a searched dimension (W79) a 3-DTE trial and a 30-DTE
trial must not share one model: the question "does price stay inside +/-X% over the life of
THIS trade" is different for each.

The bank is a lazily-filled cache ``{horizon: (predictor, status, threshold)}``:

* ``seed`` installs a model that was already trained (the walk-forward's base model, so the
  default configuration trains exactly what it always did);
* ``get`` trains any other horizon once, on demand, and remembers the outcome (a failed
  training is cached too: a trial must not retrain a model that cannot be trained);
* training runs under a save/restore of the global numpy / ``random`` state, so a model
  trained in the middle of a study cannot perturb whatever consumes that state afterwards.

It knows nothing about walk-forward, Optuna or the engine: the owners inject ``train_fn``
(horizon -> result tuple) and ``bucket_fn`` (entry DTE -> horizon), so it is exercised
directly in tests.
"""
from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np

from ait.utils.logging import get_logger

log = get_logger("backtesting.range_bank")

# (predictor | None, status, threshold_pct): the tuple _train_window_range_model returns.
RangeResult = tuple[Any, str, float]
_FALLBACK_THRESHOLD = 0.05


class RangeModelBank:
    def __init__(
        self,
        train_fn: Callable[[int], RangeResult | None],
        bucket_fn: Callable[[int | None], int],
    ) -> None:
        self._train_fn = train_fn
        self._bucket_fn = bucket_fn
        self._cache: dict[int, RangeResult] = {}

    # -- population -----------------------------------------------------------------------
    def seed(self, horizon: int, result: RangeResult) -> None:
        """Install an already-trained (or already-failed) model for ``horizon``."""
        self._cache[int(horizon)] = result

    def get(self, horizon: int) -> RangeResult:
        """The model for ``horizon``; trained once on first use."""
        h = int(horizon)
        if h in self._cache:
            return self._cache[h]
        np_state, py_state = np.random.get_state(), random.getstate()
        try:
            res = self._train_fn(h)
        except Exception as exc:  # noqa: BLE001 - one bad horizon must not kill a study
            log.warning("range_bank_train_failed", horizon=h, error=str(exc))
            res = (None, f"train_exception:{type(exc).__name__}", _FALLBACK_THRESHOLD)
        finally:
            np.random.set_state(np_state)
            random.setstate(py_state)
        if res is None or not isinstance(res, tuple) or len(res) != 3:
            res = (None, "skipped", _FALLBACK_THRESHOLD)
        self._cache[h] = res
        return res

    # -- lookup by entry DTE --------------------------------------------------------------
    def horizon_for(self, entry_dte: int | None = None) -> int:
        return int(self._bucket_fn(entry_dte))

    def for_entry_dte(self, entry_dte: int | None = None) -> RangeResult:
        return self.get(self.horizon_for(entry_dte))

    def predictor_for(self, entry_dte: int | None = None) -> Any:
        """The optimizer's per-trial hook: just the predictor (None when training failed)."""
        return self.for_entry_dte(entry_dte)[0]

    # -- introspection (tests, study) --------------------------------------------------------
    @property
    def trained_horizons(self) -> tuple[int, ...]:
        return tuple(sorted(self._cache))
