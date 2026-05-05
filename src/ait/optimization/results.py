"""OptimizationResult — wraps an Optuna study for reporting and config export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import optuna

from ait.utils.logging import get_logger

log = get_logger("optimization.results")


@dataclass
class OptimizationResult:
    """Thin wrapper around a completed Optuna study."""

    study: optuna.Study

    @property
    def best_params(self) -> dict:
        return self.study.best_params

    @property
    def best_value(self) -> float:
        return self.study.best_value

    @property
    def best_metrics(self) -> dict:
        trial = self.study.best_trial
        return {
            "value":       trial.value,
            "params":      trial.params,
            "trial_number": trial.number,
            "n_trials":    len(self.study.trials),
        }

    def summary(self, top_n: int = 5) -> str:
        """Return a formatted table of the top-N trials."""
        completed = [
            t for t in self.study.trials
            if t.state == optuna.trial.TrialState.COMPLETE
        ]
        completed.sort(key=lambda t: t.value or float("-inf"), reverse=True)
        top = completed[:top_n]

        lines = [
            "=" * 70,
            f"  OPTUNA OPTIMIZATION RESULTS  (study: {self.study.study_name})",
            "=" * 70,
            f"  Total trials:   {len(self.study.trials)}",
            f"  Best value:     {self.best_value:.4f}",
            f"  Best trial #:   {self.study.best_trial.number}",
            "-" * 70,
            f"  TOP {top_n} TRIALS:",
            f"  {'#':>5s}  {'Value':>8s}  Params",
            f"  {'---':>5s}  {'-----':>8s}  ------",
        ]
        for t in top:
            param_str = ", ".join(f"{k}={v}" for k, v in t.params.items())
            lines.append(f"  {t.number:5d}  {t.value:8.4f}  {param_str}")
        lines += [
            "-" * 70,
            "  BEST PARAMS:",
        ]
        for k, v in self.best_params.items():
            lines.append(f"    {k:30s} = {v}")
        lines.append("=" * 70)
        return "\n".join(lines)

    def apply_to_config(self, config_path: str = "config.yaml") -> None:
        """Write best params into config.yaml under strategy_overrides."""
        import yaml

        path = Path(config_path)
        data: dict = {}
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}

        data.setdefault("strategy_overrides", {})
        data["strategy_overrides"].update(self.best_params)

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)

        log.info("config_updated_with_best_params", path=config_path, params=self.best_params)

    def save(self, path: str = "reports/optimization_result.json") -> None:
        """Persist best params and metrics to a JSON file."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "study_name":   self.study.study_name,
            "best_value":   self.best_value,
            "best_params":  self.best_params,
            "best_metrics": self.best_metrics,
            "n_trials":     len(self.study.trials),
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("optimization_result_saved", path=str(out_path))
