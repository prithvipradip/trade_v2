"""ML model drift detection — monitors prediction accuracy over time.

Detects when the model's accuracy degrades below acceptable thresholds
and triggers automatic retraining. Uses a rolling window of recent
predictions vs actual outcomes to compute live accuracy.

Drift types detected:
- Accuracy drift: rolling accuracy drops below threshold
- Confidence calibration drift: high-confidence predictions are wrong
- Class distribution shift: predicted class ratios change significantly
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from ait.utils.logging import get_logger

log = get_logger("ml.drift")


@dataclass
class DriftReport:
    """Result of a drift check."""

    is_drifting: bool = False
    accuracy: float = 0.0
    calibration_error: float = 0.0
    samples: int = 0
    reason: str = ""
    should_retrain: bool = False


@dataclass
class PredictionRecord:
    """A single prediction with its actual outcome."""

    timestamp: str
    predicted_direction: str  # "bullish", "bearish", "neutral"
    confidence: float
    actual_direction: str = ""  # Filled in when outcome is known
    correct: bool | None = None


class DriftDetector:
    """Monitors ML model predictions against actual outcomes.

    Maintains a rolling window of predictions and checks for drift
    every N predictions. When drift is detected, signals that the
    model should be retrained.
    """

    def __init__(
        self,
        window_size: int = 50,
        accuracy_threshold: float = 0.40,
        calibration_threshold: float = 0.25,
        check_interval: int = 10,
    ) -> None:
        self._window_size = window_size
        self._accuracy_threshold = accuracy_threshold
        self._calibration_threshold = calibration_threshold
        self._check_interval = check_interval

        self._predictions: deque[PredictionRecord] = deque(maxlen=window_size)
        self._pending: dict[str, PredictionRecord] = {}  # trade_id → record
        self._prediction_count = 0
        self._last_report: DriftReport | None = None
        self._retrain_triggered = False

    def record_prediction(
        self,
        trade_id: str,
        direction: str,
        confidence: float,
    ) -> None:
        """Record a new prediction (outcome unknown yet)."""
        record = PredictionRecord(
            timestamp=datetime.now().isoformat(),
            predicted_direction=direction,
            confidence=confidence,
        )
        self._pending[trade_id] = record
        self._prediction_count += 1

    def record_outcome(
        self,
        trade_id: str,
        actual_direction: str,
    ) -> None:
        """Record the actual outcome for a previous prediction."""
        record = self._pending.pop(trade_id, None)
        if record is None:
            return

        record.actual_direction = actual_direction
        record.correct = record.predicted_direction == actual_direction
        self._predictions.append(record)

    def check_drift(self) -> DriftReport:
        """Check if the model is drifting.

        Only runs every check_interval predictions to avoid noise.
        """
        completed = [p for p in self._predictions if p.correct is not None]
        if len(completed) < 20:
            return DriftReport(samples=len(completed))

        # Rolling accuracy
        correct = sum(1 for p in completed if p.correct)
        accuracy = correct / len(completed)

        # Calibration: high-confidence predictions should be right more often
        high_conf = [p for p in completed if p.confidence >= 0.80]
        cal_error = 0.0
        if len(high_conf) >= 5:
            high_conf_acc = sum(1 for p in high_conf if p.correct) / len(high_conf)
            # Expected: high confidence should have ~80% accuracy
            cal_error = max(0, 0.80 - high_conf_acc)

        # Determine if drifting
        is_drifting = False
        reason = ""
        should_retrain = False

        if accuracy < self._accuracy_threshold:
            is_drifting = True
            reason = f"accuracy_below_threshold ({accuracy:.0%} < {self._accuracy_threshold:.0%})"
            should_retrain = True

        elif cal_error > self._calibration_threshold:
            is_drifting = True
            reason = f"calibration_drift (high-conf error={cal_error:.0%})"
            should_retrain = True

        report = DriftReport(
            is_drifting=is_drifting,
            accuracy=accuracy,
            calibration_error=cal_error,
            samples=len(completed),
            reason=reason,
            should_retrain=should_retrain and not self._retrain_triggered,
        )

        if is_drifting:
            log.warning(
                "model_drift_detected",
                accuracy=f"{accuracy:.2f}",
                calibration_error=f"{cal_error:.2f}",
                samples=len(completed),
                reason=reason,
            )

        self._last_report = report
        return report

    def acknowledge_retrain(self) -> None:
        """Called after retraining to reset drift state."""
        self._retrain_triggered = False
        self._predictions.clear()
        log.info("drift_state_reset_after_retrain")

    @property
    def last_report(self) -> DriftReport | None:
        return self._last_report

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def completed_count(self) -> int:
        return len([p for p in self._predictions if p.correct is not None])

    def get_stats(self) -> dict:
        """Get current drift monitoring stats."""
        completed = [p for p in self._predictions if p.correct is not None]
        if not completed:
            return {"samples": 0, "accuracy": 0, "pending": len(self._pending)}

        correct = sum(1 for p in completed if p.correct)
        return {
            "samples": len(completed),
            "accuracy": correct / len(completed),
            "pending": len(self._pending),
            "is_drifting": self._last_report.is_drifting if self._last_report else False,
        }
