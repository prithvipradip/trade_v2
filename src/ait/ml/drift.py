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

from ait.config.settings import MLConfig
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
    at most once every ``check_interval`` newly recorded predictions. When drift is detected, signals that the
    model should be retrained.

    W92: ``enabled=False`` switches the whole detector off (it records nothing and always reports "no drift"); the live
    trainer passes ``ml.drift_detection_enabled`` (default OFF). The class default is on, so a bare ``DriftDetector()``
    behaves as before. ``check_interval`` used to be stored and never read; it now throttles the full check.
    """

    def __init__(
        self,
        window_size: int | None = None,
        accuracy_threshold: float | None = None,
        calibration_threshold: float | None = None,
        check_interval: int | None = None,
        high_confidence: float | None = None,
        *,
        enabled: bool = True,
        min_samples: int | None = None,
        min_high_confidence_samples: int | None = None,
    ) -> None:
        # W69: the four drift thresholds are ml.drift_* config. None -> the
        # MLConfig model defaults (50 / 0.40 / 0.25 / 0.80, the former
        # literals), so a bare DriftDetector() is unchanged. The trainer
        # passes the configured values explicitly.
        _d = MLConfig()
        window_size = _d.drift_window_size if window_size is None else window_size
        accuracy_threshold = (_d.drift_accuracy_threshold
                              if accuracy_threshold is None else accuracy_threshold)
        calibration_threshold = (_d.drift_calibration_threshold
                                 if calibration_threshold is None else calibration_threshold)
        self._high_confidence = (_d.drift_high_confidence
                                 if high_confidence is None else high_confidence)
        self._window_size = window_size
        self._accuracy_threshold = accuracy_threshold
        self._calibration_threshold = calibration_threshold
        # W92: ml.drift_check_interval / drift_min_samples / drift_min_high_confidence_samples (were 10 [unused] / 20 / 5)
        self._check_interval = _d.drift_check_interval if check_interval is None else check_interval
        self._min_samples = _d.drift_min_samples if min_samples is None else min_samples
        self._min_high_conf_samples = (_d.drift_min_high_confidence_samples
                                       if min_high_confidence_samples is None else min_high_confidence_samples)
        self._enabled = bool(enabled)
        self._evaluated_at: int | None = None     # the prediction count at the last full evaluation (the throttle)

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
        if not self._enabled:
            return
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
        """Record the actual outcome for a previous prediction (a disabled detector holds no pending predictions)."""
        record = self._pending.pop(trade_id, None)
        if record is None:
            return

        record.actual_direction = actual_direction
        record.correct = record.predicted_direction == actual_direction
        self._predictions.append(record)

    def check_drift(self) -> DriftReport:
        """Check if the model is drifting.

        Disabled -> an empty report (never drifting). Fewer than ``min_samples`` resolved predictions -> a report with no
        verdict. Otherwise the full check runs at most once per ``check_interval`` newly recorded predictions (W92: it was
        documented but never implemented); in between, the last report is returned.
        """
        if not self._enabled:
            return DriftReport()
        completed = [p for p in self._predictions if p.correct is not None]
        if len(completed) < self._min_samples:
            return DriftReport(samples=len(completed))
        if (self._last_report is not None and self._evaluated_at is not None
                and self._prediction_count - self._evaluated_at < self._check_interval):
            return self._last_report
        self._evaluated_at = self._prediction_count

        # Rolling accuracy
        correct = sum(1 for p in completed if p.correct)
        accuracy = correct / len(completed)

        # Calibration: high-confidence predictions should be right more often
        high_conf = [p for p in completed if p.confidence >= self._high_confidence]
        cal_error = 0.0
        if len(high_conf) >= self._min_high_conf_samples:
            high_conf_acc = sum(1 for p in high_conf if p.correct) / len(high_conf)
            # Expected: high confidence should be right about as often as the
            # cohort cutoff claims (default ~80%)
            cal_error = max(0, self._high_confidence - high_conf_acc)

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
        self._evaluated_at = None            # W92: the next check evaluates immediately, whatever the throttle says
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
