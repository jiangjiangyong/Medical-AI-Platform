from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable


def _clip_probability(value: float) -> float:
    return min(1.0 - 1e-7, max(1e-7, float(value)))


def _logit(value: float) -> float:
    value = _clip_probability(value)
    return math.log(value / (1.0 - value))


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


@dataclass(frozen=True)
class ConfidenceDecision:
    status: str
    confidence: float
    abstained: bool
    reason: str


@dataclass
class CalibrationBundle:
    """Serializable temperature-scaling and abstention policy."""

    method: str = "temperature_scaling"
    temperature: float = 1.0
    accept_threshold: float = 0.55
    uncertain_threshold: float = 0.35
    version: str = "uncalibrated"
    class_temperatures: dict[str, float] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path | None) -> "CalibrationBundle":
        if path is None or not path.exists():
            return cls()
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            method=str(payload.get("method", "temperature_scaling")),
            temperature=max(1e-3, float(payload.get("temperature", 1.0))),
            accept_threshold=float(payload.get("accept_threshold", 0.55)),
            uncertain_threshold=float(payload.get("uncertain_threshold", 0.35)),
            version=str(payload.get("version", path.stem)),
            class_temperatures={
                str(key): max(1e-3, float(value))
                for key, value in dict(payload.get("class_temperatures", {})).items()
            },
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def calibrate(self, probability: float, label: str | None = None) -> float:
        temperature = self.class_temperatures.get(label or "", self.temperature)
        return _sigmoid(_logit(probability) / max(1e-3, temperature))

    def decide(self, probability: float, label: str | None = None) -> ConfidenceDecision:
        calibrated = self.calibrate(probability, label)
        if calibrated >= self.accept_threshold:
            return ConfidenceDecision("accepted", calibrated, False, "above_accept_threshold")
        if calibrated >= self.uncertain_threshold:
            return ConfidenceDecision("uncertain", calibrated, False, "below_accept_threshold")
        return ConfidenceDecision("rejected", calibrated, True, "below_uncertain_threshold")

    @classmethod
    def fit(
        cls,
        probabilities: Iterable[float],
        labels: Iterable[int],
        *,
        version: str = "calibrated",
        accept_threshold: float = 0.55,
        uncertain_threshold: float = 0.35,
        steps: int = 400,
    ) -> "CalibrationBundle":
        values = [_clip_probability(item) for item in probabilities]
        targets = [1 if int(item) else 0 for item in labels]
        if len(values) != len(targets) or not values or len(set(targets)) < 2:
            return cls(
                version=version,
                accept_threshold=accept_threshold,
                uncertain_threshold=uncertain_threshold,
            )

        log_temperature = 0.0
        learning_rate = 0.03
        for _ in range(max(1, steps)):
            gradient = 0.0
            for probability, target in zip(values, targets):
                logit = _logit(probability)
                calibrated = _sigmoid(logit / math.exp(log_temperature))
                gradient += (calibrated - target) * (-logit / math.exp(log_temperature))
            gradient /= len(values)
            log_temperature = min(4.0, max(-4.0, log_temperature - learning_rate * gradient))
        return cls(
            temperature=math.exp(log_temperature),
            accept_threshold=accept_threshold,
            uncertain_threshold=uncertain_threshold,
            version=version,
        )


def expected_calibration_error(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    bins: int = 10,
) -> float:
    values = [_clip_probability(item) for item in probabilities]
    targets = [1 if int(item) else 0 for item in labels]
    if len(values) != len(targets) or not values:
        return 0.0
    error = 0.0
    for index in range(max(1, bins)):
        lower = index / bins
        upper = (index + 1) / bins
        selected = [
            position
            for position, probability in enumerate(values)
            if (probability >= lower and probability < upper)
            or (index == bins - 1 and lower <= probability <= upper)
        ]
        if not selected:
            continue
        confidence = sum(values[position] for position in selected) / len(selected)
        accuracy = sum(targets[position] for position in selected) / len(selected)
        error += len(selected) / len(values) * abs(confidence - accuracy)
    return error


def brier_score(probabilities: Iterable[float], labels: Iterable[int]) -> float:
    values = [_clip_probability(item) for item in probabilities]
    targets = [1 if int(item) else 0 for item in labels]
    if len(values) != len(targets) or not values:
        return 0.0
    return sum((probability - target) ** 2 for probability, target in zip(values, targets)) / len(values)


def calibration_summary(
    probabilities: Iterable[float],
    labels: Iterable[int],
    bundle: CalibrationBundle | None = None,
) -> dict[str, Any]:
    values = list(probabilities)
    targets = list(labels)
    bundle = bundle or CalibrationBundle()
    calibrated = [bundle.calibrate(item) for item in values]
    return {
        "method": bundle.method,
        "version": bundle.version,
        "temperature": bundle.temperature,
        "before": {
            "ece": expected_calibration_error(values, targets),
            "brier": brier_score(values, targets),
        },
        "after": {
            "ece": expected_calibration_error(calibrated, targets),
            "brier": brier_score(calibrated, targets),
        },
    }


def reliability_bins(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    bins: int = 10,
) -> list[dict[str, Any]]:
    """Return reliability-diagram points with empty bins kept explicit."""

    values = [_clip_probability(item) for item in probabilities]
    targets = [1 if int(item) else 0 for item in labels]
    if len(values) != len(targets):
        raise ValueError("probabilities and labels must have the same length")
    bin_count = max(1, int(bins))
    result: list[dict[str, Any]] = []
    for index in range(bin_count):
        lower = index / bin_count
        upper = (index + 1) / bin_count
        selected = [
            position
            for position, probability in enumerate(values)
            if (probability >= lower and probability < upper)
            or (index == bin_count - 1 and lower <= probability <= upper)
        ]
        count = len(selected)
        confidence = (
            sum(values[position] for position in selected) / count
            if count
            else None
        )
        accuracy = (
            sum(targets[position] for position in selected) / count
            if count
            else None
        )
        result.append(
            {
                "bin": index,
                "lower": round(lower, 6),
                "upper": round(upper, 6),
                "count": count,
                "mean_confidence": confidence,
                "empirical_accuracy": accuracy,
                "gap": abs(confidence - accuracy)
                if confidence is not None and accuracy is not None
                else None,
            }
        )
    return result


def positive_acceptance_curve(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """Measure precision/recall as positive events are retained above a threshold."""

    values = [_clip_probability(item) for item in probabilities]
    targets = [1 if int(item) else 0 for item in labels]
    if len(values) != len(targets):
        raise ValueError("probabilities and labels must have the same length")
    candidates = list(thresholds or (round(index / 100, 2) for index in range(5, 100, 5)))
    positive_total = sum(targets)
    result: list[dict[str, Any]] = []
    for threshold in candidates:
        threshold = float(threshold)
        retained = [index for index, value in enumerate(values) if value >= threshold]
        true_positive = sum(targets[index] for index in retained)
        false_positive = len(retained) - true_positive
        precision = true_positive / len(retained) if retained else 0.0
        recall = true_positive / positive_total if positive_total else None
        result.append(
            {
                "threshold": threshold,
                "retained_events": len(retained),
                "coverage": len(retained) / len(values) if values else 0.0,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "precision": precision,
                "recall": recall,
                "false_discovery_rate": 1.0 - precision if retained else None,
            }
        )
    return result


def selective_coverage_risk_curve(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    thresholds: Iterable[float] | None = None,
) -> list[dict[str, Any]]:
    """Measure selective classification risk when low-confidence cases abstain."""

    values = [_clip_probability(item) for item in probabilities]
    targets = [1 if int(item) else 0 for item in labels]
    if len(values) != len(targets):
        raise ValueError("probabilities and labels must have the same length")
    candidates = list(thresholds or (round(0.5 + index * 0.05, 2) for index in range(10)))
    result: list[dict[str, Any]] = []
    confidences = [max(value, 1.0 - value) for value in values]
    predictions = [int(value >= 0.5) for value in values]
    for threshold in candidates:
        threshold = float(threshold)
        retained = [
            index for index, confidence in enumerate(confidences)
            if confidence >= threshold
        ]
        errors = sum(predictions[index] != targets[index] for index in retained)
        result.append(
            {
                "confidence_threshold": threshold,
                "retained_samples": len(retained),
                "coverage": len(retained) / len(values) if values else 0.0,
                "abstain_rate": 1.0 - (len(retained) / len(values))
                if values
                else 0.0,
                "errors": errors,
                "risk": errors / len(retained) if retained else None,
            }
        )
    return result


def select_precision_threshold(
    probabilities: Iterable[float],
    labels: Iterable[int],
    *,
    target_precision: float,
    min_events: int = 1,
    default_threshold: float = 0.55,
    max_threshold: float | None = None,
) -> dict[str, Any]:
    """Select the lowest threshold meeting an engineering precision target."""

    if not 0.0 <= target_precision <= 1.0:
        raise ValueError("target_precision must be between 0 and 1")
    if min_events < 1:
        raise ValueError("min_events must be positive")
    curve = positive_acceptance_curve(probabilities, labels)
    candidates = [
        row for row in curve
        if row["retained_events"] >= min_events
        and row["precision"] >= target_precision
        and (
            max_threshold is None
            or row["threshold"] < float(max_threshold)
        )
    ]
    selected = min(candidates, key=lambda row: row["threshold"]) if candidates else None
    threshold = float(selected["threshold"] if selected else default_threshold)
    nearest = min(
        curve,
        key=lambda row: abs(row["threshold"] - threshold),
        default=None,
    )
    return {
        "threshold": threshold,
        "target_precision": target_precision,
        "min_events": min_events,
        "constraint_met": selected is not None,
        "selection": selected or nearest,
        "fallback": selected is None,
    }
