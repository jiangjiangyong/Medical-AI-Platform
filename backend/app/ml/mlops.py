from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from app.schemas.common import VisionResult
from app.services.modelops import compare_shadow_results, distribution_psi


def drift_report(
    reference: Iterable[float],
    current: Iterable[float],
    *,
    psi_threshold: float = 0.20,
) -> dict[str, Any]:
    reference_values = list(reference)
    current_values = list(current)
    if not reference_values or not current_values:
        return {
            "status": "blocked",
            "passed": False,
            "reason": "reference_and_current_samples_required",
            "reference_count": len(reference_values),
            "current_count": len(current_values),
            "psi": None,
        }
    psi = distribution_psi(reference_values, current_values)
    return {
        "status": "completed",
        "passed": psi <= float(psi_threshold),
        "reason": "within_threshold" if psi <= float(psi_threshold) else "drift_exceeded",
        "reference_count": len(reference_values),
        "current_count": len(current_values),
        "psi": round(psi, 6),
        "threshold": float(psi_threshold),
    }


def shadow_evaluation(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, row in enumerate(rows, start=1):
        try:
            primary = VisionResult.model_validate(row.get("primary"))
            shadow = VisionResult.model_validate(row.get("shadow"))
            comparisons.append(compare_shadow_results(primary, shadow))
        except Exception as exc:
            errors.append(f"row_{index}:{type(exc).__name__}")
    denominator = len(comparisons) or 1
    return {
        "status": "completed" if comparisons and not errors else "blocked",
        "sample_count": len(comparisons),
        "error_count": len(errors),
        "errors": errors,
        "accepted_finding_agreement": round(
            sum(float(item["accepted_finding_agreement"]) for item in comparisons)
            / denominator,
            6,
        ),
        "risk_level_match_rate": round(
            sum(bool(item["risk_level_match"]) for item in comparisons) / denominator,
            6,
        ),
        "abstention_match_rate": round(
            sum(bool(item["abstention_match"]) for item in comparisons) / denominator,
            6,
        ),
    }


def promotion_gate(
    candidate: Mapping[str, Any],
    *,
    minimum_metrics: Mapping[str, float] | None = None,
    baseline: Mapping[str, Any] | None = None,
    regression_limits: Mapping[str, float] | None = None,
    drift: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    for metric, threshold in (minimum_metrics or {}).items():
        value = candidate.get(metric)
        try:
            checks[f"minimum:{metric}"] = float(value) >= float(threshold)
        except (TypeError, ValueError):
            checks[f"minimum:{metric}"] = False
        if not checks[f"minimum:{metric}"]:
            reasons.append(f"minimum_not_met:{metric}")
    for metric, allowed_drop in (regression_limits or {}).items():
        if not baseline or metric not in baseline:
            checks[f"regression:{metric}"] = False
            reasons.append(f"baseline_missing:{metric}")
            continue
        try:
            checks[f"regression:{metric}"] = float(candidate.get(metric)) >= float(
                baseline[metric]
            ) * (1.0 - float(allowed_drop))
        except (TypeError, ValueError):
            checks[f"regression:{metric}"] = False
        if not checks[f"regression:{metric}"]:
            reasons.append(f"regression_exceeded:{metric}")
    if drift is not None:
        checks["drift_gate"] = bool(drift.get("passed", False))
        if not checks["drift_gate"]:
            reasons.append("drift_gate_failed")
    return {
        "passed": bool(checks) and all(checks.values()),
        "checks": checks,
        "reasons": reasons,
    }
