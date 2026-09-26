from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import EvaluationRun, InferenceTelemetry, ModelArtifact
from app.schemas.common import VisionResult


def register_model(
    db: Session,
    *,
    name: str,
    version: str,
    kind: str,
    artifact_path: str = "",
    status: str = "candidate",
    dataset_version: str = "unversioned",
    prompt_version: str = "not_applicable",
    knowledge_base_version: str = "not_applicable",
    calibration_version: str = "not_applicable",
    metrics: dict[str, Any] | None = None,
    tags: dict[str, Any] | None = None,
) -> ModelArtifact:
    artifact = db.scalar(select(ModelArtifact).where(ModelArtifact.name == name, ModelArtifact.version == version))
    if artifact is None:
        artifact = ModelArtifact(name=name, version=version)
        db.add(artifact)
    artifact.kind = kind
    artifact.artifact_path = artifact_path
    artifact.status = status
    artifact.dataset_version = dataset_version
    artifact.prompt_version = prompt_version
    artifact.knowledge_base_version = knowledge_base_version
    artifact.calibration_version = calibration_version
    artifact.metrics_json = metrics or {}
    artifact.tags_json = tags or {}
    db.commit()
    db.refresh(artifact)
    return artifact


def record_evaluation(
    db: Session,
    *,
    task_type: str,
    split: str,
    model_name: str,
    model_version: str,
    dataset_version: str,
    metrics: dict[str, Any],
) -> EvaluationRun:
    if split != "test":
        raise ValueError("evaluation runs must be recorded against an explicit independent test split")
    run = EvaluationRun(
        task_type=task_type,
        split=split,
        model_name=model_name,
        model_version=model_version,
        dataset_version=dataset_version,
        metrics_json=metrics,
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run



def current_gpu_memory_mb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        return round(float(torch.cuda.memory_allocated()) / (1024 * 1024), 3)
    except Exception:
        return None


def record_vision_telemetry(
    db: Session,
    *,
    case_id: str | None,
    study_id: str | None,
    vision: VisionResult,
    status: str = "completed",
    gpu_memory_mb: float | None = None,
) -> InferenceTelemetry:
    confidences = [finding.confidence for finding in vision.findings]
    confidence_summary = {
        "count": len(confidences),
        "accepted": sum(finding.confidence_status == "accepted" for finding in vision.findings),
        "uncertain": sum(finding.confidence_status == "uncertain" for finding in vision.findings),
        "rejected": len(vision.rejected_findings),
        "mean": sum(confidences) / len(confidences) if confidences else 0.0,
        "abstained": vision.abstained,
        "simulated": vision.simulated,
        "provider": vision.provider,
        "calibration_version": vision.calibration_version,
    }
    telemetry = InferenceTelemetry(
        case_id=case_id,
        study_id=study_id,
        model_name=vision.model_name,
        model_version=vision.model_version,
        task_type=vision.task_type,
        status=status,
        latency_ms=float(vision.image_quality.get("inference_latency_ms", 0.0) or 0.0),
        gpu_memory_mb=gpu_memory_mb,
        quality_status=str(vision.image_quality.get("status", "not_evaluated")),
        confidence_summary=confidence_summary,
    )
    db.add(telemetry)
    return telemetry


def monitoring_summary(db: Session) -> dict[str, Any]:
    total = db.scalar(select(func.count()).select_from(InferenceTelemetry)) or 0
    failed = db.scalar(
        select(func.count()).select_from(InferenceTelemetry).where(InferenceTelemetry.status != "completed")
    ) or 0
    rows = list(db.scalars(select(InferenceTelemetry).order_by(InferenceTelemetry.created_at.desc()).limit(1000)))
    latencies = [float(row.latency_ms or 0.0) for row in rows]
    quality_counts = Counter(row.quality_status for row in rows)
    return {
        "inference_count": total,
        "failure_rate": failed / total if total else 0.0,
        "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "p95_latency_ms": _percentile(latencies, 0.95),
        "quality_status_counts": dict(quality_counts),
        "recent_window": len(rows),
    }


def distribution_psi(reference: Iterable[float], current: Iterable[float], bins: int = 10) -> float:
    reference = list(reference)
    current = list(current)
    if not reference or not current:
        return 0.0
    edges = [index / bins for index in range(bins + 1)]
    def histogram(values: list[float]) -> list[float]:
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, max(0, int(float(value) * bins)))
            counts[index] += 1
        return [(count + 1e-6) / (len(values) + bins * 1e-6) for count in counts]
    expected = histogram(reference)
    actual = histogram(current)
    return sum((now - old) * math.log(now / old) for old, now in zip(expected, actual))


def compare_shadow_results(primary: VisionResult, shadow: VisionResult) -> dict[str, Any]:
    primary_names = {finding.name for finding in primary.findings if finding.confidence_status == "accepted"}
    shadow_names = {finding.name for finding in shadow.findings if finding.confidence_status == "accepted"}
    union = primary_names | shadow_names
    intersection = primary_names & shadow_names
    return {
        "primary_model": primary.model_name,
        "shadow_model": shadow.model_name,
        "accepted_finding_agreement": len(intersection) / len(union) if union else 1.0,
        "primary_only": sorted(primary_names - shadow_names),
        "shadow_only": sorted(shadow_names - primary_names),
        "risk_level_match": primary.risk_level == shadow.risk_level,
        "abstention_match": primary.abstained == shadow.abstained,
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * percentile))))
    return ordered[position]
