from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.config import settings
from app.db.session import get_db
from app.models import (
    AuditLog,
    EvaluationRun,
    InferenceTelemetry,
    ModelArtifact,
    User,
)
from app.schemas.common import (
    EvaluationRunCreate,
    EvaluationRunRead,
    ModelArtifactCreate,
    ModelArtifactRead,
)
from app.services.dicom import DICOMwebAdapter
from app.services.modelops import monitoring_summary, record_evaluation, register_model


router = APIRouter(prefix="/admin", tags=["admin-runtime"])


@router.get("/runtime")
def runtime_dependencies(user: User = Depends(require_roles("admin"))) -> dict:
    del user
    return {
        "vision_adapter": settings.vision_adapter,
        "vision_target": settings.vision_model_name,
        "dependencies": {
            "medgemma_vlm": {
                "configured": bool(settings.vision_vlm_base_url),
                "probe_attempted": False,
            },
            "dicomweb": {
                "configured": DICOMwebAdapter().configured,
                "probe_attempted": False,
            },
            "fhir": {
                "configured": bool(settings.fhir_base_url),
                "probe_attempted": False,
            },
            "report_llm": {
                "configured": bool(settings.deepseek_api_key),
                "probe_attempted": False,
            },
            "embedding": {
                "configured": bool(settings.embedding_api_key),
                "probe_attempted": False,
            },
        },
        "resilience": {
            "timeout_seconds": settings.service_timeout_seconds,
            "retry_attempts": settings.service_retry_attempts,
            "circuit_failure_threshold": settings.service_circuit_failure_threshold,
        },
    }


@router.get("/monitoring")
def monitoring(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)) -> dict:
    del user
    return monitoring_summary(db)


def _model_artifact_read(artifact: ModelArtifact) -> dict:
    return {
        "id": artifact.id,
        "name": artifact.name,
        "version": artifact.version,
        "kind": artifact.kind,
        "artifact_path": artifact.artifact_path,
        "status": artifact.status,
        "dataset_version": artifact.dataset_version,
        "prompt_version": artifact.prompt_version,
        "knowledge_base_version": artifact.knowledge_base_version,
        "calibration_version": artifact.calibration_version,
        "metrics": artifact.metrics_json or {},
        "tags": artifact.tags_json or {},
        "created_at": artifact.created_at,
    }


@router.get("/models", response_model=list[ModelArtifactRead])
def list_models(
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> list[dict]:
    del user
    artifacts = list(
        db.scalars(select(ModelArtifact).order_by(ModelArtifact.created_at.desc()))
    )
    return [_model_artifact_read(item) for item in artifacts]


@router.post("/models", response_model=ModelArtifactRead)
def create_model(
    payload: ModelArtifactCreate,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict:
    artifact = register_model(
        db,
        name=payload.name,
        version=payload.version,
        kind=payload.kind,
        artifact_path=payload.artifact_path,
        status=payload.status,
        dataset_version=payload.dataset_version,
        prompt_version=payload.prompt_version,
        knowledge_base_version=payload.knowledge_base_version,
        calibration_version=payload.calibration_version,
        metrics=payload.metrics,
        tags=payload.tags,
    )
    db.add(
        AuditLog(
            actor_id=user.id,
            action="model.registered",
            resource_type="model_artifact",
            resource_id=artifact.id,
            details={
                "name": artifact.name,
                "version": artifact.version,
                "status": artifact.status,
            },
        )
    )
    db.commit()
    db.refresh(artifact)
    return _model_artifact_read(artifact)


def _evaluation_read(run: EvaluationRun) -> dict:
    return {
        "id": run.id,
        "task_type": run.task_type,
        "split": run.split,
        "model_name": run.model_name,
        "model_version": run.model_version,
        "dataset_version": run.dataset_version,
        "metrics": run.metrics_json or {},
        "created_at": run.created_at,
    }


@router.post("/evaluations", response_model=EvaluationRunRead)
def create_evaluation(
    payload: EvaluationRunCreate,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict:
    run = record_evaluation(
        db,
        task_type=payload.task_type,
        split=payload.split,
        model_name=payload.model_name,
        model_version=payload.model_version,
        dataset_version=payload.dataset_version,
        metrics=payload.metrics,
    )
    db.add(
        AuditLog(
            actor_id=user.id,
            action="evaluation.recorded",
            resource_type="evaluation_run",
            resource_id=run.id,
            details={"task_type": run.task_type, "split": run.split},
        )
    )
    db.commit()
    db.refresh(run)
    return _evaluation_read(run)


@router.get("/telemetry/summary")
def telemetry_summary(
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict:
    del user
    return {
        "count": db.scalar(select(func.count()).select_from(InferenceTelemetry)) or 0,
        "monitoring": monitoring_summary(db),
    }
