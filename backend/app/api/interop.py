from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import require_roles
from app.db.session import get_db
from app.models import Case, ImageStudy, Report, User, VisionAnalysis
from app.schemas.common import VisionResult
from app.services.interop import (
    build_fhir_bundle,
    interoperability_contract,
)


STAFF_ROLES = ("admin", "doctor", "health_manager")
router = APIRouter(prefix="/interop", tags=["interop"])


@router.get("/contracts")
def get_interoperability_contract(
    user: User = Depends(require_roles("admin")),
) -> dict:
    del user
    return interoperability_contract()


@router.get("/cases/{case_id}/fhir-bundle")
def get_case_fhir_bundle(
    case_id: str,
    user: User = Depends(require_roles(*STAFF_ROLES)),
    db: Session = Depends(get_db),
) -> dict:
    del user
    case = db.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="病例不存在")
    vision_record = db.scalar(
        select(VisionAnalysis)
        .where(VisionAnalysis.case_id == case.id)
        .order_by(VisionAnalysis.created_at.desc())
    )
    report = db.scalar(
        select(Report)
        .where(
            Report.case_id == case.id,
            Report.report_type == "professional",
            Report.status != "superseded",
        )
        .order_by(Report.created_at.desc())
    )
    if vision_record is None or report is None:
        raise HTTPException(
            status_code=409,
            detail="病例尚未形成可互操作的视觉结果和专业报告",
        )
    try:
        vision = VisionResult.model_validate(vision_record.result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail="视觉结果不符合互操作载荷要求") from exc
    study = db.scalar(
        select(ImageStudy)
        .where(ImageStudy.case_id == case.id)
        .order_by(ImageStudy.created_at.desc())
    )
    bundle = build_fhir_bundle(
        case_id=case.id,
        patient_reference=f"Patient/{case.patient_id}",
        report_id=report.id,
        report_status=report.status,
        report_text=report.content,
        vision=vision,
        study_reference=f"ImagingStudy/{study.id}" if study else None,
    )
    return bundle
