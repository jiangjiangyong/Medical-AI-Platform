from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user, require_roles
from app.config import settings
from app.db.session import get_db
from app.models import (
    AuditLog,
    Case,
    CaseIntake,
    FollowUpTask,
    IdentityVerification,
    ImageStudy,
    Report,
    ReportRevision,
    ReportReview,
    User,
    VisionAnalysis,
)
from app.schemas.common import (
    AnalyzeRequest,
    CaseCreate,
    CaseCreateResponse,
    CaseDetail,
    CaseSummary,
    IdentityVerificationRead,
    IdentityVerificationRequest,
    ReportEditRequest,
    ReportRead,
    ReportRevisionRead,
    ReportReviewRequest,
    UserRead,
)
from app.services.workflow import (
    add_report_revision,
    process_case_analysis_background,
    publish_patient_explanation,
    render_professional_report,
    run_case_analysis,
)

router = APIRouter(prefix="/cases", tags=["cases"])

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg", "application/pdf"}
MAX_STUDY_SIZE = 16 * 1024 * 1024
STAFF_ROLES = {"admin", "doctor", "health_manager"}


def _can_access_case(case: Case, user: User) -> bool:
    return user.role in STAFF_ROLES or case.patient_id == user.id


def _get_case(case_id: str, db: Session, user: User) -> Case:
    case = db.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="病例不存在")
    if not _can_access_case(case, user):
        raise HTTPException(status_code=403, detail="无权访问该病例")
    return case


def _hash_intake_code(code: str) -> str:
    return hashlib.sha256(code.strip().upper().encode("utf-8")).hexdigest()


def _new_intake(db: Session, case: Case, created_by: str) -> tuple[str, datetime]:
    code = f"MI-{secrets.token_hex(4).upper()}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    db.add(
        CaseIntake(
            case_id=case.id,
            code_hash=_hash_intake_code(code),
            code_hint=code[-4:],
            created_by=created_by,
            expires_at=expires_at,
        )
    )
    return code, expires_at


def _latest_verification(db: Session, case_id: str) -> IdentityVerification | None:
    return db.scalar(
        select(IdentityVerification)
        .where(IdentityVerification.case_id == case_id)
        .order_by(IdentityVerification.verified_at.desc())
    )


def _is_expired(value: datetime | None) -> bool:
    if value is None:
        return False
    normalized = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value
    return normalized < datetime.now(timezone.utc)


def _report_read(db: Session, report: Report) -> ReportRead:
    version = db.scalar(
        select(func.max(ReportRevision.version)).where(ReportRevision.report_id == report.id)
    ) or 1
    payload = ReportRead.model_validate(report).model_dump()
    payload["version"] = version
    return ReportRead.model_validate(payload)


def _patient_reports_are_hidden(db: Session, case_id: str) -> None:
    for report in db.scalars(
        select(Report).where(
            Report.case_id == case_id,
            Report.report_type == "patient",
            Report.status != "superseded",
        )
    ):
        report.status = "superseded"


@router.get("", response_model=list[CaseSummary])
def list_cases(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[Case]:
    stmt = select(Case).order_by(Case.updated_at.desc())
    if user.role == "patient":
        stmt = stmt.where(Case.patient_id == user.id)
    return list(db.scalars(stmt))


@router.get("/patients", response_model=list[UserRead])
def list_patients(
    user: User = Depends(require_roles(*STAFF_ROLES)),
    db: Session = Depends(get_db),
) -> list[User]:
    stmt = (
        select(User)
        .where(User.role == "patient", User.is_active.is_(True))
        .order_by(User.display_name, User.email)
    )
    return list(db.scalars(stmt))


@router.post("", response_model=CaseCreateResponse, status_code=status.HTTP_201_CREATED)
def create_case(
    payload: CaseCreate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CaseCreateResponse:
    patient_id = payload.patient_id or user.id
    if user.role == "patient" and patient_id != user.id:
        raise HTTPException(status_code=403, detail="患者只能为自己创建病例")
    patient = db.get(User, patient_id)
    if patient is None or patient.role != "patient":
        raise HTTPException(status_code=422, detail="patient_id 必须对应患者账号")
    case = Case(
        patient_id=patient_id,
        clinician_id=user.id if user.role in {"doctor", "health_manager"} else None,
        title=payload.title.strip(),
        modality=payload.modality,
        symptoms=payload.symptoms.strip(),
        clinical_context=payload.clinical_context,
    )
    db.add(case)
    db.flush()
    intake_code = None
    intake_expires_at = None
    if user.role in STAFF_ROLES:
        intake_code, intake_expires_at = _new_intake(db, case, user.id)
    db.add(AuditLog(actor_id=user.id, action="case.created", resource_type="case", resource_id=case.id))
    db.commit()
    db.refresh(case)
    return CaseCreateResponse(
        **CaseSummary.model_validate(case).model_dump(),
        intake_code=intake_code,
        intake_code_expires_at=intake_expires_at,
    )


@router.get("/{case_id}", response_model=CaseDetail)
def get_case(case_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> CaseDetail:
    case = _get_case(case_id, db, user)
    verification = _latest_verification(db, case.id)
    vision = db.scalar(
        select(VisionAnalysis)
        .where(VisionAnalysis.case_id == case.id)
        .order_by(VisionAnalysis.created_at.desc())
    )
    reports = list(
        db.scalars(
            select(Report)
            .where(Report.case_id == case.id, Report.status != "superseded")
            .order_by(Report.created_at.desc())
        )
    )
    followups = list(
        db.scalars(select(FollowUpTask).where(FollowUpTask.case_id == case.id).order_by(FollowUpTask.due_at))
    )
    studies = list(db.scalars(select(ImageStudy).where(ImageStudy.case_id == case.id)))
    if user.role == "patient":
        reports = [report for report in reports if report.report_type == "patient" and report.status == "published"]
        vision = None
        if verification is None:
            studies = []
    return CaseDetail(
        **CaseSummary.model_validate(case).model_dump(),
        symptoms=case.symptoms,
        clinical_context=case.clinical_context,
        vision_analysis=vision,
        reports=[_report_read(db, report) for report in reports],
        follow_ups=followups,
        studies=[
            {
                "id": study.id,
                "original_name": study.original_name,
                "content_type": study.content_type,
                "file_size": study.file_size,
                "created_at": study.created_at,
            }
            for study in studies
        ],
        identity_verified=verification is not None,
        verification_method=verification.method if verification else None,
        requires_identity_verification=user.role == "patient" and verification is None,
    )


@router.post("/{case_id}/verify-identity", response_model=IdentityVerificationRead)
def verify_identity(
    case_id: str,
    payload: IdentityVerificationRequest,
    user: User = Depends(require_roles("patient")),
    db: Session = Depends(get_db),
) -> IdentityVerificationRead:
    case = _get_case(case_id, db, user)
    existing = _latest_verification(db, case.id)
    if existing:
        return IdentityVerificationRead(
            case_id=case.id,
            verified=True,
            method=existing.method,
            verified_at=existing.verified_at,
        )
    if not payload.confirm_identity:
        raise HTTPException(status_code=422, detail="请先确认这是您本人的检查资料")
    if payload.confirmed_name.strip().casefold() != user.display_name.strip().casefold():
        raise HTTPException(status_code=422, detail="姓名与当前患者账号不一致，请核对后重试")

    intake = db.scalar(select(CaseIntake).where(CaseIntake.case_id == case.id))
    if intake:
        if not payload.intake_code.strip():
            raise HTTPException(status_code=422, detail="该检查需要输入检查编号或一次性核验码")
        if _is_expired(intake.expires_at):
            raise HTTPException(status_code=410, detail="检查核验码已过期，请联系工作人员重新生成")
        if not hmac.compare_digest(intake.code_hash, _hash_intake_code(payload.intake_code)):
            raise HTTPException(status_code=422, detail="检查编号或一次性核验码不正确")
        method = "intake_code"
        intake.used_at = datetime.now(timezone.utc)
    else:
        if payload.intake_code.strip():
            raise HTTPException(status_code=422, detail="当前病例未配置外部核验码，请确认本人身份后继续")
        method = "account_attestation"

    verification = IdentityVerification(
        case_id=case.id,
        patient_id=user.id,
        method=method,
        confirmed_name=payload.confirmed_name.strip(),
        consent_confirmed=True,
    )
    db.add(verification)
    db.add(
        AuditLog(
            actor_id=user.id,
            action="case.identity_verified",
            resource_type="case",
            resource_id=case.id,
            details={"method": method},
        )
    )
    db.commit()
    db.refresh(verification)
    return IdentityVerificationRead(
        case_id=case.id,
        verified=True,
        method=verification.method,
        verified_at=verification.verified_at,
    )


@router.post("/{case_id}/studies", status_code=status.HTTP_201_CREATED)
async def upload_study(
    case_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> dict:
    case = _get_case(case_id, db, user)
    if user.role == "patient" and _latest_verification(db, case.id) is None:
        raise HTTPException(status_code=403, detail="请先完成患者身份核验，再上传检查资料")
    content_type = (file.content_type or "").lower()
    if content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=415, detail="仅支持 JPG、PNG 或 PDF 文件")
    original_name = Path(file.filename or "study").name
    suffix = Path(original_name).suffix.lower() or ".bin"
    target = settings.storage_path / "uploads" / f"{uuid.uuid4().hex}{suffix}"
    size = 0
    with target.open("wb") as destination:
        while chunk := await file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_STUDY_SIZE:
                target.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="文件不能超过 16 MB")
            destination.write(chunk)
    if size == 0:
        target.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="不能上传空文件")

    study = ImageStudy(
        case_id=case.id,
        original_name=original_name,
        content_type=content_type,
        storage_path=str(target),
        file_size=size,
    )
    case.status = "processing"
    db.add(study)
    db.add(
        AuditLog(
            actor_id=user.id,
            action="study.uploaded",
            resource_type="study",
            resource_id=study.id,
            details={"validation": "passed", "content_type": content_type, "file_size": size},
        )
    )
    db.commit()
    db.refresh(study)
    background_tasks.add_task(process_case_analysis_background, case.id, study.id, user.id)
    return {
        "id": study.id,
        "original_name": study.original_name,
        "file_size": study.file_size,
        "status": "processing",
        "validation": "passed",
    }


@router.post("/{case_id}/analyze")
async def analyze_case(
    case_id: str,
    payload: AnalyzeRequest,
    user: User = Depends(require_roles(*STAFF_ROLES)),
    db: Session = Depends(get_db),
) -> dict:
    case = _get_case(case_id, db, user)
    case.status = "processing"
    db.commit()
    study = db.scalar(select(ImageStudy).where(ImageStudy.case_id == case.id).order_by(ImageStudy.created_at.desc()))
    result = await run_case_analysis(
        db,
        case,
        study,
        user.id,
        payload.scenario,
        agent_variant=payload.agent_variant,
    )
    return {
        "status": "success",
        "case_id": case.id,
        "risk_level": result["risk_level"],
        "vision_model": result["vision_analysis"].model_name,
        "report_model": result["model_name"],
        "professional_report": _report_read(db, result["professional_report"]),
    }


@router.patch("/{case_id}/reports/{report_id}", response_model=ReportRead)
def edit_report(
    case_id: str,
    report_id: str,
    payload: ReportEditRequest,
    user: User = Depends(require_roles(*STAFF_ROLES)),
    db: Session = Depends(get_db),
) -> ReportRead:
    case = _get_case(case_id, db, user)
    report = db.scalar(
        select(Report).where(
            Report.id == report_id,
            Report.case_id == case.id,
            Report.report_type == "professional",
            Report.status != "superseded",
        )
    )
    if report is None:
        raise HTTPException(status_code=404, detail="专业报告不存在")
    previous_status = report.status
    current_data = dict(report.structured_data or {})
    edited_data = {
        **current_data,
        "imaging_findings": [item.strip() for item in payload.imaging_findings if item.strip()],
        "preliminary_assessment": payload.preliminary_assessment.strip(),
        "recommendations": [item.strip() for item in payload.recommendations if item.strip()],
        "risk_level": payload.risk_level,
        "human_review_required": True,
        "safety_note": current_data.get("safety_note") or "本内容为 AI 辅助决策草稿，不替代医生诊断或治疗方案。",
        "evidence_summary": current_data.get("evidence_summary", []),
    }
    report.structured_data = edited_data
    report.content = render_professional_report(edited_data)
    report.status = "pending_review"
    report.reviewed_by = None
    report.reviewed_at = None
    if previous_status == "approved":
        _patient_reports_are_hidden(db, case.id)
    case.status = "pending_review"
    add_report_revision(
        db,
        report,
        source_type="doctor",
        editor_id=user.id,
        change_note=payload.change_note or "医生修改报告内容",
    )
    db.add(
        AuditLog(
            actor_id=user.id,
            action="report.edited",
            resource_type="report",
            resource_id=report.id,
            details={"previous_status": previous_status},
        )
    )
    db.commit()
    db.refresh(report)
    return _report_read(db, report)


@router.post("/{case_id}/reports/{report_id}/review", response_model=ReportRead)
async def review_report(
    case_id: str,
    report_id: str,
    payload: ReportReviewRequest,
    user: User = Depends(require_roles(*STAFF_ROLES)),
    db: Session = Depends(get_db),
) -> ReportRead:
    case = _get_case(case_id, db, user)
    report = db.scalar(
        select(Report).where(
            Report.id == report_id,
            Report.case_id == case.id,
            Report.report_type == "professional",
            Report.status != "superseded",
        )
    )
    if report is None:
        raise HTTPException(status_code=404, detail="专业报告不存在")
    if report.status not in {"draft", "pending_review", "needs_revision"}:
        raise HTTPException(status_code=409, detail="当前报告状态不允许重复审核")

    target_status = "needs_revision" if payload.status in {"needs_revision", "rejected"} else "approved"
    previous_status = report.status
    report.status = target_status
    report.reviewed_by = user.id
    report.reviewed_at = datetime.now(timezone.utc)
    current_version = db.scalar(
        select(func.max(ReportRevision.version)).where(ReportRevision.report_id == report.id)
    ) or 1
    db.add(
        ReportReview(
            report_id=report.id,
            reviewer_id=user.id,
            action="publish" if target_status == "approved" else "return",
            from_status=previous_status,
            to_status=target_status,
            version=current_version,
            note=payload.note,
        )
    )
    db.add(
        AuditLog(
            actor_id=user.id,
            action="report.reviewed",
            resource_type="report",
            resource_id=report.id,
            details={"status": target_status, "note": payload.note, "version": current_version},
        )
    )
    if target_status == "needs_revision":
        _patient_reports_are_hidden(db, case.id)
        case.status = "needs_revision"
        db.commit()
        db.refresh(report)
        return _report_read(db, report)

    case.status = "patient_explanation_ready"
    db.commit()
    try:
        await publish_patient_explanation(db, case, report, user.id)
    except Exception as exc:
        db.rollback()
        case = db.get(Case, case_id)
        report = db.get(Report, report_id)
        if case is not None:
            case.status = "approved"
            db.add(
                AuditLog(
                    actor_id=user.id,
                    action="patient_report.generation_failed",
                    resource_type="case",
                    resource_id=case_id,
                    details={"error_type": type(exc).__name__},
                )
            )
            db.commit()
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    db.refresh(report)
    return _report_read(db, report)


@router.get("/{case_id}/reports/{report_id}/revisions", response_model=list[ReportRevisionRead])
def list_report_revisions(
    case_id: str,
    report_id: str,
    user: User = Depends(require_roles(*STAFF_ROLES)),
    db: Session = Depends(get_db),
) -> list[ReportRevision]:
    case = _get_case(case_id, db, user)
    report = db.scalar(select(Report).where(Report.id == report_id, Report.case_id == case.id))
    if report is None:
        raise HTTPException(status_code=404, detail="报告不存在")
    return list(
        db.scalars(
            select(ReportRevision)
            .where(ReportRevision.report_id == report.id)
            .order_by(ReportRevision.version.desc())
        )
    )


@router.get("/studies/{study_id}/file")
def get_study_file(study_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> FileResponse:
    study = db.get(ImageStudy, study_id)
    if study is None:
        raise HTTPException(status_code=404, detail="影像不存在")
    case = db.get(Case, study.case_id)
    if case is None or not _can_access_case(case, user):
        raise HTTPException(status_code=403, detail="无权访问该影像")
    if user.role == "patient" and _latest_verification(db, case.id) is None:
        raise HTTPException(status_code=403, detail="请先完成患者身份核验")
    path = Path(study.storage_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="影像文件不存在")
    return FileResponse(path, media_type=study.content_type, filename=study.original_name)
