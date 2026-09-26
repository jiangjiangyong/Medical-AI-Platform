from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.session import SessionLocal
from app.ml.agent import AGENT_VARIANTS, AgentTraceRecorder
from app.services.resilience import ResiliencePolicy, async_call_with_resilience
from app.models.entities import AuditLog, Case, FollowUpTask, ImageStudy, Report, ReportRevision, VisionAnalysis
from app.services.knowledge import KnowledgeService
from app.services.llm import DeepSeekService
from app.services.modelops import current_gpu_memory_mb, record_vision_telemetry
from app.services.report_guard import validate_patient_report, validate_professional_report
from app.services.vision import VisionResult, get_vision_adapter


def _professional_fallback(vision: VisionResult, evidence: list[dict[str, Any]]) -> dict[str, Any]:
    finding_lines = []
    for finding in vision.findings:
        status_prefix = "待复核候选" if finding.confidence_status == "uncertain" else "模型发现"
        confidence_label = "校准置信度"
        finding_lines.append(
            f"{status_prefix}：{finding.name}（{finding.location}，{confidence_label} {finding.confidence:.2f}）"
        )
    if not finding_lines:
        finding_lines = ["当前影像未形成可供自动报告引用的确定性异常发现"]
    return {
        "schema_version": "professional-report.v1",
        "imaging_findings": finding_lines or ["当前模拟结果未见明确异常征象"],
        "preliminary_assessment": vision.impression,
        "recommendations": [
            "由具备资质的专业人员结合原始影像、既往检查和临床资料复核。",
            "如症状持续、加重或出现紧急表现，请及时寻求线下医疗帮助。",
        ],
        "risk_level": vision.risk_level,
        "human_review_required": True,
        "safety_note": "本内容为 AI 辅助决策草稿，不替代医生诊断或治疗方案。",
        "evidence_summary": [item["content"] for item in evidence],
        "abstained_findings": vision.rejected_findings,
    }


def _patient_fallback(professional: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": "patient-report.v1",
        "summary": "这份内容是对医生最终审核意见的通俗解释，不能替代医生面对面判断。",
        "what_it_means": professional.get("preliminary_assessment", "需要结合原始影像进一步判断。"),
        "what_to_do": professional.get("recommendations", []),
        "when_to_seek_help": "如果出现明显呼吸困难、持续胸痛、意识异常或症状快速加重，请及时就医。",
        "questions_for_clinician": [
            "这项影像发现是否需要与既往检查比较？",
            "下一步是否需要进一步检查或复查？",
        ],
        "safety_note": "请以医生最终审核后的意见为准。",
    }


def render_professional_report(data: dict[str, Any]) -> str:
    findings = data.get("imaging_findings") or ["未提供"]
    recommendations = data.get("recommendations") or ["请由专业人员复核"]
    lines = [
        "## 影像所见",
        *[f"- {item}" for item in findings],
        "",
        "## 初步判断",
        str(data.get("preliminary_assessment", "需要进一步复核。")),
        "",
        "## 建议",
        *[f"- {item}" for item in recommendations],
        "",
        f"风险等级：{data.get('risk_level', 'indeterminate')}",
        "",
        f"> {data.get('safety_note', '本内容为辅助决策草稿。')}",
    ]
    return "\n".join(lines)


def _render_patient(data: dict[str, Any]) -> str:
    action_items = data.get("what_to_do") or ["请等待专业人员进一步说明"]
    questions = data.get("questions_for_clinician") or []
    lines = [
        "## 这次结果怎么理解",
        str(data.get("summary", "这是一份辅助解释。")),
        "",
        "## 目前看到什么",
        str(data.get("what_it_means", "需要结合完整检查资料判断。")),
        "",
        "## 接下来可以做什么",
        *[f"- {item}" for item in action_items],
        "",
        "## 可以和医生确认的问题",
        *[f"- {item}" for item in questions],
        "",
        f"## 需要注意\n{data.get('when_to_seek_help', '')}",
        "",
        f"> {data.get('safety_note', '请以医生最终审核意见为准。')}",
    ]
    return "\n".join(lines)


def _follow_up_days(risk_level: str) -> int:
    return {"high": 7, "moderate": 30, "low": 90}.get(risk_level, 30)


def _normalize_risk_level(value: Any, fallback: str = "moderate") -> str:
    risk_level = str(value or "").lower()
    if risk_level in {"low", "moderate", "high"}:
        return risk_level
    return fallback if fallback in {"low", "moderate", "high"} else "moderate"


def _supersede_reports(db: Session, case_id: str, report_type: str | None = None) -> None:
    stmt = select(Report).where(Report.case_id == case_id, Report.status != "superseded")
    if report_type:
        stmt = stmt.where(Report.report_type == report_type)
    for report in db.scalars(stmt):
        report.status = "superseded"


def add_report_revision(
    db: Session,
    report: Report,
    *,
    source_type: str,
    editor_id: str | None,
    change_note: str = "",
) -> ReportRevision:
    db.flush()
    current_version = db.scalar(
        select(func.max(ReportRevision.version)).where(ReportRevision.report_id == report.id)
    ) or 0
    revision = ReportRevision(
        report_id=report.id,
        version=current_version + 1,
        source_type=source_type,
        editor_id=editor_id,
        content=report.content,
        structured_data=report.structured_data,
        evidence=report.evidence,
        change_note=change_note,
    )
    db.add(revision)
    return revision


def _ensure_follow_up(db: Session, case: Case, risk_level: str) -> FollowUpTask:
    due_at = datetime.now(timezone.utc) + timedelta(days=_follow_up_days(risk_level))
    active_followup = db.scalar(
        select(FollowUpTask)
        .where(
            FollowUpTask.case_id == case.id,
            FollowUpTask.status.in_(["pending", "in_progress"]),
        )
        .order_by(FollowUpTask.created_at.desc())
    )
    priority = "high" if risk_level == "high" else "normal"
    if active_followup:
        active_followup.due_at = due_at
        active_followup.priority = priority
        return active_followup
    follow_up = FollowUpTask(
        case_id=case.id,
        patient_id=case.patient_id,
        title="执行医生确认的随访计划",
        description="请按医生最终发布的意见完成复查、健康记录或线下就医安排。",
        due_at=due_at,
        priority=priority,
    )
    db.add(follow_up)
    return follow_up


_AGENT_TRACE_STAGE_MAP: dict[str, dict[str, tuple[str, str]]] = {
    "fixed_workflow": {
        "load_case": ("load_case", "case.read"),
        "quality_gate": ("quality_gate", "quality.check"),
        "vision": ("vision", "vision.analyze"),
        "retrieve": ("retrieve", "knowledge.search"),
        "draft_report": ("draft_report", "report.generate"),
        "verify_report": ("verify_report", "report.verify"),
        "human_review": ("human_review", "review.request"),
    },
    "single_agent": {
        "load_case": ("case.read", "case.read"),
        "vision": ("vision.analyze", "vision.analyze"),
        "retrieve": ("knowledge.search", "knowledge.search"),
        "draft_report": ("report.generate", "report.generate"),
        "verify_report": ("report.verify", "report.verify"),
        "human_review": ("review.request", "review.request"),
    },
    "supervisor_multi_agent": {
        "vision": ("vision_agent", "vision_agent"),
        "retrieve": ("retrieval_agent", "retrieval_agent"),
        "draft_report": ("report_agent", "report_agent"),
        "verify_report": ("verification_agent", "verification_agent"),
        "human_review": ("human_review", "human_review"),
    },
}


def _record_agent_stage(
    trace: AgentTraceRecorder,
    variant: str,
    stage: str,
    *,
    status: str = "completed",
    metadata: dict[str, Any] | None = None,
) -> None:
    mapping = _AGENT_TRACE_STAGE_MAP[variant].get(stage)
    if mapping is None:
        return
    step, tool = mapping
    trace.record(step, tool=tool, status=status, metadata=metadata)


async def run_case_analysis(
    db: Session,
    case: Case,
    study: ImageStudy | None,
    actor_id: str,
    scenario: str = "opacity",
    *,
    agent_variant: str = "fixed_workflow",
) -> dict[str, Any]:
    if agent_variant not in AGENT_VARIANTS:
        raise ValueError(f"unknown agent variant: {agent_variant}")
    image_path = Path(study.storage_path) if study else None
    vision = get_vision_adapter().analyze(image_path, scenario=scenario)
    trace = AgentTraceRecorder(
        agent_variant,
        model_name=vision.model_name,
        simulated=vision.simulated,
    )
    if agent_variant == "single_agent":
        trace.record("agent.route", tool="agent.route", metadata={"case_id": case.id})
    elif agent_variant == "supervisor_multi_agent":
        trace.record(
            "supervisor.route",
            tool="supervisor.route",
            metadata={"case_id": case.id},
        )
    _record_agent_stage(
        trace,
        agent_variant,
        "load_case",
        metadata={"case_id": case.id},
    )
    _record_agent_stage(
        trace,
        agent_variant,
        "quality_gate",
        status=(
            "completed"
            if vision.image_quality.get("is_usable", True)
            else "abstained"
        ),
        metadata={"quality_status": vision.image_quality.get("status")},
    )
    _record_agent_stage(
        trace,
        agent_variant,
        "vision",
        status="abstained" if vision.abstained else "completed",
        metadata={"provider": vision.provider},
    )
    telemetry_status = "abstained" if vision.abstained else "completed"
    record_vision_telemetry(
        db,
        case_id=case.id,
        study_id=study.id if study else None,
        vision=vision,
        status=telemetry_status,
        gpu_memory_mb=current_gpu_memory_mb(),
    )

    _supersede_reports(db, case.id)
    vision_record = VisionAnalysis(
        case_id=case.id,
        study_id=study.id if study else None,
        model_name=vision.model_name,
        schema_version=vision.schema_version,
        status=(
            "quality_rejected"
            if vision.image_quality.get("is_usable") is False
            else ("simulated" if vision.simulated else "completed")
        ),
        result=vision.model_dump(mode="json"),
    )
    db.add(vision_record)
    db.flush()

    query = "；".join([case.symptoms, vision.impression, *[finding.name for finding in vision.findings]])
    evidence = await KnowledgeService().search(db, query, limit=4)
    _record_agent_stage(
        trace,
        agent_variant,
        "retrieve",
        status="completed" if evidence else "abstained",
        metadata={"evidence_count": len(evidence)},
    )
    professional_fallback = _professional_fallback(vision, evidence)
    professional_prompt = (
        "你是医学辅助决策系统中的报告草稿模块。只根据病例资料、结构化视觉结果和证据生成 JSON。"
        "不得把概率性结果写成确诊，不得编造影像中没有的部位、尺寸、检查结果或治疗方案。"
        "必须输出合法 JSON，字段为 imaging_findings、preliminary_assessment、recommendations、"
        "risk_level、human_review_required、safety_note、evidence_summary。"
    )
    professional_user = {
        "病例": {
            "检查类型": case.modality,
            "症状": case.symptoms,
            "临床背景": case.clinical_context,
        },
        "视觉模型结果": vision.model_dump(mode="json"),
        "检索证据": evidence,
        "json_example": professional_fallback,
    }
    if vision.image_quality.get("is_usable") is False:
        professional_data, professional_model = professional_fallback, "quality-gate-fallback"
    else:
        professional_data, professional_model = await DeepSeekService().generate_json(
            professional_prompt,
            json.dumps(professional_user, ensure_ascii=False),
            professional_fallback,
        )
    _record_agent_stage(
        trace,
        agent_variant,
        "draft_report",
        status="completed",
        metadata={"model": professional_model},
    )
    guard = validate_professional_report(professional_data, professional_fallback, vision, evidence)
    professional_data = guard.data
    if guard.used_fallback:
        professional_model = f"{professional_model}+guard-fallback"
    _record_agent_stage(
        trace,
        agent_variant,
        "verify_report",
        status="completed" if guard.schema_valid and guard.fact_check_passed else "abstained",
        metadata={
            "schema_valid": guard.schema_valid,
            "fact_check_passed": guard.fact_check_passed,
            "evidence_gate": guard.evidence_gate.to_dict(),
        },
    )
    if agent_variant == "single_agent":
        _record_agent_stage(
            trace,
            agent_variant,
            "human_review",
            status="pending",
            metadata={"required": True},
        )
        trace.record(
            "agent.verify",
            tool="agent.verify",
            status="completed",
            metadata={"report_guard_passed": guard.schema_valid and guard.fact_check_passed},
        )
        trace.record("human_review", tool="human_review", status="pending")
    elif agent_variant == "supervisor_multi_agent":
        trace.record(
            "supervisor.aggregate",
            tool="supervisor.aggregate",
            status="completed",
            metadata={"report_guard_passed": guard.schema_valid and guard.fact_check_passed},
        )
        _record_agent_stage(
            trace,
            agent_variant,
            "human_review",
            status="pending",
            metadata={"required": True},
        )
    else:
        _record_agent_stage(
            trace,
            agent_variant,
            "human_review",
            status="pending",
            metadata={"required": True},
        )
    professional_data["risk_level"] = _normalize_risk_level(professional_data.get("risk_level"), vision.risk_level)
    trace_id = f"{case.id}:{vision_record.id}"
    agent_trace = trace.to_dict(trace_id=trace_id)
    generation_trace = {
        "vision_model": vision.model_name,
        "vision_model_version": vision.model_version,
        "dataset_version": vision.dataset_version,
        "calibration_version": vision.calibration_version,
        "pipeline_stages": vision.pipeline_stages,
        "agent_trace": agent_trace,
    }
    professional_data["generation_trace"] = generation_trace
    vision_result_payload = dict(vision_record.result or {})
    vision_result_payload["generation_trace"] = generation_trace
    vision_record.result = vision_result_payload
    professional_report = Report(
        case_id=case.id,
        report_type="professional",
        status="pending_review",
        content=render_professional_report(professional_data),
        structured_data=professional_data,
        evidence=evidence,
        model_name=professional_model,
    )
    db.add(professional_report)
    db.flush()
    add_report_revision(db, professional_report, source_type="model", editor_id=None, change_note="视觉模型分析生成")

    risk_level = _normalize_risk_level(professional_data.get("risk_level"), vision.risk_level)
    case.status = "pending_review"
    case.priority = "high" if risk_level == "high" else "routine"
    db.add(
        AuditLog(
            actor_id=actor_id,
            action="case.analyzed",
            resource_type="case",
            resource_id=case.id,
            details={
                "vision_model": vision.model_name,
                "professional_model": professional_model,
                "risk_level": risk_level,
                "study_id": study.id if study else None,
                "schema_valid": guard.schema_valid,
                "fact_check_passed": guard.fact_check_passed,
                "evidence_gate": guard.evidence_gate.to_dict(),
                "guard_issues": guard.issues,
                "agent_trace": agent_trace,
            },
        )
    )
    db.commit()
    db.refresh(vision_record)
    db.refresh(professional_report)
    return {
        "vision_analysis": vision_record,
        "professional_report": professional_report,
        "risk_level": risk_level,
        "model_name": professional_model,
    }


async def publish_patient_explanation(
    db: Session,
    case: Case,
    professional_report: Report,
    actor_id: str,
) -> Report:
    professional_data = dict(professional_report.structured_data or {})
    patient_fallback = _patient_fallback(professional_data)
    patient_prompt = (
        "你是患者沟通模块。只能把医生已经最终确认的专业报告转成清楚、克制、友好的中文 JSON。"
        "不得新增诊断、药物、检查或未经医生报告支持的医学事实；不得改变医生的风险判断。"
        "字段为 summary、what_it_means、what_to_do、when_to_seek_help、"
        "questions_for_clinician、safety_note。输出合法 JSON。"
    )
    patient_data, patient_model = await DeepSeekService().generate_json(
        patient_prompt,
        json.dumps(
            {
                "医生最终报告": {
                    "structured_data": professional_data,
                    "content": professional_report.content,
                },
                "json_example": patient_fallback,
            },
            ensure_ascii=False,
        ),
        patient_fallback,
    )
    patient_data, patient_issues, patient_valid = validate_patient_report(
        patient_data,
        patient_fallback,
        professional_data,
    )
    if not patient_valid:
        patient_model = f"{patient_model}+guard-fallback"
        patient_data = _patient_fallback(professional_data)
        patient_data["validation"] = {"passed": False, "issues": patient_issues}
    else:
        patient_data["validation"] = {"passed": True, "issues": patient_issues}
    _supersede_reports(db, case.id, report_type="patient")
    patient_report = Report(
        case_id=case.id,
        report_type="patient",
        status="published",
        content=_render_patient(patient_data),
        structured_data=patient_data,
        evidence=professional_report.evidence,
        model_name=patient_model,
        reviewed_by=actor_id,
        reviewed_at=datetime.now(timezone.utc),
    )
    db.add(patient_report)
    db.flush()
    add_report_revision(db, patient_report, source_type="patient_explanation", editor_id=actor_id, change_note="基于医生最终报告生成")
    risk_level = _normalize_risk_level(professional_data.get("risk_level"))
    _ensure_follow_up(db, case, risk_level)
    case.status = "published"
    db.add(
        AuditLog(
            actor_id=actor_id,
            action="patient_report.published",
            resource_type="case",
            resource_id=case.id,
            details={"professional_report_id": professional_report.id, "patient_model": patient_model},
        )
    )
    db.commit()
    db.refresh(patient_report)
    return patient_report


async def process_case_analysis_background(
    case_id: str,
    study_id: str,
    actor_id: str,
    scenario: str = "opacity",
) -> None:
    db = SessionLocal()
    try:
        case = db.get(Case, case_id)
        study = db.get(ImageStudy, study_id)
        if case is None or study is None:
            return

        async def _run_once() -> None:
            try:
                await run_case_analysis(db, case, study, actor_id, scenario=scenario)
            except Exception:
                db.rollback()
                raise

        await async_call_with_resilience(
            _run_once,
            policy=ResiliencePolicy(
                timeout_seconds=settings.service_timeout_seconds,
                attempts=max(1, settings.service_retry_attempts),
                backoff_seconds=settings.service_retry_backoff_seconds,
            ),
        )
    except Exception as exc:
        db.rollback()
        case = db.get(Case, case_id)
        if case is not None:
            case.status = "analysis_failed"
            db.add(
                AuditLog(
                    actor_id=actor_id,
                    action="case.analysis_failed",
                    resource_type="case",
                    resource_id=case_id,
                    details={"error_type": type(exc).__name__},
                )
            )
            db.commit()
    finally:
        db.close()
