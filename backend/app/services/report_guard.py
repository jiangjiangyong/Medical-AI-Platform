from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.schemas.common import VisionResult


UNCERTAINTY_MARKERS = (
    "可疑",
    "不确定",
    "待复核",
    "可能",
    "考虑",
    "不能排除",
    "建议结合",
    "未能确认",
)


class ProfessionalReportPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "professional-report.v1"
    imaging_findings: list[str] = Field(default_factory=list, max_length=20)
    preliminary_assessment: str = Field(default="需要进一步复核", max_length=5000)
    recommendations: list[str] = Field(default_factory=list, max_length=20)
    risk_level: str = Field(default="indeterminate", pattern="^(low|moderate|high|indeterminate)$")
    human_review_required: bool = True
    safety_note: str = Field(default="本内容为 AI 辅助决策草稿，不替代医生诊断或治疗方案。", max_length=2000)
    evidence_summary: list[str] = Field(default_factory=list, max_length=20)


class PatientReportPayload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: str = "patient-report.v1"
    summary: str = Field(default="这份内容需要结合医生最终审核意见理解。", max_length=3000)
    what_it_means: str = Field(default="需要结合原始影像和临床资料判断。", max_length=5000)
    what_to_do: list[str] = Field(default_factory=list, max_length=20)
    when_to_seek_help: str = Field(default="如症状明显加重，请及时就医。", max_length=3000)
    questions_for_clinician: list[str] = Field(default_factory=list, max_length=20)
    safety_note: str = Field(default="请以医生最终审核后的意见为准。", max_length=2000)


@dataclass(frozen=True)
class EvidenceGate:
    passed: bool
    abstain: bool
    reason: str
    top_score: float
    coverage: float
    covered_findings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "abstain": self.abstain,
            "reason": self.reason,
            "top_score": self.top_score,
            "coverage": self.coverage,
            "covered_findings": self.covered_findings,
        }


@dataclass(frozen=True)
class ReportGuardResult:
    data: dict[str, Any]
    schema_valid: bool
    fact_check_passed: bool
    evidence_gate: EvidenceGate
    issues: list[str]
    used_fallback: bool


def _normalize_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _normalize_professional_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    normalized["imaging_findings"] = _normalize_list(normalized.get("imaging_findings"))
    normalized["recommendations"] = _normalize_list(normalized.get("recommendations"))
    normalized["evidence_summary"] = _normalize_list(normalized.get("evidence_summary"))
    return normalized


def _normalize_patient_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload or {})
    normalized["what_to_do"] = _normalize_list(normalized.get("what_to_do"))
    normalized["questions_for_clinician"] = _normalize_list(normalized.get("questions_for_clinician"))
    return normalized


def _report_text(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def evaluate_evidence_gate(
    evidence: Iterable[dict[str, Any]],
    vision: VisionResult,
    *,
    min_score: float = 0.05,
    min_items: int = 1,
) -> EvidenceGate:
    usable = [
        item for item in evidence
        if str(item.get("content", "")).strip()
        and float(item.get("score", 0.0) or 0.0) >= min_score
    ]
    top_score = max((float(item.get("score", 0.0) or 0.0) for item in usable), default=0.0)
    findings = [finding.name for finding in vision.findings]
    if not findings:
        return EvidenceGate(
            passed=bool(usable),
            abstain=not bool(usable),
            reason="no_structured_findings" if usable else "no_evidence",
            top_score=top_score,
            coverage=1.0 if usable else 0.0,
            covered_findings=[],
        )
    evidence_text = "\n".join(str(item.get("content", "")) for item in usable)
    covered = [name for name in findings if name and name in evidence_text]
    coverage = len(covered) / len(findings)
    passed = len(usable) >= min_items and coverage >= 0.5
    return EvidenceGate(
        passed=passed,
        abstain=not passed,
        reason="passed" if passed else "insufficient_finding_evidence",
        top_score=top_score,
        coverage=coverage,
        covered_findings=covered,
    )


def _fact_check_professional_payload(
    payload: dict[str, Any],
    vision: VisionResult,
) -> list[str]:
    text = _report_text(payload)
    issues: list[str] = []
    for finding in vision.findings:
        if finding.confidence_status != "uncertain":
            continue
        if finding.name in text:
            sentences = re.split(r"[。！？.!?\n]", text)
            matching_sentences = [sentence for sentence in sentences if finding.name in sentence]
            if matching_sentences and not all(
                any(marker in sentence for marker in UNCERTAINTY_MARKERS)
                for sentence in matching_sentences
            ):
                issues.append(f"uncertain_finding_written_as_fact:{finding.name}")
    for rejected_name in vision.rejected_findings:
        if rejected_name and rejected_name in text:
            issues.append(f"rejected_finding_reintroduced:{rejected_name}")
    if not payload.get("human_review_required", True):
        issues.append("human_review_required_must_be_true")
    return issues


def validate_professional_report(
    payload: dict[str, Any],
    fallback: dict[str, Any],
    vision: VisionResult,
    evidence: Iterable[dict[str, Any]],
) -> ReportGuardResult:
    normalized = _normalize_professional_payload(payload)
    fallback_normalized = _normalize_professional_payload(fallback)
    issues: list[str] = []
    schema_valid = True
    try:
        validated = ProfessionalReportPayload.model_validate(normalized)
        data = validated.model_dump()
    except ValidationError as exc:
        schema_valid = False
        issues.append("professional_schema_validation_failed")
        issues.extend(error["type"] for error in exc.errors())
        data = ProfessionalReportPayload.model_validate(fallback_normalized).model_dump()
    fact_issues = _fact_check_professional_payload(data, vision)
    issues.extend(fact_issues)
    fact_check_passed = not fact_issues
    if not fact_check_passed:
        data = ProfessionalReportPayload.model_validate(fallback_normalized).model_dump()
    evidence_gate = evaluate_evidence_gate(evidence, vision)
    data.update(
        {
            "schema_version": "professional-report.v1",
            "human_review_required": True,
            "evidence_gate": evidence_gate.to_dict(),
            "fact_check": {
                "passed": fact_check_passed,
                "issues": fact_issues,
            },
        }
    )
    data["evidence_trace"] = [
        {
            "evidence_id": item.get("evidence_id"),
            "source_name": item.get("source_name"),
            "chunk_index": item.get("chunk_index"),
            "rank": item.get("rank"),
            "score": item.get("score"),
        }
        for item in evidence
        if str(item.get("content", "")).strip()
    ]
    if evidence_gate.abstain:
        data["evidence_abstention"] = "证据不足，未将检索结果作为确定性医学事实。"
    return ReportGuardResult(
        data=data,
        schema_valid=schema_valid,
        fact_check_passed=fact_check_passed,
        evidence_gate=evidence_gate,
        issues=issues,
        used_fallback=(not schema_valid or not fact_check_passed),
    )


def validate_patient_report(
    payload: dict[str, Any],
    fallback: dict[str, Any],
    professional: dict[str, Any],
) -> tuple[dict[str, Any], list[str], bool]:
    normalized = _normalize_patient_payload(payload)
    fallback_normalized = _normalize_patient_payload(fallback)
    issues: list[str] = []
    try:
        data = PatientReportPayload.model_validate(normalized).model_dump()
        schema_valid = True
    except ValidationError as exc:
        schema_valid = False
        issues.append("patient_schema_validation_failed")
        issues.extend(error["type"] for error in exc.errors())
        data = PatientReportPayload.model_validate(fallback_normalized).model_dump()
    professional_text = _report_text(professional)
    generated_text = _report_text(data)
    if not data.get("safety_note"):
        issues.append("patient_safety_note_missing")
    if not professional_text:
        issues.append("patient_source_report_missing")
    professional_findings = [str(item) for item in professional.get("imaging_findings", [])]
    professional_recommendations = [str(item) for item in professional.get("recommendations", [])]
    rejected = [str(item) for item in professional.get("abstained_findings", [])]
    for name in rejected:
        if name and name in generated_text:
            issues.append(f"patient_reintroduced_rejected_finding:{name}")
    # The patient text may paraphrase approved findings, but a rejected or
    # absent finding name must never become a new positive assertion.
    if professional_findings and not any(item in generated_text for item in professional_findings):
        if not any(marker in generated_text for marker in UNCERTAINTY_MARKERS):
            issues.append("patient_findings_not_traceable_to_professional_report")
    if professional_recommendations and not data.get("what_to_do"):
        issues.append("patient_action_items_missing")
    if issues and not schema_valid:
        data = PatientReportPayload.model_validate(fallback_normalized).model_dump()
    data["schema_version"] = "patient-report.v1"
    data["source_report_validated"] = bool(professional_text)
    return data, issues, schema_valid and not issues
