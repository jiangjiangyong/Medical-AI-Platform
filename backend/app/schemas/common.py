from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class UserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    email: str
    display_name: str
    role: str
    is_active: bool
    created_at: datetime


class RegisterRequest(BaseModel):
    email: str = Field(min_length=5, max_length=190)
    password: str = Field(min_length=8, max_length=128)
    display_name: str = Field(min_length=1, max_length=120)


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserRead


class CaseCreate(BaseModel):
    patient_id: str | None = None
    title: str = Field(default="胸部 X 光辅助决策病例", min_length=1, max_length=190)
    modality: str = Field(default="chest_xray", max_length=64)
    symptoms: str = Field(default="", max_length=5000)
    clinical_context: dict[str, Any] = Field(default_factory=dict)


class CaseSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    patient_id: str
    title: str
    modality: str
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime


class CaseCreateResponse(CaseSummary):
    intake_code: str | None = None
    intake_code_expires_at: datetime | None = None


class IdentityVerificationRequest(BaseModel):
    confirmed_name: str = Field(min_length=1, max_length=120)
    intake_code: str = Field(default="", max_length=64)
    confirm_identity: bool = Field(default=False)


class IdentityVerificationRead(BaseModel):
    case_id: str
    verified: bool
    method: str
    verified_at: datetime


class VisionFinding(BaseModel):
    name: str
    location: str = "未明确"
    confidence: float = Field(ge=0, le=1)
    raw_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence_status: Literal["accepted", "uncertain", "rejected"] = "accepted"
    bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    calibration_version: str = "uncalibrated"
    evidence: str = ""
    severity: str = "indeterminate"


class VisionResult(BaseModel):
    schema_version: str = "1.0"
    model_name: str
    model_version: str = "unversioned"
    dataset_version: str = "unversioned"
    task_type: Literal["classification", "detection", "segmentation"] = "detection"
    provider: str = "local"
    simulated: bool = False
    image_quality: dict[str, Any]
    findings: list[VisionFinding]
    impression: str
    risk_level: str
    needs_human_review: bool
    calibration_version: str = "uncalibrated"
    abstained: bool = False
    rejected_findings: list[str] = Field(default_factory=list)
    pipeline_stages: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class VisionAnalysisRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    study_id: str | None
    model_name: str
    schema_version: str
    status: str
    result: dict[str, Any]
    created_at: datetime


class ReportRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    report_type: str
    status: str
    content: str
    structured_data: dict[str, Any]
    evidence: list[Any]
    model_name: str
    reviewed_by: str | None
    reviewed_at: datetime | None
    version: int = 1
    created_at: datetime


class ReportReviewRequest(BaseModel):
    status: str = Field(pattern="^(approved|needs_revision|rejected)$")
    note: str = Field(default="", max_length=2000)


class ReportEditRequest(BaseModel):
    imaging_findings: list[str] = Field(default_factory=list, max_length=20)
    preliminary_assessment: str = Field(min_length=1, max_length=5000)
    recommendations: list[str] = Field(default_factory=list, max_length=20)
    risk_level: str = Field(pattern="^(low|moderate|high|indeterminate)$")
    change_note: str = Field(default="", max_length=2000)


class ReportRevisionRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    report_id: str
    version: int
    source_type: str
    editor_id: str | None
    content: str
    structured_data: dict[str, Any]
    evidence: list[Any]
    change_note: str
    created_at: datetime


class FollowUpRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    case_id: str
    patient_id: str
    title: str
    description: str
    due_at: datetime | None
    status: str
    priority: str
    created_at: datetime
    updated_at: datetime


class FollowUpUpdate(BaseModel):
    status: str = Field(pattern="^(pending|in_progress|completed|cancelled)$")


class CaseDetail(CaseSummary):
    symptoms: str
    clinical_context: dict[str, Any]
    vision_analysis: VisionAnalysisRead | None = None
    reports: list[ReportRead] = Field(default_factory=list)
    follow_ups: list[FollowUpRead] = Field(default_factory=list)
    studies: list[dict[str, Any]] = Field(default_factory=list)
    identity_verified: bool = False
    verification_method: str | None = None
    requires_identity_verification: bool = False


class AnalyzeRequest(BaseModel):
    scenario: str = Field(default="opacity", pattern="^(normal|opacity|nodule|urgent)$")
    agent_variant: Literal[
        "fixed_workflow",
        "single_agent",
        "supervisor_multi_agent",
    ] = "fixed_workflow"


class DashboardSummary(BaseModel):
    total_cases: int
    pending_review: int
    active_followups: int
    high_priority: int
    recent_cases: list[CaseSummary]


class KnowledgeIndexRequest(BaseModel):
    source_name: str = Field(min_length=1, max_length=255)
    content: str = Field(min_length=1, max_length=100000)


class ModelArtifactCreate(BaseModel):
    name: str = Field(min_length=1, max_length=190)
    version: str = Field(min_length=1, max_length=120)
    kind: str = Field(pattern="^(vision|embedding|llm|reranker|prompt|dataset|calibration)$")
    artifact_path: str = Field(default="", max_length=500)
    status: str = Field(default="candidate", pattern="^(candidate|staging|production|retired)$")
    dataset_version: str = Field(default="unversioned", max_length=120)
    prompt_version: str = Field(default="not_applicable", max_length=120)
    knowledge_base_version: str = Field(default="not_applicable", max_length=120)
    calibration_version: str = Field(default="not_applicable", max_length=120)
    metrics: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)


class ModelArtifactRead(ModelArtifactCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime


class EvaluationRunCreate(BaseModel):
    task_type: str = Field(min_length=1, max_length=32)
    split: str = Field(default="test", pattern="^test$")
    model_name: str = Field(min_length=1, max_length=190)
    model_version: str = Field(min_length=1, max_length=120)
    dataset_version: str = Field(min_length=1, max_length=120)
    metrics: dict[str, Any] = Field(default_factory=dict)


class EvaluationRunRead(EvaluationRunCreate):
    model_config = ConfigDict(from_attributes=True)

    id: str
    created_at: datetime
