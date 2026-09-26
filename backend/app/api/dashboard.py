from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import Case, FollowUpTask, User
from app.schemas.common import DashboardSummary, CaseSummary

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
def dashboard_summary(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> DashboardSummary:
    case_query = select(Case)
    followup_query = select(FollowUpTask)
    if user.role == "patient":
        case_query = case_query.where(Case.patient_id == user.id)
        followup_query = followup_query.where(FollowUpTask.patient_id == user.id)
    cases = list(db.scalars(case_query))
    followups = list(db.scalars(followup_query))
    recent = sorted(cases, key=lambda item: item.updated_at, reverse=True)[:5]
    return DashboardSummary(
        total_cases=len(cases),
        pending_review=sum(1 for item in cases if item.status in {"pending_review", "needs_revision"}),
        active_followups=sum(1 for item in followups if item.status in {"pending", "in_progress"}),
        high_priority=sum(1 for item in cases if item.priority == "high"),
        recent_cases=[CaseSummary.model_validate(item) for item in recent],
    )

