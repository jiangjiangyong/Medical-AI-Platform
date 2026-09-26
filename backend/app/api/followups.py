from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.cases import _can_access_case
from app.api.deps import get_current_user
from app.db.session import get_db
from app.models import AuditLog, Case, FollowUpTask, User
from app.schemas.common import FollowUpRead, FollowUpUpdate

router = APIRouter(prefix="/follow-ups", tags=["follow-ups"])


@router.get("", response_model=list[FollowUpRead])
def list_followups(user: User = Depends(get_current_user), db: Session = Depends(get_db)) -> list[FollowUpTask]:
    stmt = select(FollowUpTask).order_by(FollowUpTask.due_at)
    if user.role == "patient":
        stmt = stmt.where(FollowUpTask.patient_id == user.id)
    return list(db.scalars(stmt))


@router.patch("/{task_id}", response_model=FollowUpRead)
def update_followup(
    task_id: str,
    payload: FollowUpUpdate,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> FollowUpTask:
    task = db.get(FollowUpTask, task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="随访任务不存在")
    case = db.get(Case, task.case_id)
    if case is None or not _can_access_case(case, user):
        raise HTTPException(status_code=403, detail="无权更新该随访任务")
    task.status = payload.status
    db.add(AuditLog(actor_id=user.id, action="followup.updated", resource_type="followup", resource_id=task.id, details={"status": task.status}))
    db.commit()
    db.refresh(task)
    return task

