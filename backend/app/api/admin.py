from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config import settings
from app.api.deps import require_roles
from app.db.session import get_db
from app.models import AuditLog, Case, FollowUpTask, KnowledgeChunk, User
from app.schemas.common import KnowledgeIndexRequest
from app.services.knowledge import KnowledgeService

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/overview")
def overview(user: User = Depends(require_roles("admin")), db: Session = Depends(get_db)) -> dict:
    return {
        "users": db.scalar(select(func.count()).select_from(User)) or 0,
        "cases": db.scalar(select(func.count()).select_from(Case)) or 0,
        "followups": db.scalar(select(func.count()).select_from(FollowUpTask)) or 0,
        "knowledge_chunks": db.scalar(select(func.count()).select_from(KnowledgeChunk)) or 0,
        "vision_adapter": settings.vision_adapter,
        "vision_target": settings.vision_model_name,
    }


@router.get("/audit-logs")
def audit_logs(
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> list[dict]:
    logs = list(
        db.scalars(
            select(AuditLog)
            .order_by(AuditLog.created_at.desc())
            .limit(30)
        )
    )
    actor_ids = {item.actor_id for item in logs if item.actor_id}
    actors = {
        actor.id: actor.display_name
        for actor in db.scalars(select(User).where(User.id.in_(actor_ids)))
    } if actor_ids else {}
    return [
        {
            "id": item.id,
            "action": item.action,
            "resource_type": item.resource_type,
            "resource_id": item.resource_id,
            "details": item.details,
            "actor_name": actors.get(item.actor_id, "系统") if item.actor_id else "系统",
            "created_at": item.created_at,
        }
        for item in logs
    ]


@router.post("/knowledge/index")
async def index_knowledge(
    payload: KnowledgeIndexRequest,
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict:
    return await KnowledgeService().index_text(db, payload.source_name, payload.content)


@router.post("/knowledge/index-defaults")
async def index_default_knowledge(
    user: User = Depends(require_roles("admin")),
    db: Session = Depends(get_db),
) -> dict:
    from pathlib import Path

    directory = Path(__file__).resolve().parents[2] / "knowledge"
    return {"items": await KnowledgeService().index_directory(db, directory)}
