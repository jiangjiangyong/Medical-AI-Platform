from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import admin, admin_ops, auth, cases, dashboard, followups, interop
from app.config import settings
from app.db.schema import ensure_database_schema
from app.db.session import engine
from app.models import entities as _entities


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        ensure_database_schema(engine)
        app.state.database_ready = True
    except Exception as exc:
        app.state.database_ready = False
        app.state.database_error = str(exc)
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="医生辅助决策、患者解释和健康随访的医学影像平台原型。",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_prefix = "/api/v1"
app.include_router(auth.router, prefix=api_prefix)
app.include_router(cases.router, prefix=api_prefix)
app.include_router(followups.router, prefix=api_prefix)
app.include_router(dashboard.router, prefix=api_prefix)
app.include_router(admin.router, prefix=api_prefix)
app.include_router(admin_ops.router, prefix=api_prefix)
app.include_router(interop.router, prefix=api_prefix)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if getattr(app.state, "database_ready", False) else "degraded",
        "service": settings.app_name,
        "database_ready": getattr(app.state, "database_ready", False),
        "vision_adapter": settings.vision_adapter,
        "vision_target": settings.vision_model_name,
    }
