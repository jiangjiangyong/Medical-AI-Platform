from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import select

from app.core.security import hash_password
from app.db.schema import ensure_database_schema
from app.db.session import SessionLocal, engine
from app.models import User
from app.services.knowledge import KnowledgeService


def seed_users() -> None:
    db = SessionLocal()
    try:
        ensure_database_schema(engine)
        users = [
            ("admin@medical.local", "系统管理员", "admin", "Admin123!"),
            ("doctor@medical.local", "李医生", "doctor", "Doctor123!"),
            ("manager@medical.local", "健康管理师", "health_manager", "Manager123!"),
            ("patient@medical.local", "体验患者", "patient", "Patient123!"),
        ]
        for email, display_name, role, password in users:
            if db.scalar(select(User).where(User.email == email)):
                continue
            db.add(User(email=email, display_name=display_name, role=role, password_hash=hash_password(password)))
        db.commit()
    finally:
        db.close()


async def seed_knowledge() -> None:
    db = SessionLocal()
    try:
        directory = ROOT / "knowledge"
        result = await KnowledgeService().index_directory(db, directory)
        print(result)
    finally:
        db.close()


if __name__ == "__main__":
    seed_users()
    asyncio.run(seed_knowledge())
    print("Demo users and default knowledge indexed.")
