from __future__ import annotations

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

from app.db.base import Base
from app.models import entities as _entities  # noqa: F401 - populate metadata


_SQLITE_IMAGE_STUDY_COLUMNS = {
    "source_type": "VARCHAR(32) NOT NULL DEFAULT 'upload'",
    "modality": "VARCHAR(32) NOT NULL DEFAULT 'chest_xray'",
    "study_instance_uid": "VARCHAR(128)",
    "series_instance_uid": "VARCHAR(128)",
    "sop_instance_uid": "VARCHAR(128)",
    "metadata_json": "JSON NOT NULL DEFAULT '{}'",
}

_SQLITE_IMAGE_STUDY_INDEXES = {
    "ix_image_studies_study_instance_uid": "study_instance_uid",
    "ix_image_studies_series_instance_uid": "series_instance_uid",
    "ix_image_studies_sop_instance_uid": "sop_instance_uid",
}


def ensure_database_schema(engine: Engine) -> None:
    """Create missing tables and apply safe additive SQLite compatibility fixes."""
    Base.metadata.create_all(bind=engine)
    if engine.dialect.name != "sqlite":
        return

    inspector = inspect(engine)
    if not inspector.has_table("image_studies"):
        return
    existing = {column["name"] for column in inspector.get_columns("image_studies")}
    with engine.begin() as connection:
        for name, definition in _SQLITE_IMAGE_STUDY_COLUMNS.items():
            if name not in existing:
                connection.execute(
                    text(f"ALTER TABLE image_studies ADD COLUMN {name} {definition}")
                )
        for index_name, column_name in _SQLITE_IMAGE_STUDY_INDEXES.items():
            connection.execute(
                text(
                    f"CREATE INDEX IF NOT EXISTS {index_name} "
                    f"ON image_studies ({column_name})"
                )
            )
