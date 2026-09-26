from __future__ import annotations

from sqlalchemy import create_engine, inspect, text

from app.db.schema import ensure_database_schema


def test_sqlite_legacy_image_studies_schema_is_upgraded(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE TABLE image_studies ("
                "id VARCHAR(36) PRIMARY KEY, "
                "case_id VARCHAR(36) NOT NULL, "
                "original_name VARCHAR(255) NOT NULL, "
                "content_type VARCHAR(100) NOT NULL, "
                "storage_path VARCHAR(500) NOT NULL, "
                "file_size INTEGER NOT NULL DEFAULT 0, "
                "created_at DATETIME"
                ")"
            )
        )

    ensure_database_schema(engine)

    columns = {column["name"] for column in inspect(engine).get_columns("image_studies")}
    assert {
        "source_type",
        "modality",
        "study_instance_uid",
        "series_instance_uid",
        "sop_instance_uid",
        "metadata_json",
    }.issubset(columns)
