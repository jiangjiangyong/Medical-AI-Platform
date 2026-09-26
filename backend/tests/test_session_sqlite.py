from pathlib import Path

from app.db.session import _prepare_sqlite_parent


def test_file_backed_sqlite_parent_is_created(tmp_path: Path) -> None:
    database_path = tmp_path / "nested" / "storage" / "platform.db"

    _prepare_sqlite_parent(f"sqlite:///{database_path}")

    assert database_path.parent.is_dir()
