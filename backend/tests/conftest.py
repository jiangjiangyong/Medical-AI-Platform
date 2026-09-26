from __future__ import annotations

import atexit
import os
import sys
import tempfile
from pathlib import Path

test_database = Path(tempfile.gettempdir()) / f"medical_imaging_platform_test_{os.getpid()}.sqlite3"
test_database.unlink(missing_ok=True)
os.environ["DATABASE_URL"] = f"sqlite:///{test_database.as_posix()}"
os.environ["DEEPSEEK_API_KEY"] = ""
os.environ["EMBEDDING_API_KEY"] = ""


@atexit.register
def remove_test_database() -> None:
    try:
        test_database.unlink(missing_ok=True)
    except PermissionError:
        # Windows may release the SQLite handle after interpreter shutdown.
        pass


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
