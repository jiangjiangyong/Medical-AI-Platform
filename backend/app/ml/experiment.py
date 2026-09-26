from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable, Mapping


EXPERIMENT_VERSION_FIELDS = (
    "dataset_version",
    "model_version",
    "prompt_version",
    "retriever_version",
    "knowledge_base_version",
    "code_commit",
    "experiment_id",
)
ARTIFACT_FILENAMES = (
    "config.json",
    "metrics.json",
    "environment.json",
    "error_cases.jsonl",
    "summary.md",
)
VALID_STATUSES = {"not_run", "blocked", "completed", "failed"}
_EXPERIMENT_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{2,127}$")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def validate_experiment_id(experiment_id: str) -> str:
    value = str(experiment_id).strip()
    if not _EXPERIMENT_ID_PATTERN.fullmatch(value):
        raise ValueError(
            "experiment_id must contain 3-128 lowercase letters, numbers, '.', '_' or '-'; "
            f"got {experiment_id!r}"
        )
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def resolve_code_commit(repo_root: Path) -> str:
    """Return the current commit without inventing a version when Git is absent."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    commit = result.stdout.strip()
    return commit or "unknown"


def capture_environment(repo_root: Path) -> dict[str, Any]:
    """Capture reproducibility context without requiring optional ML packages."""

    package_names = (
        "fastapi",
        "numpy",
        "Pillow",
        "pydicom",
        "pydantic",
        "pytest",
        "torch",
        "torchvision",
        "ultralytics",
        "opencv-python",
    )
    packages: dict[str, str] = {}
    for name in package_names:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = "not_installed"

    gpu: dict[str, Any] = {"available": False, "devices": []}
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        result = None
    if result is not None:
        devices = []
        for line in result.stdout.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) == 3:
                devices.append(
                    {
                        "name": parts[0],
                        "memory_total_mib": parts[1],
                        "driver_version": parts[2],
                    }
                )
        gpu = {"available": bool(devices), "devices": devices}

    return {
        "captured_at": utc_now_iso(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "repo_root": str(repo_root.resolve()),
        "code_commit": resolve_code_commit(repo_root),
        "packages": packages,
        "gpu": gpu,
    }


@dataclass(frozen=True)
class ExperimentMetadata:
    dataset_version: str
    model_version: str
    prompt_version: str
    retriever_version: str
    knowledge_base_version: str
    code_commit: str
    experiment_id: str

    def __post_init__(self) -> None:
        validate_experiment_id(self.experiment_id)
        for field_name in EXPERIMENT_VERSION_FIELDS[:-1]:
            value = getattr(self, field_name)
            if not str(value).strip():
                raise ValueError(f"{field_name} must not be empty")

    def to_dict(self) -> dict[str, str]:
        return {
            field_name: str(getattr(self, field_name))
            for field_name in EXPERIMENT_VERSION_FIELDS
        }


@dataclass(frozen=True)
class ExperimentArtifacts:
    root: Path
    config_path: Path
    metrics_path: Path
    environment_path: Path
    error_cases_path: Path
    summary_path: Path


@dataclass
class ExperimentStore:
    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root).resolve()

    def path_for(self, experiment_id: str) -> Path:
        return self.root / validate_experiment_id(experiment_id)

    def create(
        self,
        *,
        config: Mapping[str, Any],
        metrics: Mapping[str, Any],
        environment: Mapping[str, Any],
        error_cases: Iterable[Mapping[str, Any]],
        summary: str,
        overwrite: bool = False,
    ) -> ExperimentArtifacts:
        experiment_id = validate_experiment_id(str(config.get("experiment_id", "")))
        target = self.path_for(experiment_id)
        if target.exists() and not overwrite:
            raise FileExistsError(
                f"experiment directory already exists: {target}; use overwrite explicitly"
            )
        target.mkdir(parents=True, exist_ok=True)

        _write_json(target / "config.json", dict(config))
        _write_json(target / "metrics.json", dict(metrics))
        _write_json(target / "environment.json", dict(environment))
        with (target / "error_cases.jsonl").open("w", encoding="utf-8") as handle:
            for case in error_cases:
                handle.write(
                    json.dumps(dict(case), ensure_ascii=False, sort_keys=True, default=str)
                    + "\n"
                )
        (target / "summary.md").write_text(summary.rstrip() + "\n", encoding="utf-8")

        return ExperimentArtifacts(
            root=target,
            config_path=target / "config.json",
            metrics_path=target / "metrics.json",
            environment_path=target / "environment.json",
            error_cases_path=target / "error_cases.jsonl",
            summary_path=target / "summary.md",
        )
