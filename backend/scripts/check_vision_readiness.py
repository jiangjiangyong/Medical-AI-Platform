from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from importlib.util import find_spec
from pathlib import Path
from typing import Iterable


SKIP_PARTS = {".git", ".venv", "node_modules", "__pycache__", "storage", "artifacts"}
IMAGE_SUFFIXES = {".dcm", ".dicom", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MANIFEST_SUFFIXES = {".csv", ".json", ".jsonl", ".parquet"}
WEIGHT_SUFFIXES = {".pt", ".pth", ".onnx", ".engine"}


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            relative_parts = path.relative_to(root).parts
        except ValueError:
            continue
        if any(part in SKIP_PARTS for part in relative_parts):
            continue
        yield path


def candidates(root: Path, suffixes: set[str], name_hint: str | None = None) -> list[str]:
    values = []
    for path in iter_files(root):
        if path.suffix.lower() not in suffixes:
            continue
        if name_hint and name_hint.lower() not in path.name.lower():
            continue
        values.append(str(path))
    return sorted(values)


def gpu_snapshot() -> dict[str, object]:
    command = shutil.which("nvidia-smi")
    if not command:
        return {"available": False, "devices": []}
    try:
        result = subprocess.run(
            [
                command,
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return {"available": False, "devices": []}
    devices = []
    for line in result.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) == 3:
            devices.append(
                {
                    "name": fields[0],
                    "memory_total_mib": fields[1],
                    "driver_version": fields[2],
                }
            )
    return {"available": bool(devices), "devices": devices}


def inspect_readiness(project_root: Path, data_root: Path | None) -> dict[str, object]:
    project_root = project_root.resolve()
    roots = [project_root]
    if data_root is not None:
        roots.append(data_root.resolve())

    manifest_files = sorted(
        {
            path
            for root in roots
            for path in candidates(root, MANIFEST_SUFFIXES, "manifest")
        }
    )
    weight_files = sorted(
        {
            path
            for root in roots
            for path in candidates(root, WEIGHT_SUFFIXES)
        }
    )
    image_files = sorted(
        {
            path
            for root in roots
            for path in candidates(root, IMAGE_SUFFIXES)
        }
    )
    blockers = []
    if data_root is None:
        blockers.append("no external dataset root was provided")
    elif not data_root.exists():
        blockers.append(f"external dataset root does not exist: {data_root}")
    if not manifest_files:
        blockers.append("no dataset manifest was found")
    if not weight_files:
        blockers.append("no model checkpoint or inference engine was found")
    if find_spec("torch") is None:
        blockers.append("torch is not installed in the server virtual environment")
    if find_spec("ultralytics") is None:
        blockers.append("ultralytics is not installed in the server virtual environment")
    if not image_files:
        blockers.append("no image or DICOM files were found outside ignored demo storage")

    return {
        "project_root": str(project_root),
        "data_root": str(data_root.resolve()) if data_root is not None else None,
        "manifest_files": manifest_files,
        "weight_files": weight_files,
        "image_file_count": len(image_files),
        "image_files_sample": image_files[:20],
        "python_packages": {
            "torch": find_spec("torch") is not None,
            "ultralytics": find_spec("ultralytics") is not None,
        },
        "gpu": gpu_snapshot(),
        "ready_for_real_vision_baseline": not blockers,
        "blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only check for the real vision baseline prerequisites."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    result = inspect_readiness(args.project_root, args.data_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
