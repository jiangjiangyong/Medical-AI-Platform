from __future__ import annotations

"""Read-only MedGemma-4B access, dependency and memory preflight."""

import argparse
import importlib.util
import json
import os
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _has_token() -> bool:
    return bool(
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it"))
    parser.add_argument("--model-dir", default=os.getenv("MEDGEMMA_MODEL_DIR", ""))
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "artifacts" / "experiments" / "phase5-medgemma-preflight.json")
    args = parser.parse_args()

    package_checks = {
        name: bool(importlib.util.find_spec(name))
        for name in ("torch", "transformers", "accelerate", "bitsandbytes", "fastapi", "uvicorn", "multipart")
    }
    gpu: dict[str, Any] = {"available": False}
    try:
        import torch

        gpu["available"] = bool(torch.cuda.is_available())
        if gpu["available"]:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
            gpu.update(
                {
                    "name": torch.cuda.get_device_name(0),
                    "total_gib": round(total_bytes / 1024**3, 3),
                    "free_gib": round(free_bytes / 1024**3, 3),
                    "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                }
            )
    except Exception as exc:
        gpu["error"] = type(exc).__name__

    source = Path(args.model_dir) if args.model_dir else None
    local_weights = False
    local_files: list[str] = []
    if source is not None:
        local_files = [path.name for path in source.glob("*.safetensors")]
        local_weights = source.is_dir() and (source / "config.json").is_file() and bool(local_files)
    disk = shutil.disk_usage(PROJECT_ROOT)
    missing: list[str] = []
    if not all(package_checks.values()):
        missing.append("inference_dependencies")
    if source is not None and not local_weights:
        missing.append("local_model_weights")
    if source is None and not _has_token():
        missing.append("huggingface_access_token_and_terms")
    if not gpu.get("available"):
        missing.append("cuda_gpu")
    if gpu.get("free_gib", 0.0) < 18.0:
        missing.append("safe_gpu_headroom")
    result = {
        "schema_version": "medgemma-preflight.v1",
        "status": "ready_to_load" if not missing else "blocked",
        "model_id": args.model_id,
        "model_dir": args.model_dir,
        "token_configured": _has_token(),
        "packages": package_checks,
        "gpu": gpu,
        "disk": {
            "free_gib": round(disk.free / 1024**3, 3),
            "total_gib": round(disk.total / 1024**3, 3),
        },
        "local_weights": {"ready": local_weights, "files": local_files},
        "safe_profile": {
            "dtype": "float16",
            "gpu_memory_cap": "18GiB",
            "max_new_tokens": 256,
            "concurrency": 1,
            "oom_fallback": "4bit_if_needed",
        },
        "missing": list(dict.fromkeys(missing)),
        "note": "Read-only check; it does not download or load model weights.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["status"] == "ready_to_load" else 2


if __name__ == "__main__":
    raise SystemExit(main())
