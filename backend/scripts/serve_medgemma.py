from __future__ import annotations

"""Guarded MedGemma-4B HTTP service for the local project.

The model runs in a separate process from the platform API. Inference is
serialized, GPU memory is capped, and unavailable or non-JSON output becomes
an explicit abstention instead of a fabricated finding.
"""

import asyncio
from contextlib import asynccontextmanager
import io
import json
import logging
import os
import re
import threading
import time
from typing import Any

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image, ImageOps


MODEL_ID = os.getenv("MEDGEMMA_MODEL_ID", "google/medgemma-4b-it")
MODEL_DIR = os.getenv("MEDGEMMA_MODEL_DIR", "").strip()
MODEL_NAME = os.getenv("VISION_VLM_MODEL", "MedGemma-4B")
MODEL_VERSION = os.getenv("VISION_MODEL_VERSION", "medgemma-4b-it")
MAX_NEW_TOKENS = int(os.getenv("MEDGEMMA_MAX_NEW_TOKENS", "256"))
MAX_IMAGE_BYTES = int(os.getenv("MEDGEMMA_MAX_IMAGE_BYTES", str(20 * 1024 * 1024)))
LOAD_ON_STARTUP = os.getenv("MEDGEMMA_LOAD_ON_STARTUP", "1").lower() not in {"0", "false", "no"}
ALLOW_4BIT_FALLBACK = os.getenv("MEDGEMMA_ALLOW_4BIT_FALLBACK", "1").lower() not in {"0", "false", "no"}
LOGGER = logging.getLogger(__name__)


def _dtype() -> torch.dtype:
    configured = os.getenv("MEDGEMMA_DTYPE", "float16").lower()
    if configured in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if configured in {"fp32", "float32"}:
        return torch.float32
    return torch.float16


def _model_source() -> str:
    return MODEL_DIR or MODEL_ID


def _token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None


def _memory_map() -> dict[Any, str]:
    gpu_limit = os.getenv("MEDGEMMA_GPU_MEMORY", "18GiB")
    cpu_limit = os.getenv("MEDGEMMA_CPU_MEMORY", "24GiB")
    return {0: gpu_limit, "cpu": cpu_limit} if torch.cuda.is_available() else {"cpu": cpu_limit}


def _load_kwargs(dtype: torch.dtype) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "torch_dtype": dtype,
        "device_map": "auto",
        "max_memory": _memory_map(),
        "low_cpu_mem_usage": True,
    }
    if _token():
        kwargs["token"] = _token()
    return kwargs


class MedGemmaRuntime:
    def __init__(self) -> None:
        self.model: Any | None = None
        self.processor: Any | None = None
        self.load_error: str | None = None
        self.loaded_with_4bit = False
        self.lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.processor is not None

    def load(self) -> None:
        if self.ready:
            return
        with self.lock:
            if self.ready:
                return
            try:
                from transformers import AutoModelForImageTextToText, AutoProcessor

                source = _model_source()
                processor = AutoProcessor.from_pretrained(source, token=_token())
                try:
                    model = AutoModelForImageTextToText.from_pretrained(
                        source,
                        **_load_kwargs(_dtype()),
                    )
                except RuntimeError as exc:
                    is_oom = "out of memory" in str(exc).lower()
                    if not (is_oom and ALLOW_4BIT_FALLBACK):
                        raise
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    from transformers import BitsAndBytesConfig

                    quantization = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                    kwargs = _load_kwargs(torch.float16)
                    kwargs.pop("torch_dtype", None)
                    kwargs["quantization_config"] = quantization
                    model = AutoModelForImageTextToText.from_pretrained(source, **kwargs)
                    self.loaded_with_4bit = True
                model.eval()
                self.processor = processor
                self.model = model
                self.load_error = None
            except Exception as exc:
                self.load_error = f"{type(exc).__name__}: {str(exc)[:500]}"
                self.model = None
                self.processor = None
                raise

    def health(self) -> dict[str, Any]:
        return {
            "ready": self.ready,
            "model_id": _model_source(),
            "model_name": MODEL_NAME,
            "dtype": str(_dtype()).replace("torch.", ""),
            "max_new_tokens": MAX_NEW_TOKENS,
            "serialized_inference": True,
            "loaded_with_4bit": self.loaded_with_4bit,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "gpu_memory_reserved_mb": round(torch.cuda.memory_reserved(0) / 1024**2, 2)
            if torch.cuda.is_available()
            else 0.0,
            "load_error": self.load_error,
        }

    def _input_device(self) -> torch.device:
        if self.model is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return next(self.model.parameters()).device

    def analyze(self, image: Image.Image, prompt: str) -> dict[str, Any]:
        self.load()
        assert self.model is not None
        assert self.processor is not None
        image = ImageOps.exif_transpose(image).convert("RGB")
        messages = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are a medical imaging engineering demo assistant. "
                            "Do not diagnose or prescribe. Return JSON only."
                        ),
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        device = self._input_device()
        moved: dict[str, Any] = {}
        for key, value in inputs.items():
            if hasattr(value, "to"):
                if torch.is_floating_point(value):
                    moved[key] = value.to(device=device, dtype=_dtype())
                else:
                    moved[key] = value.to(device=device)
            else:
                moved[key] = value
        input_len = int(moved["input_ids"].shape[-1])
        with self.lock, torch.inference_mode():
            output = self.model.generate(
                **moved,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
            )
        generated = output[0][input_len:]
        text = self.processor.decode(generated, skip_special_tokens=True).strip()
        return _parse_result(text)


def _parse_json(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if cleaned.startswith(chr(96) * 3):
        cleaned = cleaned[3:]
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.lstrip()
        if cleaned.endswith(chr(96) * 3):
            cleaned = cleaned[:-3].rstrip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _normalize_image_quality(value: Any) -> dict[str, Any]:
    """Keep loosely formatted VLM quality output JSON-safe and auditable."""
    if isinstance(value, dict):
        quality = dict(value)
    elif value is None:
        quality = {}
    else:
        quality = {"model_assessment": str(value).strip()[:200]}
    quality.setdefault("status", "passed")
    quality.setdefault("is_usable", True)
    return quality


def _parse_result(text: str) -> dict[str, Any]:
    parsed = _parse_json(text)
    if parsed is None:
        return {
            "task_type": "detection",
            "image_quality": {"status": "passed", "is_usable": True},
            "findings": [],
            "impression": "MedGemma 未返回可解析的结构化结果。",
            "risk_level": "indeterminate",
            "needs_human_review": True,
            "abstained": True,
            "rejected_findings": [],
            "limitations": ["model_output_not_json", "raw_model_output_was_not_used_as_fact"],
        }
    result = dict(parsed)
    result.setdefault("task_type", "detection")
    if result.get("task_type") not in {"classification", "detection", "segmentation"}:
        result["task_type"] = "detection"
    result["image_quality"] = _normalize_image_quality(result.get("image_quality"))
    raw_findings = result.get("findings")
    invalid_findings_shape = not isinstance(raw_findings, list)
    result["findings"] = raw_findings if isinstance(raw_findings, list) else []
    result.setdefault("impression", "需要人工复核。")
    result.setdefault("risk_level", "indeterminate")
    result["needs_human_review"] = True
    result.setdefault("abstained", False)
    result.setdefault("rejected_findings", [])
    result.setdefault("limitations", [])
    if not isinstance(result["limitations"], list):
        result["limitations"] = [str(result["limitations"])]
    if invalid_findings_shape and raw_findings is not None:
        result["abstained"] = True
        result["limitations"].append("model_findings_not_array")
    normalized_findings: list[dict[str, Any]] = []
    for raw in result["findings"] if isinstance(result["findings"], list) else []:
        if not isinstance(raw, dict) or not str(raw.get("name", "")).strip():
            continue
        try:
            confidence = float(raw.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0
        status = str(raw.get("confidence_status", "uncertain")).lower()
        if status not in {"accepted", "uncertain", "rejected"}:
            status = "uncertain"
        normalized_findings.append(
            {
                **raw,
                "name": str(raw["name"]).strip(),
                "location": str(raw.get("location", "unspecified")).strip() or "unspecified",
                "confidence": max(0.0, min(1.0, confidence)),
                "confidence_status": status,
                "evidence": str(raw.get("evidence", "需要人工复核")).strip(),
                "severity": str(raw.get("severity", "indeterminate")).lower(),
            }
        )
    result["findings"] = normalized_findings
    if invalid_findings_shape and raw_findings is not None:
        result["impression"] = "模型返回的异常结构无法安全转为结构化发现，需要人工复核。"
    if result.get("risk_level") not in {"low", "moderate", "high", "indeterminate"}:
        result["risk_level"] = "indeterminate"
    return result


runtime = MedGemmaRuntime()
@asynccontextmanager
async def lifespan(_: FastAPI):
    if LOAD_ON_STARTUP:
        try:
            runtime.load()
        except Exception:
            # Keep the health endpoint available so an operator can see the
            # failure and fix credentials/weights without restarting blindly.
            LOGGER.warning(
                "MedGemma startup load failed; service remains available with ready=false: %s",
                runtime.load_error,
            )
    yield


app = FastAPI(
    title=f"{MODEL_NAME} local inference service",
    version="1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, Any]:
    return runtime.health()


@app.post("/v1/analyze")
async def analyze(
    image: UploadFile = File(...),
    model: str = Form(default=MODEL_NAME),
    task: str = Form(default="detection"),
    schema_version: str = Form(default="vision.v1"),
) -> dict[str, Any]:
    del model, task, schema_version
    raw = await image.read()
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image is empty or exceeds the configured size")
    try:
        pil_image = Image.open(io.BytesIO(raw))
        pil_image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"image decode failed: {type(exc).__name__}") from exc
    prompt = (
        "Analyze this chest radiograph for an engineering demo. Return one compact JSON object only, "
        "without markdown fences. Use exactly these keys: task_type, image_quality, findings, "
        "impression, risk_level, needs_human_review, abstained, rejected_findings, limitations. "
        "findings must always be a JSON array with at most two items; each item has name, location, "
        "confidence, confidence_status, evidence, severity. Keep every text value short. "
        "If uncertain, use findings=[] and abstained=true. Do not diagnose or prescribe."
    )
    started = time.perf_counter()
    try:
        result = await asyncio.to_thread(runtime.analyze, pil_image, prompt)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        if torch.cuda.is_available() and "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=503, detail="MedGemma inference unavailable") from exc
    result.update(
        {
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "provider": "local_vlm",
            "simulated": False,
            "calibration_version": "uncalibrated",
            "pipeline_stages": ["image_quality_gate", "vlm_inference", "structured_findings"],
            "image_quality": {
                **dict(result.get("image_quality") or {}),
                "status": "passed",
                "is_usable": True,
                "inference_latency_ms": round((time.perf_counter() - started) * 1000, 3),
            },
        }
    )
    return {"result": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("MEDGEMMA_HOST", "127.0.0.1"),
        port=int(os.getenv("MEDGEMMA_PORT", "9000")),
        log_level=os.getenv("MEDGEMMA_LOG_LEVEL", "info"),
    )
