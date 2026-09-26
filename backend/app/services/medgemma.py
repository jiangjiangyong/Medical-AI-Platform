from __future__ import annotations

import mimetypes
import time
from pathlib import Path
from typing import Any

import httpx

from app.config import settings
from app.ml.quality import QualityGateConfig, assess_image_quality
from app.ml.vlm import normalize_vlm_payload
from app.schemas.common import VisionResult
from app.services.dicom import DICOMError, prepare_inference_image
from app.services.resilience import (
    CircuitBreaker,
    CircuitOpenError,
    ResiliencePolicy,
    call_with_resilience,
)


_VLM_BREAKER = CircuitBreaker(
    failure_threshold=settings.service_circuit_failure_threshold,
    recovery_seconds=settings.service_circuit_recovery_seconds,
)


class MedGemmaHTTPClient:
    """HTTP client for a private MedGemma service.

    The service contract is POST base_url plus api_path with multipart field
    image and form fields model, task and schema_version. The JSON response is
    a VisionResult object or an object with a result field.
    """

    def __init__(self) -> None:
        self.base_url = settings.vision_vlm_base_url.rstrip("/")
        self.api_path = settings.vision_vlm_api_path or "/v1/analyze"
        self.model_name = settings.vision_vlm_model or "MedGemma-4B-local"
        self.timeout = settings.vision_vlm_timeout_seconds

    @property
    def endpoint(self) -> str:
        if self.api_path.startswith("/"):
            return f"{self.base_url}{self.api_path}"
        return f"{self.base_url}/{self.api_path}"

    def analyze(
        self,
        image_path: Path | None,
        *,
        scenario: str = "opacity",
    ) -> VisionResult:
        del scenario
        started = time.perf_counter()
        if image_path is None:
            return self._abstain("image_missing", started, is_usable=False)
        if not image_path.exists():
            return self._abstain("image_not_found", started, is_usable=False)
        try:
            inference_path, dicom_metadata = prepare_inference_image(
                image_path,
                settings.storage_path / "derived",
            )
        except DICOMError:
            return self._abstain("dicom_decode_error", started, is_usable=False)
        quality = assess_image_quality(inference_path, self._quality_config())
        quality_payload = quality.to_dict()
        if not quality.is_usable:
            quality_payload["inference_latency_ms"] = 0.0
            return self._abstain(
                "quality_rejected",
                started,
                is_usable=False,
                quality=quality_payload,
            )
        if dicom_metadata:
            quality_payload["dicom_metadata"] = dicom_metadata.to_dict()
        if not self.base_url:
            return self._abstain(
                "service_not_configured",
                started,
                is_usable=True,
                quality=quality_payload,
            )
        try:
            payload = call_with_resilience(
                lambda: self._request_json(inference_path),
                policy=ResiliencePolicy(
                    timeout_seconds=self.timeout,
                    attempts=max(1, settings.service_retry_attempts),
                    backoff_seconds=settings.service_retry_backoff_seconds,
                ),
                breaker=_VLM_BREAKER,
                retry_exceptions=(httpx.HTTPError, OSError),
            )
            result = normalize_vlm_payload(
                payload,
                model_name=self.model_name,
                model_version=settings.vision_model_version,
                dataset_version=settings.vision_dataset_version,
            )
            quality_payload.update(result.image_quality)
            quality_payload["status"] = "passed"
            quality_payload["is_usable"] = True
            quality_payload["inference_latency_ms"] = round(
                (time.perf_counter() - started) * 1000, 3
            )
            return result.model_copy(
                update={
                    "model_name": self.model_name,
                    "image_quality": quality_payload,
                    "pipeline_stages": list(
                        dict.fromkeys(
                            [
                                "image_quality_gate",
                                "vlm_http_request",
                                "vlm_inference",
                                *result.pipeline_stages,
                            ]
                        )
                    ),
                }
            )
        except CircuitOpenError:
            return self._abstain("service_circuit_open", started, quality=quality_payload)
        except httpx.HTTPError:
            return self._abstain("service_request_failed", started, quality=quality_payload)
        except (OSError, ValueError, TypeError, KeyError):
            return self._abstain("invalid_service_response", started, quality=quality_payload)

    def _request_json(self, inference_path: Path) -> Any:
        content_type = mimetypes.guess_type(inference_path.name)[0]
        with inference_path.open("rb") as handle:
            headers = {"Accept": "application/json"}
            if settings.vision_vlm_token:
                headers["Authorization"] = f"Bearer {settings.vision_vlm_token}"
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    self.endpoint,
                    headers=headers,
                    files={
                        "image": (
                            inference_path.name,
                            handle,
                            content_type or "application/octet-stream",
                        )
                    },
                    data={
                        "model": self.model_name,
                        "task": settings.vision_task,
                        "schema_version": "vision.v1",
                    },
                )
                response.raise_for_status()
                return response.json()

    @staticmethod
    def _quality_config() -> QualityGateConfig:
        return QualityGateConfig(
            min_width=settings.quality_min_width,
            min_height=settings.quality_min_height,
            min_contrast=settings.quality_min_contrast,
            min_sharpness=settings.quality_min_sharpness,
            min_dynamic_range=settings.quality_min_dynamic_range,
            max_blank_fraction=settings.quality_max_blank_fraction,
        )

    def _abstain(
        self,
        reason: str,
        started: float,
        *,
        is_usable: bool = True,
        quality: dict[str, Any] | None = None,
    ) -> VisionResult:
        quality_payload = dict(quality or {})
        quality_payload.setdefault("status", reason)
        quality_payload["is_usable"] = is_usable
        quality_payload["inference_latency_ms"] = round(
            (time.perf_counter() - started) * 1000, 3
        )
        return VisionResult(
            model_name=self.model_name,
            model_version=settings.vision_model_version,
            dataset_version=settings.vision_dataset_version,
            task_type="detection",
            provider="medgemma_remote",
            simulated=False,
            image_quality=quality_payload,
            findings=[],
            impression="真实视觉服务未返回可供系统引用的确定性结果。",
            risk_level="indeterminate",
            needs_human_review=True,
            calibration_version="uncalibrated",
            abstained=True,
            pipeline_stages=["image_quality_gate", "vlm_service_preflight", "vision_abstention"],
            limitations=[
                f"视觉服务状态：{reason}。",
                "系统未把缺失或失败的视觉结果写成医学事实，需要人工复核。",
            ],
        )
