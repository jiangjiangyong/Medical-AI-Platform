from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class QualityGateConfig:
    """Conservative pre-inference checks for chest radiographs."""

    min_width: int = 512
    min_height: int = 512
    min_contrast: float = 8.0
    min_sharpness: float = 4.0
    min_dynamic_range: float = 20.0
    max_blank_fraction: float = 0.985


@dataclass
class ImageQualityResult:
    status: str
    is_usable: bool
    score: float
    metrics: dict[str, Any] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)
    source: str = "image_quality_gate"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _result(
    status: str,
    is_usable: bool,
    score: float,
    *,
    metrics: dict[str, Any] | None = None,
    reasons: list[str] | None = None,
) -> ImageQualityResult:
    return ImageQualityResult(
        status=status,
        is_usable=is_usable,
        score=max(0.0, min(1.0, float(score))),
        metrics=metrics or {},
        reasons=reasons or [],
    )


def assess_image_quality(
    image_path: Path | None,
    config: QualityGateConfig | None = None,
) -> ImageQualityResult:
    """Assess whether an image is suitable for model inference.

    The gate intentionally fails closed for a real image path. A missing path
    is treated as ``not_evaluated`` so that the legacy mock/demo workflow can
    still be exercised without pretending that quality was measured.
    """

    config = config or QualityGateConfig()
    if image_path is None:
        return _result(
            "not_evaluated",
            True,
            0.0,
            reasons=["image_not_provided"],
        )
    if not image_path.exists():
        return _result("missing", False, 0.0, reasons=["file_not_found"])
    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".dcm", ".dicom"}:
        return _result("unsupported", False, 0.0, reasons=["unsupported_image_format"])

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - exercised in minimal deployments
        return _result(
            "dependency_missing",
            False,
            0.0,
            reasons=[f"quality_dependencies_missing:{type(exc).__name__}"],
        )

    if image_path.suffix.lower() in {".dcm", ".dicom"}:
        return _result(
            "dicom_pending",
            False,
            0.0,
            reasons=["dicom_decoder_not_configured"],
        )

    try:
        with Image.open(image_path) as image:
            image.load()
            grayscale = np.asarray(image.convert("L"), dtype=np.float32)
            width, height = image.size
    except Exception as exc:
        return _result(
            "decode_error",
            False,
            0.0,
            reasons=[f"image_decode_failed:{type(exc).__name__}"],
        )

    if grayscale.size == 0:
        return _result("empty", False, 0.0, reasons=["empty_pixel_array"])

    contrast = float(grayscale.std())
    percentile_low, percentile_high = np.percentile(grayscale, [1, 99]).tolist()
    dynamic_range = float(percentile_high - percentile_low)
    vertical_gradient = np.diff(grayscale, axis=0)
    horizontal_gradient = np.diff(grayscale, axis=1)
    sharpness = float(
        (np.var(vertical_gradient) if vertical_gradient.size else 0.0)
        + (np.var(horizontal_gradient) if horizontal_gradient.size else 0.0)
    )
    blank_fraction = float(
        np.mean((grayscale <= 2.0) | (grayscale >= 253.0))
    )

    checks = {
        "resolution": width >= config.min_width and height >= config.min_height,
        "contrast": contrast >= config.min_contrast,
        "sharpness": sharpness >= config.min_sharpness,
        "dynamic_range": dynamic_range >= config.min_dynamic_range,
        "blank_fraction": blank_fraction <= config.max_blank_fraction,
    }
    reasons = [f"{name}_below_threshold" for name, passed in checks.items() if not passed]
    score = sum(1.0 for passed in checks.values() if passed) / len(checks)
    metrics = {
        "width": width,
        "height": height,
        "contrast_std": round(contrast, 4),
        "sharpness_gradient_variance": round(sharpness, 4),
        "dynamic_range_p01_p99": round(dynamic_range, 4),
        "blank_fraction": round(blank_fraction, 6),
        "checks": checks,
        "thresholds": asdict(config),
    }
    return _result("passed" if not reasons else "failed", not reasons, score, metrics=metrics, reasons=reasons)
