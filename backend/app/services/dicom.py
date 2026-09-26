from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings


class DICOMError(RuntimeError):
    pass


def _hash_identifier(value: Any) -> str | None:
    if value in (None, ""):
        return None
    raw = f"{settings.dicom_hash_salt}:{str(value)}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


@dataclass
class DICOMMetadata:
    study_instance_uid: str | None = None
    series_instance_uid: str | None = None
    sop_instance_uid: str | None = None
    modality: str | None = None
    body_part_examined: str | None = None
    view_position: str | None = None
    rows: int | None = None
    columns: int | None = None
    photometric_interpretation: str | None = None
    number_of_frames: int | None = None
    patient_name_hash: str | None = None
    patient_id_hash: str | None = None
    safe_tags: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_dicom(path: Path) -> DICOMMetadata:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - dependency is part of backend requirements
        raise DICOMError("pydicom is not installed") from exc
    try:
        dataset = pydicom.dcmread(str(path), stop_before_pixels=True, force=False)
    except Exception as exc:
        raise DICOMError(f"unable to parse DICOM: {type(exc).__name__}") from exc

    def value(name: str) -> Any:
        item = getattr(dataset, name, None)
        return str(item) if item not in (None, "") else None

    def integer(name: str) -> int | None:
        item = getattr(dataset, name, None)
        try:
            return int(item) if item is not None else None
        except (TypeError, ValueError):
            return None

    safe_tags = {
        "StudyDate": value("StudyDate"),
        "StudyTime": value("StudyTime"),
        "AccessionNumber": value("AccessionNumber"),
        "StudyDescription": value("StudyDescription"),
        "SeriesDescription": value("SeriesDescription"),
        "InstitutionName": value("InstitutionName"),
    }
    safe_tags = {key: item for key, item in safe_tags.items() if item is not None}
    return DICOMMetadata(
        study_instance_uid=value("StudyInstanceUID"),
        series_instance_uid=value("SeriesInstanceUID"),
        sop_instance_uid=value("SOPInstanceUID"),
        modality=value("Modality"),
        body_part_examined=value("BodyPartExamined"),
        view_position=value("ViewPosition"),
        rows=integer("Rows"),
        columns=integer("Columns"),
        photometric_interpretation=value("PhotometricInterpretation"),
        number_of_frames=integer("NumberOfFrames"),
        patient_name_hash=_hash_identifier(getattr(dataset, "PatientName", None)),
        patient_id_hash=_hash_identifier(getattr(dataset, "PatientID", None)),
        safe_tags=safe_tags,
    )


def dicom_to_png(path: Path, output_path: Path) -> Path:
    """Materialize the first DICOM frame as a normalized grayscale PNG."""

    try:
        import numpy as np
        import pydicom
        from PIL import Image
    except ImportError as exc:  # pragma: no cover
        raise DICOMError("numpy, Pillow and pydicom are required for DICOM pixels") from exc
    try:
        dataset = pydicom.dcmread(str(path), force=False)
        pixels = dataset.pixel_array
    except Exception as exc:
        raise DICOMError(f"unable to decode DICOM pixels: {type(exc).__name__}") from exc
    array = np.asarray(pixels)
    if array.ndim > 2:
        array = array[0]
    array = array.astype(np.float32)
    low, high = np.percentile(array, [1, 99])
    if high <= low:
        low, high = float(array.min()), float(array.max())
    if high <= low:
        normalized = np.zeros_like(array, dtype=np.uint8)
    else:
        normalized = np.clip((array - low) / (high - low) * 255.0, 0, 255).astype(np.uint8)
    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        normalized = 255 - normalized
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(normalized, mode="L").save(output_path)
    return output_path


def prepare_inference_image(path: Path, derived_dir: Path) -> tuple[Path, DICOMMetadata | None]:
    if path.suffix.lower() not in {".dcm", ".dicom"}:
        return path, None
    metadata = parse_dicom(path)
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    return dicom_to_png(path, derived_dir / f"{digest}.png"), metadata


class DICOMwebAdapter:
    """Small DICOMweb client for QIDO-RS/WADO-RS/STOW-RS integration."""

    def __init__(self, base_url: str | None = None, bearer_token: str | None = None) -> None:
        self.base_url = (base_url or settings.dicomweb_base_url).rstrip("/")
        self.bearer_token = bearer_token or settings.dicomweb_token

    @property
    def configured(self) -> bool:
        return bool(self.base_url)

    def _require_configured(self) -> None:
        if not self.configured:
            raise DICOMError(
                "DICOMweb base URL is not configured; external connectivity was not attempted"
            )

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}"} if self.bearer_token else {}

    async def query_studies(self, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        self._require_configured()
        import httpx

        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.get(
                f"{self.base_url}/studies",
                params=params or {},
                headers=self._headers(),
            )
            response.raise_for_status()
            payload = response.json()
            return payload if isinstance(payload, list) else []

    async def fetch_instance(self, study_uid: str, series_uid: str, sop_uid: str) -> bytes:
        self._require_configured()
        import httpx

        url = f"{self.base_url}/studies/{study_uid}/series/{series_uid}/instances/{sop_uid}"
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.content

    async def store_instance(self, content: bytes) -> dict[str, Any]:
        self._require_configured()
        import httpx

        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{self.base_url}/studies",
                content=content,
                headers={**self._headers(), "Content-Type": "application/dicom"},
            )
            response.raise_for_status()
            return {"status_code": response.status_code, "body": response.text[:2000]}
