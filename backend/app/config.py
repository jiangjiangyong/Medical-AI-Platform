from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "storage"
DEFAULT_DATABASE_URL = f"sqlite:///{(DEFAULT_STORAGE_DIR / 'medical_imaging_platform.db').as_posix()}"
DEFAULT_CREDENTIALS_FILE = PROJECT_ROOT / "api_credentials.txt"


def _parse_credentials_file(path: Path) -> dict[str, str]:
    """Read simple KEY=VALUE credentials without copying secrets into the repo."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line and ":" not in line:
            continue
        separator = "=" if "=" in line else ":"
        key, value = line.split(separator, 1)
        key = key.strip().upper()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


class Settings(BaseModel):
    app_name: str = Field(default="医学影像辅助决策与健康随访平台")
    app_env: str = Field(default="development")
    api_credentials_file: str = Field(default=str(DEFAULT_CREDENTIALS_FILE))
    database_url: str = Field(default=DEFAULT_DATABASE_URL)
    jwt_secret: str = Field(default="dev-only-change-this-secret")
    jwt_expire_minutes: int = Field(default=720)
    storage_dir: str = Field(default=str(DEFAULT_STORAGE_DIR))
    vision_adapter: str = Field(default="mock")
    vision_model_name: str = Field(default="MedGemma-4B-local-adapter")
    vision_model_path: str = Field(default="")
    vision_device: str = Field(default="auto")
    vision_task: str = Field(default="detection")
    vision_model_version: str = Field(default="unversioned")
    vision_dataset_version: str = Field(default="unversioned")
    vision_calibration_path: str = Field(default="")
    vision_vlm_base_url: str = Field(default="")
    vision_vlm_api_path: str = Field(default="/v1/analyze")
    vision_vlm_token: str = Field(default="")
    vision_vlm_model: str = Field(default="MedGemma-4B-local")
    vision_vlm_timeout_seconds: float = Field(default=60.0, gt=0)
    service_timeout_seconds: float = Field(default=45.0, gt=0)
    service_retry_attempts: int = Field(default=2, ge=0, le=5)
    service_retry_backoff_seconds: float = Field(default=0.25, ge=0, le=10)
    service_circuit_failure_threshold: int = Field(default=3, ge=1, le=20)
    service_circuit_recovery_seconds: float = Field(default=30.0, gt=0)
    performance_p95_budget_ms: float = Field(default=3000.0, gt=0)
    performance_failure_rate_budget: float = Field(default=0.02, ge=0, le=1)
    mlops_drift_psi_threshold: float = Field(default=0.20, ge=0)
    vision_confidence_threshold: float = Field(default=0.10, ge=0, le=1)
    vision_accept_threshold: float = Field(default=0.55, ge=0, le=1)
    vision_uncertain_threshold: float = Field(default=0.35, ge=0, le=1)
    vision_iou_threshold: float = Field(default=0.45, ge=0, le=1)
    quality_min_width: int = Field(default=512, ge=1)
    quality_min_height: int = Field(default=512, ge=1)
    quality_min_contrast: float = Field(default=8.0, ge=0)
    quality_min_sharpness: float = Field(default=4.0, ge=0)
    quality_min_dynamic_range: float = Field(default=20.0, ge=0)
    quality_max_blank_fraction: float = Field(default=0.985, ge=0, le=1)
    prompt_version: str = Field(default="prompt-v1")
    knowledge_base_version: str = Field(default="kb-unversioned")
    rag_version: str = Field(default="hybrid-rag-v1")
    rag_candidate_multiplier: int = Field(default=5, ge=1, le=50)
    rag_rrf_k: int = Field(default=60, ge=1)
    rag_min_evidence_score: float = Field(default=0.0, ge=0)
    rag_reranker_path: str = Field(default="")
    dicom_hash_salt: str = Field(default="development-only-dicom-salt")
    dicomweb_base_url: str = Field(default="")
    dicomweb_token: str = Field(default="")
    fhir_base_url: str = Field(default="")
    fhir_token: str = Field(default="")
    embedding_max_chars: int = Field(default=280)
    embedding_chunk_overlap: int = Field(default=40)
    embedding_batch_size: int = Field(default=8)
    cors_origins: str = Field(default="http://localhost:5173,http://127.0.0.1:5173")
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_model: str = Field(default="deepseek-chat")
    embedding_api_key: str = Field(default="")
    embedding_url: str = Field(default="https://api.siliconflow.cn/v1/embeddings")
    embedding_model: str = Field(default="BAAI/bge-large-zh-v1.5")

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    @property
    def storage_path(self) -> Path:
        path = Path(self.storage_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "uploads").mkdir(parents=True, exist_ok=True)
        (path / "knowledge").mkdir(parents=True, exist_ok=True)
        return path


def _env_value(name: str, default: Any) -> Any:
    return os.getenv(name, default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def load_settings() -> Settings:
    credential_file = Path(
        os.getenv(
            "API_CREDENTIALS_FILE",
            str(DEFAULT_CREDENTIALS_FILE),
        )
    )
    external = _parse_credentials_file(credential_file)
    return Settings(
        app_name=_env_value("APP_NAME", "医学影像辅助决策与健康随访平台"),
        app_env=_env_value("APP_ENV", "development"),
        api_credentials_file=str(credential_file),
        database_url=_env_value(
            "DATABASE_URL",
            DEFAULT_DATABASE_URL,
        ),
        jwt_secret=_env_value("JWT_SECRET", "dev-only-change-this-secret"),
        jwt_expire_minutes=int(_env_value("JWT_EXPIRE_MINUTES", 720)),
        storage_dir=_env_value(
            "STORAGE_DIR",
            str(DEFAULT_STORAGE_DIR),
        ),
        vision_adapter=_env_value("VISION_ADAPTER", "mock"),
        vision_model_name=_env_value(
            "VISION_MODEL_NAME", "MedGemma-4B-local-adapter"
        ),
        vision_model_path=_env_value("VISION_MODEL_PATH", ""),
        vision_device=_env_value("VISION_DEVICE", "auto"),
        vision_task=_env_value("VISION_TASK", "detection"),
        vision_model_version=_env_value("VISION_MODEL_VERSION", "unversioned"),
        vision_dataset_version=_env_value("VISION_DATASET_VERSION", "unversioned"),
        vision_calibration_path=_env_value("VISION_CALIBRATION_PATH", ""),
        vision_vlm_base_url=_env_value("VISION_VLM_BASE_URL", ""),
        vision_vlm_api_path=_env_value("VISION_VLM_API_PATH", "/v1/analyze"),
        vision_vlm_token=_env_value("VISION_VLM_TOKEN", ""),
        vision_vlm_model=_env_value("VISION_VLM_MODEL", "MedGemma-4B-local"),
        vision_vlm_timeout_seconds=float(_env_value("VISION_VLM_TIMEOUT_SECONDS", 60.0)),
        service_timeout_seconds=float(_env_value("SERVICE_TIMEOUT_SECONDS", 45.0)),
        service_retry_attempts=int(_env_value("SERVICE_RETRY_ATTEMPTS", 2)),
        service_retry_backoff_seconds=float(
            _env_value("SERVICE_RETRY_BACKOFF_SECONDS", 0.25)
        ),
        service_circuit_failure_threshold=int(
            _env_value("SERVICE_CIRCUIT_FAILURE_THRESHOLD", 3)
        ),
        service_circuit_recovery_seconds=float(
            _env_value("SERVICE_CIRCUIT_RECOVERY_SECONDS", 30.0)
        ),
        performance_p95_budget_ms=float(_env_value("PERFORMANCE_P95_BUDGET_MS", 3000.0)),
        performance_failure_rate_budget=float(
            _env_value("PERFORMANCE_FAILURE_RATE_BUDGET", 0.02)
        ),
        mlops_drift_psi_threshold=float(_env_value("MLOPS_DRIFT_PSI_THRESHOLD", 0.20)),
        vision_confidence_threshold=float(_env_value("VISION_CONFIDENCE_THRESHOLD", 0.10)),
        vision_accept_threshold=float(_env_value("VISION_ACCEPT_THRESHOLD", 0.55)),
        vision_uncertain_threshold=float(_env_value("VISION_UNCERTAIN_THRESHOLD", 0.35)),
        vision_iou_threshold=float(_env_value("VISION_IOU_THRESHOLD", 0.45)),
        quality_min_width=int(_env_value("QUALITY_MIN_WIDTH", 512)),
        quality_min_height=int(_env_value("QUALITY_MIN_HEIGHT", 512)),
        quality_min_contrast=float(_env_value("QUALITY_MIN_CONTRAST", 8.0)),
        quality_min_sharpness=float(_env_value("QUALITY_MIN_SHARPNESS", 4.0)),
        quality_min_dynamic_range=float(_env_value("QUALITY_MIN_DYNAMIC_RANGE", 20.0)),
        quality_max_blank_fraction=float(_env_value("QUALITY_MAX_BLANK_FRACTION", 0.985)),
        prompt_version=_env_value("PROMPT_VERSION", "prompt-v1"),
        knowledge_base_version=_env_value("KNOWLEDGE_BASE_VERSION", "kb-unversioned"),
        rag_version=_env_value("RAG_VERSION", "hybrid-rag-v1"),
        rag_candidate_multiplier=int(_env_value("RAG_CANDIDATE_MULTIPLIER", 5)),
        rag_rrf_k=int(_env_value("RAG_RRF_K", 60)),
        rag_min_evidence_score=float(_env_value("RAG_MIN_EVIDENCE_SCORE", 0.0)),
        rag_reranker_path=_env_value("RAG_RERANKER_PATH", ""),
        dicom_hash_salt=_env_value("DICOM_HASH_SALT", "development-only-dicom-salt"),
        dicomweb_base_url=_env_value("DICOMWEB_BASE_URL", ""),
        dicomweb_token=_env_value("DICOMWEB_TOKEN", ""),
        fhir_base_url=_env_value("FHIR_BASE_URL", ""),
        fhir_token=_env_value("FHIR_TOKEN", ""),
        embedding_max_chars=int(_env_value("EMBEDDING_MAX_CHARS", 280)),
        embedding_chunk_overlap=int(_env_value("EMBEDDING_CHUNK_OVERLAP", 40)),
        embedding_batch_size=int(_env_value("EMBEDDING_BATCH_SIZE", 8)),
        cors_origins=_env_value(
            "CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
        ),
        deepseek_api_key=_env_value(
            "DEEPSEEK_API_KEY", external.get("DEEPSEEK_API_KEY", "")
        ),
        deepseek_base_url=_env_value(
            "DEEPSEEK_BASE_URL", external.get("BASE_URL", "https://api.deepseek.com")
        ),
        deepseek_model=_env_value(
            "DEEPSEEK_MODEL", external.get("MODEL_NAME", "deepseek-chat")
        ),
        embedding_api_key=_env_value(
            "EMBEDDING_API_KEY", external.get("EMBEDDING_API_KEY", "")
        ),
        embedding_url=_env_value(
            "EMBEDDING_URL",
            external.get("EMBEDDING_URL", "https://api.siliconflow.cn/v1/embeddings"),
        ),
        embedding_model=_env_value(
            "EMBEDDING_MODEL", external.get("EMBEDDING_MODEL", "BAAI/bge-large-zh-v1.5")
        ),
    )


settings = load_settings()
