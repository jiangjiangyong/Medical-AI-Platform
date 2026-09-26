from __future__ import annotations

import asyncio

import pytest

from app.db.session import SessionLocal
from app.ml.mlops import drift_report, promotion_gate, shadow_evaluation
from app.ml.performance import performance_gate, summarize_performance
from app.schemas.common import VisionFinding, VisionResult
from app.services.dicom import DICOMError, DICOMwebAdapter
from app.services.modelops import record_vision_telemetry
from app.services.interop import build_fhir_bundle, validate_fhir_bundle
from app.services.resilience import CircuitBreaker, CircuitOpenError, ResiliencePolicy, call_with_resilience


def _vision(model_name: str) -> VisionResult:
    return VisionResult(
        model_name=model_name,
        model_version="v1",
        dataset_version="test",
        task_type="detection",
        provider="test",
        simulated=False,
        image_quality={"status": "passed", "is_usable": True},
        findings=[
            VisionFinding(
                name="opacity",
                location="right",
                confidence=0.9,
                confidence_status="accepted",
            )
        ],
        impression="review required",
        risk_level="moderate",
        needs_human_review=True,
    )


def test_fhir_bundle_contract_contains_required_resources() -> None:
    bundle = build_fhir_bundle(
        case_id="case",
        patient_reference="Patient/patient",
        report_id="report",
        report_status="pending_review",
        report_text="draft",
        vision=_vision("primary"),
    )
    validation = validate_fhir_bundle(bundle)
    assert validation["valid"] is True
    assert {"DiagnosticReport", "Observation"} <= set(validation["resource_types"])


def test_dicomweb_without_url_does_not_attempt_network() -> None:
    adapter = DICOMwebAdapter(base_url="")
    assert adapter.configured is False
    with pytest.raises(DICOMError):
        asyncio.run(adapter.query_studies())


def test_resilience_retries_and_opens_circuit() -> None:
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 2:
            raise RuntimeError("temporary")
        return "ok"

    breaker = CircuitBreaker(failure_threshold=1, recovery_seconds=60)
    assert call_with_resilience(
        flaky,
        policy=ResiliencePolicy(attempts=2, backoff_seconds=0),
        breaker=breaker,
    ) == "ok"
    assert calls["count"] == 2
    breaker.record_failure()
    with pytest.raises(CircuitOpenError):
        call_with_resilience(
            lambda: "unreachable",
            policy=ResiliencePolicy(attempts=1, backoff_seconds=0),
            breaker=breaker,
        )


def test_performance_summary_and_gate() -> None:
    metrics = summarize_performance(
        [
            {"latency_ms": 10, "status": "completed", "gpu_memory_mb": 100},
            {"latency_ms": 20, "status": "completed", "gpu_memory_mb": 120},
            {"latency_ms": 100, "status": "failed", "gpu_memory_mb": 130},
        ],
        duration_seconds=1,
    )
    assert metrics["sample_count"] == 3
    assert metrics["p95_latency_ms"] == 100.0
    gate = performance_gate(
        metrics,
        p95_budget_ms=150,
        failure_rate_budget=0.5,
    )
    assert gate["passed"] is True


def test_mlops_gates_and_shadow_metrics() -> None:
    drift = drift_report([0.1, 0.2, 0.3], [0.1, 0.2, 0.3])
    assert drift["status"] == "completed"
    shadow = shadow_evaluation(
        [{"primary": _vision("primary").model_dump(), "shadow": _vision("shadow").model_dump()}]
    )
    assert shadow["sample_count"] == 1
    gate = promotion_gate(
        {"task_success": 0.9},
        minimum_metrics={"task_success": 0.8},
        drift=drift,
    )
    assert gate["passed"] is True


def test_telemetry_preserves_provenance_and_gpu_memory() -> None:
    db = SessionLocal()
    try:
        telemetry = record_vision_telemetry(
            db,
            case_id=None,
            study_id=None,
            vision=_vision("primary"),
            status="completed",
            gpu_memory_mb=123.5,
        )
        db.flush()
        assert telemetry.gpu_memory_mb == 123.5
        assert telemetry.confidence_summary["simulated"] is False
        assert telemetry.confidence_summary["provider"] == "test"
        assert telemetry.confidence_summary["calibration_version"] == "uncalibrated"
    finally:
        db.rollback()
        db.close()
