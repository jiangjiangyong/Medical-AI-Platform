from __future__ import annotations

import json

import pytest

from app.ml.benchmarks import BENCHMARK_NAMES, build_status_metrics
from app.ml.experiment import ARTIFACT_FILENAMES, ExperimentMetadata, ExperimentStore


def _metadata(experiment_id: str = "test-vision-001") -> ExperimentMetadata:
    return ExperimentMetadata(
        dataset_version="dataset-v0",
        model_version="model-v0",
        prompt_version="prompt-v0",
        retriever_version="retriever-v0",
        knowledge_base_version="kb-v0",
        code_commit="abc123",
        experiment_id=experiment_id,
    )


def test_benchmark_suite_contains_all_required_tracks() -> None:
    assert BENCHMARK_NAMES == ("vision", "retrieval", "report", "agent", "system")


def test_blocked_metrics_never_claim_comparable_results() -> None:
    metrics = build_status_metrics(
        "vision",
        status="blocked",
        reason="real data is not available",
    )
    assert metrics["valid_for_comparison"] is False
    assert metrics["metrics_available"] is False
    assert metrics["sample_count"] is None
    assert metrics["metrics"] == {}


def test_experiment_store_writes_fixed_artifact_contract(tmp_path) -> None:
    metadata = _metadata()
    config = {
        **metadata.to_dict(),
        "benchmark": "vision",
        "status": "blocked",
        "reason": "real data is not available",
    }
    artifacts = ExperimentStore(tmp_path).create(
        config=config,
        metrics=build_status_metrics(
            "vision",
            status="blocked",
            reason="real data is not available",
        ),
        environment={"code_commit": "abc123"},
        error_cases=[],
        summary="blocked",
    )

    assert {path.name for path in artifacts.root.iterdir()} == set(ARTIFACT_FILENAMES)
    stored_config = json.loads(artifacts.config_path.read_text(encoding="utf-8"))
    assert stored_config["experiment_id"] == "test-vision-001"
    assert stored_config["dataset_version"] == "dataset-v0"
    assert artifacts.error_cases_path.read_text(encoding="utf-8") == ""


def test_experiment_id_rejects_path_traversal() -> None:
    with pytest.raises(ValueError):
        _metadata("../outside")
