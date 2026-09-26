from __future__ import annotations

import pytest

from app.ml.agent import (
    FIXED_WORKFLOW_STEPS,
    agent_benchmark,
    plan_task,
)
from app.ml.rag import (
    metadata_matches,
    retrieval_benchmark_v2,
    retrieval_decision,
    rewrite_query,
)
from app.ml.report_validation import (
    benchmark_report_variants,
    report_validation_metrics,
)
from app.ml.vlm import (
    compare_vision_branches,
    normalize_vlm_payload,
)
from app.schemas.common import VisionResult
from app.services.medgemma import MedGemmaHTTPClient


def _vision(name: str, confidence_status: str = "accepted") -> VisionResult:
    return VisionResult(
        model_name=name,
        image_quality={"status": "passed", "is_usable": True},
        findings=[
            {
                "name": "肺部结节",
                "location": "右上肺野",
                "confidence": 0.8,
                "confidence_status": confidence_status,
            }
        ],
        impression="需要专业人员复核。",
        risk_level="moderate",
        needs_human_review=True,
    )


def test_vlm_payload_is_normalized_to_safe_shared_schema() -> None:
    result = normalize_vlm_payload(
        {
            "findings": [
                {
                    "name": "肺部结节",
                    "confidence": 0.72,
                    "confidence_status": "accepted",
                }
            ],
            "impression": "发现需要复核的区域。",
            "risk_level": "moderate",
        },
        model_name="medgemma-test",
        model_version="v1",
        dataset_version="test",
    )
    assert result.provider == "medgemma_remote"
    assert result.simulated is False
    assert result.needs_human_review is True
    assert "vlm_inference" in result.pipeline_stages


def test_vlm_branch_comparison_only_calls_explicit_consensus() -> None:
    comparison = compare_vision_branches(_vision("yolo"), _vision("vlm"))
    assert comparison["comparison"]["consensus_count"] == 1
    assert comparison["comparison"]["agreement"] == 1.0
    assert comparison["fusion_policy"]["does_not_replace_independent_outputs"] is True


def test_unconfigured_medgemma_client_abstains_without_image() -> None:
    result = MedGemmaHTTPClient().analyze(None)
    assert result.abstained is True
    assert result.needs_human_review is True
    assert result.image_quality["is_usable"] is False


def test_rag_rewrite_filter_decision_and_ndcg() -> None:
    assert rewrite_query("咳嗽", context={"modality": "chest_xray"}) == "咳嗽 chest_xray"
    assert metadata_matches({"modality": "chest_xray"}, {"modality": "chest_xray"})
    assert not metadata_matches({"modality": "ct"}, {"modality": "chest_xray"})
    decision = retrieval_decision(
        [{"evidence_id": "a", "content": "证据", "score": 0.1}],
        min_score=0.2,
    )
    assert decision["abstain"] is True
    metrics = retrieval_benchmark_v2(
        [{"relevant": ["a"], "retrieved": ["b", "a"]}],
        k_values=[2],
    )
    assert metrics["metrics"]["2"]["ndcg"] > 0
    assert metrics["metrics"]["2"]["abstain_rate"] == 0


def test_report_metrics_measure_grounding_and_unsupported_claims() -> None:
    rows = [
        {
            "variant": "prompt_only",
            "schema_valid": True,
            "reference_findings": ["肺部结节"],
            "reported_findings": ["肺部结节"],
            "relevant_evidence_ids": ["kb#1"],
            "cited_evidence_ids": ["kb#1"],
            "unsupported_claims": [],
            "fact_consistency": True,
            "hallucination": False,
        },
        {
            "variant": "rag_llm",
            "schema_valid": True,
            "reference_findings": ["肺部结节"],
            "reported_findings": ["肺部结节", "肺癌"],
            "relevant_evidence_ids": ["kb#1"],
            "cited_evidence_ids": [],
            "unsupported_claims": ["肺癌"],
            "fact_consistency": False,
            "hallucination": True,
        },
    ]
    metrics = report_validation_metrics(rows)
    assert metrics["finding_coverage"] == 1.0
    assert metrics["evidence_coverage"] == 0.5
    assert metrics["unsupported_claim_rate"] > 0
    grouped = benchmark_report_variants(rows)
    assert grouped["prompt_only"]["schema_valid_rate"] == 1.0
    assert grouped["rag_structured_findings"]["sample_count"] == 0


def test_agent_benchmark_keeps_fixed_workflow_explicit() -> None:
    plan = plan_task({"required_tools": ["knowledge.search"]}, variant="fixed_workflow")
    assert plan.steps == FIXED_WORKFLOW_STEPS
    metrics = agent_benchmark(
        [
            {
                "variant": "fixed_workflow",
                "task_success": True,
                "tool_selection_accuracy": True,
                "evidence_grounding": True,
                "hallucination": False,
                "agent_steps": len(FIXED_WORKFLOW_STEPS),
                "latency_ms": 10,
                "token_usage": 20,
            }
        ]
    )
    assert metrics["fixed_workflow"]["task_success"] == 1.0
    assert metrics["fixed_workflow"]["failure_rate"] == 0.0
