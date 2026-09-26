from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from app.ml.experiment import VALID_STATUSES


@dataclass(frozen=True)
class BenchmarkDefinition:
    name: str
    purpose: str
    primary_metrics: tuple[str, ...]
    required_inputs: tuple[str, ...]
    frozen_split: str


BENCHMARK_DEFINITIONS: dict[str, BenchmarkDefinition] = {
    "vision": BenchmarkDefinition(
        name="vision",
        purpose="Evaluate image quality, classification or detection quality on a frozen patient-level test set.",
        primary_metrics=(
            "precision",
            "recall",
            "f1",
            "mAP50",
            "mAP50-95",
            "per_class_metrics",
            "confusion_matrix",
            "false_positive_analysis",
            "false_negative_analysis",
        ),
        required_inputs=(
            "patient-level train/valid/test manifest",
            "frozen test annotations",
            "model checkpoint",
            "prediction rows",
        ),
        frozen_split="test",
    ),
    "retrieval": BenchmarkDefinition(
        name="retrieval",
        purpose="Measure whether relevant medical evidence is retrieved before report generation.",
        primary_metrics=(
            "recall@k",
            "MRR",
            "hit_rate",
            "nDCG",
            "abstain_rate",
        ),
        required_inputs=(
            "query set",
            "relevant evidence labels",
            "knowledge-base version",
            "retrieval outputs",
        ),
        frozen_split="retrieval_test",
    ),
    "report": BenchmarkDefinition(
        name="report",
        purpose="Measure structured report quality, grounding, schema validity and unsupported claims.",
        primary_metrics=(
            "finding_coverage",
            "fact_consistency",
            "evidence_coverage",
            "hallucination_rate",
            "schema_valid_rate",
            "unsupported_claim_rate",
        ),
        required_inputs=(
            "frozen case prompts",
            "reference findings or report labels",
            "retrieved evidence",
            "generated structured reports",
        ),
        frozen_split="report_test",
    ),
    "agent": BenchmarkDefinition(
        name="agent",
        purpose="Compare fixed workflow, single-agent and multi-agent execution on the same tasks.",
        primary_metrics=(
            "task_success",
            "tool_selection_accuracy",
            "evidence_grounding",
            "hallucination_rate",
            "agent_steps",
            "latency_ms",
            "token_usage",
            "failure_rate",
        ),
        required_inputs=(
            "frozen task set",
            "expected tool traces",
            "workflow or agent execution traces",
            "grounding labels",
        ),
        frozen_split="agent_test",
    ),
    "system": BenchmarkDefinition(
        name="system",
        purpose="Measure service reliability and inference cost before and after engineering changes.",
        primary_metrics=(
            "p50_latency_ms",
            "p95_latency_ms",
            "throughput",
            "gpu_memory_mb",
            "cold_start_ms",
            "failure_rate",
        ),
        required_inputs=(
            "fixed request mix",
            "service version",
            "load-test traces",
            "resource telemetry",
        ),
        frozen_split="system_test",
    ),
}

BENCHMARK_NAMES = tuple(BENCHMARK_DEFINITIONS)


def get_benchmark_definition(name: str) -> BenchmarkDefinition:
    try:
        return BENCHMARK_DEFINITIONS[str(name)]
    except KeyError as exc:
        choices = ", ".join(BENCHMARK_NAMES)
        raise ValueError(f"unknown benchmark {name!r}; expected one of: {choices}") from exc


def build_status_metrics(
    benchmark: str,
    *,
    status: str,
    reason: str,
) -> dict[str, Any]:
    definition = get_benchmark_definition(benchmark)
    if status not in VALID_STATUSES:
        raise ValueError(f"unsupported experiment status: {status}")
    if status != "completed" and not reason.strip():
        raise ValueError("a non-completed benchmark must include a reason")
    return {
        "benchmark": definition.name,
        "status": status,
        "valid_for_comparison": status == "completed",
        "reason": reason,
        "sample_count": None,
        "metrics_available": False,
        "metrics": {},
    }


def render_summary(
    config: Mapping[str, Any],
    metrics: Mapping[str, Any],
    definition: BenchmarkDefinition,
) -> str:
    status = str(metrics.get("status", config.get("status", "not_run")))
    reason = str(metrics.get("reason", ""))
    lines = [
        f"# Experiment {config.get('experiment_id', 'unknown')}",
        "",
        f"- Benchmark: {definition.name}",
        f"- Status: {status}",
        f"- Dataset version: {config.get('dataset_version', 'unknown')}",
        f"- Model version: {config.get('model_version', 'unknown')}",
        f"- Code commit: {config.get('code_commit', 'unknown')}",
        "",
        "## Decision",
        "",
    ]
    if status == "completed":
        lines.append("This artifact is eligible for comparison only after the frozen test evaluation is verified.")
    else:
        lines.append("No medical benchmark result is reported because the evaluation has not completed.")
    if reason:
        lines.extend(["", "## Reason", "", reason])
    lines.extend(
        [
            "",
            "## Required primary metrics",
            "",
            ", ".join(f"{metric}" for metric in definition.primary_metrics),
            "",
            "The fixed artifact layout is config.json, metrics.json, environment.json, "
            "error_cases.jsonl, and summary.md.",
        ]
    )
    return "\n".join(lines)
