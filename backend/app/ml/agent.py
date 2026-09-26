from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Any


AGENT_VARIANTS = (
    "fixed_workflow",
    "single_agent",
    "supervisor_multi_agent",
)

FIXED_WORKFLOW_STEPS = (
    "load_case",
    "quality_gate",
    "vision",
    "retrieve",
    "draft_report",
    "verify_report",
    "human_review",
)

TRACE_SCHEMA_VERSION = "agent-observed-trace.v1"


@dataclass
class AgentTraceRecorder:
    """Collect an observed execution trace without storing image bytes."""

    variant: str
    model_name: str = "unknown"
    simulated: bool = False

    def __post_init__(self) -> None:
        self._started = perf_counter()
        self._events: list[dict[str, Any]] = []

    def record(
        self,
        step: str,
        *,
        tool: str = "",
        status: str = "completed",
        started_at_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._events.append(
            {
                "step": str(step),
                "tool": str(tool or step),
                "status": str(status),
                "duration_ms": round(
                    max(0.0, perf_counter() - self._started) * 1000, 3
                ),
                "started_at_ms": started_at_ms,
                "metadata": dict(metadata or {}),
            }
        )

    @property
    def events(self) -> list[dict[str, Any]]:
        return [dict(event) for event in self._events]

    def to_dict(self, *, trace_id: str = "") -> dict[str, Any]:
        return {
            "schema_version": TRACE_SCHEMA_VERSION,
            "trace_id": trace_id,
            "variant": self.variant,
            "model_name": self.model_name,
            "simulated": self.simulated,
            "events": self.events,
            "observed_steps": [event["step"] for event in self._events],
            "observed_tools": [event["tool"] for event in self._events],
        }


def expected_trace(variant: str, task: Mapping[str, Any] | None = None) -> list[str]:
    """Return the ordered step contract used to judge an observed trace."""
    return list(plan_task(task or {}, variant=variant).steps)


def _trace_steps(value: Any) -> list[str]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    steps: list[str] = []
    for item in value:
        if isinstance(item, Mapping):
            item = item.get("step") or item.get("tool") or ""
        text = str(item).strip()
        if text:
            steps.append(text)
    return steps


def trace_alignment(expected: Any, observed: Any) -> dict[str, Any]:
    expected_steps = _trace_steps(expected)
    observed_steps = _trace_steps(observed)
    if not expected_steps:
        return {
            "expected_count": 0,
            "observed_count": len(observed_steps),
            "matched_count": 0,
            "coverage": 0.0 if observed_steps else 1.0,
            "exact_match": not observed_steps,
        }
    matched = sum(
        1 for index, step in enumerate(expected_steps) if index < len(observed_steps) and observed_steps[index] == step
    )
    return {
        "expected_count": len(expected_steps),
        "observed_count": len(observed_steps),
        "matched_count": matched,
        "coverage": matched / len(expected_steps),
        "exact_match": expected_steps == observed_steps,
    }


@dataclass(frozen=True)
class AgentPlan:
    variant: str
    steps: tuple[str, ...]
    selected_tools: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "steps": list(self.steps),
            "selected_tools": list(self.selected_tools),
        }


def _unique(values: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return tuple(result)


def plan_task(
    task: Mapping[str, Any],
    *,
    variant: str,
) -> AgentPlan:
    if variant not in AGENT_VARIANTS:
        raise ValueError(f"unknown agent variant: {variant}")
    required = _unique(
        task.get("required_tools", ())
        if isinstance(task.get("required_tools", ()), Iterable)
        else ()
    )
    if variant == "fixed_workflow":
        steps = FIXED_WORKFLOW_STEPS
        selected = _unique(
            (
                "case.read",
                "vision.analyze",
                "knowledge.search",
                "report.generate",
                "report.verify",
                "review.request",
            )
        )
    elif variant == "single_agent":
        steps = _unique(("agent.route", *required, "agent.verify", "human_review"))
        selected = _unique(required or ("case.read", "vision.analyze", "knowledge.search"))
    else:
        steps = _unique(
            (
                "supervisor.route",
                "vision_agent",
                "retrieval_agent",
                "report_agent",
                "verification_agent",
                "supervisor.aggregate",
                "human_review",
            )
        )
        selected = _unique(required or ("vision.analyze", "knowledge.search", "report.verify"))
    return AgentPlan(variant=variant, steps=steps, selected_tools=selected)


def agent_case_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    steps = float(row.get("agent_steps", 0) or 0)
    latency = float(row.get("latency_ms", 0) or 0)
    tokens = float(row.get("token_usage", 0) or 0)
    trace = trace_alignment(
        row.get("expected_tool_trace", []), row.get("observed_tool_trace", [])
    )
    return {
        "task_success": bool(row.get("task_success", False)),
        "tool_selection_accuracy": bool(
            row.get("tool_selection_accuracy", row.get("tool_selection_correct", False))
        ),
        "evidence_grounding": bool(row.get("evidence_grounding", False)),
        "hallucination": bool(row.get("hallucination", False)),
        "agent_steps": steps,
        "latency_ms": latency,
        "token_usage": tokens,
        "failure": bool(row.get("failure", not bool(row.get("task_success", False)))),
        "observed_trace_present": bool(trace["observed_count"]),
        "trace_coverage": float(trace["coverage"]),
        "trace_exact_match": bool(trace["exact_match"]),
    }


def agent_benchmark(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {name: [] for name in AGENT_VARIANTS}
    for row in rows:
        variant = str(row.get("variant", "")).strip()
        if variant in grouped:
            grouped[variant].append(agent_case_metrics(row))
    result: dict[str, Any] = {}
    for variant, cases in grouped.items():
        count = len(cases) or 1
        result[variant] = {
            "sample_count": len(cases),
            "task_success": sum(item["task_success"] for item in cases) / count,
            "tool_selection_accuracy": sum(
                item["tool_selection_accuracy"] for item in cases
            ) / count,
            "evidence_grounding": sum(item["evidence_grounding"] for item in cases) / count,
            "hallucination_rate": sum(item["hallucination"] for item in cases) / count,
            "agent_steps": sum(item["agent_steps"] for item in cases) / count,
            "latency_ms": sum(item["latency_ms"] for item in cases) / count,
            "token_usage": sum(item["token_usage"] for item in cases) / count,
            "failure_rate": sum(item["failure"] for item in cases) / count,
            "observed_trace_presence_rate": sum(
                item["observed_trace_present"] for item in cases
            ) / count,
            "trace_coverage": sum(item["trace_coverage"] for item in cases) / count,
            "trace_exact_match_rate": sum(item["trace_exact_match"] for item in cases) / count,
        }
    return result
