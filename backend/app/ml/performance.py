from __future__ import annotations

from collections.abc import Iterable, Mapping
from math import ceil
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return default


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(_number(value) for value in values)
    if not ordered:
        return 0.0
    q = min(1.0, max(0.0, float(quantile)))
    index = min(len(ordered) - 1, max(0, ceil(q * len(ordered)) - 1))
    return round(ordered[index], 3)


def summarize_performance(
    rows: Iterable[Mapping[str, Any]],
    *,
    duration_seconds: float | None = None,
    cold_start_ms: float | None = None,
) -> dict[str, Any]:
    records = list(rows)
    latencies = [_number(row.get("latency_ms")) for row in records]
    memory = [
        _number(row.get("gpu_memory_mb"))
        for row in records
        if row.get("gpu_memory_mb") is not None
    ]
    successes = [
        row
        for row in records
        if str(row.get("status", "completed")).lower()
        in {"completed", "success", "ok"}
        and not bool(row.get("failure", False))
    ]
    sample_count = len(records)
    observed_seconds = _number(duration_seconds)
    if observed_seconds <= 0 and latencies:
        observed_seconds = sum(latencies) / 1000.0
    return {
        "sample_count": sample_count,
        "success_count": len(successes),
        "failure_rate": round(
            (sample_count - len(successes)) / sample_count if sample_count else 0.0,
            6,
        ),
        "p50_latency_ms": percentile(latencies, 0.50),
        "p95_latency_ms": percentile(latencies, 0.95),
        "mean_latency_ms": round(sum(latencies) / sample_count, 3) if sample_count else 0.0,
        "throughput_per_second": round(
            len(successes) / observed_seconds if observed_seconds > 0 else 0.0,
            6,
        ),
        "gpu_memory_mb": round(max(memory), 3) if memory else None,
        "cold_start_ms": (
            round(_number(cold_start_ms), 3) if cold_start_ms is not None else None
        ),
        "duration_seconds": round(observed_seconds, 6),
    }


def performance_gate(
    candidate: Mapping[str, Any],
    *,
    p95_budget_ms: float,
    failure_rate_budget: float,
    baseline: Mapping[str, Any] | None = None,
    max_p95_regression: float = 0.20,
    min_throughput_ratio: float = 0.80,
) -> dict[str, Any]:
    checks: dict[str, bool] = {
        "sample_available": int(candidate.get("sample_count", 0) or 0) > 0,
        "p95_within_budget": _number(candidate.get("p95_latency_ms"))
        <= _number(p95_budget_ms),
        "failure_rate_within_budget": _number(candidate.get("failure_rate"))
        <= min(1.0, max(0.0, float(failure_rate_budget))),
    }
    if baseline:
        baseline_p95 = _number(baseline.get("p95_latency_ms"))
        candidate_p95 = _number(candidate.get("p95_latency_ms"))
        baseline_throughput = _number(baseline.get("throughput_per_second"))
        candidate_throughput = _number(candidate.get("throughput_per_second"))
        checks["p95_not_regressed"] = (
            baseline_p95 <= 0
            or candidate_p95 <= baseline_p95 * (1.0 + max_p95_regression)
        )
        checks["throughput_not_regressed"] = (
            baseline_throughput <= 0
            or candidate_throughput >= baseline_throughput * min_throughput_ratio
        )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "p95_budget_ms": float(p95_budget_ms),
        "failure_rate_budget": float(failure_rate_budget),
        "baseline_compared": baseline is not None,
    }


def performance_status(metrics: Mapping[str, Any]) -> str:
    return "completed" if int(metrics.get("sample_count", 0) or 0) > 0 else "blocked"
