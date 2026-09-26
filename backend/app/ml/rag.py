from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any

from app.ml.retrieval import filter_evidence


def rewrite_query(
    query: str,
    *,
    context: Mapping[str, Any] | None = None,
) -> str:
    """Normalize a query while preserving only user/case supplied terms."""
    parts = [str(query or "").strip()]
    for key in ("modality", "body_region", "symptoms", "clinical_context"):
        if context and context.get(key):
            value = context[key]
            if isinstance(value, Mapping):
                value = " ".join(
                    f"{name}:{item}" for name, item in sorted(value.items())
                )
            parts.append(str(value).strip())
    seen: set[str] = set()
    result: list[str] = []
    for part in parts:
        if not part:
            continue
        normalized = " ".join(part.split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return " ".join(result)


def metadata_matches(
    metadata: Mapping[str, Any] | None,
    filters: Mapping[str, Any] | None,
) -> bool:
    if not filters:
        return True
    metadata = metadata or {}
    for key, expected in filters.items():
        actual = metadata.get(key)
        if isinstance(expected, (list, tuple, set)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False
    return True


def retrieval_decision(
    evidence: Iterable[Mapping[str, Any]],
    *,
    limit: int = 4,
    min_score: float = 0.0,
) -> dict[str, Any]:
    selected = filter_evidence(evidence, limit=limit, min_score=min_score)
    top_score = max(
        (float(item.get("score", 0.0) or 0.0) for item in selected),
        default=0.0,
    )
    abstain = not selected or top_score < min_score
    return {
        "evidence": selected if not abstain else [],
        "abstain": abstain,
        "reason": "insufficient_evidence" if abstain else "evidence_available",
        "top_score": top_score,
    }


def _ndcg(retrieved: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    gains = sum(
        1.0 / math.log2(rank + 1)
        for rank, item in enumerate(retrieved[:k], start=1)
        if item in relevant
    )
    ideal = sum(
        1.0 / math.log2(rank + 1)
        for rank in range(1, min(k, len(relevant)) + 1)
    )
    return gains / ideal if ideal else 0.0


def retrieval_benchmark_v2(
    cases: Iterable[Mapping[str, Any]],
    *,
    k_values: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate hand-labelled retrieval cases with nDCG and abstention."""
    rows = list(cases)
    ks = sorted({int(value) for value in k_values if int(value) > 0})
    result: dict[str, dict[str, float]] = {}
    for k in ks:
        hits: list[float] = []
        recalls: list[float] = []
        reciprocal: list[float] = []
        ndcgs: list[float] = []
        abstains: list[float] = []
        for row in rows:
            relevant = {str(item) for item in row.get("relevant", [])}
            retrieved = [str(item) for item in row.get("retrieved", [])]
            top = retrieved[:k]
            matched = [item for item in top if item in relevant]
            hits.append(float(bool(matched)))
            recalls.append(len(set(matched)) / len(relevant) if relevant else 0.0)
            reciprocal.append(
                1.0 / (top.index(matched[0]) + 1) if matched else 0.0
            )
            ndcgs.append(_ndcg(retrieved, relevant, k))
            abstains.append(float(bool(row.get("abstain")) or not retrieved))
        denominator = len(rows) or 1
        result[str(k)] = {
            "hit_rate": sum(hits) / denominator,
            "recall_at_k": sum(recalls) / denominator,
            "mrr": sum(reciprocal) / denominator,
            "ndcg": sum(ndcgs) / denominator,
            "abstain_rate": sum(abstains) / denominator,
        }
    return {"queries": len(rows), "metrics": result}


def benchmark_retrieval_variants(
    rows: Iterable[Mapping[str, Any]],
    *,
    variants: Iterable[str] = ("bm25", "dense", "hybrid", "hybrid_reranker"),
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {str(name): [] for name in variants}
    for row in rows:
        retrieved_by_method = row.get("retrieved_by_method", {})
        for name in grouped:
            retrieved = (
                retrieved_by_method.get(name, [])
                if isinstance(retrieved_by_method, Mapping)
                else []
            )
            grouped[name].append(
                {
                    "relevant": row.get("relevant", []),
                    "retrieved": retrieved,
                    "abstain": not bool(retrieved),
                }
            )
    return {
        name: retrieval_benchmark_v2(cases)
        for name, cases in grouped.items()
    }
