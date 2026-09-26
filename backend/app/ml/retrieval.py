from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Mapping


TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+")


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall((text or "").lower())


def bm25_score(
    query: str,
    document: str,
    corpus: Iterable[str],
    *,
    k1: float = 1.5,
    b: float = 0.75,
) -> float:
    documents = [tokenize(item) for item in corpus]
    tokens = tokenize(document)
    query_tokens = set(tokenize(query))
    if not tokens or not query_tokens or not documents:
        return 0.0
    document_frequency = Counter(
        token for item in documents for token in set(item)
    )
    average_length = sum(len(item) for item in documents) / len(documents)
    term_frequency = Counter(tokens)
    score = 0.0
    for term in query_tokens:
        frequency = term_frequency.get(term, 0)
        if not frequency:
            continue
        df = document_frequency.get(term, 0)
        idf = math.log(1.0 + (len(documents) - df + 0.5) / (df + 0.5))
        denominator = frequency + k1 * (1.0 - b + b * len(tokens) / max(1.0, average_length))
        score += idf * frequency * (k1 + 1.0) / denominator
    return score


def reciprocal_rank_fusion(
    rankings: Mapping[str, list[str]],
    *,
    k: int = 60,
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings.values():
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return scores


def filter_evidence(
    evidence: Iterable[Mapping[str, Any]],
    *,
    limit: int = 4,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    filtered: list[dict[str, Any]] = []
    for item in sorted(evidence, key=lambda value: float(value.get("score", 0.0) or 0.0), reverse=True):
        content = str(item.get("content", "")).strip()
        identifier = str(item.get("evidence_id") or f"{item.get('source_name')}#{item.get('chunk_index')}")
        if not content or identifier in seen or float(item.get("score", 0.0) or 0.0) < min_score:
            continue
        seen.add(identifier)
        filtered.append(dict(item))
        if len(filtered) >= limit:
            break
    for rank, item in enumerate(filtered, start=1):
        item["rank"] = rank
    return filtered


def retrieval_benchmark(
    cases: Iterable[Mapping[str, Any]],
    *,
    k_values: Iterable[int] = (1, 3, 5),
) -> dict[str, Any]:
    """Evaluate retrieved evidence against hand-labelled evidence IDs."""

    rows = list(cases)
    ks = sorted(set(int(value) for value in k_values if int(value) > 0))
    metrics: dict[str, dict[str, float]] = {}
    for k in ks:
        hit_values: list[float] = []
        recall_values: list[float] = []
        reciprocal_values: list[float] = []
        for row in rows:
            relevant = set(str(item) for item in row.get("relevant", []))
            retrieved = [str(item) for item in row.get("retrieved", [])]
            top_k = retrieved[:k]
            hits = [item for item in top_k if item in relevant]
            hit_values.append(float(bool(hits)))
            recall_values.append(len(set(hits)) / len(relevant) if relevant else 0.0)
            reciprocal_values.append(1.0 / (top_k.index(hits[0]) + 1) if hits else 0.0)
        metrics[str(k)] = {
            "hit_rate": sum(hit_values) / len(hit_values) if hit_values else 0.0,
            "recall_at_k": sum(recall_values) / len(recall_values) if recall_values else 0.0,
            "mrr": sum(reciprocal_values) / len(reciprocal_values) if reciprocal_values else 0.0,
        }
    return {"queries": len(rows), "metrics": metrics}
