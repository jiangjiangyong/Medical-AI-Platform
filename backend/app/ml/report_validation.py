from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


REPORT_VARIANTS = (
    "prompt_only",
    "rag_llm",
    "rag_structured_findings",
    "rag_structured_findings_verification",
)


def _strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, Iterable) or isinstance(value, (bytes, dict)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _matches(reference: str, candidate: str) -> bool:
    left = reference.casefold().strip()
    right = candidate.casefold().strip()
    return bool(left and right and (left in right or right in left))


def _finding_coverage(reference: list[str], reported: list[str]) -> float:
    if not reference:
        return 1.0 if not reported else 0.0
    return sum(
        any(_matches(item, candidate) for candidate in reported)
        for item in reference
    ) / len(reference)


def _evidence_coverage(relevant: list[str], cited: list[str]) -> float:
    relevant_set = {item for item in relevant if item}
    cited_set = {item for item in cited if item}
    if not relevant_set:
        return 1.0 if not cited_set else 0.0
    return len(relevant_set & cited_set) / len(relevant_set)


def report_case_metrics(row: Mapping[str, Any]) -> dict[str, Any]:
    reference = _strings(row.get("reference_findings"))
    reported = _strings(row.get("reported_findings"))
    relevant_evidence = _strings(row.get("relevant_evidence_ids"))
    cited_evidence = _strings(row.get("cited_evidence_ids"))
    unsupported = _strings(row.get("unsupported_claims"))
    claim_count = len(_strings(row.get("reported_claims"))) or len(reported) or 1
    hallucination = row.get("hallucination")
    if hallucination is None:
        hallucination = bool(unsupported)
    fact_consistency = row.get("fact_consistency")
    if fact_consistency is None:
        fact_consistency = row.get("fact_check_passed", not bool(unsupported))
    return {
        "schema_valid": bool(row.get("schema_valid", False)),
        "fact_consistency": bool(fact_consistency),
        "finding_coverage": _finding_coverage(reference, reported),
        "evidence_coverage": _evidence_coverage(relevant_evidence, cited_evidence),
        "unsupported_claim_count": len(unsupported),
        "claim_count": claim_count,
        "hallucination": bool(hallucination),
        "unsupported_claims": unsupported,
    }


def report_validation_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    cases = list(rows)
    per_case = [report_case_metrics(row) for row in cases]
    count = len(per_case) or 1
    unsupported = sum(item["unsupported_claim_count"] for item in per_case)
    claims = sum(item["claim_count"] for item in per_case) or 1
    return {
        "sample_count": len(per_case),
        "schema_valid_rate": sum(item["schema_valid"] for item in per_case) / count,
        "fact_consistency": sum(item["fact_consistency"] for item in per_case) / count,
        "finding_coverage": sum(item["finding_coverage"] for item in per_case) / count,
        "evidence_coverage": sum(item["evidence_coverage"] for item in per_case) / count,
        "unsupported_claim_rate": unsupported / claims,
        "hallucination_rate": sum(item["hallucination"] for item in per_case) / count,
        "unsupported_claim_count": unsupported,
        "claim_count": claims if per_case else 0,
    }


def benchmark_report_variants(
    rows: Iterable[Mapping[str, Any]],
    *,
    variants: Iterable[str] = REPORT_VARIANTS,
) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {str(name): [] for name in variants}
    for row in rows:
        variant = str(row.get("variant", "")).strip()
        if variant in grouped:
            grouped[variant].append(row)
    return {
        variant: report_validation_metrics(case_rows)
        for variant, case_rows in grouped.items()
    }
