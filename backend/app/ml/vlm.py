from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from pydantic import ValidationError

from app.schemas.common import VisionResult


VLM_ABLATIONS = (
    "yolo_only",
    "vlm_only",
    "yolo_plus_vlm_independent",
    "yolo_plus_vlm_fused",
    "yolo_plus_vlm_rerank",
    "yolo_plus_vlm_verified",
)


def normalize_vlm_payload(
    payload: Mapping[str, Any],
    *,
    model_name: str,
    model_version: str,
    dataset_version: str,
) -> VisionResult:
    """Validate a private VLM response without allowing free-form findings."""
    if not isinstance(payload, Mapping):
        raise ValueError("VLM response must be a JSON object")
    candidate = payload.get("result", payload)
    if not isinstance(candidate, Mapping):
        raise ValueError("VLM response result must be a JSON object")
    data = dict(candidate)
    data.setdefault("model_name", model_name)
    data.setdefault("model_version", model_version)
    data.setdefault("dataset_version", dataset_version)
    data.setdefault("task_type", "detection")
    data.setdefault("image_quality", {"status": "passed", "is_usable": True})
    data.setdefault("findings", [])
    data.setdefault("impression", "真实视觉模型未提供可确认的影像印象。")
    data.setdefault("risk_level", "indeterminate")
    data.setdefault("needs_human_review", True)
    data.setdefault("calibration_version", "uncalibrated")
    data.setdefault("abstained", False)
    data.setdefault("rejected_findings", [])
    data.setdefault("pipeline_stages", [])
    data.setdefault("limitations", [])
    data["provider"] = "medgemma_remote"
    data["simulated"] = False
    data["needs_human_review"] = True
    stages = data.get("pipeline_stages")
    if not isinstance(stages, (list, tuple)):
        stages = []
    data["pipeline_stages"] = list(
        dict.fromkeys(["image_quality_gate", "vlm_inference", *stages])
    )
    try:
        return VisionResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError("VLM response does not match VisionResult schema") from exc


def _finding_key(name: Any) -> str:
    return str(name or "").strip().casefold()


def _accepted_names(result: VisionResult) -> set[str]:
    return {
        _finding_key(finding.name)
        for finding in result.findings
        if finding.confidence_status == "accepted" and finding.name
    }


def compare_vision_branches(
    yolo: VisionResult,
    vlm: VisionResult,
) -> dict[str, Any]:
    """Keep branch outputs independent and expose only explicit agreement."""
    yolo_names = _accepted_names(yolo)
    vlm_names = _accepted_names(vlm)
    union = yolo_names | vlm_names
    intersection = yolo_names & vlm_names
    return {
        "schema_version": "vision-branches.v1",
        "branches": {
            "yolo": yolo.model_dump(mode="json"),
            "vlm": vlm.model_dump(mode="json"),
        },
        "comparison": {
            "yolo_accepted_count": len(yolo_names),
            "vlm_accepted_count": len(vlm_names),
            "consensus_count": len(intersection),
            "agreement": len(intersection) / len(union) if union else 1.0,
            "yolo_only": sorted(yolo_names - vlm_names),
            "vlm_only": sorted(vlm_names - yolo_names),
            "risk_level_match": yolo.risk_level == vlm.risk_level,
            "abstention_match": yolo.abstained == vlm.abstained,
        },
        "fusion_policy": {
            "name": "conservative_consensus_v1",
            "rule": "A finding is consensus only when both branches explicitly accept the same normalized name.",
            "does_not_replace_independent_outputs": True,
        },
    }


def ablation_plan() -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "description": description,
            "requires": requirements,
        }
        for name, description, requirements in (
            ("yolo_only", "YOLO branch alone", ["yolo_checkpoint", "frozen_test"]),
            ("vlm_only", "VLM branch alone", ["vlm_service", "frozen_test"]),
            (
                "yolo_plus_vlm_independent",
                "Two branches reported independently",
                ["yolo_checkpoint", "vlm_service", "frozen_test"],
            ),
            (
                "yolo_plus_vlm_fused",
                "Conservative name-level consensus fusion",
                ["yolo_checkpoint", "vlm_service", "frozen_test"],
            ),
            (
                "yolo_plus_vlm_rerank",
                "VLM evidence used to rerank YOLO candidates",
                ["yolo_checkpoint", "vlm_service", "frozen_test", "fusion_policy"],
            ),
            (
                "yolo_plus_vlm_verified",
                "Fusion followed by schema/fact/evidence verification",
                [
                    "yolo_checkpoint",
                    "vlm_service",
                    "frozen_test",
                    "report_verifier",
                ],
            ),
        )
    ]


@dataclass(frozen=True)
class VLMInputAudit:
    missing: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.missing

    def to_dict(self) -> dict[str, Any]:
        return {"ready": self.ready, "missing": list(self.missing)}


def audit_vlm_inputs(
    *,
    yolo_checkpoint: str = "",
    vlm_base_url: str = "",
    frozen_test_manifest: str = "",
    report_verifier: bool = False,
) -> VLMInputAudit:
    missing: list[str] = []
    if not yolo_checkpoint:
        missing.append("yolo_checkpoint")
    if not vlm_base_url:
        missing.append("vlm_service")
    if not frozen_test_manifest:
        missing.append("frozen_test")
    if not report_verifier:
        missing.append("report_verifier")
    return VLMInputAudit(tuple(missing))
