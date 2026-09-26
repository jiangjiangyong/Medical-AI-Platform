from __future__ import annotations

from typing import Any

from app.config import settings
from app.schemas.common import VisionResult


class FHIRAdapter:
    """Map platform entities into interoperable FHIR resources."""

    def diagnostic_report(
        self,
        *,
        case_id: str,
        patient_reference: str,
        report_id: str,
        report_status: str,
        report_text: str,
        study_reference: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "resourceType": "DiagnosticReport",
            "id": report_id,
            "status": "final" if report_status == "approved" else "preliminary",
            "code": {
                "coding": [
                    {
                        "system": "http://loinc.org",
                        "code": "18748-4",
                        "display": "Diagnostic imaging study",
                    }
                ],
                "text": "胸部影像辅助报告",
            },
            "subject": {"reference": patient_reference},
            "conclusion": report_text,
            "extension": [
                {
                    "url": "https://medical-imaging-platform.example/fhir/StructureDefinition/case-id",
                    "valueString": case_id,
                },
                {
                    "url": "https://medical-imaging-platform.example/fhir/StructureDefinition/report-model",
                    "valueString": settings.vision_model_name,
                },
            ],
        }
        if study_reference:
            result["imagingStudy"] = {"reference": study_reference}
        return result

    def observations(
        self,
        *,
        patient_reference: str,
        vision: VisionResult,
    ) -> list[dict[str, Any]]:
        observations: list[dict[str, Any]] = []
        for index, finding in enumerate(vision.findings, start=1):
            observations.append(
                {
                    "resourceType": "Observation",
                    "id": f"{vision.model_version}-{index}",
                    "status": "preliminary",
                    "code": {"text": finding.name},
                    "subject": {"reference": patient_reference},
                    "valueCodeableConcept": {"text": finding.confidence_status},
                    "component": [
                        {"code": {"text": "calibrated-confidence"}, "valueDecimal": finding.confidence},
                        {"code": {"text": "location"}, "valueString": finding.location},
                    ],
                    "note": [{"text": finding.evidence}],
                }
            )
        return observations
