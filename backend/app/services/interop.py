from __future__ import annotations

from typing import Any, Mapping

from app.schemas.common import VisionResult
from app.services.dicom import DICOMwebAdapter
from app.services.fhir import FHIRAdapter


def build_fhir_bundle(
    *,
    case_id: str,
    patient_reference: str,
    report_id: str,
    report_status: str,
    report_text: str,
    vision: VisionResult,
    study_reference: str | None = None,
) -> dict[str, Any]:
    """Build a standard FHIR collection Bundle from the platform's traceable result."""
    adapter = FHIRAdapter()
    report = adapter.diagnostic_report(
        case_id=case_id,
        patient_reference=patient_reference,
        report_id=report_id,
        report_status=report_status,
        report_text=report_text,
        study_reference=study_reference,
    )
    observations = adapter.observations(
        patient_reference=patient_reference,
        vision=vision,
    )
    resources = [report, *observations]
    entries = [
        {
            "fullUrl": f"urn:uuid:{resource.get('id', f'resource-{index}')}",
            "resource": resource,
        }
        for index, resource in enumerate(resources, start=1)
    ]
    return {
        "resourceType": "Bundle",
        "type": "collection",
        "entry": entries,
        "meta": {
            "tag": [
                {
                    "system": "https://medical-imaging-platform.example/fhir/tags",
                    "code": "demo-interoperability-payload",
                    "display": "标准互操作 Demo 载荷，未证明真实医院连接",
                }
            ]
        },
    }


def validate_fhir_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    issues: list[str] = []
    if bundle.get("resourceType") != "Bundle":
        issues.append("resourceType_must_be_Bundle")
    if bundle.get("type") != "collection":
        issues.append("bundle_type_must_be_collection")
    entries = bundle.get("entry")
    if not isinstance(entries, list) or not entries:
        issues.append("entry_must_be_non_empty_list")
        entries = []
    resource_types: list[str] = []
    full_urls: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, Mapping):
            issues.append(f"entry_{index}_must_be_object")
            continue
        resource = entry.get("resource")
        if not isinstance(resource, Mapping):
            issues.append(f"entry_{index}_resource_missing")
            continue
        resource_type = str(resource.get("resourceType", "")).strip()
        if not resource_type:
            issues.append(f"entry_{index}_resource_type_missing")
        else:
            resource_types.append(resource_type)
        full_url = str(entry.get("fullUrl", "")).strip()
        if not full_url:
            issues.append(f"entry_{index}_full_url_missing")
        elif full_url in full_urls:
            issues.append(f"entry_{index}_duplicate_full_url")
        full_urls.add(full_url)
    required_types = {"DiagnosticReport", "Observation"}
    missing_types = sorted(required_types - set(resource_types))
    issues.extend(f"required_resource_missing:{item}" for item in missing_types)
    return {
        "valid": not issues,
        "issues": issues,
        "resource_count": len(entries),
        "resource_types": resource_types,
        "external_connectivity_validated": False,
    }


def interoperability_contract() -> dict[str, Any]:
    return {
        "dicom": {
            "standard": "DICOMweb",
            "operations": ["QIDO-RS query", "WADO-RS fetch", "STOW-RS store"],
            "configured": DICOMwebAdapter().configured,
        },
        "fhir": {
            "standard": "FHIR R4",
            "resources": ["Bundle", "DiagnosticReport", "Observation"],
            "payload_validation": "local_schema_contract",
        },
        "scope": "standard-adapter-demo",
        "hospital_connectivity_claim": False,
    }
