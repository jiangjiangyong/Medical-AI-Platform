from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from app.ml.calibration import CalibrationBundle, expected_calibration_error
from app.ml.manifest import ManifestRecord, class_balance_summary, patient_level_split
from app.ml.metrics import BoundingBox, binary_classification_metrics, detection_map
from app.ml.quality import QualityGateConfig, assess_image_quality
from app.ml.retrieval import reciprocal_rank_fusion, retrieval_benchmark
from app.services.modelops import distribution_psi
from app.schemas.common import VisionResult
from app.services.report_guard import validate_professional_report


def _records() -> list[ManifestRecord]:
    return [
        ManifestRecord(
            image_path=f"p{patient}-image.png",
            patient_id=f"p{patient}",
            labels={"opacity": float(patient % 2 == 0)},
        )
        for patient in range(1, 10)
    ]


def test_patient_level_split_keeps_independent_patients() -> None:
    splits = patient_level_split(_records(), seed=7)
    owners = {record.patient_id: split for split, records in splits.items() for record in records}
    assert set(splits) == {"train", "valid", "test"}
    assert all(splits.values())
    assert len(owners) == 9
    assert all(record.split == split for split, records in splits.items() for record in records)
    assert class_balance_summary(_records())["classes"]["opacity"]["positive"] == 4


def test_patient_level_split_preserves_balance_for_unique_patients() -> None:
    records = [
        ManifestRecord(
            image_path=f"unique-{index}.png",
            patient_id=f"unique-patient-{index}",
            labels={"opacity": float(index < 20)},
        )
        for index in range(100)
    ]

    splits = patient_level_split(records, seed=11)
    assert {split: len(rows) for split, rows in splits.items()} == {
        "train": 70,
        "valid": 15,
        "test": 15,
    }
    assert {
        split: sum(record.labels["opacity"] >= 0.5 for record in rows)
        for split, rows in splits.items()
    } == {"train": 14, "valid": 3, "test": 3}


def test_quality_gate_passes_good_image_and_rejects_blank_image(tmp_path: Path) -> None:
    good = np.tile(np.arange(768, dtype=np.uint16), (768, 1)).astype(np.uint8)
    good_path = tmp_path / "good.png"
    Image.fromarray(good, mode="L").save(good_path)
    passed = assess_image_quality(good_path, QualityGateConfig(min_sharpness=0))
    assert passed.status == "passed"
    assert passed.is_usable is True

    blank_path = tmp_path / "blank.png"
    Image.fromarray(np.zeros((768, 768), dtype=np.uint8), mode="L").save(blank_path)
    rejected = assess_image_quality(blank_path)
    assert rejected.is_usable is False
    assert "contrast_below_threshold" in rejected.reasons


def test_classification_metrics_and_detection_map() -> None:
    metrics = binary_classification_metrics([0, 1, 1, 0], [0.1, 0.8, 0.7, 0.2])
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["auroc"] == 1.0

    predictions = {
        "study-1": [BoundingBox("nodule", 10, 10, 30, 30, confidence=0.9)],
    }
    targets = {
        "study-1": [BoundingBox("nodule", 10, 10, 30, 30)],
    }
    detection = detection_map(predictions, targets, ["nodule"], iou_thresholds=[0.5])
    assert detection["map50"] == 1.0


def test_calibration_and_low_confidence_rejection() -> None:
    bundle = CalibrationBundle.fit(
        [0.55, 0.60, 0.65, 0.95, 0.10, 0.20, 0.30, 0.40],
        [0, 0, 1, 1, 0, 0, 0, 1],
        version="calibration-test",
    )
    assert bundle.version == "calibration-test"
    assert abs(expected_calibration_error([0.1, 0.9], [0, 1]) - 0.1) < 1e-9
    assert bundle.decide(0.05).status == "rejected"
    assert bundle.decide(0.05).abstained is True


def test_report_guard_keeps_uncertain_finding_uncertain() -> None:
    vision = VisionResult(
        model_name="test-model",
        image_quality={"status": "passed", "is_usable": True},
        findings=[
            {
                "name": "肺部结节",
                "location": "右上肺野",
                "confidence": 0.4,
                "confidence_status": "uncertain",
            }
        ],
        impression="存在需要复核的可疑区域。",
        risk_level="moderate",
        needs_human_review=True,
    )
    fallback = {
        "imaging_findings": ["待复核候选：肺部结节"],
        "preliminary_assessment": "不能仅凭当前结果确定病因。",
        "recommendations": ["由专业人员复核"],
        "risk_level": "moderate",
        "human_review_required": True,
        "safety_note": "本内容为辅助决策草稿。",
        "evidence_summary": ["肺部影像异常需要结合原始影像复核。"],
    }
    guarded = validate_professional_report(
        {**fallback, "imaging_findings": ["已确诊肺部结节"]},
        fallback,
        vision,
        [{"source_name": "test.md", "chunk_index": 0, "content": "肺部结节需要复核。", "score": 0.8}],
    )
    assert guarded.fact_check_passed is False
    assert guarded.used_fallback is True
    assert "待复核" in guarded.data["imaging_findings"][0]


def test_hybrid_retrieval_metrics_and_drift_metric() -> None:
    fused = reciprocal_rank_fusion(
        {"dense": ["a", "b"], "bm25": ["b", "a"]},
        k=60,
    )
    assert fused["a"] == fused["b"]
    benchmark = retrieval_benchmark(
        [
            {"relevant": ["a"], "retrieved": ["b", "a"]},
            {"relevant": ["b"], "retrieved": ["b", "c"]},
        ],
        k_values=[1, 2],
    )
    assert benchmark["metrics"]["1"]["hit_rate"] == 0.5
    assert benchmark["metrics"]["2"]["recall_at_k"] == 1.0
    assert distribution_psi([0.1, 0.2, 0.3], [0.1, 0.2, 0.3]) == 0.0


def test_patient_level_calibration_split_is_deterministic_and_disjoint() -> None:
    from app.ml.manifest import patient_level_calibration_split

    records = [
        ManifestRecord(
            image_path=f"image-{patient}-{view}.png",
            patient_id=f"patient-{patient}",
            labels={"opacity": float(patient % 2 == 0)},
            split="valid",
        )
        for patient in range(20)
        for view in range(2)
    ]
    first = patient_level_calibration_split(records, fit_ratio=0.6, seed=17)
    second = patient_level_calibration_split(records, fit_ratio=0.6, seed=17)

    first_ids = {
        split: [record.patient_id for record in rows]
        for split, rows in first.items()
    }
    second_ids = {
        split: [record.patient_id for record in rows]
        for split, rows in second.items()
    }
    assert first_ids == second_ids
    assert set(first_ids["calibration_fit"]).isdisjoint(
        first_ids["calibration_eval"]
    )
    assert all(record.split == "valid" for record in records)
    assert all(
        record.split == split
        for split, rows in first.items()
        for record in rows
    )


def test_reliability_and_selective_curves_keep_empty_bins_and_risk() -> None:
    from app.ml.calibration import (
        reliability_bins,
        selective_coverage_risk_curve,
        select_precision_threshold,
    )

    bins = reliability_bins([0.1, 0.9], [0, 1], bins=4)
    assert len(bins) == 4
    assert bins[0]["count"] == 1
    assert bins[1]["count"] == 0
    assert bins[-1]["count"] == 1

    curve = selective_coverage_risk_curve([0.1, 0.9], [0, 1])
    assert curve[0]["coverage"] == 1.0
    assert curve[0]["risk"] == 0.0

    policy = select_precision_threshold(
        [0.1, 0.8, 0.9],
        [0, 1, 1],
        target_precision=0.8,
        min_events=1,
    )
    assert policy["constraint_met"] is True
    assert policy["threshold"] <= 0.2
