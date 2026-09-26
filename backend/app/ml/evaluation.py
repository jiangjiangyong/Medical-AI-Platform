from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Mapping

from app.ml.calibration import CalibrationBundle, calibration_summary
from app.ml.manifest import ManifestRecord
from app.ml.metrics import detection_map, multilabel_classification_metrics


def assert_independent_test_split(records: Iterable[ManifestRecord]) -> None:
    records = list(records)
    test_patients = {record.patient_id for record in records if record.split == "test"}
    overlap = {
        record.patient_id
        for record in records
        if record.split != "test" and record.patient_id in test_patients
    }
    if overlap:
        raise ValueError(f"test leakage detected for patients: {sorted(overlap)}")
    if not test_patients:
        raise ValueError("no independent test records found")


def evaluate_prediction_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    class_names: Iterable[str],
    threshold: float = 0.5,
    calibration: CalibrationBundle | None = None,
) -> dict[str, Any]:
    rows = list(rows)
    if not rows:
        raise ValueError("prediction rows are empty")
    if any(str(row.get("split", "test")) != "test" for row in rows):
        raise ValueError("vision evaluation must run on the independent test split only")
    patient_ids = [str(row.get("patient_id") or "") for row in rows]
    if any(not patient_id for patient_id in patient_ids):
        raise ValueError("every prediction row must include patient_id")
    if len(patient_ids) != len(set(patient_ids)):
        raise ValueError("test predictions contain multiple rows for one patient; aggregate at patient level first")

    names = list(class_names)
    true_rows = [dict(row.get("labels") or {}) for row in rows]
    score_rows = [dict(row.get("scores") or {}) for row in rows]
    result: dict[str, Any] = {
        "split": "test",
        "samples": len(rows),
        "patients": len(set(patient_ids)),
        "classification": multilabel_classification_metrics(
            true_rows,
            score_rows,
            names,
            threshold=threshold,
        ),
    }
    if calibration is not None:
        calibration_rows = {}
        for name in names:
            calibration_rows[name] = calibration_summary(
                [row.get(name, 0.0) for row in score_rows],
                [row.get(name, 0) for row in true_rows],
                calibration,
            )
        result["calibration"] = calibration_rows

    predictions = {
        str(row.get("image_id") or index): row.get("pred_boxes") or []
        for index, row in enumerate(rows)
    }
    targets = {
        str(row.get("image_id") or index): row.get("target_boxes") or []
        for index, row in enumerate(rows)
    }
    if any(predictions.values()) or any(targets.values()):
        result["detection"] = detection_map(
            predictions,
            targets,
            names,
            iou_thresholds=[0.5 + index * 0.05 for index in range(10)],
        )
    return result


def evaluate_manifest_predictions(
    records: Iterable[ManifestRecord],
    predictions: Iterable[Mapping[str, Any]],
    *,
    class_names: Iterable[str],
    threshold: float = 0.5,
    calibration: CalibrationBundle | None = None,
) -> dict[str, Any]:
    records = list(records)
    assert_independent_test_split(records)
    test_records = {record.image_path: record for record in records if record.split == "test"}
    rows = []
    for prediction in predictions:
        image_path = str(prediction.get("image_path") or prediction.get("image_id") or "")
        record = test_records.get(image_path)
        if record is None:
            raise ValueError(f"prediction does not belong to the independent test manifest: {image_path}")
        rows.append(
            {
                **prediction,
                "image_id": prediction.get("image_id") or image_path,
                "patient_id": record.patient_id,
                "split": "test",
                "labels": record.labels,
                "target_boxes": record.boxes,
            }
        )
    return evaluate_prediction_rows(
        rows,
        class_names=class_names,
        threshold=threshold,
        calibration=calibration,
    )
