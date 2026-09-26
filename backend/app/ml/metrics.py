from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def auroc(y_true: Iterable[int], y_score: Iterable[float]) -> float | None:
    pairs = sorted(zip((1 if int(value) else 0 for value in y_true), y_score), key=lambda item: item[1])
    positives = sum(label for label, _ in pairs)
    negatives = len(pairs) - positives
    if not pairs or positives == 0 or negatives == 0:
        return None
    rank_sum = 0.0
    index = 0
    while index < len(pairs):
        end = index + 1
        while end < len(pairs) and pairs[end][1] == pairs[index][1]:
            end += 1
        average_rank = (index + 1 + end) / 2.0
        rank_sum += average_rank * sum(label for label, _ in pairs[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def binary_classification_metrics(
    y_true: Iterable[int],
    y_score: Iterable[float],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    targets = [1 if int(value) else 0 for value in y_true]
    scores = [float(value) for value in y_score]
    if len(targets) != len(scores):
        raise ValueError("y_true and y_score must have the same length")
    predictions = [int(value >= threshold) for value in scores]
    tp = sum(target == 1 and prediction == 1 for target, prediction in zip(targets, predictions))
    tn = sum(target == 0 and prediction == 0 for target, prediction in zip(targets, predictions))
    fp = sum(target == 0 and prediction == 1 for target, prediction in zip(targets, predictions))
    fn = sum(target == 1 and prediction == 0 for target, prediction in zip(targets, predictions))
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    return {
        "support": len(targets),
        "positive_support": sum(targets),
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": _safe_divide(2 * precision * recall, precision + recall),
        "auroc": auroc(targets, scores),
        "confusion": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    }


def multilabel_classification_metrics(
    y_true: Iterable[Mapping[str, int | float]],
    y_score: Iterable[Mapping[str, float]],
    class_names: Iterable[str],
    *,
    threshold: float = 0.5,
) -> dict[str, Any]:
    true_rows = list(y_true)
    score_rows = list(y_score)
    if len(true_rows) != len(score_rows):
        raise ValueError("y_true and y_score must have the same length")
    per_class: dict[str, Any] = {}
    for name in class_names:
        per_class[name] = binary_classification_metrics(
            [row.get(name, 0) for row in true_rows],
            [row.get(name, 0.0) for row in score_rows],
            threshold=threshold,
        )
    valid = [item for item in per_class.values() if item["auroc"] is not None]
    return {
        "per_class": per_class,
        "macro": {
            "precision": _safe_divide(sum(item["precision"] for item in per_class.values()), len(per_class)),
            "recall": _safe_divide(sum(item["recall"] for item in per_class.values()), len(per_class)),
            "f1": _safe_divide(sum(item["f1"] for item in per_class.values()), len(per_class)),
            "auroc": _safe_divide(sum(item["auroc"] for item in valid), len(valid)),
        },
    }


@dataclass(frozen=True)
class BoundingBox:
    label: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "BoundingBox":
        coordinates = value.get("bbox") or value
        if isinstance(coordinates, Mapping):
            return cls(
                label=str(value.get("label") or value.get("name") or "unknown"),
                x1=float(coordinates["x1"]),
                y1=float(coordinates["y1"]),
                x2=float(coordinates["x2"]),
                y2=float(coordinates["y2"]),
                confidence=float(value.get("confidence", 1.0)),
            )
        values = list(coordinates)
        return cls(
            label=str(value.get("label") or value.get("name") or "unknown"),
            x1=float(values[0]),
            y1=float(values[1]),
            x2=float(values[2]),
            y2=float(values[3]),
            confidence=float(value.get("confidence", 1.0)),
        )


def box_iou(left: BoundingBox, right: BoundingBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left.x2 - left.x1) * max(0.0, left.y2 - left.y1)
    right_area = max(0.0, right.x2 - right.x1) * max(0.0, right.y2 - right.y1)
    return _safe_divide(intersection, left_area + right_area - intersection)


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    envelope = precisions[:]
    for index in range(len(envelope) - 2, -1, -1):
        envelope[index] = max(envelope[index], envelope[index + 1])
    augmented_recall = [0.0, *recalls, 1.0]
    augmented_precision = [0.0, *envelope, 0.0]
    return sum(
        (augmented_recall[index + 1] - augmented_recall[index]) * augmented_precision[index + 1]
        for index in range(len(augmented_recall) - 1)
    )


def detection_map(
    predictions: Mapping[str, Iterable[BoundingBox | Mapping[str, Any]]],
    targets: Mapping[str, Iterable[BoundingBox | Mapping[str, Any]]],
    class_names: Iterable[str] | None = None,
    *,
    iou_thresholds: Iterable[float] = (0.5,),
) -> dict[str, Any]:
    """Compute per-class AP and mAP for one or more IoU thresholds."""

    def normalize(values: Iterable[BoundingBox | Mapping[str, Any]]) -> list[BoundingBox]:
        return [item if isinstance(item, BoundingBox) else BoundingBox.from_mapping(item) for item in values]

    normalized_predictions = {key: normalize(value) for key, value in predictions.items()}
    normalized_targets = {key: normalize(value) for key, value in targets.items()}
    names = sorted(
        set(class_names or ())
        | {box.label for boxes in normalized_predictions.values() for box in boxes}
        | {box.label for boxes in normalized_targets.values() for box in boxes}
    )
    threshold_results: dict[str, Any] = {}
    for threshold in iou_thresholds:
        per_class: dict[str, float | None] = {}
        for name in names:
            ground_truth_count = sum(
                1 for boxes in normalized_targets.values() for box in boxes if box.label == name
            )
            if ground_truth_count == 0:
                per_class[name] = None
                continue
            candidates = sorted(
                [
                    (box.confidence, image_id, box)
                    for image_id, boxes in normalized_predictions.items()
                    for box in boxes
                    if box.label == name
                ],
                key=lambda item: item[0],
                reverse=True,
            )
            matched: dict[str, set[int]] = {}
            true_positives: list[int] = []
            false_positives: list[int] = []
            for _, image_id, prediction in candidates:
                image_targets = [
                    (index, box)
                    for index, box in enumerate(normalized_targets.get(image_id, []))
                    if box.label == name
                ]
                best = max(
                    image_targets,
                    key=lambda item: box_iou(prediction, item[1]),
                    default=None,
                )
                if best and box_iou(prediction, best[1]) >= threshold and best[0] not in matched.setdefault(image_id, set()):
                    matched[image_id].add(best[0])
                    true_positives.append(1)
                    false_positives.append(0)
                else:
                    true_positives.append(0)
                    false_positives.append(1)
            cumulative_tp = 0
            cumulative_fp = 0
            recalls: list[float] = []
            precisions: list[float] = []
            for true_positive, false_positive in zip(true_positives, false_positives):
                cumulative_tp += true_positive
                cumulative_fp += false_positive
                recalls.append(cumulative_tp / ground_truth_count)
                precisions.append(_safe_divide(cumulative_tp, cumulative_tp + cumulative_fp))
            per_class[name] = _average_precision(recalls, precisions)
        valid = [value for value in per_class.values() if value is not None]
        threshold_results[f"{threshold:.2f}"] = {
            "per_class": per_class,
            "map": _safe_divide(sum(valid), len(valid)),
        }
    maps = [value["map"] for value in threshold_results.values()]
    return {
        "iou_thresholds": [float(value) for value in iou_thresholds],
        "by_iou": threshold_results,
        "map": _safe_divide(sum(maps), len(maps)),
        "map50": threshold_results.get("0.50", {}).get("map"),
    }
