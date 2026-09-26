from __future__ import annotations

import csv
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping


@dataclass
class ManifestRecord:
    image_path: str
    patient_id: str
    labels: dict[str, float] = field(default_factory=dict)
    boxes: list[dict[str, Any]] = field(default_factory=list)
    study_id: str | None = None
    split: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ManifestRecord":
        labels_value = payload.get("labels", {})
        if isinstance(labels_value, str):
            try:
                labels_value = json.loads(labels_value)
            except json.JSONDecodeError:
                labels_value = {
                    item.strip(): 1.0
                    for item in labels_value.replace(";", ",").split(",")
                    if item.strip()
                }
        if isinstance(labels_value, list):
            labels_value = {str(item): 1.0 for item in labels_value}
        labels = {str(key): float(value) for key, value in dict(labels_value or {}).items()}

        boxes_value = payload.get("boxes", [])
        if isinstance(boxes_value, str):
            try:
                boxes_value = json.loads(boxes_value)
            except json.JSONDecodeError:
                boxes_value = []
        return cls(
            image_path=str(payload.get("image_path") or payload.get("path") or ""),
            patient_id=str(payload.get("patient_id") or "").strip(),
            labels=labels,
            boxes=[dict(item) for item in (boxes_value or [])],
            study_id=str(payload["study_id"]) if payload.get("study_id") else None,
            split=str(payload["split"]) if payload.get("split") else None,
            metadata=dict(payload.get("metadata") or {}),
        )

    def to_mapping(self) -> dict[str, Any]:
        return asdict(self)


def load_manifest(path: Path) -> list[ManifestRecord]:
    if path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("records", [])
    records = [ManifestRecord.from_mapping(row) for row in rows]
    validate_manifest(records)
    return records


def save_manifest(records: Iterable[ManifestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.to_mapping(), ensure_ascii=False) + "\n")


def validate_manifest(records: Iterable[ManifestRecord]) -> None:
    records = list(records)
    if not records:
        raise ValueError("manifest is empty")
    for index, record in enumerate(records):
        if not record.image_path:
            raise ValueError(f"record {index} has no image_path")
        if not record.patient_id:
            raise ValueError(f"record {index} has no patient_id")


def _group_records(records: Iterable[ManifestRecord]) -> dict[str, list[ManifestRecord]]:
    groups: dict[str, list[ManifestRecord]] = {}
    for record in records:
        groups.setdefault(record.patient_id, []).append(record)
    return groups


def _allocate_bucket_counts(
    size: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    """Allocate one class-signature bucket using largest remainders."""
    splits = tuple(ratios)
    raw = {split: size * ratios[split] for split in splits}
    counts = {split: int(raw[split]) for split in splits}
    remainder = size - sum(counts.values())
    order = sorted(
        splits,
        key=lambda split: (-(raw[split] - counts[split]), splits.index(split)),
    )
    for split in order[:remainder]:
        counts[split] += 1
    return counts


def _patient_label_signature(
    patient_records: list[ManifestRecord],
    class_names: list[str],
) -> tuple[str, ...]:
    return tuple(
        name
        for name in class_names
        if any(record.labels.get(name, 0.0) >= 0.5 for record in patient_records)
    )


def patient_level_split(
    records: Iterable[ManifestRecord],
    *,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[ManifestRecord]]:
    """Create deterministic, approximately stratified splits without patient leakage."""
    records = list(records)
    validate_manifest(records)
    ratios = {"train": train_ratio, "valid": valid_ratio, "test": test_ratio}
    if any(value <= 0 for value in ratios.values()) or abs(sum(ratios.values()) - 1.0) > 1e-6:
        raise ValueError("split ratios must be positive and sum to 1")

    groups = _group_records(records)
    if len(groups) < 3:
        raise ValueError("at least three unique patients are required for an independent test split")

    class_names = sorted({name for record in records for name in record.labels})
    buckets: dict[tuple[str, ...], list[tuple[str, list[ManifestRecord]]]] = {}
    for patient_id, patient_records in groups.items():
        signature = _patient_label_signature(patient_records, class_names)
        buckets.setdefault(signature, []).append((patient_id, patient_records))

    rng = random.Random(seed)
    assigned: dict[str, list[tuple[str, list[ManifestRecord]]]] = {
        split: [] for split in ratios
    }
    for signature in sorted(buckets):
        bucket = buckets[signature]
        rng.shuffle(bucket)
        allocation = _allocate_bucket_counts(len(bucket), ratios)
        offset = 0
        for split in ratios:
            count = allocation[split]
            assigned[split].extend(bucket[offset : offset + count])
            offset += count

    # Small datasets can leave a split empty after per-bucket rounding.
    for split in ratios:
        if assigned[split]:
            continue
        donor = max(
            (candidate for candidate in ratios if len(assigned[candidate]) > 1),
            key=lambda candidate: len(assigned[candidate]),
            default=None,
        )
        if donor is None:
            raise ValueError("unable to create three non-empty patient-level splits")
        assigned[split].append(assigned[donor].pop())

    result = {
        split: [
            record
            for _, patient_records in assigned[split]
            for record in patient_records
        ]
        for split in ratios
    }
    for split, split_records in result.items():
        for record in split_records:
            record.split = split
    validate_split_disjointness(result)
    return result


def validate_split_disjointness(splits: Mapping[str, Iterable[ManifestRecord]]) -> None:
    owners: dict[str, str] = {}
    for split, records in splits.items():
        if not list(records):
            raise ValueError(f"split {split} is empty")
        for record in records:
            previous = owners.get(record.patient_id)
            if previous and previous != split:
                raise ValueError(f"patient leakage detected: {record.patient_id} in {previous} and {split}")
            owners[record.patient_id] = split


def class_balance_summary(
    records: Iterable[ManifestRecord],
    class_names: Iterable[str] | None = None,
) -> dict[str, Any]:
    records = list(records)
    names = sorted(set(class_names or ()) | {name for record in records for name in record.labels})
    summary: dict[str, Any] = {}
    for name in names:
        positive = sum(1 for record in records if record.labels.get(name, 0.0) >= 0.5)
        negative = max(0, len(records) - positive)
        summary[name] = {
            "positive": positive,
            "negative": negative,
            "prevalence": positive / len(records) if records else 0.0,
            "pos_weight": negative / max(1, positive),
        }
    return {"samples": len(records), "classes": summary}


def patient_level_calibration_split(
    records: Iterable[ManifestRecord],
    *,
    source_split: str = "valid",
    fit_ratio: float = 0.6,
    seed: int = 42,
) -> dict[str, list[ManifestRecord]]:
    """Create a deterministic patient-disjoint calibration fit/eval split.

    The source split is never mixed with train or test records. Records are
    grouped by patient and approximately stratified by their label signature,
    so a calibration bundle can be fitted on ``calibration_fit`` and assessed
    on ``calibration_eval`` without fitting on the evaluation rows.
    """

    import hashlib
    from dataclasses import replace

    records = list(records)
    if not 0.0 < fit_ratio < 1.0:
        raise ValueError("fit_ratio must be between 0 and 1")
    candidates = [record for record in records if record.split == source_split]
    if len(candidates) < 2:
        raise ValueError(
            f"at least two records in source split {source_split!r} are required"
        )
    groups = _group_records(candidates)
    if len(groups) < 2:
        raise ValueError("at least two unique patients are required")

    class_names = sorted({name for record in candidates for name in record.labels})
    buckets: dict[tuple[str, ...], list[tuple[str, list[ManifestRecord]]]] = {}
    for patient_id, patient_records in groups.items():
        signature = _patient_label_signature(patient_records, class_names)
        buckets.setdefault(signature, []).append((patient_id, patient_records))

    assigned: dict[str, list[tuple[str, list[ManifestRecord]]]] = {
        "calibration_fit": [],
        "calibration_eval": [],
    }
    for signature in sorted(buckets):
        bucket = sorted(
            buckets[signature],
            key=lambda item: hashlib.sha256(
                f"{seed}:{item[0]}".encode("utf-8")
            ).hexdigest(),
        )
        fit_count = int(round(len(bucket) * fit_ratio))
        if len(bucket) > 1:
            fit_count = min(len(bucket) - 1, max(1, fit_count))
        else:
            fit_count = 1 if fit_ratio >= 0.5 else 0
        assigned["calibration_fit"].extend(bucket[:fit_count])
        assigned["calibration_eval"].extend(bucket[fit_count:])

    for split in assigned:
        if assigned[split]:
            continue
        donor = "calibration_fit" if split == "calibration_eval" else "calibration_eval"
        if len(assigned[donor]) <= 1:
            raise ValueError("unable to create two non-empty calibration splits")
        assigned[split].append(assigned[donor].pop())

    result = {
        split: [
            replace(record, split=split)
            for _, patient_records in assigned[split]
            for record in patient_records
        ]
        for split in assigned
    }
    validate_split_disjointness(result)
    return result
