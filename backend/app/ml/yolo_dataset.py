from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Iterable

from app.ml.manifest import ManifestRecord, class_balance_summary
from app.ml.metrics import BoundingBox


def prepare_yolo_dataset(
    records: Iterable[ManifestRecord],
    output_dir: Path,
    class_names: list[str],
) -> Path:
    """Materialize a YOLO detection dataset from the patient-split manifest."""

    records = list(records)
    if any(record.split not in {"train", "valid", "test"} for record in records):
        raise ValueError("every record must have train, valid or test split before YOLO conversion")
    output_dir.mkdir(parents=True, exist_ok=True)
    class_to_id = {name: index for index, name in enumerate(class_names)}
    for split in ("train", "valid", "test"):
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        source = Path(record.image_path)
        if not source.exists():
            raise FileNotFoundError(source)
        try:
            from PIL import Image

            with Image.open(source) as image:
                width, height = image.size
        except Exception as exc:
            raise ValueError(f"unable to read image dimensions for {source}") from exc
        stable_name = f"{hashlib.sha256(str(source).encode('utf-8')).hexdigest()[:12]}_{source.name}"
        image_target = output_dir / "images" / str(record.split) / stable_name
        shutil.copy2(source, image_target)
        label_target = output_dir / "labels" / str(record.split) / f"{Path(stable_name).stem}.txt"
        lines = []
        for raw_box in record.boxes:
            box = raw_box if isinstance(raw_box, BoundingBox) else BoundingBox.from_mapping(raw_box)
            if box.label not in class_to_id:
                raise ValueError(f"unknown class {box.label!r} in {source}")
            center_x = ((box.x1 + box.x2) / 2.0) / width
            center_y = ((box.y1 + box.y2) / 2.0) / height
            box_width = (box.x2 - box.x1) / width
            box_height = (box.y2 - box.y1) / height
            values = [center_x, center_y, box_width, box_height]
            if not all(0.0 <= value <= 1.0 for value in values):
                raise ValueError(f"box is outside image bounds for {source}: {raw_box}")
            lines.append(f"{class_to_id[box.label]} " + " ".join(f"{value:.8f}" for value in values))
        label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    dataset_yaml = output_dir / "dataset.yaml"
    dataset_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.as_posix()}",
                "train: images/train",
                "val: images/valid",
                "test: images/test",
                f"nc: {len(class_names)}",
                f"names: {json.dumps(class_names, ensure_ascii=False)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (output_dir / "class_balance.json").write_text(
        json.dumps(class_balance_summary(records, class_names), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset_yaml
