from __future__ import annotations

import time
from pathlib import Path
from typing import Protocol

from app.config import settings
from app.ml.calibration import CalibrationBundle
from app.ml.quality import QualityGateConfig, assess_image_quality
from app.schemas.common import VisionResult
from app.services.dicom import DICOMError, prepare_inference_image
from app.services.medgemma import MedGemmaHTTPClient


class VisionAdapter(Protocol):
    model_name: str

    def analyze(self, image_path: Path | None, scenario: str = "opacity") -> VisionResult:
        ...


class MockMedGemmaAdapter:
    """Contract-compatible simulator for a future private MedGemma service."""

    model_name = "MedGemma-4B-local-adapter"

    def analyze(self, image_path: Path | None, scenario: str = "opacity") -> VisionResult:
        quality = assess_image_quality(image_path, _quality_config())
        quality_payload = quality.to_dict()
        limitations = [
            "当前为模拟适配器，未对原始影像执行真实推理。",
            "结果只用于演示工作流，不能作为临床诊断依据。",
        ]
        if image_path is not None and not quality.is_usable:
            # The mock adapter preserves the legacy demo path, but records that
            # quality could not be used as a production gate.
            quality_payload["is_usable"] = True
            quality_payload["enforcement"] = "bypassed_for_mock"
            limitations.append("模拟适配器未阻断质量不合格文件；真实模型会在推理前拒识。")
        common_limitations = [
            *limitations,
        ]
        if scenario == "normal":
            findings = []
            impression = "模拟结果：未发现明确异常影像征象。"
            risk_level = "low"
        elif scenario == "nodule":
            findings = [
                {
                    "name": "肺部结节样影",
                    "location": "右上肺野",
                    "confidence": 0.78,
                    "evidence": "模拟模型在局部发现边界相对集中的结节样密度改变。",
                    "severity": "moderate",
                }
            ]
            impression = "模拟结果：存在需要结合既往影像和进一步检查复核的结节样影。"
            risk_level = "moderate"
        elif scenario == "urgent":
            findings = [
                {
                    "name": "胸部异常影像征象",
                    "location": "左侧肺野",
                    "confidence": 0.86,
                    "evidence": "模拟模型发现需要优先人工复核的异常区域。",
                    "severity": "high",
                }
            ]
            impression = "模拟结果：发现高优先级异常提示，建议尽快由专业人员复核。"
            risk_level = "high"
        else:
            findings = [
                {
                    "name": "肺部混浊影",
                    "location": "右下肺野",
                    "confidence": 0.82,
                    "evidence": "模拟模型发现局部密度增高区域，无法仅凭该结果确定病因。",
                    "severity": "moderate",
                }
            ]
            impression = "模拟结果：存在需要结合临床信息进一步复核的肺部混浊影。"
            risk_level = "moderate"
        return VisionResult(
            model_name=self.model_name,
            model_version=settings.vision_model_version,
            dataset_version=settings.vision_dataset_version,
            task_type=settings.vision_task if settings.vision_task in {"classification", "detection", "segmentation"} else "detection",
            provider="local",
            simulated=True,
            image_quality={
                **quality_payload,
                "view": "PA/AP 未确认",
                "message": "模拟结果：质量门控信息仅作演示，真实模型需在推理前强制执行。",
            },
            findings=findings,
            impression=impression,
            risk_level=risk_level,
            needs_human_review=True,
            calibration_version="uncalibrated",
            abstained=False,
            pipeline_stages=["image_quality_gate", "mock_structured_findings"],
            limitations=common_limitations,
        )


class YoloVisionAdapter:
    """Real YOLO detection adapter with quality and confidence gates.

    The heavyweight Ultralytics import is intentionally lazy. A deployment
    using the legacy mock adapter therefore does not need the YOLO runtime.
    """

    def __init__(self) -> None:
        self.model_name = settings.vision_model_name or "YOLOv8-medical"
        self.model_path = Path(settings.vision_model_path) if settings.vision_model_path else None
        calibration_path = Path(settings.vision_calibration_path) if settings.vision_calibration_path else None
        self.calibrator = CalibrationBundle.load(calibration_path)
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self.model_path is None:
            raise RuntimeError("VISION_MODEL_PATH must point to a trained YOLO checkpoint")
        if not self.model_path.exists():
            raise RuntimeError(f"YOLO checkpoint does not exist: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "ultralytics is not installed; install backend/requirements-vision.txt on the inference host"
            ) from exc
        self._model = YOLO(str(self.model_path))
        return self._model

    def analyze(self, image_path: Path | None, scenario: str = "opacity") -> VisionResult:
        del scenario
        if image_path is None:
            raise RuntimeError("real YOLO inference requires an uploaded image")
        started = time.perf_counter()
        try:
            inference_path, dicom_metadata = prepare_inference_image(
                image_path,
                settings.storage_path / "derived",
            )
        except DICOMError as exc:
            return VisionResult(
                model_name=self.model_name,
                model_version=settings.vision_model_version,
                dataset_version=settings.vision_dataset_version,
                task_type="detection",
                provider="ultralytics",
                simulated=False,
                image_quality={"status": "dicom_decode_error", "is_usable": False, "reasons": [str(exc)]},
                findings=[],
                impression="DICOM 像素无法解码，系统拒绝自动生成异常判断。",
                risk_level="indeterminate",
                needs_human_review=True,
                calibration_version=self.calibrator.version,
                abstained=True,
                pipeline_stages=["dicom_adapter", "quality_abstention"],
                limitations=["DICOM 文件已接收但无法转换为推理像素，需要人工复核。"],
            )
        quality = assess_image_quality(inference_path, _quality_config())
        if not quality.is_usable:
            return VisionResult(
                model_name=self.model_name,
                model_version=settings.vision_model_version,
                dataset_version=settings.vision_dataset_version,
                task_type="detection",
                provider="ultralytics",
                simulated=False,
                image_quality={
                    **quality.to_dict(),
                    "inference_latency_ms": 0.0,
                    "dicom_metadata": dicom_metadata.to_dict() if dicom_metadata else None,
                },
                findings=[],
                impression="影像质量门控未通过，系统拒绝自动生成异常判断。",
                risk_level="indeterminate",
                needs_human_review=True,
                calibration_version=self.calibrator.version,
                abstained=True,
                rejected_findings=[],
                pipeline_stages=["image_quality_gate", "quality_abstention"],
                limitations=[
                    "质量门控未通过，未调用检测模型。",
                    "需要重新获取或由专业人员直接复核原始影像。",
                ],
            )

        model = self._load_model()
        device = None if settings.vision_device == "auto" else settings.vision_device
        results = model.predict(
            source=str(inference_path),
            device=device,
            conf=settings.vision_confidence_threshold,
            iou=settings.vision_iou_threshold,
            verbose=False,
        )
        result = results[0]
        names = getattr(result, "names", {}) or {}
        findings = []
        rejected_count = 0
        boxes = getattr(result, "boxes", None)
        rejected_findings: list[str] = []
        if boxes is not None:
            xyxy = boxes.xyxy.tolist()
            confidences = boxes.conf.tolist()
            class_ids = boxes.cls.tolist()
            for coordinates, raw_confidence, class_id in zip(xyxy, confidences, class_ids):
                label = str(names.get(int(class_id), int(class_id))) if isinstance(names, dict) else str(int(class_id))
                decision = self.calibrator.decide(float(raw_confidence), label)
                if decision.abstained:
                    rejected_count += 1
                    rejected_findings.append(label)
                    continue
                x1, y1, x2, y2 = [round(float(value), 2) for value in coordinates]
                findings.append(
                    {
                        "name": label,
                        "location": f"bbox[{x1}, {y1}, {x2}, {y2}]",
                        "confidence": round(decision.confidence, 6),
                        "raw_confidence": round(float(raw_confidence), 6),
                        "confidence_status": decision.status,
                        "bbox": [x1, y1, x2, y2],
                        "calibration_version": self.calibrator.version,
                        "evidence": "YOLO 检测框经过质量门控和置信度校准；位置需结合原始影像复核。",
                        "severity": "high" if decision.status == "accepted" else "indeterminate",
                    }
                )

        uncertain_count = sum(item["confidence_status"] == "uncertain" for item in findings)
        accepted_count = len(findings) - uncertain_count
        abstained = not findings and rejected_count > 0
        if accepted_count:
            impression = "检测模型发现至少一处达到接受阈值的局部异常，需由专业人员复核。"
            risk_level = "high"
        elif uncertain_count:
            impression = "检测模型仅发现低于接受阈值的可疑区域，系统保留为不确定结果。"
            risk_level = "moderate"
        elif abstained:
            impression = "检测结果低于拒识策略阈值，系统未将其写入确定性发现。"
            risk_level = "indeterminate"
        else:
            impression = "当前检测模型未输出达到记录阈值的局部异常。"
            risk_level = "low"
        limitations = [
            "AI 输出仅作为辅助决策，不能替代医生诊断。",
        ]
        if self.calibrator.version == "uncalibrated":
            limitations.append("未加载独立校准文件，置信度仅为模型原始分数的温度缩放结果。")
        if uncertain_count or rejected_count:
            limitations.append("存在不确定或被拒识的候选框，报告生成模块不得将其改写为确定事实。")
        elapsed_ms = round((time.perf_counter() - started) * 1000, 3)
        quality_payload = quality.to_dict()
        quality_payload["inference_latency_ms"] = elapsed_ms
        quality_payload["rejected_detection_count"] = rejected_count
        if dicom_metadata:
            quality_payload["dicom_metadata"] = dicom_metadata.to_dict()
        return VisionResult(
            model_name=self.model_name,
            model_version=settings.vision_model_version,
            dataset_version=settings.vision_dataset_version,
            task_type="detection",
            provider="ultralytics",
            simulated=False,
            image_quality=quality_payload,
            findings=findings,
            impression=impression,
            risk_level=risk_level,
            needs_human_review=True,
            calibration_version=self.calibrator.version,
            abstained=abstained,
            rejected_findings=rejected_findings,
            pipeline_stages=[
                "image_quality_gate",
                "yolo_detection",
                "confidence_calibration",
                "low_confidence_abstention",
                "structured_findings",
            ],
            limitations=limitations,
        )


class MedGemmaLocalAdapter:
    """Reserved interface for a private MedGemma 4B inference service."""

    def __init__(self) -> None:
        self.client = MedGemmaHTTPClient()
        self.model_name = self.client.model_name

    def analyze(self, image_path: Path | None, scenario: str = "opacity") -> VisionResult:
        return self.client.analyze(image_path, scenario=scenario)


def get_vision_adapter() -> VisionAdapter:
    if settings.vision_adapter in {"yolov8", "yolo", "ultralytics"}:
        return YoloVisionAdapter()
    if settings.vision_adapter == "medgemma_local":
        return MedGemmaLocalAdapter()
    return MockMedGemmaAdapter()


def _quality_config() -> QualityGateConfig:
    return QualityGateConfig(
        min_width=settings.quality_min_width,
        min_height=settings.quality_min_height,
        min_contrast=settings.quality_min_contrast,
        min_sharpness=settings.quality_min_sharpness,
        min_dynamic_range=settings.quality_min_dynamic_range,
        max_blank_fraction=settings.quality_max_blank_fraction,
    )
