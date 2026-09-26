"""Reusable medical-imaging experiment primitives.

The modules in this package are deliberately independent from the HTTP layer so
that dataset preparation and evaluation can be reproduced from the command
line or from a scheduled training job.
"""

from app.ml.calibration import CalibrationBundle, ConfidenceDecision
from app.ml.manifest import ManifestRecord, patient_level_split
from app.ml.quality import ImageQualityResult, QualityGateConfig, assess_image_quality

__all__ = [
    "CalibrationBundle",
    "ConfidenceDecision",
    "ImageQualityResult",
    "ManifestRecord",
    "QualityGateConfig",
    "assess_image_quality",
    "patient_level_split",
]

from app.ml.benchmarks import (
    BENCHMARK_DEFINITIONS,
    BENCHMARK_NAMES,
    BenchmarkDefinition,
)
from app.ml.experiment import (
    ARTIFACT_FILENAMES,
    EXPERIMENT_VERSION_FIELDS,
    ExperimentMetadata,
    ExperimentStore,
)
