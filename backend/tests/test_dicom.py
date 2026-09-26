from __future__ import annotations

from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from app.services.dicom import dicom_to_png, parse_dicom


def _write_dicom(path: Path) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = pydicom.uid.SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.PatientName = "Sensitive^Patient"
    dataset.PatientID = "patient-001"
    dataset.StudyInstanceUID = generate_uid()
    dataset.SeriesInstanceUID = generate_uid()
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.Modality = "CR"
    dataset.ViewPosition = "PA"
    dataset.Rows = 768
    dataset.Columns = 768
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 0
    dataset.PixelData = (np.arange(768 * 768, dtype=np.uint16) % 4096).tobytes()
    dataset.save_as(path)


def test_dicom_metadata_is_safe_and_pixels_are_materialized(tmp_path: Path) -> None:
    source = tmp_path / "study.dcm"
    output = tmp_path / "derived" / "study.png"
    _write_dicom(source)

    metadata = parse_dicom(source)
    assert metadata.modality == "CR"
    assert metadata.view_position == "PA"
    assert metadata.patient_name_hash
    assert metadata.patient_id_hash
    assert "PatientName" not in metadata.safe_tags
    assert metadata.study_instance_uid

    result = dicom_to_png(source, output)
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
