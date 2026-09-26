from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

from app.core.security import hash_password
from app.db.session import SessionLocal
from app.main import app
from app.models import User


def _register(client: TestClient, role: str) -> tuple[str, dict]:
    suffix = uuid4().hex[:8]
    passwords = {"patient": "Patient123!", "doctor": "Doctor123!"}
    email = f"{role}-{suffix}@example.com"
    display_name = "集成测试患者" if role == "patient" else "集成测试医生"
    if role == "doctor":
        db = SessionLocal()
        try:
            user = User(email=email, password_hash=hash_password(passwords[role]), display_name=display_name, role=role)
            db.add(user)
            db.commit()
            db.refresh(user)
            user_id = user.id
        finally:
            db.close()
        login = client.post("/api/v1/auth/login", json={"email": email, "password": passwords[role]})
        assert login.status_code == 200
        return user_id, {"Authorization": f"Bearer {login.json()['access_token']}"}
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": email,
            "password": passwords[role],
            "display_name": display_name,
        },
    )
    assert response.status_code == 201
    token = response.json()["access_token"]
    return response.json()["user"]["id"], {"Authorization": f"Bearer {token}"}


def test_patient_case_flow_requires_review_before_release() -> None:
    with TestClient(app) as client:
        _, patient_headers = _register(client, "patient")
        _, doctor_headers = _register(client, "doctor")

        created = client.post(
            "/api/v1/cases",
            headers=patient_headers,
            json={"title": "测试胸片病例", "symptoms": "持续咳嗽"},
        )
        assert created.status_code == 201
        case_id = created.json()["id"]

        blocked = client.post(
            f"/api/v1/cases/{case_id}/analyze",
            headers=patient_headers,
            json={"scenario": "opacity"},
        )
        assert blocked.status_code == 403

        verified = client.post(
            f"/api/v1/cases/{case_id}/verify-identity",
            headers=patient_headers,
            json={
                "confirmed_name": "集成测试患者",
                "confirm_identity": True,
            },
        )
        assert verified.status_code == 200

        uploaded = client.post(
            f"/api/v1/cases/{case_id}/studies",
            headers=patient_headers,
            files={"file": ("chest.png", b"demo-image", "image/png")},
        )
        assert uploaded.status_code == 201
        assert uploaded.json()["status"] == "processing"

        doctor_detail = client.get(f"/api/v1/cases/{case_id}", headers=doctor_headers)
        assert doctor_detail.status_code == 200
        assert doctor_detail.json()["status"] in {"processing", "pending_review"}

        if doctor_detail.json()["status"] == "processing":
            analyzed = client.post(
                f"/api/v1/cases/{case_id}/analyze",
                headers=doctor_headers,
                json={"scenario": "opacity"},
            )
            assert analyzed.status_code == 200
            doctor_detail = client.get(f"/api/v1/cases/{case_id}", headers=doctor_headers)

        professional_report = next(
            report for report in doctor_detail.json()["reports"] if report["report_type"] == "professional"
        )
        assert professional_report["status"] == "pending_review"
        assert not any(report["report_type"] == "patient" for report in doctor_detail.json()["reports"])

        edited = client.patch(
            f"/api/v1/cases/{case_id}/reports/{professional_report['id']}",
            headers=doctor_headers,
            json={
                "imaging_findings": ["医生确认：右下肺野局部密度增高"],
                "preliminary_assessment": "建议结合症状和既往影像进一步判断。",
                "recommendations": ["由专业人员结合原始影像复核。"],
                "risk_level": "moderate",
                "change_note": "集成测试修改",
            },
        )
        assert edited.status_code == 200
        assert edited.json()["version"] == 2
        assert edited.json()["status"] == "pending_review"

        approved = client.post(
            f"/api/v1/cases/{case_id}/reports/{professional_report['id']}/review",
            headers=doctor_headers,
            json={"status": "approved", "note": "确认发布"},
        )
        assert approved.status_code == 200

        final_doctor_detail = client.get(f"/api/v1/cases/{case_id}", headers=doctor_headers).json()
        assert final_doctor_detail["status"] == "published"
        assert any(report["report_type"] == "patient" and report["status"] == "published" for report in final_doctor_detail["reports"])
        assert len(final_doctor_detail["follow_ups"]) == 1

        patient_detail = client.get(f"/api/v1/cases/{case_id}", headers=patient_headers).json()
        assert patient_detail["vision_analysis"] is None
        assert len(patient_detail["reports"]) == 1
        assert patient_detail["reports"][0]["report_type"] == "patient"

        revisions = client.get(
            f"/api/v1/cases/{case_id}/reports/{professional_report['id']}/revisions",
            headers=doctor_headers,
        )
        assert revisions.status_code == 200
        assert [item["version"] for item in revisions.json()] == [2, 1]


def test_staff_intake_code_is_required_for_patient_upload() -> None:
    with TestClient(app) as client:
        patient_id, patient_headers = _register(client, "patient")
        _, doctor_headers = _register(client, "doctor")

        created = client.post(
            "/api/v1/cases",
            headers=doctor_headers,
            json={"patient_id": patient_id, "title": "工作人员预登记胸片"},
        )
        assert created.status_code == 201
        case_id = created.json()["id"]
        intake_code = created.json()["intake_code"]
        assert intake_code.startswith("MI-")

        missing_code = client.post(
            f"/api/v1/cases/{case_id}/verify-identity",
            headers=patient_headers,
            json={"confirmed_name": "集成测试患者", "confirm_identity": True},
        )
        assert missing_code.status_code == 422

        verified = client.post(
            f"/api/v1/cases/{case_id}/verify-identity",
            headers=patient_headers,
            json={
                "confirmed_name": "集成测试患者",
                "intake_code": intake_code,
                "confirm_identity": True,
            },
        )
        assert verified.status_code == 200
