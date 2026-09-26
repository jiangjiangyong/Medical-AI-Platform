from __future__ import annotations

from io import BytesIO

from fastapi.testclient import TestClient
from PIL import Image

from scripts import serve_medgemma


def test_model_load_oserror_returns_explicit_503(monkeypatch) -> None:
    monkeypatch.setattr(serve_medgemma, "LOAD_ON_STARTUP", False)

    def unavailable(*args, **kwargs):
        raise OSError("model weights are unavailable")

    monkeypatch.setattr(serve_medgemma.runtime, "analyze", unavailable)
    buffer = BytesIO()
    Image.effect_noise((512, 512), 48).convert("L").save(buffer, format="PNG")
    buffer.seek(0)

    with TestClient(serve_medgemma.app) as client:
        response = client.post(
            "/v1/analyze",
            files={"image": ("test.png", buffer, "image/png")},
        )

    assert response.status_code == 503
    assert response.json()["detail"] == "MedGemma inference unavailable"
