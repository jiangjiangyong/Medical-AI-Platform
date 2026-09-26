from pathlib import Path

from PIL import Image

from app.services.medgemma import MedGemmaHTTPClient


def _make_contract_image(path: Path) -> None:
    image = Image.effect_noise((512, 512), 48).convert("L")
    image.save(path, format="PNG")


def test_medgemma_http_client_normalizes_real_contract(monkeypatch, tmp_path):
    image_path = tmp_path / "contract.png"
    _make_contract_image(image_path)

    client = MedGemmaHTTPClient()
    client.base_url = "http://contract-test"
    client.api_path = "/v1/analyze"
    monkeypatch.setattr(
        client,
        "_request_json",
        lambda _: {
            "result": {
                "task_type": "detection",
                "image_quality": {"status": "passed", "is_usable": True},
                "findings": [
                    {
                        "name": "contract-finding",
                        "location": "test-field",
                        "confidence": 0.91,
                        "confidence_status": "accepted",
                        "evidence": "contract-test-only",
                        "severity": "moderate",
                    }
                ],
                "impression": "contract test only",
                "risk_level": "moderate",
                "needs_human_review": True,
                "abstained": False,
                "rejected_findings": [],
                "limitations": ["contract_test"],
            }
        },
    )

    result = client.analyze(image_path)

    assert result.provider == "medgemma_remote"
    assert result.simulated is False
    assert result.abstained is False
    assert result.findings[0].name == "contract-finding"
    assert result.image_quality["status"] == "passed"
    assert result.image_quality["inference_latency_ms"] >= 0
    assert "vlm_http_request" in result.pipeline_stages

