import io
import base64
import os

from fastapi.testclient import TestClient
from PIL import Image

os.environ["USE_MOCK_MODEL"] = "True"

import main
from scripts.init_model import _class_row_indices


def _png_bytes(size: tuple[int, int] = (96, 64)) -> bytes:
    image = Image.new("RGB", size, (80, 140, 210))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_predict_returns_result_and_logs_scan():
    main._recent_scans.clear()
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("portrait.png", _png_bytes(), "image/png")},
        )

        assert response.status_code == 200
        result = response.json()
        assert result["label"] in {"Real", "Synthetic"}
        assert result["is_deepfake"] is (result["label"] == "Synthetic")
        assert 0 <= result["confidence"] <= 1
        assert result["confidence_percentage"] == round(result["confidence"] * 100, 2)
        assert result["attention_heatmap"].startswith("data:image/jpeg;base64,")
        assert result["raw_heatmap"].startswith("data:image/jpeg;base64,")
        heatmap = Image.open(
            io.BytesIO(base64.b64decode(result["raw_heatmap"].split(",", 1)[1]))
        )
        assert heatmap.size == (96, 64)
        assert result["filename"] == "portrait.png"
        assert result["id"]

        history = client.get("/scans")
        assert history.status_code == 200
        assert history.json()[0]["id"] == result["id"]


def test_predict_rejects_unsupported_file_type():
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("portrait.gif", b"GIF89a", "image/gif")},
        )

    assert response.status_code == 415


def test_health_reports_model_mode_and_checkpoint_status():
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    health = response.json()
    model_status = main.get_model_status()
    assert health["mock_mode"] is model_status["mock_mode"]
    assert health["model_mode"] == (
        "mock" if model_status["mock_mode"] else "vit_base_patch16_224"
    )
    assert health["model_name"] == model_status["model_name"]
    assert isinstance(health["device"], str)
    assert health["checkpoint_loaded"] is (not model_status["mock_mode"])


def test_hugging_face_labels_are_mapped_to_real_then_synthetic():
    assert _class_row_indices(
        {"id2label": {"0": "Deepfake", "1": "Realism"}}
    ) == (1, 0)


def test_vercel_origin_is_allowed_without_allowing_arbitrary_sites():
    with TestClient(main.app) as client:
        vercel_response = client.options(
            "/predict",
            headers={
                "Origin": "https://veriface-ai-preview.vercel.app",
                "Access-Control-Request-Method": "POST",
            },
        )
        render_response = client.options(
            "/predict",
            headers={
                "Origin": "https://veriface-ai-1.onrender.com",
                "Access-Control-Request-Method": "POST",
            },
        )
        other_response = client.options(
            "/predict",
            headers={
                "Origin": "https://untrusted.example",
                "Access-Control-Request-Method": "POST",
            },
        )

    assert vercel_response.headers["access-control-allow-origin"] == (
        "https://veriface-ai-preview.vercel.app"
    )
    assert render_response.headers["access-control-allow-origin"] == (
        "https://veriface-ai-1.onrender.com"
    )
    assert "access-control-allow-origin" not in other_response.headers


def test_large_image_visualization_is_downscaled():
    with TestClient(main.app) as client:
        response = client.post(
            "/predict",
            files={"file": ("large.png", _png_bytes((1920, 1080)), "image/png")},
        )

    assert response.status_code == 200
    heatmap = Image.open(
        io.BytesIO(
            base64.b64decode(response.json()["raw_heatmap"].split(",", 1)[1])
        )
    )
    assert heatmap.size == (1600, 900)
