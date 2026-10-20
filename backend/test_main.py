import io
import base64

from fastapi.testclient import TestClient
from PIL import Image

import main
import model
from scripts.init_model import _class_row_indices


def _png_bytes(size: tuple[int, int] = (96, 64)) -> bytes:
    image = Image.new("RGB", size, (80, 140, 210))
    stream = io.BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def _prediction_result() -> dict:
    return {
        "label": "Real",
        "is_deepfake": False,
        "confidence": 0.91,
        "confidence_percentage": 91.0,
        "attention_heatmap": "",
        "raw_heatmap": "",
        "execution_time_ms": 12.5,
        "model_name": "vit_base_patch16_224",
        "mode": "production_vit",
        "patch_analysis": {
            "patch_count": 196,
            "grid_size": "14x14",
            "highest_activation_zone": "Highest ViT attention patch",
        },
    }


def test_predict_returns_result_and_logs_scan(monkeypatch):
    monkeypatch.setattr(main, "predict_image", lambda _image_bytes: _prediction_result())
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


def test_health_reports_model_warming_up_until_first_prediction():
    with TestClient(main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    health = response.json()
    assert health["status"] == "warming_up"
    assert health["model_name"] == "vit_base_patch16_224"
    assert isinstance(health["device"], str)
    assert health["checkpoint_loaded"] is False


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


def test_attention_heatmap_is_encoded_at_source_image_size():
    image = Image.new("RGB", (1920, 1080), (80, 140, 210))
    image_bgr = model.cv2.cvtColor(model.np.asarray(image), model.cv2.COLOR_RGB2BGR)
    overlay, raw_heatmap = model._encode_heatmaps(
        image_bgr, model.np.zeros((14, 14), dtype=model.np.float32)
    )

    for data_url in (overlay, raw_heatmap):
        assert data_url.startswith("data:image/jpeg;base64,")
        heatmap = Image.open(
            io.BytesIO(base64.b64decode(data_url.split(",", 1)[1]))
        )
        assert heatmap.size == (1920, 1080)
