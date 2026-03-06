import io

from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.triton_client import TritonResult


def _png_bytes(size: tuple[int, int] = (32, 32), color: tuple[int, int, int] = (100, 140, 210)) -> bytes:
    image = Image.new("RGB", size, color)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_returns_200_when_mock_mode(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(
        main.triton_client,
        "readiness",
        lambda model_name=None: {
            "mode": "mock",
            "server_live": True,
            "server_ready": True,
            "model_ready": True,
            "model_name": model_name or "real_esrgan_x2",
        },
    )
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["model_ready"] is True


def test_ready_returns_503_when_model_not_ready(monkeypatch) -> None:
    from app import main

    monkeypatch.setattr(
        main.triton_client,
        "readiness",
        lambda model_name=None: {
            "mode": "triton",
            "server_live": True,
            "server_ready": True,
            "model_ready": False,
            "model_name": model_name or "real_esrgan_x2",
        },
    )
    client = TestClient(app)
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["detail"]["model_ready"] is False


def test_upscale_success(monkeypatch) -> None:
    from app import main

    upscaled = _png_bytes((64, 64), (70, 200, 120))

    monkeypatch.setattr(
        main.triton_client,
        "upscale",
        lambda raw_image, model_name, scale_factor: TritonResult(
            upscaled_image=upscaled,
            width=64,
            height=64,
            compute_latency_ms=12.5,
        ),
    )

    client = TestClient(app)
    payload = _png_bytes()
    files = {"image": ("test.png", payload, "image/png")}
    data = {"model_name": "real_esrgan_x2", "scale_factor": "2.0"}
    response = client.post("/upscale", files=files, data=data)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["inference_time_ms"] == 12.5
    assert isinstance(result["image_data"], str) and len(result["image_data"]) > 16


def test_upscale_rejects_unsupported_image(monkeypatch) -> None:
    from app import main

    # Upscale is not called, but keep behavior explicit for test clarity.
    monkeypatch.setattr(main.triton_client, "upscale", lambda *_args, **_kwargs: None)

    client = TestClient(app)
    files = {"image": ("note.txt", b"not-an-image", "text/plain")}
    data = {"model_name": "real_esrgan_x2", "scale_factor": "2.0"}
    response = client.post("/upscale", files=files, data=data)
    assert response.status_code == 400


def test_anime_hayao_success(monkeypatch) -> None:
    from app import main

    styled = _png_bytes((48, 48), (140, 90, 190))

    monkeypatch.setattr(
        main.triton_client,
        "upscale",
        lambda raw_image, model_name, scale_factor: TritonResult(
            upscaled_image=styled,
            width=48,
            height=48,
            compute_latency_ms=9.3,
        ),
    )

    client = TestClient(app)
    payload = _png_bytes()
    files = {"image": ("test.png", payload, "image/png")}
    response = client.post("/anime/hayao", files=files)

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "success"
    assert result["inference_time_ms"] == 9.3

