import base64
import io
import os
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from .schemas import UpscaleResponseModel
from .triton_client import TritonClient, TritonClientError

app = FastAPI(title="RAInder Gateway API", version="0.1.0")
triton_client = TritonClient()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/x-png"}
ALLOWED_FORMATS = {"JPEG", "PNG"}

raw_origins = os.getenv(
    "CORS_ORIGINS",
    (
        "http://localhost:5173,http://127.0.0.1:5173,"
        "http://localhost:5174,http://127.0.0.1:5174"
    ),
)
cors_origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(model_name: str | None = None) -> dict[str, bool | str]:
    try:
        details = triton_client.readiness(model_name=model_name)
    except TritonClientError as exc:
        raise HTTPException(status_code=503, detail=f"Inference readiness check failed: {exc}") from exc

    if not (details["server_live"] and details["server_ready"] and details["model_ready"]):
        raise HTTPException(status_code=503, detail=details)
    return details


@app.post("/upscale", response_model=UpscaleResponseModel)
async def upscale(
    image: UploadFile = File(...),
    model_name: str = Form(""),
    scale_factor: float = Form(2.0),
) -> UpscaleResponseModel:
    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    # Prefer content validation over MIME-only checks; some clients send image/x-png.
    if image.content_type and image.content_type.lower() not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")
    try:
        with Image.open(io.BytesIO(raw)) as parsed:
            fmt = (parsed.format or "").upper()
        if fmt not in ALLOWED_FORMATS:
            raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    if scale_factor not in {2.0, 4.0}:
        raise HTTPException(status_code=400, detail="scale_factor must be 2.0 or 4.0.")

    selected_model = model_name.strip()
    if not selected_model:
        selected_model = "real_esrgan_x2" if scale_factor == 2.0 else "real_esrgan_x4"

    try:
        result = triton_client.upscale(
            raw_image=raw, model_name=selected_model, scale_factor=scale_factor
        )
    except TritonClientError as exc:
        raise HTTPException(status_code=502, detail=f"Inference failed: {exc}") from exc

    return UpscaleResponseModel(
        image_data=base64.b64encode(result.upscaled_image).decode("ascii"),
        inference_time_ms=result.compute_latency_ms,
        status="success",
    )
