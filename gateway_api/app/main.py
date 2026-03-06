import base64
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .schemas import UpscaleResponseModel
from .triton_client import TritonClient

app = FastAPI(title="RAInder Gateway API", version="0.1.0")
triton_client = TritonClient()

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upscale", response_model=UpscaleResponseModel)
async def upscale(
    image: UploadFile = File(...),
    model_name: str = Form("real-esrgan"),
    scale_factor: float = Form(2.0),
) -> UpscaleResponseModel:
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail="Only JPEG and PNG are supported.")

    raw = await image.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds 10 MB limit.")

    try:
        result = triton_client.upscale(
            raw_image=raw, model_name=model_name, scale_factor=scale_factor
        )
    except NotImplementedError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc

    return UpscaleResponseModel(
        image_data=base64.b64encode(result.upscaled_image).decode("ascii"),
        inference_time_ms=result.compute_latency_ms,
        status="success",
    )
