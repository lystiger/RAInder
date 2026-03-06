from pydantic import BaseModel, Field


class UpscaleResponseModel(BaseModel):
    image_data: str = Field(..., description="Base64-encoded upscaled image")
    inference_time_ms: float = Field(..., ge=0)
    status: str = Field(..., description="success or failure")
