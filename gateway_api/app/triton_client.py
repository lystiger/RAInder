from dataclasses import dataclass
import os
from time import perf_counter

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

from .image_utils import decode_image_to_chw_fp32, encode_chw_fp32_to_png_bytes


@dataclass
class TritonResult:
    upscaled_image: bytes
    width: int
    height: int
    compute_latency_ms: float


class TritonClientError(Exception):
    pass


class TritonClient:
    def __init__(self) -> None:
        self.triton_url = os.getenv("TRITON_URL", "localhost:8001")
        self.default_model_name = os.getenv("TRITON_MODEL_NAME", "super_resolution_model")
        self.mock_mode = os.getenv("TRITON_MOCK", "false").lower() in {"1", "true", "yes"}
        self.client = grpcclient.InferenceServerClient(url=self.triton_url)

    def upscale(self, raw_image: bytes, model_name: str, scale_factor: float) -> TritonResult:
        if self.mock_mode:
            return self._mock_upscale(raw_image, scale_factor)

        request_model = model_name if model_name else self.default_model_name
        chw = decode_image_to_chw_fp32(raw_image)

        infer_input = grpcclient.InferInput("input", list(chw.shape), "FP32")
        infer_input.set_data_from_numpy(chw.astype(np.float32))
        output_req = grpcclient.InferRequestedOutput("output")

        started = perf_counter()
        try:
            response = self.client.infer(
                model_name=request_model,
                inputs=[infer_input],
                outputs=[output_req],
            )
        except InferenceServerException as exc:
            raise TritonClientError(str(exc)) from exc

        duration_ms = (perf_counter() - started) * 1000.0

        output = response.as_numpy("output")
        if output is None:
            raise TritonClientError("Triton returned no output tensor named 'output'.")

        output_chw = self._normalize_output_shape(output)
        upscaled_image, width, height = encode_chw_fp32_to_png_bytes(output_chw)

        return TritonResult(
            upscaled_image=upscaled_image,
            width=width,
            height=height,
            compute_latency_ms=duration_ms,
        )

    def _normalize_output_shape(self, output: np.ndarray) -> np.ndarray:
        if output.ndim == 3:
            return output
        if output.ndim == 4 and output.shape[0] == 1:
            return output[0]
        raise TritonClientError(
            f"Unexpected output tensor shape from Triton: {output.shape}."
        )

    def _mock_upscale(self, raw_image: bytes, scale_factor: float) -> TritonResult:
        from PIL import Image
        import io

        if scale_factor <= 0:
            scale_factor = 2.0

        with Image.open(io.BytesIO(raw_image)) as img:
            rgb = img.convert("RGB")
            new_w = max(1, int(rgb.width * scale_factor))
            new_h = max(1, int(rgb.height * scale_factor))
            out = rgb.resize((new_w, new_h), Image.Resampling.BICUBIC)
            buf = io.BytesIO()
            out.save(buf, format="PNG")

        return TritonResult(
            upscaled_image=buf.getvalue(),
            width=new_w,
            height=new_h,
            compute_latency_ms=0.0,
        )
