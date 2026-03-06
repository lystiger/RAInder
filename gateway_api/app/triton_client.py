from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import InferenceServerException

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - validated by runtime mode checks
    ort = None

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
        self.backend = os.getenv("INFERENCE_BACKEND", "triton").lower()
        self.triton_url = os.getenv("TRITON_URL", "localhost:8001")
        self.default_model_name = os.getenv("TRITON_MODEL_NAME", "real_esrgan_x2")
        self.mock_mode = os.getenv("TRITON_MOCK", "false").lower() in {"1", "true", "yes"}
        if self.mock_mode:
            self.backend = "mock"

        self.client: grpcclient.InferenceServerClient | None = None
        self.local_sessions: dict[str, Any] = {}
        self.local_input_names: dict[str, str] = {}
        self.local_output_names: dict[str, str] = {}

        if self.backend == "triton":
            self.client = grpcclient.InferenceServerClient(url=self.triton_url)
        elif self.backend == "onnx_local":
            self._init_onnx_local()
        elif self.backend != "mock":
            raise TritonClientError(
                f"Unsupported INFERENCE_BACKEND='{self.backend}'. Use triton, onnx_local, or mock."
            )

    def readiness(self, model_name: str | None = None) -> dict[str, bool | str]:
        """Return inference backend readiness details for API probes."""
        if self.backend == "mock":
            return {
                "mode": "mock",
                "server_live": True,
                "server_ready": True,
                "model_ready": True,
                "model_name": model_name or self.default_model_name,
            }

        selected_model = model_name or self.default_model_name
        if self.backend == "onnx_local":
            return {
                "mode": "onnx_local",
                "server_live": True,
                "server_ready": True,
                "model_ready": selected_model in self.local_sessions,
                "model_name": selected_model,
            }

        try:
            if self.client is None:
                raise TritonClientError("Triton client is not initialized.")
            server_live = bool(self.client.is_server_live())
            server_ready = bool(self.client.is_server_ready())
            model_ready = bool(self.client.is_model_ready(selected_model))
        except InferenceServerException as exc:
            raise TritonClientError(str(exc)) from exc

        return {
            "mode": "triton",
            "server_live": server_live,
            "server_ready": server_ready,
            "model_ready": model_ready,
            "model_name": selected_model,
        }

    def upscale(self, raw_image: bytes, model_name: str, scale_factor: float) -> TritonResult:
        if self.backend == "mock":
            return self._mock_upscale(raw_image, scale_factor)
        if self.backend == "onnx_local":
            return self._onnx_local_upscale(raw_image, model_name, scale_factor)

        request_model = model_name if model_name else self.default_model_name
        chw = decode_image_to_chw_fp32(raw_image)

        infer_input = grpcclient.InferInput("input", list(chw.shape), "FP32")
        infer_input.set_data_from_numpy(chw.astype(np.float32))
        output_req = grpcclient.InferRequestedOutput("output")

        started = perf_counter()
        try:
            if self.client is None:
                raise TritonClientError("Triton client is not initialized.")
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

    def _init_onnx_local(self) -> None:
        if ort is None:
            raise TritonClientError(
                "onnxruntime is not installed. Install requirements or set INFERENCE_BACKEND=mock."
            )

        root = Path(__file__).resolve().parents[2]
        model_paths = {
            "real_esrgan_x2": Path(
                os.getenv(
                    "ONNX_MODEL_X2_PATH",
                    str(root / "model_repo" / "real_esrgan_x2" / "1" / "model.onnx"),
                )
            ),
            "real_esrgan_x4": Path(
                os.getenv(
                    "ONNX_MODEL_X4_PATH",
                    str(root / "model_repo" / "real_esrgan_x4" / "1" / "model.onnx"),
                )
            ),
        }

        for model_name, model_path in model_paths.items():
            if not model_path.exists():
                continue
            session = ort.InferenceSession(
                str(model_path),
                providers=["CPUExecutionProvider"],
            )
            self.local_sessions[model_name] = session
            self.local_input_names[model_name] = session.get_inputs()[0].name
            self.local_output_names[model_name] = session.get_outputs()[0].name

        if not self.local_sessions:
            raise TritonClientError(
                "INFERENCE_BACKEND=onnx_local but no ONNX models were found. "
                "Check ONNX_MODEL_X2_PATH/ONNX_MODEL_X4_PATH."
            )

    def _onnx_local_upscale(self, raw_image: bytes, model_name: str, scale_factor: float) -> TritonResult:
        request_model = model_name.strip() if model_name else ""
        if not request_model:
            request_model = "real_esrgan_x2" if scale_factor == 2.0 else "real_esrgan_x4"

        session = self.local_sessions.get(request_model)
        if session is None:
            available = ", ".join(sorted(self.local_sessions))
            raise TritonClientError(
                f"Model '{request_model}' is not loaded for onnx_local backend. "
                f"Available: {available or '(none)'}."
            )

        chw = decode_image_to_chw_fp32(raw_image).astype(np.float32)
        started = perf_counter()
        output = self._run_onnx_session(request_model, chw)
        duration_ms = (perf_counter() - started) * 1000.0

        output_chw = self._normalize_output_shape(output)
        upscaled_image, width, height = encode_chw_fp32_to_png_bytes(output_chw)
        return TritonResult(
            upscaled_image=upscaled_image,
            width=width,
            height=height,
            compute_latency_ms=duration_ms,
        )

    def _run_onnx_session(self, model_name: str, chw: np.ndarray) -> np.ndarray:
        session = self.local_sessions[model_name]
        input_name = self.local_input_names[model_name]
        output_name = self.local_output_names[model_name]

        attempts = [np.expand_dims(chw, axis=0), chw]
        last_exc: Exception | None = None
        for tensor in attempts:
            try:
                outputs = session.run([output_name], {input_name: tensor})
                return outputs[0]
            except Exception as exc:  # pragma: no cover - depends on model signature
                last_exc = exc

        raise TritonClientError(
            f"ONNX local inference failed for model '{model_name}': {last_exc}"
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
