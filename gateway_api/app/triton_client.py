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


@dataclass(frozen=True)
class ModelProfile:
    input_name: str
    output_name: str
    input_layout: str  # nchw or nhwc
    output_layout: str  # nchw, nhwc, chw
    input_rank: int
    tile_hw: tuple[int, int] | None = None
    input_range: str = "zero_to_one"
    output_range: str = "zero_to_one"


MODEL_PROFILES: dict[str, ModelProfile] = {
    "real_esrgan_x2": ModelProfile(
        input_name="input",
        output_name="output",
        input_layout="nchw",
        output_layout="nchw",
        input_rank=4,
        tile_hw=(64, 64),
    ),
    "real_esrgan_x4": ModelProfile(
        input_name="image",
        output_name="upscaled_image",
        input_layout="nchw",
        output_layout="chw",
        input_rank=4,
        tile_hw=(128, 128),
    ),
    "anime_gan_hayao": ModelProfile(
        input_name="generator_input:0",
        output_name="generator/G_MODEL/out_layer/Tanh:0",
        input_layout="nhwc",
        output_layout="nhwc",
        input_rank=4,
        input_range="neg_one_to_one",
        output_range="neg_one_to_one",
    ),
}


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
        self.local_input_shapes: dict[str, list[Any]] = {}
        self.local_output_names: dict[str, str] = {}
        self.local_output_shapes: dict[str, list[Any]] = {}

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

        request_model = model_name.strip() if model_name else self.default_model_name
        profile = MODEL_PROFILES.get(
            request_model,
            ModelProfile(
                input_name="input",
                output_name="output",
                input_layout="nchw",
                output_layout="nchw",
                input_rank=3,
            ),
        )
        chw = decode_image_to_chw_fp32(raw_image)

        started = perf_counter()
        if profile.tile_hw is not None:
            output_chw = self._triton_tiled_upscale(request_model, chw, profile, scale_factor)
        else:
            output_chw = self._triton_single_infer(request_model, chw, profile)
        duration_ms = (perf_counter() - started) * 1000.0

        upscaled_image, width, height = encode_chw_fp32_to_png_bytes(output_chw)

        return TritonResult(
            upscaled_image=upscaled_image,
            width=width,
            height=height,
            compute_latency_ms=duration_ms,
        )

    def _triton_single_infer(
        self,
        model_name: str,
        chw: np.ndarray,
        profile: ModelProfile,
    ) -> np.ndarray:
        tensor = self._prepare_triton_input(chw, profile)

        infer_input = grpcclient.InferInput(profile.input_name, list(tensor.shape), "FP32")
        infer_input.set_data_from_numpy(tensor)
        output_req = grpcclient.InferRequestedOutput(profile.output_name)

        try:
            if self.client is None:
                raise TritonClientError("Triton client is not initialized.")
            response = self.client.infer(
                model_name=model_name,
                inputs=[infer_input],
                outputs=[output_req],
            )
        except InferenceServerException as exc:
            raise TritonClientError(str(exc)) from exc

        output = response.as_numpy(profile.output_name)
        if output is None:
            raise TritonClientError(
                f"Triton returned no output tensor named '{profile.output_name}' for model '{model_name}'."
            )

        return self._normalize_output(output, profile)

    def _triton_tiled_upscale(
        self,
        model_name: str,
        chw: np.ndarray,
        profile: ModelProfile,
        fallback_scale: float,
    ) -> np.ndarray:
        if profile.tile_hw is None:
            return self._triton_single_infer(model_name, chw, profile)

        tile_h, tile_w = profile.tile_hw
        in_h, in_w = chw.shape[1], chw.shape[2]
        output_scale = float(fallback_scale if fallback_scale > 0 else 1.0)
        out_h = int(round(in_h * output_scale))
        out_w = int(round(in_w * output_scale))
        output_canvas = np.zeros((3, out_h, out_w), dtype=np.float32)

        inferred_scale: float | None = None

        for top in range(0, in_h, tile_h):
            for left in range(0, in_w, tile_w):
                bottom = min(top + tile_h, in_h)
                right = min(left + tile_w, in_w)
                patch = chw[:, top:bottom, left:right]

                pad_h = tile_h - patch.shape[1]
                pad_w = tile_w - patch.shape[2]
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="edge",
                    )

                patch_out = self._triton_single_infer(model_name, patch, profile)

                if inferred_scale is None and tile_h > 0 and tile_w > 0:
                    sh = patch_out.shape[1] / tile_h
                    sw = patch_out.shape[2] / tile_w
                    if sh > 0 and abs(sh - sw) < 1e-6:
                        inferred_scale = sh
                        out_h = int(round(in_h * inferred_scale))
                        out_w = int(round(in_w * inferred_scale))
                        output_canvas = np.zeros((3, out_h, out_w), dtype=np.float32)

                scale = inferred_scale if inferred_scale is not None else output_scale
                valid_out_h = int(round((bottom - top) * scale))
                valid_out_w = int(round((right - left) * scale))

                y0 = int(round(top * scale))
                x0 = int(round(left * scale))
                y1 = min(y0 + valid_out_h, out_h)
                x1 = min(x0 + valid_out_w, out_w)

                output_canvas[:, y0:y1, x0:x1] = patch_out[:, : (y1 - y0), : (x1 - x0)]

        return output_canvas

    def _prepare_triton_input(self, chw: np.ndarray, profile: ModelProfile) -> np.ndarray:
        normalized = self._apply_input_range(chw, profile)
        if profile.input_layout == "nchw":
            if profile.input_rank == 4:
                return np.expand_dims(normalized, axis=0).astype(np.float32)
            if profile.input_rank == 3:
                return normalized.astype(np.float32)
        elif profile.input_layout == "nhwc":
            hwc = np.transpose(normalized, (1, 2, 0))
            if profile.input_rank == 4:
                return np.expand_dims(hwc, axis=0).astype(np.float32)
            if profile.input_rank == 3:
                return hwc.astype(np.float32)

        raise TritonClientError(
            f"Unsupported Triton input format for profile: layout={profile.input_layout}, "
            f"rank={profile.input_rank}."
        )

    def _normalize_output(self, output: np.ndarray, profile: ModelProfile) -> np.ndarray:
        if profile.output_layout == "nchw":
            if output.ndim == 4:
                chw = output[0]
                return self._apply_output_range(chw, profile)
            if output.ndim == 3:
                return self._apply_output_range(output, profile)
        elif profile.output_layout == "nhwc":
            if output.ndim == 4:
                chw = np.transpose(output[0], (2, 0, 1))
                return self._apply_output_range(chw, profile)
            if output.ndim == 3:
                chw = np.transpose(output, (2, 0, 1))
                return self._apply_output_range(chw, profile)
        elif profile.output_layout == "chw":
            if output.ndim == 3:
                return self._apply_output_range(output, profile)
            if output.ndim == 4:
                return self._apply_output_range(output[0], profile)

        raise TritonClientError(
            f"Unexpected output tensor shape from Triton for layout '{profile.output_layout}': {output.shape}."
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
            "anime_gan_hayao": Path(
                os.getenv(
                    "ONNX_MODEL_ANIME_HAYAO_PATH",
                    str(root / "model_repo" / "AnimeGANv2_Hayao.onnx"),
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
            input_meta = session.get_inputs()[0]
            self.local_input_names[model_name] = input_meta.name
            self.local_input_shapes[model_name] = list(input_meta.shape)
            output_meta = session.get_outputs()[0]
            self.local_output_names[model_name] = output_meta.name
            self.local_output_shapes[model_name] = list(output_meta.shape)

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
        input_shape = self.local_input_shapes[request_model]
        tile_hw = self._extract_static_hw(input_shape)
        if tile_hw is None:
            output = self._run_onnx_session(request_model, chw)
            output_chw = self._prepare_onnx_output_to_chw(request_model, output)
        else:
            output_chw = self._onnx_local_tiled_upscale(request_model, chw, tile_hw, scale_factor)
        duration_ms = (perf_counter() - started) * 1000.0

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
        input_shape = self.local_input_shapes[model_name]
        expected_rank = len(input_shape)
        tensor = self._prepare_onnx_input(model_name, chw, input_shape)
        if tensor.ndim != expected_rank:
            raise TritonClientError(
                f"ONNX input rank mismatch for model '{model_name}': "
                f"prepared rank={tensor.ndim}, expected rank={expected_rank}, shape={input_shape}"
            )
        try:
            outputs = session.run([output_name], {input_name: tensor})
            return outputs[0]
        except Exception as exc:  # pragma: no cover - depends on model signature
            raise TritonClientError(
                f"ONNX local inference failed for model '{model_name}': {exc}"
            ) from exc

    def _onnx_local_tiled_upscale(
        self,
        model_name: str,
        chw: np.ndarray,
        tile_hw: tuple[int, int],
        fallback_scale: float,
    ) -> np.ndarray:
        tile_h, tile_w = tile_hw
        in_h, in_w = chw.shape[1], chw.shape[2]
        output_shape = self.local_output_shapes[model_name]
        scale = self._infer_scale_factor(tile_h, tile_w, output_shape, fallback_scale)

        out_h = int(round(in_h * scale))
        out_w = int(round(in_w * scale))
        output_canvas = np.zeros((3, out_h, out_w), dtype=np.float32)

        for top in range(0, in_h, tile_h):
            for left in range(0, in_w, tile_w):
                bottom = min(top + tile_h, in_h)
                right = min(left + tile_w, in_w)
                patch = chw[:, top:bottom, left:right]

                pad_h = tile_h - patch.shape[1]
                pad_w = tile_w - patch.shape[2]
                if pad_h > 0 or pad_w > 0:
                    patch = np.pad(
                        patch,
                        ((0, 0), (0, pad_h), (0, pad_w)),
                        mode="edge",
                    )

                patch_out_raw = self._run_onnx_session(model_name, patch)
                patch_out = self._prepare_onnx_output_to_chw(model_name, patch_out_raw)
                valid_out_h = int(round((bottom - top) * scale))
                valid_out_w = int(round((right - left) * scale))
                y0 = int(round(top * scale))
                x0 = int(round(left * scale))
                y1 = min(y0 + valid_out_h, out_h)
                x1 = min(x0 + valid_out_w, out_w)
                output_canvas[:, y0:y1, x0:x1] = patch_out[:, : (y1 - y0), : (x1 - x0)]

        return output_canvas

    def _prepare_onnx_output_to_chw(self, model_name: str, output: np.ndarray) -> np.ndarray:
        output_shape = self.local_output_shapes[model_name]
        profile = MODEL_PROFILES.get(model_name, ModelProfile("", "", "nchw", "nchw", 4))
        if output.ndim == 4:
            # NCHW
            if self._dim_equals(output_shape, 1, 3) or output.shape[1] == 3:
                return self._apply_output_range(output[0], profile)
            # NHWC
            if self._dim_equals(output_shape, 3, 3) or output.shape[3] == 3:
                chw = np.transpose(output[0], (2, 0, 1))
                return self._apply_output_range(chw, profile)
            return self._apply_output_range(output[0], profile)
        if output.ndim == 3:
            if self._dim_equals(output_shape, 0, 3) or output.shape[0] == 3:
                return self._apply_output_range(output, profile)
            if self._dim_equals(output_shape, 2, 3) or output.shape[2] == 3:
                chw = np.transpose(output, (2, 0, 1))
                return self._apply_output_range(chw, profile)
            return self._apply_output_range(output, profile)
        raise TritonClientError(f"Unsupported ONNX output rank: {output.ndim} for model '{model_name}'.")

    def _extract_static_hw(self, input_shape: list[Any]) -> tuple[int, int] | None:
        rank = len(input_shape)
        if rank != 4:
            return None
        # NCHW
        if self._dim_equals(input_shape, 1, 3):
            h = input_shape[2]
            w = input_shape[3]
        # NHWC
        elif self._dim_equals(input_shape, 3, 3):
            h = input_shape[1]
            w = input_shape[2]
        else:
            h = input_shape[2]
            w = input_shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return (h, w)
        return None

    def _infer_scale_factor(
        self,
        in_h: int,
        in_w: int,
        output_shape: list[Any],
        fallback_scale: float,
    ) -> float:
        if len(output_shape) == 4:
            # NCHW
            if self._dim_equals(output_shape, 1, 3):
                oh, ow = output_shape[2], output_shape[3]
            # NHWC
            elif self._dim_equals(output_shape, 3, 3):
                oh, ow = output_shape[1], output_shape[2]
            else:
                oh, ow = output_shape[2], output_shape[3]
            if isinstance(oh, int) and isinstance(ow, int) and oh > 0 and ow > 0:
                sh = oh / in_h
                sw = ow / in_w
                if sh > 0 and abs(sh - sw) < 1e-6:
                    return sh
        if len(output_shape) == 3:
            if self._dim_equals(output_shape, 0, 3):
                oh, ow = output_shape[1], output_shape[2]
            elif self._dim_equals(output_shape, 2, 3):
                oh, ow = output_shape[0], output_shape[1]
            else:
                oh, ow = output_shape[1], output_shape[2]
            if isinstance(oh, int) and isinstance(ow, int) and oh > 0 and ow > 0:
                sh = oh / in_h
                sw = ow / in_w
                if sh > 0 and abs(sh - sw) < 1e-6:
                    return sh
        return float(fallback_scale if fallback_scale > 0 else 1.0)

    def _prepare_onnx_input(
        self,
        model_name: str,
        chw: np.ndarray,
        input_shape: list[Any],
    ) -> np.ndarray:
        profile = MODEL_PROFILES.get(model_name, ModelProfile("", "", "nchw", "nchw", len(input_shape)))
        normalized = self._apply_input_range(chw, profile)
        rank = len(input_shape)
        if rank == 4:
            # Prefer NCHW when channel dimension is at index 1.
            if self._dim_equals(input_shape, 1, 3):
                return np.expand_dims(normalized, axis=0).astype(np.float32)
            # NHWC when channel dimension is at index 3.
            if self._dim_equals(input_shape, 3, 3):
                hwc = np.transpose(normalized, (1, 2, 0))
                return np.expand_dims(hwc, axis=0).astype(np.float32)
            # Default to NCHW for unknown symbolic shapes.
            return np.expand_dims(normalized, axis=0).astype(np.float32)

        if rank == 3:
            if self._dim_equals(input_shape, 0, 3):
                return normalized.astype(np.float32)
            if self._dim_equals(input_shape, 2, 3):
                return np.transpose(normalized, (1, 2, 0)).astype(np.float32)
            return normalized.astype(np.float32)

        raise TritonClientError(f"Unsupported ONNX input rank: {rank} (shape={input_shape}).")

    def _apply_input_range(self, chw: np.ndarray, profile: ModelProfile) -> np.ndarray:
        if profile.input_range == "neg_one_to_one":
            return (chw * 2.0) - 1.0
        return chw

    def _apply_output_range(self, chw: np.ndarray, profile: ModelProfile) -> np.ndarray:
        if profile.output_range == "neg_one_to_one":
            return (chw + 1.0) / 2.0
        return chw

    @staticmethod
    def _dim_equals(shape: list[Any], idx: int, value: int) -> bool:
        dim = shape[idx]
        if isinstance(dim, int):
            return dim == value
        if isinstance(dim, str):
            try:
                return int(dim) == value
            except ValueError:
                return False
        return False

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
