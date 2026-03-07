#!/usr/bin/env python3
import argparse
import io
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[2]
GATEWAY_ROOT = ROOT / "gateway_api"
if str(GATEWAY_ROOT) not in sys.path:
    sys.path.insert(0, str(GATEWAY_ROOT))

from app.triton_client import TritonClient


DEFAULT_OUTPUT_DIR = ROOT / "evaluation_outputs"
MODEL_RUNS = (
    ("real_esrgan_x2", 2.0),
    ("real_esrgan_x4", 4.0),
    ("anime_gan_hayao", 1.0),
)


@dataclass
class Metrics:
    model_name: str
    scale_factor: float
    width: int
    height: int
    latency_ms: float
    psnr_vs_bicubic: float
    ssim_vs_bicubic: float
    mae_vs_bicubic: float
    entropy: float
    sharpness_laplacian_var: float
    edge_density: float
    saturation_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run all local models and report image metrics.")
    parser.add_argument(
        "--image",
        default=str(ROOT / "test.png"),
        help="Input image path. Defaults to ./test.png",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for generated images and metrics JSON.",
    )
    return parser.parse_args()


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_to_float_hwc(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.float32) / 255.0


def grayscale(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("L"), dtype=np.float32) / 255.0


def compute_psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    if mse <= 1e-12:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def compute_ssim(a: np.ndarray, b: np.ndarray) -> float:
    # Global SSIM is sufficient here for relative comparison against bicubic.
    c1 = (0.01 ** 2)
    c2 = (0.03 ** 2)
    mu_a = float(a.mean())
    mu_b = float(b.mean())
    sigma_a = float(a.var())
    sigma_b = float(b.var())
    sigma_ab = float(((a - mu_a) * (b - mu_b)).mean())
    numerator = (2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)
    denominator = (mu_a**2 + mu_b**2 + c1) * (sigma_a + sigma_b + c2)
    return numerator / denominator if denominator else 1.0


def compute_entropy(image: Image.Image) -> float:
    hist = np.asarray(image.convert("L").histogram(), dtype=np.float64)
    probs = hist / hist.sum()
    nonzero = probs[probs > 0]
    return float(-(nonzero * np.log2(nonzero)).sum())


def compute_laplacian_variance(image: Image.Image) -> float:
    gray = image.convert("L").filter(ImageFilter.FIND_EDGES)
    arr = np.asarray(gray, dtype=np.float32)
    return float(arr.var())


def compute_edge_density(image: Image.Image) -> float:
    gray = grayscale(image)
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    mag = np.sqrt(gx**2 + gy**2)
    return float((mag > 0.08).mean())


def compute_saturation_mean(image: Image.Image) -> float:
    arr = image_to_float_hwc(image)
    max_rgb = arr.max(axis=2)
    min_rgb = arr.min(axis=2)
    denom = np.where(max_rgb == 0.0, 1.0, max_rgb)
    saturation = (max_rgb - min_rgb) / denom
    return float(saturation.mean())


def evaluate_run(
    client: TritonClient,
    input_path: Path,
    model_name: str,
    scale_factor: float,
    output_dir: Path,
) -> Metrics:
    raw = input_path.read_bytes()
    result = client._onnx_local_upscale(raw, model_name, scale_factor)
    output_path = output_dir / f"{input_path.stem}_{model_name}.png"
    output_path.write_bytes(result.upscaled_image)

    output_image = load_rgb(output_path)
    baseline = load_rgb(input_path).resize(output_image.size, Image.Resampling.BICUBIC)

    out_arr = image_to_float_hwc(output_image)
    base_arr = image_to_float_hwc(baseline)

    return Metrics(
        model_name=model_name,
        scale_factor=scale_factor,
        width=output_image.width,
        height=output_image.height,
        latency_ms=result.compute_latency_ms,
        psnr_vs_bicubic=compute_psnr(out_arr, base_arr),
        ssim_vs_bicubic=compute_ssim(out_arr, base_arr),
        mae_vs_bicubic=float(np.mean(np.abs(out_arr - base_arr))),
        entropy=compute_entropy(output_image),
        sharpness_laplacian_var=compute_laplacian_variance(output_image),
        edge_density=compute_edge_density(output_image),
        saturation_mean=compute_saturation_mean(output_image),
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.image).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"Input image not found: {input_path}")

    os.environ["INFERENCE_BACKEND"] = "onnx_local"
    os.environ["TRITON_MOCK"] = "false"
    client = TritonClient()

    metrics = [
        evaluate_run(client, input_path, model_name, scale_factor, output_dir)
        for model_name, scale_factor in MODEL_RUNS
    ]

    metrics_path = output_dir / f"{input_path.stem}_metrics.json"
    metrics_path.write_text(
        json.dumps([asdict(item) for item in metrics], indent=2),
        encoding="utf-8",
    )

    print(f"Input: {input_path}")
    print(f"Outputs: {output_dir}")
    print()
    for item in metrics:
        print(
            f"{item.model_name}: size={item.width}x{item.height}, "
            f"latency_ms={item.latency_ms:.2f}, "
            f"psnr_vs_bicubic={item.psnr_vs_bicubic:.2f}, "
            f"ssim_vs_bicubic={item.ssim_vs_bicubic:.4f}, "
            f"mae_vs_bicubic={item.mae_vs_bicubic:.4f}, "
            f"entropy={item.entropy:.3f}, "
            f"sharpness={item.sharpness_laplacian_var:.2f}, "
            f"edge_density={item.edge_density:.4f}, "
            f"saturation_mean={item.saturation_mean:.4f}"
        )

    print()
    print(f"Metrics JSON: {metrics_path}")


if __name__ == "__main__":
    main()
