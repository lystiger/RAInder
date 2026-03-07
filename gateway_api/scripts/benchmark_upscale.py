#!/usr/bin/env python3
import argparse
import csv
import json
import mimetypes
import shutil
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "real_esrgan_x2": {"endpoint": "/upscale", "scale_factor": 2.0},
    "real_esrgan_x4": {"endpoint": "/upscale", "scale_factor": 4.0},
    "anime_gan_hayao": {"endpoint": "/anime/hayao", "scale_factor": 1.0},
}


@dataclass
class RunSample:
    api_latency_ms: float
    inference_latency_ms: float
    gpu_util_percent: float | None
    gpu_memory_used_mb: float | None
    gpu_memory_total_mb: float | None


@dataclass
class BenchmarkSummary:
    image: str
    model_name: str
    endpoint: str
    scale_factor: float
    runs: int
    warmup: int
    api_latency_mean_ms: float
    api_latency_median_ms: float
    api_latency_p95_ms: float
    api_latency_min_ms: float
    api_latency_max_ms: float
    inference_latency_mean_ms: float
    inference_latency_median_ms: float
    inference_latency_p95_ms: float
    inference_latency_min_ms: float
    inference_latency_max_ms: float
    throughput_images_per_sec: float
    gpu_util_mean_percent: float | None
    gpu_memory_used_mean_mb: float | None
    gpu_memory_peak_mb: float | None
    gpu_memory_total_mb: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark gateway models for target GPU baselining.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Gateway API base URL")
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="One or more PNG/JPEG input image paths",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["real_esrgan_x2", "real_esrgan_x4"],
        choices=sorted(MODEL_SPECS),
        help="Models to benchmark",
    )
    parser.add_argument("--runs", type=int, default=20, help="Measured runs per image/model")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs per image/model")
    parser.add_argument(
        "--output-json",
        default="benchmark_results.json",
        help="Path to write summary JSON",
    )
    parser.add_argument(
        "--output-csv",
        default="benchmark_results.csv",
        help="Path to write summary CSV",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="HTTP timeout per request in seconds",
    )
    parser.add_argument(
        "--skip-gpu-sampling",
        action="store_true",
        help="Do not query nvidia-smi during measured runs",
    )
    return parser.parse_args()


def percentile_p95(values: list[float]) -> float:
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100)[94]


def maybe_sample_gpu() -> dict[str, float] | None:
    if shutil.which("nvidia-smi") is None:
        return None

    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None

    line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    if not line:
        return None

    try:
        util, used, total = [float(part.strip()) for part in line.split(",")[:3]]
    except ValueError:
        return None

    return {
        "gpu_util_percent": util,
        "gpu_memory_used_mb": used,
        "gpu_memory_total_mb": total,
    }


def request_spec(model_name: str) -> tuple[str, dict[str, str]]:
    spec = MODEL_SPECS[model_name]
    endpoint = spec["endpoint"]
    if endpoint == "/upscale":
        data = {
            "model_name": model_name,
            "scale_factor": str(spec["scale_factor"]),
        }
    else:
        data = {}
    return endpoint, data


def run_case(
    client: httpx.Client,
    api_url: str,
    image_path: Path,
    model_name: str,
    runs: int,
    warmup: int,
    sample_gpu: bool,
) -> BenchmarkSummary:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    payload = image_path.read_bytes()
    mime_type, _ = mimetypes.guess_type(str(image_path))
    content_type = mime_type or "application/octet-stream"
    endpoint, data = request_spec(model_name)
    samples: list[RunSample] = []
    measured_started = 0.0

    for index in range(warmup + runs):
        if index == warmup:
            measured_started = time.perf_counter()
        started = time.perf_counter()
        response = client.post(
            f"{api_url.rstrip('/')}{endpoint}",
            files={"image": (image_path.name, payload, content_type)},
            data=data,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        if response.status_code != 200:
            raise RuntimeError(
                f"{model_name} on {image_path} failed at run {index + 1} "
                f"with status {response.status_code}: {response.text}"
            )

        body = response.json()
        inference_ms = float(body["inference_time_ms"])

        if index >= warmup:
            gpu = maybe_sample_gpu() if sample_gpu else None
            samples.append(
                RunSample(
                    api_latency_ms=elapsed_ms,
                    inference_latency_ms=inference_ms,
                    gpu_util_percent=None if gpu is None else gpu["gpu_util_percent"],
                    gpu_memory_used_mb=None if gpu is None else gpu["gpu_memory_used_mb"],
                    gpu_memory_total_mb=None if gpu is None else gpu["gpu_memory_total_mb"],
                )
            )

    measured_elapsed = time.perf_counter() - measured_started
    api_latencies = [item.api_latency_ms for item in samples]
    inference_latencies = [item.inference_latency_ms for item in samples]
    gpu_utils = [item.gpu_util_percent for item in samples if item.gpu_util_percent is not None]
    gpu_memory_used = [item.gpu_memory_used_mb for item in samples if item.gpu_memory_used_mb is not None]
    gpu_memory_total = [item.gpu_memory_total_mb for item in samples if item.gpu_memory_total_mb is not None]

    return BenchmarkSummary(
        image=str(image_path),
        model_name=model_name,
        endpoint=endpoint,
        scale_factor=float(MODEL_SPECS[model_name]["scale_factor"]),
        runs=runs,
        warmup=warmup,
        api_latency_mean_ms=statistics.fmean(api_latencies),
        api_latency_median_ms=statistics.median(api_latencies),
        api_latency_p95_ms=percentile_p95(api_latencies),
        api_latency_min_ms=min(api_latencies),
        api_latency_max_ms=max(api_latencies),
        inference_latency_mean_ms=statistics.fmean(inference_latencies),
        inference_latency_median_ms=statistics.median(inference_latencies),
        inference_latency_p95_ms=percentile_p95(inference_latencies),
        inference_latency_min_ms=min(inference_latencies),
        inference_latency_max_ms=max(inference_latencies),
        throughput_images_per_sec=(runs / measured_elapsed) if measured_elapsed > 0 else 0.0,
        gpu_util_mean_percent=(statistics.fmean(gpu_utils) if gpu_utils else None),
        gpu_memory_used_mean_mb=(statistics.fmean(gpu_memory_used) if gpu_memory_used else None),
        gpu_memory_peak_mb=(max(gpu_memory_used) if gpu_memory_used else None),
        gpu_memory_total_mb=(max(gpu_memory_total) if gpu_memory_total else None),
    )


def print_summary(summary: BenchmarkSummary) -> None:
    print(
        f"{summary.model_name} | image={Path(summary.image).name} | "
        f"api_mean={summary.api_latency_mean_ms:.2f} ms | "
        f"infer_mean={summary.inference_latency_mean_ms:.2f} ms | "
        f"throughput={summary.throughput_images_per_sec:.3f} img/s | "
        f"gpu_mem_peak={summary.gpu_memory_peak_mb if summary.gpu_memory_peak_mb is not None else 'n/a'} MB"
    )


def write_json(path: Path, summaries: list[BenchmarkSummary]) -> None:
    payload = [asdict(summary) for summary in summaries]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, summaries: list[BenchmarkSummary]) -> None:
    rows = [asdict(summary) for summary in summaries]
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    images = [Path(item).resolve() for item in args.images]
    summaries: list[BenchmarkSummary] = []
    sample_gpu = not args.skip_gpu_sampling

    with httpx.Client(timeout=args.timeout) as client:
        for image_path in images:
            for model_name in args.models:
                summary = run_case(
                    client=client,
                    api_url=args.api_url,
                    image_path=image_path,
                    model_name=model_name,
                    runs=args.runs,
                    warmup=args.warmup,
                    sample_gpu=sample_gpu,
                )
                summaries.append(summary)
                print_summary(summary)

    json_path = Path(args.output_json).resolve()
    csv_path = Path(args.output_csv).resolve()
    write_json(json_path, summaries)
    write_csv(csv_path, summaries)
    print(f"JSON written to {json_path}")
    print(f"CSV written to {csv_path}")


if __name__ == "__main__":
    main()
