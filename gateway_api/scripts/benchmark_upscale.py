#!/usr/bin/env python3
import argparse
import statistics
import time
from pathlib import Path

import httpx


def run_benchmark(
    api_url: str,
    image_path: Path,
    runs: int,
    warmup: int,
    scale_factor: float,
    model_name: str,
) -> None:
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    payload = image_path.read_bytes()
    durations = []

    with httpx.Client(timeout=60.0) as client:
        for i in range(warmup + runs):
            started = time.perf_counter()
            response = client.post(
                f"{api_url.rstrip('/')}/upscale",
                files={"image": (image_path.name, payload, "image/png")},
                data={"scale_factor": str(scale_factor), "model_name": model_name},
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0

            if response.status_code != 200:
                raise RuntimeError(
                    f"Run {i + 1} failed with status {response.status_code}: {response.text}"
                )

            if i >= warmup:
                durations.append(elapsed_ms)

    print(f"Runs: {runs} (warmup: {warmup})")
    print(f"API: {api_url}")
    print(f"Image: {image_path}")
    print(f"Model: {model_name}, Scale: {scale_factor}")
    print(f"mean_ms={statistics.fmean(durations):.2f}")
    print(f"median_ms={statistics.median(durations):.2f}")
    print(f"p95_ms={statistics.quantiles(durations, n=100)[94]:.2f}")
    print(f"min_ms={min(durations):.2f}")
    print(f"max_ms={max(durations):.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark /upscale endpoint latency.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Gateway API base URL")
    parser.add_argument("--image", required=True, help="Path to PNG/JPEG image")
    parser.add_argument("--runs", type=int, default=20, help="Measured runs")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs")
    parser.add_argument("--scale-factor", type=float, default=2.0, choices=[2.0, 4.0])
    parser.add_argument("--model-name", default="super_resolution_model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        api_url=args.api_url,
        image_path=Path(args.image),
        runs=args.runs,
        warmup=args.warmup,
        scale_factor=args.scale_factor,
        model_name=args.model_name,
    )
