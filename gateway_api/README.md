# gateway_api

FastAPI API Gateway for RAInder.

## Endpoints

- `GET /health`: basic health probe.
- `GET /ready`: readiness probe for Triton and selected model.
- `POST /upscale`: accepts image upload and forwards inference request to Triton over gRPC.
- `POST /anime/hayao`: anime style transfer using `anime_gan_hayao`.

## Run

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --log-config logging.ini
```

## Environment Variables

- `INFERENCE_BACKEND` (default: `triton`): `triton`, `onnx_local`, or `mock`.
- `TRITON_URL` (default: `localhost:8001`): Triton gRPC endpoint.
- `TRITON_MODEL_NAME` (default: `real_esrgan_x2`): fallback model name.
- `TRITON_MOCK` (default: `false`): if true, use bicubic mock upscale path.
- `ONNX_MODEL_X2_PATH`: local ONNX path used in `onnx_local` mode.
- `ONNX_MODEL_X4_PATH`: local ONNX path used in `onnx_local` mode.
- `ONNX_MODEL_ANIME_HAYAO_PATH`: local ONNX path for anime Hayao model.
- `CORS_ORIGINS`: comma-separated allowed web origins.

## Non-GPU Local Run (Ubuntu Laptop)

Use ONNX Runtime CPU backend instead of Triton:

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
INFERENCE_BACKEND=onnx_local uvicorn app.main:app --reload
```

If you only need UI/API flow validation without real model inference:

```bash
INFERENCE_BACKEND=mock uvicorn app.main:app --reload
```

## Testing

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q
```

## Benchmark

```bash
cd gateway_api
source .venv/bin/activate
python scripts/benchmark_upscale.py \
  --images ../test.png \
  --models real_esrgan_x2 real_esrgan_x4 anime_gan_hayao \
  --runs 20 \
  --warmup 3 \
  --output-json benchmark_results.json \
  --output-csv benchmark_results.csv
```

Notes:

- `real_esrgan_x2` and `real_esrgan_x4` use `POST /upscale`
- `anime_gan_hayao` uses `POST /anime/hayao`
- GPU metrics are sampled with `nvidia-smi` when available
