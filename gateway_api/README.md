# gateway_api

FastAPI API Gateway for RAInder.

## Endpoints

- `GET /health`: basic health probe.
- `GET /ready`: readiness probe for Triton and selected model.
- `POST /upscale`: accepts image upload and forwards inference request to Triton over gRPC.

## Run

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Environment Variables

- `TRITON_URL` (default: `localhost:8001`): Triton gRPC endpoint.
- `TRITON_MODEL_NAME` (default: `super_resolution_model`): fallback model name.
- `TRITON_MOCK` (default: `false`): if true, use bicubic mock upscale path.
- `CORS_ORIGINS`: comma-separated allowed web origins.

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
python scripts/benchmark_upscale.py --image ../docs/Architecture.png --runs 20 --warmup 3
```
