# gateway_api

FastAPI API Gateway for RAInder.

## Endpoints

- `GET /health`: basic health probe.
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

## Next Implementation Steps

1. Generate Python gRPC stubs from `proto/upscale.proto`.
2. Wire model-specific pre/postprocessing for Real-ESRGAN.
3. Add integration tests against a running Triton container.
