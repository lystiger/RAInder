# gateway_api

FastAPI API Gateway for RAInder.

## Endpoints

- `GET /health`: basic health probe.
- `POST /upscale`: accepts image upload and forwards inference request to Triton (stubbed for now).

## Run

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Next Implementation Steps

1. Generate Python gRPC stubs from `proto/upscale.proto`.
2. Implement `app/triton_client.py` to call Triton `Process`.
3. Add preprocessing (NCHW FP32) and postprocessing logic.
