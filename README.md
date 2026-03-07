# RAInder

Custom AI upscaling playground with Triton and UI.

## Repository Layout

- `docs/`: project specs, contracts, and architecture docs.
- `gateway_api/`: FastAPI API Gateway scaffold.
- `model_repo/`: Triton model repository scaffold.
- `webapp/`: frontend scaffold.

## Current Status

- M1 complete: architecture docs, sequence diagrams, and initial gRPC contract.
- M2 complete: Triton model repository, Docker Compose wiring, and local Triton runtime validation.
- M3 complete: FastAPI gateway with `/health`, `/ready`, `/upscale`, anime route support, and automated API coverage.
- M4 bootstrap complete: React webapp with upload flow, model selection, and before/after comparison UI.

## Recent Progress

- Triton Docker Compose stack has been validated successfully on a target PC.
- Added `onnx_local` CPU fallback for local evaluation without Triton/GPU.
- Added regression coverage for AnimeGAN normalization/output handling.
- Added `gateway_api/scripts/evaluate_models.py` for repeatable local model comparison runs.

## Quick Start (Gateway)

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

CPU-only laptop mode (no Triton/GPU):

```bash
cd gateway_api
source .venv/bin/activate
INFERENCE_BACKEND=onnx_local uvicorn app.main:app --reload
```

## Quick Start (Webapp)

```bash
cd webapp
npm install
npm run dev
```

## Quick Start (Docker Compose)

Requirements:

- Docker + Docker Compose
- NVIDIA Container Toolkit (for Triton GPU usage)

```bash
docker compose up --build
```

Service URLs:

- Gateway API: `http://localhost:8000`
- Webapp: `http://localhost:5173`
- Triton gRPC: `localhost:8001`
- Triton HTTP (mapped): `http://localhost:8003`
