# RAInder

Custom AI upscaling playground with Triton and UI.

## Repository Layout

- `docs/`: project specs, contracts, and architecture docs.
- `gateway_api/`: FastAPI API Gateway scaffold.
- `model_repo/`: Triton model repository scaffold.
- `webapp/`: frontend scaffold.

## Current Status

- M1 bootstrap complete: `docs/system_design.md` and initial gRPC contract.
- M2 bootstrap complete: Triton `config.pbtxt` scaffold.
- M3 bootstrap complete: FastAPI `/health` and `/upscale` endpoint skeleton.

## Quick Start (Gateway)

```bash
cd gateway_api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
