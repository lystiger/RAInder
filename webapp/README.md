# webapp

React frontend for RAInder.

## Features

- Image upload form (JPG/PNG, up to 10 MB).
- Model name and scale factor controls.
- Before/after comparison slider.
- Inference metrics display (`inference_time_ms`, status).

## API Target

- POST `/upscale` on the FastAPI API Gateway.

## Run

```bash
cd webapp
npm install
npm run dev
```

Optional environment variable:

- `VITE_API_BASE_URL` (default `http://localhost:8000`)
