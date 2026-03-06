# webapp

React frontend for RAInder.

## Features

- Image upload form (JPG/PNG, up to 10 MB).
- Model selection for Real-ESRGAN and anime filters.
- Scale factor controls for Real-ESRGAN (`2x`, `4x`).
- Before/after comparison slider.
- Inference metrics display (`inference_time_ms`, status).

## API Target

- POST `/upscale` for Real-ESRGAN models.
- POST `/anime/hayao` for `anime_gan_hayao`.

## Run

```bash
cd webapp
npm install
npm run dev
```

Optional environment variable:

- `VITE_API_BASE_URL` (default `http://localhost:8000`)
