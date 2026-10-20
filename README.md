# VeriFace AI

VeriFace AI is a Next.js interface and FastAPI service for image analysis with a
ViT attention visualization. All predictions use a fine-tuned real/synthetic
ViT-Base classifier. The backend loads the configured checkpoint lazily on the
first prediction and downloads/converts the configured Hugging Face deepfake
classifier if the local checkpoint is missing.

> **Model limitation:** predictions are probabilistic and should not be used as
> the sole basis for forensic, safety, or production decisions.

## Run locally

1. Start the API and frontend together from the project root:

   ```powershell
   docker compose up --build
   ```

   The first `/predict` request downloads roughly 350 MB of Hugging Face
   weights if no local checkpoint exists. `/health` returns immediately with
   `status: "warming_up"` until the model is loaded.
   The app is available at `http://localhost:3000`; health and interactive API
   documentation are at `http://localhost:8000/health` and
   `http://localhost:8000/docs`.

2. To run directly with a local Python environment, from the repository root:

   ```powershell
   py -3.14 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
   .\.venv\Scripts\python.exe -m pip install -r backend\requirements-model.txt
   Set-Location backend
   ..\.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   In a second terminal:

   ```powershell
   Set-Location frontend
   npm ci
   npm run dev
   ```

   The frontend uses `http://localhost:8000` as its default API URL.

## Configuration

The backend reads `backend/.env` when run directly, or the project-root `.env`
when run with Docker Compose. Both local env files configure a default
Hugging Face ViT-Base deepfake detector. `MODEL_URL` may point to a compatible
PyTorch state-dictionary file instead; set `HF_MODEL_REPO` to a different public
Hugging Face ViT-Base image classifier. Its `id2label` must identify real and
synthetic/fake classes, or set `HF_CLASS_ORDER` to the source row order (for
example, `fake,real`). The bootstrap remaps the classifier to VeriFace class
index `0 = Real`, `1 = Synthetic`; local checkpoints are loaded strictly before
inference. For persistent history, set `SUPABASE_URL` and
`SUPABASE_SERVICE_ROLE_KEY` on the backend and
run [`supabase/schema.sql`](./supabase/schema.sql) in the Supabase SQL editor.
Keep the service-role key out of frontend environment variables and source
control. The five most recent scan records are shared through the public
history UI; avoid using sensitive filenames.

The frontend can optionally use the Supabase anon key for history fallback;
configure `NEXT_PUBLIC_SUPABASE_URL` and `NEXT_PUBLIC_SUPABASE_ANON_KEY` in
`frontend/.env.local`. `NEXT_PUBLIC_API_URL` should point to the deployed API
for frontend deployments. Set it as a build-time environment variable on the
frontend host. The deployed Render frontend
(`https://veriface-ai-1.onrender.com`) is allowed by default. Set
`CORS_ORIGINS` to add any other trusted frontend origins, comma-separated.
Vercel preview origins under `vercel.app` are also allowed by the backend's
default CORS origin pattern.

For Render, connect this repository and use the Docker runtime with the
repository root as the service root and `./Dockerfile` as the Dockerfile path.
The root Dockerfile copies and builds the backend from that repository-root
context. Alternatively, set the service Root Directory to `backend` and use
its `./Dockerfile`. Set `SUPABASE_URL` to the project URL shown above and add
`SUPABASE_SERVICE_ROLE_KEY` using Render's secret environment-variable UI.
Never put the service-role key in GitHub, Vercel, or a `NEXT_PUBLIC_` variable.

`MODEL_CHECKPOINT` defaults to `backend/models/vit_deepfake.pth` for direct
local runs and `/models/vit_deepfake.pth` in Docker. Compose mounts
`backend/models` writable so the first prediction can persist downloaded
weights. PyTorch, torchvision, and timm are installed as CPU packages.

## Verify inference

```powershell
Invoke-RestMethod http://localhost:8000/health
curl.exe -X POST -F "file=@C:\path\to\portrait.jpg" http://localhost:8000/predict
```

Health reports a `status` of `warming_up`, `ready`, or `error`, as well as
`model_name`, `device`, and `checkpoint_loaded`. The prediction response
includes Base64 JPEG overlay and JET heatmap data URLs; real attention rollout
averages all heads and propagates attention with residual connections through
all 12 transformer blocks.

## API

- `GET /health` — immediate service and model-loading status.
- `POST /predict` — JPEG/PNG multipart upload using form field `file`; returns
  the verdict, confidence, latency, and Base64 attention heatmaps.
- `GET /scans?limit=5` — most recent scans from Supabase or the local fallback
  buffer.

Uploads are limited to 10 MB and 40 megapixels.
