# VeriFace AI

VeriFace AI is a Next.js interface and FastAPI service for image analysis with a
ViT attention visualization. By default, the API runs in mock mode so the full
upload-to-history flow can be exercised without downloading model weights.

> **Model limitation:** mock results are simulated and must not be used for
> forensic, safety, or production decisions. The repository does not include a
> deepfake-trained model. Real mode requires a fine-tuned two-class
> `vit_base_patch16_224` state dictionary with class index `0 = Real` and
> `1 = Synthetic`. ImageNet-pretrained weights alone do not detect deepfakes.

## Run locally

1. Start the API from the project root:

   ```powershell
   docker compose up --build
   ```

   API health and interactive documentation are available at
   `http://localhost:8000/health` and `http://localhost:8000/docs`.

2. Install and start the frontend:

   ```powershell
   cd frontend
   Copy-Item .env.local.example .env.local
   npm install
   npm run dev
   ```

   Visit `http://localhost:3000`. The default API URL is
   `http://localhost:8000`.

## Configuration

The backend reads `backend/.env` when run directly, or the project-root `.env`
when run with Docker Compose. Start from `backend/.env.example` for direct
execution, or copy `.env.example` to `.env` for Compose. For persistent
history, set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` on the backend and
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

To enable real ViT inference:

1. Set `USE_MOCK_MODEL=False`.
2. Provide a compatible, fine-tuned binary checkpoint at a path visible to the
   API process.
3. Set `MODEL_CHECKPOINT` to that path. With Compose, mount the checkpoint
   under `backend/models` and use its container path under `/models`.

The backend rejects real-mode startup if a checkpoint is missing or invalid;
it does not silently report simulated predictions as model output.
The default Docker image intentionally omits PyTorch to keep mock-mode builds
small and quick. For real inference, build with
`docker compose build --build-arg INSTALL_VIT_MODEL=true backend`, then start
with `INSTALL_VIT_MODEL=true`, `USE_MOCK_MODEL=False`, and the checkpoint
environment variable configured.

## API

- `GET /health` — service and model-mode status.
- `POST /predict` — JPEG/PNG multipart upload using form field `file`; returns
  the verdict, confidence, latency, and Base64 attention heatmaps.
- `GET /scans?limit=5` — most recent scans from Supabase or the local fallback
  buffer.

Uploads are limited to 10 MB and 40 megapixels.
