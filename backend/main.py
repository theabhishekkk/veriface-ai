"""FastAPI application for VeriFace AI inference and scan history."""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool

load_dotenv()

from model import MODEL_CHECKPOINT, USE_MOCK_MODEL, load_vit_model, predict_image

logger = logging.getLogger("veriface.api")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
if MAX_IMAGE_BYTES < 1:
    raise ValueError("MAX_IMAGE_BYTES must be a positive integer.")
MAX_CONCURRENT_INFERENCES = int(os.getenv("MAX_CONCURRENT_INFERENCES", "1"))
if MAX_CONCURRENT_INFERENCES < 1:
    raise ValueError("MAX_CONCURRENT_INFERENCES must be a positive integer.")
_inference_slots = asyncio.Semaphore(MAX_CONCURRENT_INFERENCES)
origins = [
    origin.strip()
    for origin in os.getenv(
        "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(",")
    if origin.strip()
]
origin_regex = os.getenv("CORS_ORIGIN_REGEX", r"https://[a-z0-9-]+\.vercel\.app")

supabase_client = None
supabase_url = os.getenv(
    "SUPABASE_URL", "https://cowtcfcjpauyfvyvwfxg.supabase.co"
).strip()
supabase_key = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    or os.getenv("SUPABASE_KEY", "").strip()
    or os.getenv("SUPABASE_ANON_KEY", "").strip()
)
if supabase_url and supabase_key:
    try:
        from supabase import create_client

        supabase_client = create_client(supabase_url, supabase_key)
    except Exception:
        logger.exception("Could not initialize Supabase; scan history will be in memory.")

_recent_scans: list[dict[str, Any]] = []


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    id: str
    filename: str
    label: Literal["Real", "Synthetic"]
    is_deepfake: bool
    confidence: float
    confidence_percentage: float
    attention_heatmap: str
    raw_heatmap: str
    execution_time_ms: float
    model_name: str
    mode: str
    patch_analysis: dict[str, Any]
    created_at: str


class ScanItem(BaseModel):
    id: str
    filename: str
    label: str
    is_deepfake: bool
    confidence: float
    execution_time_ms: float
    created_at: str


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    if not USE_MOCK_MODEL:
        await run_in_threadpool(load_vit_model)
    logger.info(
        "VeriFace API started (mode=%s, supabase=%s)",
        "mock" if USE_MOCK_MODEL else "vit_base_patch16_224",
        bool(supabase_client),
    )
    yield


app = FastAPI(
    title="VeriFace AI API",
    description="Vision Transformer deepfake analysis with attention-map explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=origin_regex,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/")
async def root() -> dict[str, str]:
    return {"service": "VeriFace AI API", "status": "online", "docs": "/docs"}


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "healthy",
        "model_mode": "mock" if USE_MOCK_MODEL else "vit_base_patch16_224",
        "checkpoint_configured": bool(MODEL_CHECKPOINT),
        "supabase_configured": supabase_client is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)) -> PredictResponse:
    filename = os.path.basename((file.filename or "upload").replace("\\", "/"))
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")
    if file.content_type not in {"image/jpeg", "image/png", "image/jpg", None}:
        raise HTTPException(status_code=415, detail="Upload a JPEG or PNG image.")

    image_bytes = await file.read(MAX_IMAGE_BYTES + 1)
    await file.close()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        limit_mb = MAX_IMAGE_BYTES / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {limit_mb:g} MB upload limit.",
        )

    try:
        async with _inference_slots:
            result = await run_in_threadpool(predict_image, image_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Image inference failed")
        raise HTTPException(status_code=503, detail="Inference is unavailable.") from exc

    created_at = datetime.now(timezone.utc).isoformat()
    scan = {
        "id": str(uuid4()),
        "filename": filename,
        "label": result["label"],
        "is_deepfake": result["is_deepfake"],
        "confidence": result["confidence"],
        "execution_time_ms": result["execution_time_ms"],
        "created_at": created_at,
    }
    if supabase_client is not None:
        try:
            response = await run_in_threadpool(_insert_scan, scan)
            if response.data and response.data[0].get("id") is not None:
                scan["id"] = str(response.data[0]["id"])
        except Exception:
            logger.exception("Failed to persist scan to Supabase; retaining it in local history.")
            _recent_scans.insert(0, scan)
    else:
        _recent_scans.insert(0, scan)
    del _recent_scans[50:]

    return PredictResponse(**{**result, **scan})


@app.get("/scans", response_model=list[ScanItem])
async def recent_scans(limit: int = Query(default=5, ge=1, le=50)) -> list[ScanItem]:
    if supabase_client is not None:
        try:
            response = await run_in_threadpool(_query_recent_scans, limit)
            if response.data:
                return [
                    ScanItem(**{**item, "id": str(item["id"])})
                    for item in response.data
                ]
        except Exception:
            logger.exception("Failed to read scan history from Supabase.")
    return [ScanItem(**item) for item in _recent_scans[:limit]]


def _insert_scan(scan: dict[str, Any]) -> Any:
    return supabase_client.table("scans").insert(
        {key: value for key, value in scan.items() if key != "id"}
    ).execute()


def _query_recent_scans(limit: int) -> Any:
    return (
        supabase_client.table("scans")
        .select("id,filename,label,is_deepfake,confidence,execution_time_ms,created_at")
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
