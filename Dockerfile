FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    TORCH_NUM_THREADS=1 \
    HF_MODEL_REPO=prithivMLmods/Deep-Fake-Detector-v2-Model \
    MODEL_CHECKPOINT=/models/vit_deepfake.pth

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY backend/requirements-model.txt /app/requirements-model.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
      "torch>=2.2.0,<3.0" "torchvision>=0.17.0,<1.0" && \
    pip install --no-cache-dir -r /app/requirements-model.txt

COPY backend/main.py backend/model.py /app/
COPY backend/scripts /app/scripts

RUN mkdir -p /models && useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
    CMD curl -f "http://localhost:${PORT}/health" || exit 1

CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
