# VeriFace AI - Backend Production Dockerfile
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV USE_MOCK_MODEL=False
ENV HF_MODEL_REPO=prithivMLmods/Deep-Fake-Detector-v2-Model
ENV MODEL_CHECKPOINT=/models/vit_deepfake.pth
ARG INSTALL_VIT_MODEL=true
ENV INSTALL_VIT_MODEL=${INSTALL_VIT_MODEL}

WORKDIR /app

# Install runtime dependencies required for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python requirements
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY requirements-model.txt /app/
RUN if [ "$INSTALL_VIT_MODEL" = "true" ]; then \
      pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.2.0,<3.0" "torchvision>=0.17.0,<1.0" && \
      pip install --no-cache-dir -r /app/requirements-model.txt; \
    fi

# Copy application source code
COPY main.py model.py /app/
COPY scripts /app/scripts

# Run the API as an unprivileged user.
RUN mkdir -p /models && useradd --create-home --uid 10001 appuser && \
    chown -R appuser:appuser /app /models
USER appuser

# Expose API port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5m --retries=3 \
  CMD curl -f http://localhost:8000/health || exit 1

# Start FastAPI server via uvicorn
CMD ["sh", "-c", "python /app/scripts/init_model.py && exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
