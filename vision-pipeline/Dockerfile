# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE
# Default to a CPU PyTorch image when BASE_IMAGE is not provided.
# For GPU-capable hosts the build can override this ARG to e.g.
#   pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
FROM ${BASE_IMAGE}

# Make pip faster and deterministic for layer caching
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ONLY_BINARY=":all:" \
    PIP_PREFER_BINARY=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen2.5vl:3b

WORKDIR /app

# Install system deps needed by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxcb1 \
    libxcb-xinerama0 \
    libxcb-render0 \
    libxcb-shape0 \
    libxcb-randr0 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-icccm4 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install only essential system dependencies (Debian/Ubuntu base inside PyTorch image)
RUN python3 -m pip install --upgrade pip \
 && pip install --only-binary=:all: --no-cache-dir -r requirements.txt
RUN pip install --upgrade bitsandbytes
# Copy application code
COPY vision_app.py .

# Expose port
EXPOSE 5000

# Health check for container orchestration
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:5000/health || exit 1

# Run the application
CMD ["uvicorn", "vision_app:app", "--host", "0.0.0.0", "--port", "5000"]
