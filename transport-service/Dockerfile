# Multi-stage build to reduce image size
# Stage 1: Build dependencies
FROM python:3.9-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir --user torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image (much smaller, no models included)
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code (NO MODELS - downloaded at runtime)
COPY app.py .
COPY utils/ utils/
COPY requirements.txt .

# Create model directory (will be populated at runtime)
RUN mkdir -p model

# Make sure scripts are available in PATH
ENV PATH=/root/.local/bin:$PATH

# Environment variables for model download
ENV MODEL_REPO=Travion-Reserch-Project/ml-services
ENV MODEL_RELEASE_TAG=latest
ENV MODEL_CACHE_DIR=/app/model

# Create non-root user
RUN useradd -m -u 1001 mluser && \
    chown -R mluser:mluser /app
USER mluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Models will be downloaded on startup before app starts
CMD ["python", "app.py"]
