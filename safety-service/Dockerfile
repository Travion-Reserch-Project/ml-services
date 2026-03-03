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
RUN pip install --upgrade pip

# Install dependencies (only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

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
COPY routes/ routes/
COPY requirements.txt .

# Create model and data directories (will be populated at runtime)
RUN mkdir -p model data

# Make sure scripts are available in PATH
ENV PATH=/root/.local/bin:$PATH

# Environment variables for model download
ENV MODEL_SOURCE=mlflow
ENV MODEL_CACHE_DIR=/app/model

# Create non-root user
RUN useradd -m -u 1001 mluser && \
    chown -R mluser:mluser /app
USER mluser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8003/api/safety/health || exit 1

# Models will be downloaded on startup before app starts
CMD ["python", "app.py"]
