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
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Environment variables for model download
ENV MODEL_SOURCE=mlflow
ENV MODEL_CACHE_DIR=/app/model

# Create non-root user
RUN useradd -m -u 1001 mluser && \
    chown -R mluser:mluser /app
USER mluser

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=2 \
  CMD curl -f http://localhost:8003/api/safety/health || exit 1

# Start the application with uvicorn
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8003"]
