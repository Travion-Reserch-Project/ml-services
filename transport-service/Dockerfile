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

# Install Python dependencies in separate layers for better caching
RUN pip install --upgrade pip

# Install PyTorch CPU separately (cached layer)
RUN pip install --no-cache-dir --user torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies (only rebuilds if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image (much smaller, no models included)
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user first
RUN useradd -m -u 1001 mluser

# Copy Python packages from builder to user location
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code (NO MODELS - downloaded at runtime)
COPY app.py .
COPY routes.py .
COPY download_model.py .
COPY utils/ utils/
COPY data/ data/
COPY requirements.txt .

# Create model directory (will be populated at runtime)
RUN mkdir -p model

# Change ownership to mluser
RUN chown -R mluser:mluser /app /home/mluser/.local

# Make sure scripts are available in PATH
ENV PATH=/home/mluser/.local/bin:$PATH
ENV PYTHONPATH=/app

# Environment variables for model download
ENV MODEL_REPO=Travion-Reserch-Project/ml-services
ENV MODEL_RELEASE_TAG=latest
ENV MODEL_CACHE_DIR=/app/model

# Switch to non-root user
USER mluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

# Download model from DagsHub (or use existing) then start app
CMD ["sh", "-c", "python download_model.py && python app.py"]
