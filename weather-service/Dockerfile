# syntax=docker/dockerfile:1.7

# Dockerfile for Weather Service
# Computer Vision-based UV risk and Fitzpatrick skin type detection
# Python 3.11 (required for contourpy==1.3.3)

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libopenblas-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Install PyTorch CPU (separate layer for caching)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Copy and install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libopenblas0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app.py .
COPY utils/ utils/
COPY requirements.txt .

# Create models and data directories
RUN mkdir -p models data

# Make sure scripts are in PATH
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -u 1001 mluser && \
    chown -R mluser:mluser /app
USER mluser

EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=2 \
  CMD curl -f http://localhost:8002/health || exit 1

# Run the application
CMD ["python", "app.py"]
