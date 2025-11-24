# Multi-stage build to reduce image size
# Stage 1: Build dependencies
FROM python:3.9-slim as builder

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
    pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image (much smaller)
FROM python:3.9-slim

WORKDIR /app

# Install only runtime dependencies (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code (models included via Git LFS when available)
COPY . .

# Make sure scripts are available in PATH
ENV PATH=/root/.local/bin:$PATH

# Check for available models (non-blocking for development)
RUN python -c "import os, glob; \
    models = glob.glob('model/*.pth') + glob.glob('model/*.pkl'); \
    print('Found models:', models) if models else print('No models yet - will be added via Git LFS')"

# Create non-root user
RUN useradd -m -u 1001 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "app.py"]
