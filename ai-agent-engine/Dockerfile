# syntax=docker/dockerfile:1.7

# Dockerfile for AI Agent Engine
# LangGraph agentic workflow with LLM integration
# Python 3.10+

FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python build tools
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel

# Copy and install requirements
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user before copying dependencies
RUN useradd -m -u 1001 mluser

# Copy Python packages from builder into mluser home
COPY --from=builder /root/.local /home/mluser/.local

# Copy application code
COPY app/ app/
COPY requirements.txt .

# Create necessary directories
RUN mkdir -p models vector_db data logs

# Make sure scripts are in PATH
ENV PATH=/home/mluser/.local/bin:$PATH
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Ensure app and user dependencies are owned by mluser
RUN chown -R mluser:mluser /app /home/mluser/.local
USER mluser

EXPOSE 8004

# Health check
HEALTHCHECK --interval=45s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -f http://localhost:8004/api/v1/health || exit 1

# Run the application
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8004", "--workers", "2"]
