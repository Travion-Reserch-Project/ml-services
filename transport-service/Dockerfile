FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code (models included via Git LFS when available)
COPY . .

# Check for available models (non-blocking for development)
RUN python -c "import os; \
    import glob; \
    models = glob.glob('model/*.pth') + glob.glob('model/*.pkl'); \
    if models: \
        print('✅ Found models:', models); \
    else: \
        print('⚠️  No models found in model/ directory'); \
        print('   Models will be added via Git LFS when trained'); \
    "

# Create non-root user
RUN useradd -m -u 1001 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "app.py"]
