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

# Copy application code (models included via Git LFS)
COPY . .

# Validate that required model files exist (pulled from Git LFS)
RUN python -c "import os; \
    models = ['model/transport_gnn_model.pth', 'model/risk_model.pkl']; \
    missing = [m for m in models if not os.path.exists(m)]; \
    print(f'⚠️  Missing models: {missing}' if missing else '✅ All models present'); \
    exit(1 if missing else 0)"

# Create non-root user
RUN useradd -m -u 1001 mluser && chown -R mluser:mluser /app
USER mluser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:8001/health || exit 1

CMD ["python", "app.py"]
