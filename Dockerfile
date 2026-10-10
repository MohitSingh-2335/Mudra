# Dockerfile - Mudra FastAPI REST Inference Engine

FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable real-time log flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies (OpenMP runtime for XGBoost/SHAP and curl for healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (leverages Docker layer caching)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy application source code and artifacts
COPY config.py .
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY data/processed/ ./data/processed/

# Security: Create and switch to non-root user
RUN useradd -m -u 1000 mudrauser && \
    chown -R mudrauser:mudrauser /app
USER mudrauser

# Expose FastAPI port
EXPOSE 8000

# Container health monitoring probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the high-performance async server
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
