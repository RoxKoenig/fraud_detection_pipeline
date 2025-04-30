# mlflow.Dockerfile: MLflow RBAC Proxy using FastAPI
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# System deps (for SSL, pip builds, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy FastAPI app
COPY main.py .

# Install dependencies
RUN pip install --no-cache-dir fastapi uvicorn httpx

# ✅ Use port 8000 for consistency
EXPOSE 8000

# ✅ Start FastAPI on port 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
