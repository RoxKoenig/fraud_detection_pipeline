# Base image: Use Python 3.11 for MLflow compatibility
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and required Python dependencies
RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    psycopg2-binary \
    setuptools

# Create a directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Expose the MLflow server port
EXPOSE 5001

# Start MLflow server
CMD ["mlflow", "server", \
     "--backend-store-uri", "postgresql://admin:password@fraud_detection_db:5432/mlflow", \
     "--default-artifact-root", "/mlflow/artifacts", \
     "--host", "0.0.0.0", \
     "--port", "5001"]

