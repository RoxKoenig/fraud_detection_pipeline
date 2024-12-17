# Base image
FROM python:3.12-slim

# Install required dependencies
RUN apt-get update && apt-get install -y \
    curl \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow, psycopg2-binary, and setuptools (fix missing pkg_resources)
RUN pip install --no-cache-dir \
    mlflow==2.10.0 \
    psycopg2-binary \
    setuptools

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Expose the MLflow server port
EXPOSE 5001

# Start MLflow server with PostgreSQL readiness check
CMD ["/bin/bash", "-c", " \
    echo 'Checking PostgreSQL readiness...'; \
    for i in {1..30}; do \
        pg_isready -h fraud_detection_db -p 5432 -U admin && break || sleep 5; \
        echo 'Retrying PostgreSQL connection...'; \
    done; \
    echo 'PostgreSQL is ready. Starting MLflow server...'; \
    mlflow server \
        --backend-store-uri postgresql://admin:password@fraud_detection_db:5432 \
        --default-artifact-root /mlflow/artifacts \
        --host 0.0.0.0 --port 5001 || { echo 'MLflow server failed to start!'; exit 1; }"]

