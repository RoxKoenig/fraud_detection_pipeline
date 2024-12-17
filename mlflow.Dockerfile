# Base image
FROM python:3.12-slim

# Install MLflow and PostgreSQL connector
RUN pip install --no-cache-dir mlflow==2.10.0 psycopg2-binary

# Install curl for health checks
RUN apt-get update && apt-get install -y curl

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Expose the MLflow server port
EXPOSE 5001

# Verify backend database connection before starting MLflow
CMD ["bash", "-c", "
    echo 'Waiting for PostgreSQL to be ready...';
    for i in {1..30}; do
        pg_isready -h fraud_detection_db -p 5432 -U admin && break || sleep 2;
    done;
    echo 'Starting MLflow server...';
    mlflow server \
        --backend-store-uri postgresql://admin:password@fraud_detection_db:5432 \
        --default-artifact-root /mlflow/artifacts \
        --host 0.0.0.0 --port 5001
"]

