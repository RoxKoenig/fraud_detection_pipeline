# mlflow.Dockerfile

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install MLflow and psycopg2-binary for PostgreSQL
RUN pip install --no-cache-dir mlflow psycopg2-binary

# Expose MLflow UI port
EXPOSE 5001

# No CMD here: docker-compose.yml will define how MLflow is started
