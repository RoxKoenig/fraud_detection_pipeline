# Base image
FROM python:3.12-slim

# Install MLflow and required dependencies
RUN pip install --no-cache-dir mlflow==2.10.0 psycopg2-binary

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Expose the MLflow server port
EXPOSE 5001

# Start the MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "postgresql://admin:password@fraud_detection_db:5432", "--default-artifact-root", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5001"]

