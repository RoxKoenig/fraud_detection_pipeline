# Base image
FROM python:3.12.3

# Set working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt \
    gunicorn \
    mlflow \
    psycopg2-binary

# Set MLflow Tracking URI as an environment variable
ENV MLFLOW_TRACKING_URI="postgresql://admin:password@fraud_detection_db:5432/mlflow"

# Create directory for MLflow artifacts
RUN mkdir -p /mlflow/artifacts

# Copy the project files into the container
COPY . .

# Expose Flask and MLflow server ports
EXPOSE 5000 5001

# Start Flask app and MLflow server
CMD ["bash", "-c", \
    "gunicorn -w 4 -b 0.0.0.0:5000 app:app & \
     mlflow server --backend-store-uri postgresql://admin:password@fraud_detection_db:5432 --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5001"]

