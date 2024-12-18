import joblib
from flask import Flask, request, jsonify
import numpy as np
from pydantic import BaseModel
import os
import subprocess
import time
import requests

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5001"

# Path to the locally saved model
LOCAL_MODEL_PATH = "./fraud_detection_model.pkl"

# Flask App Initialization
app = Flask(__name__)

# Load the locally saved model
def load_local_model(model_path):
    """
    Load the model from a local file path.
    """
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print(f"✅ Model loaded successfully from local path: {model_path}")
            return model
        else:
            raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    except Exception as e:
        print(f"❌ Error loading local model: {e}")
        return None

# Load the model
model = load_local_model(LOCAL_MODEL_PATH)

# Pydantic input validation
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction_request = PredictionRequest(**data)
        features = np.array([[prediction_request.feature1,
                              prediction_request.feature2,
                              prediction_request.feature3,
                              prediction_request.feature4]])
        if model is None:
            raise ValueError("Model is not loaded.")
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Health check endpoint
@app.route('/')
def home():
    return f"Fraud Detection Model is running and connected locally. MLflow URI: {MLFLOW_TRACKING_URI}"

if __name__ == '__main__':
    # Use Gunicorn to run the app
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"Starting the app using Gunicorn on port {port}...")

    try:
        # Check if the port is in use and free it
        print("Checking if port is in use...")
        subprocess.run(f"sudo fuser -k {port}/tcp", shell=True, check=False)

        # Start Gunicorn server in the background
        print("Starting Gunicorn server...")
        gunicorn_command = f"gunicorn --workers 4 --bind 0.0.0.0:{port} deploy_model:app"
        gunicorn_process = subprocess.Popen(gunicorn_command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Wait for server to start
        time.sleep(5)  # Give the server a few seconds to boot

        # Verify if the server is up by calling the root endpoint
        try:
            response = requests.get(f"http://127.0.0.1:{port}")
            if response.status_code == 200:
                print(f"✅ Server is running successfully on port {port}.")
                print(response.text)
                gunicorn_process.terminate()  # Stop the Gunicorn server for CI/CD
                print("✅ Gunicorn server terminated after successful verification.")
            else:
                print(f"❌ Server failed to start. HTTP Status: {response.status_code}")
                gunicorn_process.terminate()
                exit(1)  # Mark failure for the pipeline
        except requests.ConnectionError:
            print("❌ Unable to connect to the Gunicorn server.")
            gunicorn_process.terminate()
            exit(1)

    except subprocess.CalledProcessError as e:
        print(f"❌ Error while starting Gunicorn: {e}")
        exit(1)
