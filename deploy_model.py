import joblib
from flask import Flask, request, jsonify
import numpy as np
from pydantic import BaseModel
import os
import subprocess

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5001"

# Path to the locally saved model (ensure retrain.py saves the model here)
LOCAL_MODEL_PATH = "./fraud_detection_model.pkl"

# Initialize Flask app
app = Flask(__name__)

# Function to load the locally saved model
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

# Function to free up the port
def free_port(port):
    """
    Free up the port if it's in use.
    """
    try:
        print(f"Checking if port {port} is in use...")
        output = subprocess.check_output(f"lsof -i :{port}", shell=True, stderr=subprocess.DEVNULL).decode()
        if output:
            print(f"Port {port} is in use. Terminating process...")
            subprocess.run(f"lsof -ti:{port} | xargs kill -9", shell=True, check=False)
            print(f"✅ Port {port} freed successfully.")
    except subprocess.CalledProcessError:
        print(f"✅ Port {port} was not in use.")
    except Exception as e:
        print(f"⚠️ Error freeing port {port}: {e}")

# Load the model
model = load_local_model(LOCAL_MODEL_PATH)

# Pydantic class for input validation
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict endpoint to receive input and return predictions.
    """
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

# Root endpoint
@app.route('/')
def home():
    """
    Health check endpoint.
    """
    return f"Fraud Detection Model is running and connected locally. MLflow URI: {MLFLOW_TRACKING_URI}"

if __name__ == '__main__':
    # Define the port and free it if in use
    port = int(os.environ.get("FLASK_PORT", 5000))
    print(f"Starting the app using Gunicorn on port {port}...")
    try:
        # Free up the port before starting Gunicorn
        free_port(port)

        # Run the application using Gunicorn
        subprocess.run(
            f"gunicorn --workers 4 --bind 0.0.0.0:{port} deploy_model:app",
            shell=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Error while starting Gunicorn: {e}")
