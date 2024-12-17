import mlflow
import joblib
from flask import Flask, request, jsonify
import numpy as np
from pydantic import BaseModel
import os

# MLflow Tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Path to the locally saved model (ensure retrain.py saves the model here)
LOCAL_MODEL_PATH = "./fraud_detection_model.pkl"

# Initialize Flask app
app = Flask(__name__)

# Load the locally saved model
def load_local_model(model_path):
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

# Load the model locally instead of using MLflow registry
model = load_local_model(LOCAL_MODEL_PATH)

# Pydantic for input validation
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

# Root endpoint
@app.route('/')
def home():
    return f"Fraud Detection Model is running and connected to MLflow at {MLFLOW_TRACKING_URI}"

if __name__ == '__main__':
    port = int(os.environ.get("FLASK_PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
