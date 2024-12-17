import os
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import numpy as np
from pydantic import BaseModel

# Set MLflow Tracking URI
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize Flask app
app = Flask(__name__)

# Load the MLflow model (modify to local artifact path)
try:
    # Option 1: From the registry (ensure paths are correct inside the container)
    model = mlflow.pyfunc.load_model("models:/fraud_detection_model/2")

    # Option 2: Load directly from artifacts folder (as fallback)
    # model = mlflow.sklearn.load_model("/mlflow/artifacts/fraud_detection_model_v2")
    print(f"✅ Model successfully loaded from MLflow tracking server at {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"❌ Error loading the model: {e}")
    model = None

# Input validation with Pydantic
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model is not loaded. Please check the MLflow server connection."}), 500

    try:
        data = request.get_json()
        prediction_request = PredictionRequest(**data)

        features = np.array([[prediction_request.feature1,
                              prediction_request.feature2,
                              prediction_request.feature3,
                              prediction_request.feature4]])
        prediction = model.predict(features)
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return f"✅ Fraud Detection Model is running and connected to MLflow at {MLFLOW_TRACKING_URI}"

if __name__ == '__main__':
    PORT = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=PORT, debug=False)
