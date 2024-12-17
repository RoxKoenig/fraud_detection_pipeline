import os
import mlflow
import mlflow.sklearn
from flask import Flask, request, jsonify
import numpy as np
from pydantic import BaseModel

# Set the MLflow tracking URI (include the port)
MLFLOW_TRACKING_URI = "http://localhost:5001"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Initialize Flask app
app = Flask(__name__)

# Load the MLflow model
# Update the model URI based on your setup
try:
    model = mlflow.sklearn.load_model("models:/fraud_detection_model/1")  # Replace with the correct model URI
    print(f"Model successfully loaded from MLflow tracking server at {MLFLOW_TRACKING_URI}")
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Define a class to handle the incoming request data (using Pydantic for validation)
class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    feature3: float
    feature4: float
    # Add all the necessary features that your model requires

# API route to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse the incoming JSON request
        data = request.get_json()
        
        # Extract features from the incoming data
        prediction_request = PredictionRequest(**data)  # Using Pydantic to validate input
        
        # Convert features to the correct format for prediction (2D numpy array)
        features = np.array([[prediction_request.feature1, 
                              prediction_request.feature2, 
                              prediction_request.feature3, 
                              prediction_request.feature4]])
        
        # Predict using the loaded model
        if model is None:
            raise ValueError("Model is not loaded. Check the MLflow server connection.")
        
        prediction = model.predict(features)
        
        # Return the prediction result as a JSON response
        return jsonify({"prediction": int(prediction[0])})

    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)}), 400

# Root route to confirm the app is running
@app.route('/')
def home():
    return f"Fraud Detection Model is running and connected to MLflow at {MLFLOW_TRACKING_URI}"

# Run the app (use host='0.0.0.0' for production)
if __name__ == '__main__':
    app.run(debug=True)  # Set debug to False in production
