import logging
import pandas as pd
from flask import Flask, request, jsonify, abort
import joblib
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)

# Initialize Flask app
app = Flask(__name__)

# Load the trained pipeline
MODEL_PATH = "fraud_detection_model.pkl"
try:
    pipeline = joblib.load(MODEL_PATH)
    logging.info("Pipeline loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load pipeline: {e}")
    raise e

# API Key
API_KEY = "my_secret_api_key"

@app.before_request
def authenticate():
    api_key = request.headers.get("x-api-key")
    if api_key != API_KEY:
        abort(401, description="Unauthorized")

@app.route('/')
def index():
    return "Fraud Detection API is running. Use the /predict endpoint to make predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        if not input_data:
            abort(400, description="Invalid input: No JSON payload found.")
        
        # Convert input JSON to DataFrame
        input_df = pd.DataFrame([input_data])

        # Make predictions using the pipeline
        prediction_prob = pipeline.predict_proba(input_df)[:, 1]
        is_fraud = pipeline.predict(input_df)

        return jsonify({
            "fraud_probability": float(prediction_prob[0]),
            "is_fraud": bool(is_fraud[0]),
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
