from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from typing import List
import pickle
import numpy as np
import logging
import os

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model
MODEL_PATH = os.getenv("MODEL_PATH", "model.pkl")
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

# Flask App
app = Flask(__name__)

# Simple Auth Token
AUTH_TOKEN = os.getenv("AUTH_TOKEN", "mysecrettoken")

# Pydantic Schema for Input Validation
class PredictionRequest(BaseModel):
    features: List[float]

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "API is live"}), 200

@app.route("/predict", methods=["POST"])
def predict():
    # Auth check
    token = request.headers.get("Authorization")
    if token != f"Bearer {AUTH_TOKEN}":
        logger.warning("Unauthorized access attempt")
        return jsonify({"error": "Unauthorized"}), 401

    # Parse and validate input
    try:
        req_data = request.get_json(force=True)
        validated_data = PredictionRequest(**req_data)
        features = np.array(validated_data.features).reshape(1, -1)
        logger.info(f"Received input: {validated_data.features}")
    except ValidationError as e:
        logger.error(f"Validation error: {e}")
        return jsonify({"error": "Invalid input", "details": e.errors()}), 400
    except Exception as e:
        logger.exception("Unexpected error during input parsing")
        return jsonify({"error": str(e)}), 500

    # Make prediction
    try:
        prediction = model.predict(features)
        logger.info(f"Prediction: {prediction.tolist()}")
        return jsonify({"prediction": prediction.tolist()}), 200
    except Exception as e:
        logger.exception("Model prediction failed")
        return jsonify({"error": "Model prediction failed", "details": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)  # Gunicorn will override this in prod
