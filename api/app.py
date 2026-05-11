from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import time

app = Flask(__name__)
CORS(app)

model = joblib.load(os.path.join(os.path.dirname(__file__), "../model/banknote_model.pkl"))

query_count = 0
start_time = time.time()

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running"})

@app.route("/predict", methods=["POST"])
def predict():
    global query_count
    data = request.get_json()

    if "features" not in data:
        return jsonify({"error": "Missing features"}), 400

    features = data["features"]
    if len(features) != 4:
        return jsonify({"error": "Expected 4 features: variance, skewness, curtosis, entropy"}), 400

    X = np.array(features).reshape(1, -1)
    prediction = int(model.predict(X)[0])
    label = "Fake" if prediction == 1 else "Real"
    query_count += 1

    return jsonify({"prediction": prediction, "label": label})

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify({
        "total_queries": query_count,
        "uptime_seconds": round(time.time() - start_time, 1)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)