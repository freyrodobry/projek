import os
import csv
import joblib
import numpy as np
from datetime import datetime
from collections import deque
from flask import Flask, request, jsonify
from flask_cors import CORS

# ========== CONFIG ==========

LOG_CSV = "logs/fire_data.csv"
MODEL_FILE = "model_random_forest.pkl"
ENCODER_FILE = "label_encoder.pkl"
MAX_HISTORY = 200

# ============================

app = Flask(__name__)
CORS(app)

# Load ML model
model = joblib.load(MODEL_FILE)
label_encoder = joblib.load(ENCODER_FILE)

# In-memory history for dashboard
history = deque(maxlen=MAX_HISTORY)

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)
if not os.path.exists(LOG_CSV):
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "temp", "hum", "gas", "flame", "status"])


# ========== ML PREDICT ENDPOINT ==========
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    suhu = float(data.get("suhu", 0))
    kelembapan = float(data.get("kelembapan", 0))
    gas = float(data.get("gas", 0))
    flame = float(data.get("flame", 0))

    X = np.array([[suhu, kelembapan, gas, flame]])
    pred = model.predict(X)[0]
    status = label_encoder.inverse_transform([pred])[0].upper()

    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Save to memory for dashboard
    entry = {
        "timestamp": timestamp,
        "temp": suhu,
        "hum": kelembapan,
        "gas": gas,
        "flame": flame,
        "status": status
    }

    history.append(entry)

    # Save CSV
    with open(LOG_CSV, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, suhu, kelembapan, gas, flame, status])

    return jsonify({
        "status": status,
        "prediction": int(pred)
    })


# ========== MONITORING ENDPOINT ==========
@app.route("/latest")
def latest():
    hist_list = list(history)
    last = hist_list[-1] if hist_list else {}
    return jsonify({
        "last": last,
        "history": hist_list
    })


# ========== HEALTH CHECK ==========
@app.route("/")
def home():
    return "API Deteksi Dini Kebakaran Online"

@app.route("/health")
def health():
    return "OK"


# ========== RUN APP ==========
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
