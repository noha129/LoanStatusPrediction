# app.py
import os
import logging
from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Paths (use __file__ so it works inside container)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# Load artifacts (fail early with clear message if missing)
try:
    logger.info("Loading model from %s", MODEL_PATH)
    model = pickle.load(open(MODEL_PATH, "rb"))
except Exception as e:
    logger.exception("Failed to load model.pkl: %s", e)
    raise

try:
    logger.info("Loading scaler from %s", SCALER_PATH)
    scaler = joblib.load(SCALER_PATH)
except Exception as e:
    logger.exception("Failed to load scaler.pkl: %s", e)
    raise

ENCODING = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
}
FEATURE_ORDER = ['Credit_History', 'Married', 'CoapplicantIncome', 'Education', 'Dependents', 'Gender']

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None
    if request.method == "POST":
        # Collect form data
        row = {
            'Credit_History': float(request.form['Credit_History']),
            'Married': request.form['Married'],
            'CoapplicantIncome': float(request.form['CoapplicantIncome']),
            'Education': request.form['Education'],
            'Dependents': request.form['Dependents'],
            'Gender': request.form['Gender'],
        }
        df_new = pd.DataFrame([row])

        # Encode
        for col, mapping in ENCODING.items():
            df_new[col] = df_new[col].map(mapping)

        # Ensure column order and shape
        X = df_new[FEATURE_ORDER].values

        # Scale & Predict
        X_scaled = scaler.transform(X)
        pred = model.predict(X_scaled)[0]
        prediction_text = "Approved ✅" if pred == 1 else "Rejected ❌"

    return render_template("index.html", prediction_text=prediction_text)

# Local dev only (Gunicorn will be used in production)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
