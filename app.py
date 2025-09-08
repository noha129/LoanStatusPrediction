from flask import Flask, render_template, request
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

# Load artifacts
model = pickle.load(open("model.pkl", "rb"))
scaler = joblib.load("scaler.pkl")

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

        # Scale
        df_scaled = scaler.transform(df_new[FEATURE_ORDER])

        # Predict
        pred = model.predict(df_scaled)[0]
        prediction_text = "Approved ✅" if pred == 1 else "Rejected ❌"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run(debug=True)
