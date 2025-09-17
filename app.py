# import streamlit as st
# import pandas as pd
# import pickle
# import joblib

# # ----------------------
# # Load trained artifacts
# # ----------------------
# model = pickle.load(open("model.pkl", "rb"))
# scaler = joblib.load("scaler.pkl")

# ENCODING = {
#     'Gender': {'Male': 1, 'Female': 0},
#     'Married': {'Yes': 1, 'No': 0},
#     'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4},
#     'Education': {'Graduate': 1, 'Not Graduate': 0},
# }
# FEATURE_ORDER = ['Credit_History', 'Married', 'CoapplicantIncome', 'Education', 'Dependents', 'Gender']

# # ----------------------
# # Streamlit UI
# # ----------------------
# st.title("🏦 Loan Status Prediction App")

# st.write("Fill in the details below to check loan approval:")

# credit_history = st.selectbox("Credit History", [1.0, 0.0])
# married = st.selectbox("Married", ["Yes", "No"])
# coapp_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)
# education = st.selectbox("Education", ["Graduate", "Not Graduate"])
# dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
# gender = st.selectbox("Gender", ["Male", "Female"])

# if st.button("🔮 Predict"):
#     # Create input row
#     row = {
#         'Credit_History': float(credit_history),
#         'Married': married,
#         'CoapplicantIncome': float(coapp_income),
#         'Education': education,
#         'Dependents': dependents,
#         'Gender': gender,
#     }
#     df_new = pd.DataFrame([row])

#     # Encode categorical columns
#     for col, mapping in ENCODING.items():
#         df_new[col] = df_new[col].map(mapping)

#     # Scale features
#     df_scaled = scaler.transform(df_new[FEATURE_ORDER])

#     # Predict
#     pred = model.predict(df_scaled)[0]

#     if pred == 1:
#         st.success("✅ Loan Approved!")
#     else:
#         st.error("❌ Loan Rejected")



import streamlit as st
import pandas as pd
import pickle
import joblib

# ----------------------
# Load trained artifacts
# ----------------------
model = pickle.load(open("model.pkl", "rb"))
scaler = joblib.load("scaler.pkl")

ENCODING = {
    'Gender': {'Male': 1, 'Female': 0},
    'Married': {'Yes': 1, 'No': 0},
    'Dependents': {'0': 0, '1': 1, '2': 2, '3+': 4},
    'Education': {'Graduate': 1, 'Not Graduate': 0},
}
FEATURE_ORDER = ['Credit_History', 'Married', 'CoapplicantIncome', 'Education', 'Dependents', 'Gender']

# ----------------------
# Streamlit App
# ----------------------
def main():
    st.title("🏦 Loan Status Prediction")
    st.subheader("Fill the details")


    # Row 1: Credit History & Married
    col1, col2 = st.columns(2)
    with col1:
        credit_history = st.selectbox("Credit History", (1.0, 0.0))
    with col2:
        married = st.selectbox("Married", ("Yes", "No"))

    # Row 2: Education & Dependents
    col3, col4 = st.columns(2)
    with col3:
        education = st.selectbox("Education", ("Graduate", "Not Graduate"))
    with col4:
        dependents = st.selectbox("Dependents", ("0", "1", "2", "3+"))

    # Row 3: Gender & Coapplicant Income
    col5, col6 = st.columns(2)
    with col5:
        gender = st.selectbox("Gender", ("Male", "Female"))
    with col6:
        coapp_income = st.number_input("Coapplicant Income", min_value=0.0, value=0.0, step=100.0)

    # Button for prediction
    if st.button("🔮 Predict Loan Status"):
        row = {
            'Credit_History': float(credit_history),
            'Married': married,
            'CoapplicantIncome': float(coapp_income),
            'Education': education,
            'Dependents': dependents,
            'Gender': gender,
        }
        df_new = pd.DataFrame([row])

        # Encode categorical columns
        for col, mapping in ENCODING.items():
            df_new[col] = df_new[col].map(mapping)

        # Scale features
        df_scaled = scaler.transform(df_new[FEATURE_ORDER])

        # Predict
        pred = model.predict(df_scaled)[0]

        # Show result
        if pred == 1:
            st.success("✅ Loan Approved!")
        else:
            st.error("❌ Loan Rejected")

if __name__ == "__main__":
    main()
