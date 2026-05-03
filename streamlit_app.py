import streamlit as st
import requests

st.title("💳 Credit Risk / Fraud Detection App")

st.write("Enter customer details to predict risk")

# Create 21 inputs (since your model expects 21 features)
features = []

for i in range(21):
    value = st.number_input(f"Feature {i+1}", value=0.0)
    features.append(value)

if st.button("Predict"):

    url = "https://credit-risk-default-prediction.onrender.com/predict"

    response = requests.post(url, json={"features": features})

    if response.status_code == 200:
        result = response.json()
        
        st.success(f"Prediction: {result['prediction']}")
        st.write(f"Fraud Probability: {result['fraud_probability']:.4f}")
    else:
        st.error("Error connecting to API")