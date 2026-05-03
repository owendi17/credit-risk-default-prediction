from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load("fraud_model.pkl")

@app.route("/")
def home():
    return "Fraud Detection API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Expecting: {"features": [v1, v2, v3, ...]}
    features = np.array(data["features"]).reshape(1, -1)

    prediction = model.predict(features)
    probability = model.predict_proba(features)[0][1]

    return jsonify({
        "prediction": int(prediction[0]),
        "fraud_probability": float(probability)
    })

if __name__ == "__main__":
    app.run(debug=True)