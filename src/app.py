from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("../models/iris_classifier.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    # Expecting JSON with keys: sepal_length, sepal_width, petal_length, petal_width
    X = pd.DataFrame([data])
    pred = model.predict(X)[0]
    proba = model.predict_proba(X).max()
    return jsonify({
        "predicted_species": pred,
        "confidence": round(float(proba), 3)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
