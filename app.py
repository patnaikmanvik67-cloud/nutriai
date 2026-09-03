
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

pipeline = joblib.load("nutriai_model_tiny.pkl")


@app.route("/")
def home():
    return "NutriAI server is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age = int(data["Age"])
        cls = int(data["Class"])
        sex = data["Sex"]
        height = float(data["Height_cm"])
        weight = float(data["Weight_kg"])
        activity = data["Activity_Level"]

        child = pd.DataFrame([{
            "Age": age,
            "Class": cls,
            "Sex": sex,
            "Height_cm": height,
            "Weight_kg": weight,
            "Activity_Level": activity
        }])

        prediction = pipeline.predict(child)[0]

        return jsonify({
            "Energy_kcal": round(float(prediction[0]), 1),
            "Protein_g": round(float(prediction[1]), 1),
            "Carbohydrate_g": round(float(prediction[2]), 1),
            "Fat_g": round(float(prediction[3]), 1)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
