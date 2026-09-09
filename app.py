
from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# ==============================
# LOAD ML MODELS
# ==============================

pipeline = joblib.load("nutriai_model_tiny.pkl")
serving_pipeline = joblib.load("nutriai_serving_model.pkl")


# ==============================
# HOME
# ==============================

@app.route("/")
def home():
    return "NutriAI server is running!"


# ==============================
# NUTRITION PREDICTION
# ==============================

@app.route("/predict", methods=["GET"])
def predict():
    try:
        data = request.args

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


# ==============================
# ML SERVING PREDICTION
# ==============================

@app.route("/serving", methods=["GET"])
def serving():
    try:
        data = request.args

        age = int(data["Age"])
        cls = int(data["Class"])
        sex = data["Sex"]
        height = float(data["Height_cm"])
        weight = float(data["Weight_kg"])
        activity = data["Activity_Level"]

        # Multiple food selections
        foods = []

        for key in ["Food1", "Food2", "Food3"]:
            food = data.get(key)

            if food and food.strip():
                foods.append(food.strip())

        if not foods:
            return jsonify({
                "error": "Please select at least one food"
            }), 400

        results = []

        for food in foods:

            child_food = pd.DataFrame([{
                "Age": age,
                "Class": cls,
                "Sex": sex,
                "Height_cm": height,
                "Weight_kg": weight,
                "Activity_Level": activity,
                "Food_Selected": food
            }])

            prediction = serving_pipeline.predict(child_food)[0]

            results.append({
                "food": food,
                "Serving_g": round(float(prediction), 1)
            })

        return jsonify({
            "Age": age,
            "Class": cls,
            "servings": results,
            "note": "Educational ML-based serving estimate"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ==============================
# RUN SERVER
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)