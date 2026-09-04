from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

pipeline = joblib.load("nutriai_model_tiny.pkl")


NUTRITION_REFERENCES = {
    "Primary": {
        "classes": "1-5",
        "scheme": "PM POSHAN",
        "energy_kcal": "450 kcal",
        "protein_g": "12 g"
    },
    "Upper Primary": {
        "classes": "6-8",
        "scheme": "PM POSHAN",
        "energy_kcal": "700 kcal",
        "protein_g": "20 g"
    }
}


MENU = {
    "Monday": {
        "food": ["Rice", "Dal", "Seasonal Vegetable"],
        "description": "Rice with dal and a vegetable preparation"
    },
    "Tuesday": {
        "food": ["Roti", "Dal", "Seasonal Vegetable"],
        "description": "Wheat-based meal with dal and vegetables"
    },
    "Wednesday": {
        "food": ["Rice", "Sambar", "Vegetable"],
        "description": "Rice with pulse-based sambar and vegetables"
    },
    "Thursday": {
        "food": ["Khichdi", "Vegetable"],
        "description": "Rice-pulse khichdi with vegetables"
    },
    "Friday": {
        "food": ["Rice", "Rajma/Chana", "Seasonal Vegetable"],
        "description": "Rice with a pulse preparation and vegetables"
    }
}


@app.route("/")
def home():
    return "NutriAI server is running!"


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


@app.route("/menu", methods=["GET"])
def menu():
    try:
        day = request.args.get("Day")
        cls = int(request.args.get("Class"))

        if day not in MENU:
            return jsonify({"error": "Invalid day"}), 400

        if 1 <= cls <= 5:
            category = "Primary"
            scheme = NUTRITION_REFERENCES["Primary"]

            quantities = {
                "Food_grains_g": "100 g",
                "Pulses_g": "20 g",
                "Vegetables_g": "50 g",
                "Oil_fat_g": "5 g"
            }

        elif 6 <= cls <= 8:
            category = "Upper Primary"
            scheme = NUTRITION_REFERENCES["Upper Primary"]

            quantities = {
                "Food_grains_g": "150 g",
                "Pulses_g": "30 g",
                "Vegetables_g": "75 g",
                "Oil_fat_g": "7.5 g"
            }

        else:
            return jsonify({
                "error": "PM POSHAN menu is available for Classes 1-8"
            }), 400

        return jsonify({
            "day": day,
            "class": cls,
            "category": category,
            "scheme_applicable": scheme["scheme"],
            "menu": MENU[day]["food"],
            "description": MENU[day]["description"],
            "reference_nutrition": {
                "Energy_kcal": scheme["energy_kcal"],
                "Protein_g": scheme["protein_g"]
            },
            "reference_quantities": quantities
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)