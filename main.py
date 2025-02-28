import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

final_model = joblib.load("best_model.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

feature_names = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running on Render"})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [float(data[f]) for f in feature_names]
        features_array = np.array([features]).reshape(1, -1)

        prediction = model.predict(features_array)
        species_predicted = label_encoders["species"].inverse_transform([prediction[0]])[0]

        return jsonify({"prediction": species_predicted})
    
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
