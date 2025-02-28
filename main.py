import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# โหลดโมเดล
final_model = joblib.load("best_model.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"message": "Flask API is running on Render"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        print(f"🔹 Received Data: {data}")  # ✅ ดูค่าที่ส่งมา API

        # ✅ ตรวจสอบว่ามีค่าครบทุกช่อง
        required_fields = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
        for field in required_fields:
            if field not in data or data[field] == "":
                error_msg = f"Missing value for {field}"
                print(f"❌ {error_msg}")
                return jsonify({"error": error_msg}), 400

        # ✅ ตรวจสอบค่าของ `sex`
        if data["sex"] not in ["Male", "Female"]:
            error_msg = "Invalid sex value, must be 'Male' or 'Female'"
            print(f"❌ {error_msg}")
            return jsonify({"error": error_msg}), 400

        # ✅ แปลงค่า `sex`
        encoded_sex = label_encoders["sex"].transform([data["sex"]])[0]
        print(f"✅ Encoded sex: {encoded_sex}")

        # ✅ แปลงค่าทั้งหมด
        features = [
            float(data["bill_length_mm"]),
            float(data["bill_depth_mm"]),
            float(data["flipper_length_mm"]),
            float(data["body_mass_g"]),
            encoded_sex
        ]
        print(f"✅ Features: {features}")

        # ✅ ทำการทำนาย
        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)
        species_predicted = label_encoders["species"].inverse_transform([prediction[0]])[0]

        print(f"🎯 Prediction: {species_predicted}")
        return jsonify({"prediction": species_predicted})

    except Exception as e:
        print(f"❌ Server Error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
