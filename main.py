import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# โหลดโมเดล
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

        # ตรวจสอบว่ามีค่าครบทุกช่องหรือไม่
        required_fields = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
        for field in required_fields:
            if field not in data or data[field] == "":
                return jsonify({"error": f"Missing value for {field}"}), 400

        # ตรวจสอบค่าที่ได้รับ
        print(f"Received data: {data}")

        # แปลงค่าเพศเป็นค่าที่โมเดลเข้าใจ
        if data["sex"] not in ["Male", "Female"]:
            return jsonify({"error": "Invalid sex value, must be 'Male' or 'Female'"}), 400

        encoded_sex = label_encoders["sex"].transform([data["sex"]])[0]

        # จัดการข้อมูลสำหรับโมเดล
        features = [
            float(data["bill_length_mm"]),
            float(data["bill_depth_mm"]),
            float(data["flipper_length_mm"]),
            float(data["body_mass_g"]),
            encoded_sex
        ]

        # คำนวณผลการทำนาย
        features_array = np.array([features]).reshape(1, -1)
        prediction = model.predict(features_array)
        species_predicted = label_encoders["species"].inverse_transform([prediction[0]])[0]

        return jsonify({"prediction": species_predicted})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
