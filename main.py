import joblib
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# 📌 โหลดโมเดลที่ฝึกไว้
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
        print(f"🔹 Received Data: {data}")

        # ✅ ตรวจสอบค่าที่ LabelEncoder ใช้
        print(f"🔍 LabelEncoder for sex: {label_encoders['sex'].classes_}")

        # ✅ ตรวจสอบว่า `sex` ที่ส่งมาตรงกับ LabelEncoder หรือไม่
        if data["sex"] not in label_encoders["sex"].classes_:
            return jsonify({
                "error": f"Invalid sex value: {data['sex']}, must be one of {label_encoders['sex'].classes_}"
            }), 400

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
