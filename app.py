import joblib
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# 🔥 ตั้งค่า Firebase
cred = credentials.Certificate("firebase_config.json")  # ใส่ไฟล์ Service Account JSON
firebase_admin.initialize_app(cred)
db = firestore.client()

# โหลดโมเดล
final_model = joblib.load("best_model.pkl")
model = final_model["model"]
label_encoders = final_model["label_encoders"]

feature_names = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]

app = Flask(__name__, static_folder="public")
CORS(app)  # เปิดให้ใช้ API ได้ทุกที่

@app.route("/")
def home():
    return send_from_directory("public", "index.html")  # ส่งไฟล์ Frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        features = [float(data[f]) for f in feature_names]
        features_array = np.array([features]).reshape(1, -1)

        prediction = model.predict(features_array)
        species_predicted = label_encoders["species"].inverse_transform([prediction[0]])[0]

        # 🔥 บันทึกผลลัพธ์ลง Firestore
        data["species_predicted"] = species_predicted
        db.collection("predictions").add(data)

        return jsonify({"prediction": species_predicted})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# ใช้ Firebase Functions
@functions_framework.http
def flask_app(request):
    return app(request)
