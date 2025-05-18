from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

# File paths
MODEL_PATH = r"C:\Users\akeer\Downloads\streamlit\groundwater_level_model.pkl"
ENCODER_PATH = r"C:\Users\akeer\Downloads\streamlit\label_encoder.pkl"
DATA_PATH = r"DWLR_dataset.csv"  # Update as needed

# Load and preprocess data once on startup
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', parse_dates=["Date & Time"])
df = df.dropna(subset=["Water Level (m)", "Latitude", "Longitude", "Village", "District", "State"])
df["Month"] = df["Date & Time"].dt.month

# Monthly average water level
monthly = df.groupby(["State", "District", "Village", "Latitude", "Longitude", "Month"])["Water Level (m)"].mean().reset_index()

# Classify levels based on quantiles per village
def classify_level(row, thresholds):
    if row["Water Level (m)"] < thresholds[0]:
        return "Low"
    elif row["Water Level (m)"] < thresholds[1]:
        return "Moderate"
    else:
        return "High"

monthly["Level"] = None
for village, group in monthly.groupby("Village"):
    q1 = group["Water Level (m)"].quantile(0.33)
    q2 = group["Water Level (m)"].quantile(0.66)
    idx = group.index
    monthly.loc[idx, "Level"] = group.apply(lambda x: classify_level(x, [q1, q2]), axis=1)

# Encode target
le = LabelEncoder()
monthly["Level_Code"] = le.fit_transform(monthly["Level"])

# Train or load model
features = ["Latitude", "Longitude", "Month"]
X = monthly[features]
y = monthly["Level_Code"]

if os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH):
    model = joblib.load(MODEL_PATH)
    le = joblib.load(ENCODER_PATH)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

# === API ROUTE ===
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        lat = data.get("latitude")
        lon = data.get("longitude")

        if lat is None or lon is None:
            return jsonify({"error": "Missing latitude or longitude"}), 400
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({"error": "Invalid latitude or longitude values"}), 400

        result = []
        for month in range(1, 13):
            input_df = pd.DataFrame([[lat, lon, month]], columns=["Latitude", "Longitude", "Month"])
            prediction_code = model.predict(input_df)[0]
            level_label = le.inverse_transform([prediction_code])[0]
            result.append({"month": month, "level": level_label})

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# === Optional: Serve index.html ===
@app.route("/")
def serve_index():
    return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True)
