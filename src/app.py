from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.preprocessing import LabelEncoder


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DATA_PATH = PROJECT_ROOT / "data" / "employees.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "salary_prediction_model.joblib"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

data = pd.read_csv(DATA_PATH)
model, feature_cols = joblib.load(MODEL_PATH)

feature_frame = data[feature_cols].copy()
categorical_columns = feature_frame.select_dtypes(include=["object"]).columns.tolist()
numeric_columns = [column for column in feature_cols if column not in categorical_columns]

encoders = {}
for column in categorical_columns:
    encoder = LabelEncoder()
    encoder.fit(feature_frame[column].astype(str))
    encoders[column] = encoder

salary_encoder = LabelEncoder()
salary_encoder.fit(data["salary"].astype(str))

category_options = {
    column: encoders[column].classes_.tolist()
    for column in categorical_columns
}

default_values = {}
for column in numeric_columns:
    median_value = data[column].median()
    default_values[column] = int(median_value) if float(median_value).is_integer() else round(float(median_value), 2)

for column in categorical_columns:
    default_values[column] = category_options[column][0]


def encode_payload(payload: dict) -> pd.DataFrame:
    normalized = {}
    for column in feature_cols:
        if column in categorical_columns:
            value = payload.get(column)
            if value is None:
                raise ValueError(f"Missing required field: {column}")
            normalized[column] = int(encoders[column].transform([str(value)])[0])
        else:
            raw_value = payload.get(column)
            if raw_value is None or raw_value == "":
                raise ValueError(f"Missing required field: {column}")
            normalized[column] = float(raw_value)

    return pd.DataFrame([normalized], columns=feature_cols)


@app.route("/")
def index():
    initial_state = {
        "featureCols": feature_cols,
        "categoryOptions": category_options,
        "defaultValues": default_values,
        "salaryLabels": salary_encoder.classes_.tolist(),
        "featureCount": len(feature_cols),
    }
    return render_template("index.html", initial_state=initial_state)


@app.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        encoded = encode_payload(payload)
        prediction = int(model.predict(encoded)[0])
        probability = float(model.predict_proba(encoded)[0][prediction])
        salary_label = str(salary_encoder.inverse_transform([prediction])[0])

        return jsonify(
            {
                "prediction": salary_label,
                "confidence": probability,
                "confidencePercent": round(probability * 100, 2),
                "encodedFeatures": encoded.iloc[0].to_dict(),
            }
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/api/meta")
def meta():
    return jsonify(
        {
            "featureCols": feature_cols,
            "categoryOptions": category_options,
            "defaultValues": default_values,
            "salaryLabels": salary_encoder.classes_.tolist(),
        }
    )


if __name__ == "__main__":
    app.run(debug=True)