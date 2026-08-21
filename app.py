"""
Stroke Risk Prediction - Flask Web Application
Loads the pre-trained model + preprocessing objects and serves a
form-based prediction interface.

NOTE: This is a decision-support ML prototype, NOT a medical
diagnostic tool. It does not replace professional medical advice.
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------
# Load trained artifacts
# ---------------------------------------------------------
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
bmi_imputer = joblib.load(os.path.join(BASE_DIR, "bmi_imputer.pkl"))
feature_columns = joblib.load(os.path.join(BASE_DIR, "feature_columns.pkl"))
numerical_cols = joblib.load(os.path.join(BASE_DIR, "numerical_cols.pkl"))
categorical_cols = joblib.load(os.path.join(BASE_DIR, "categorical_cols.pkl"))
model_name = joblib.load(os.path.join(BASE_DIR, "model_name.pkl"))
feature_importance = joblib.load(os.path.join(BASE_DIR, "feature_importance.pkl"))


def preprocess_input(form):
    """Convert raw form data into the exact feature format the model expects."""
    raw = {
        "gender": form.get("gender"),
        "age": float(form.get("age")),
        "hypertension": int(form.get("hypertension")),
        "heart_disease": int(form.get("heart_disease")),
        "ever_married": form.get("ever_married"),
        "work_type": form.get("work_type"),
        "Residence_type": form.get("Residence_type"),
        "avg_glucose_level": float(form.get("avg_glucose_level")),
        "bmi": form.get("bmi"),
        "smoking_status": form.get("smoking_status"),
    }

    df = pd.DataFrame([raw])

    # Handle missing BMI the same way as training
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    df["bmi"] = bmi_imputer.transform(df[["bmi"]])

    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Align columns with training feature set (fill any missing dummy cols with 0)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Scale numerical columns
    df_encoded[numerical_cols] = scaler.transform(df_encoded[numerical_cols])

    return df_encoded


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        X = preprocess_input(request.form)
        proba = model.predict_proba(X)[0][1]
        prediction = int(proba >= 0.5)

        risk_percent = round(proba * 100, 1)

        if risk_percent >= 60:
            risk_level = "High"
            risk_class = "high"
        elif risk_percent >= 30:
            risk_level = "Moderate"
            risk_class = "moderate"
        else:
            risk_level = "Low"
            risk_class = "low"

        # Clean and format the loaded feature importance data for the chart layout
        chart_importance = {}
        for feature, weight in feature_importance.items():
            clean_name = feature.replace('_', ' ').title()
            chart_importance[clean_name] = round(float(weight) * 100, 1)

        # Sort features so the strongest drivers show first
        sorted_importance = dict(sorted(chart_importance.items(), key=lambda item: item[1], reverse=True))

        result = {
            "prediction": prediction,
            "risk_percent": risk_percent,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "model_name": model_name,
            "importance_data": sorted_importance
        }
        return render_template("result.html", result=result, form=request.form)

    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/about")
def about():
    return render_template("about.html", feature_importance=feature_importance, model_name=model_name)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
