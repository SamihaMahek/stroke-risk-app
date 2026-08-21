"""
Stroke Risk Prediction - Model Training Script
Trains Logistic Regression and Random Forest classifiers on the
healthcare stroke dataset, evaluates them (with focus on recall due
to class imbalance), and saves the best model + preprocessing objects
for use in the Flask app.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

RANDOM_STATE = 42

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
df = pd.read_csv("healthcare-dataset-stroke-data.csv")
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Drop id column (not predictive)
df = df.drop(columns=["id"])

# ---------------------------------------------------------
# 2. Clean data
# ---------------------------------------------------------
# bmi has some 'N/A' strings -> convert to NaN then impute
df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")

# Drop the single 'Other' gender row if present (too rare to encode reliably)
df = df[df["gender"] != "Other"].reset_index(drop=True)

print("Missing values before imputation:\n", df.isnull().sum())

# Impute missing BMI with median
bmi_imputer = SimpleImputer(strategy="median")
df["bmi"] = bmi_imputer.fit_transform(df[["bmi"]])

# ---------------------------------------------------------
# 3. Feature / target split
# ---------------------------------------------------------
categorical_cols = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
numerical_cols = ["age", "avg_glucose_level", "bmi"]
binary_cols = ["hypertension", "heart_disease"]

X = df.drop(columns=["stroke"])
y = df["stroke"]

# One-hot encode categorical columns
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Keep track of the exact column order the model expects
feature_columns = X_encoded.columns.tolist()

# ---------------------------------------------------------
# 4. Train/test split (stratified because of class imbalance)
# ---------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------
# 5. Scale numerical features
# ---------------------------------------------------------
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ---------------------------------------------------------
# 6. Train models
# ---------------------------------------------------------
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"\n--- {name} ---")
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

log_reg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
log_reg.fit(X_train, y_train)
lr_metrics = evaluate("Logistic Regression", log_reg, X_test, y_test)

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
rf.fit(X_train, y_train)
rf_metrics = evaluate("Random Forest", rf, X_test, y_test)

# ---------------------------------------------------------
# 7. Feature importance (Random Forest)
# ---------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=feature_columns).sort_values(ascending=False)
print("\nTop feature importances (Random Forest):")
print(importances.head(10))

# ---------------------------------------------------------
# 8. Select final model
# Prioritize recall since missing a high-risk patient is costlier
# than a false alarm in a healthcare screening context.
# ---------------------------------------------------------
final_model = rf if rf_metrics["recall"] >= lr_metrics["recall"] else log_reg
final_model_name = "Random Forest" if final_model is rf else "Logistic Regression"
print(f"\nSelected final model: {final_model_name}")

# ---------------------------------------------------------
# 9. Save artifacts for Flask app
# ---------------------------------------------------------
joblib.dump(final_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(bmi_imputer, "bmi_imputer.pkl")
joblib.dump(feature_columns, "feature_columns.pkl")
joblib.dump(numerical_cols, "numerical_cols.pkl")
joblib.dump(categorical_cols, "categorical_cols.pkl")
joblib.dump(final_model_name, "model_name.pkl")
joblib.dump(importances.head(10).to_dict(), "feature_importance.pkl")

print("\nSaved: model.pkl, scaler.pkl, bmi_imputer.pkl, feature_columns.pkl,")
print("       numerical_cols.pkl, categorical_cols.pkl, model_name.pkl, feature_importance.pkl")
