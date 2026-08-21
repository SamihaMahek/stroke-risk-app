# StrokeSight — Stroke Risk Prediction

A Flask web app that serves a trained ML classifier for stroke-risk
estimation, built for the Intel® Unnati — AI for Medical Intelligence
internship project.

## What's inside

- `train_model.py` — loads the healthcare stroke dataset, cleans it,
  trains Logistic Regression and Random Forest with `class_weight="balanced"`,
  evaluates both on accuracy/precision/recall/F1/ROC-AUC, and saves the
  better-recall model + preprocessing objects (`.pkl` files).
- `app.py` — Flask app: form page (`/`), prediction endpoint (`/predict`),
  and a model/methodology page (`/about`).
- `templates/` — HTML pages (Jinja2).
- `static/style.css` — styling.
- `healthcare-dataset-stroke-data.csv` — training data (5,110 records,
  the standard public Kaggle stroke-prediction dataset).
- `model.pkl`, `scaler.pkl`, `bmi_imputer.pkl`, `feature_columns.pkl`,
  `numerical_cols.pkl`, `categorical_cols.pkl`, `model_name.pkl`,
  `feature_importance.pkl` — pre-trained artifacts the Flask app loads
  at startup (already generated — you don't need to retrain).

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser.

## Retraining (optional)

If you want to regenerate the model artifacts (e.g. after changing
`train_model.py`):

```bash
python train_model.py
```

This overwrites the `.pkl` files that `app.py` loads.

## Deploy to Render (same flow as SkyRoute)

1. Push this folder to a new GitHub repository.
2. On [Render](https://render.com), click **New → Web Service** and
   connect that repo.
3. Settings:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app` (already set via `Procfile`)
4. Deploy. Render will give you a live URL like
   `https://strokesight-xxxx.onrender.com`.

## Important note

This is a machine-learning **decision-support prototype** built for an
academic project. It is not a certified medical device and does not
diagnose stroke. It estimates statistical risk patterns learned from a
historical dataset. Always consult a qualified physician for medical
concerns — this is stated on every page of the app as well.
