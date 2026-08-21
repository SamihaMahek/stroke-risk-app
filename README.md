# StrokeSight: Machine Learning-Based Stroke Risk Assessment Portal

[![Python Version](https://shields.io)](https://python.org)
[![Framework](https://shields.io)](https://palletsprojects.com)
[![License](https://shields.io)](LICENSE)
[![Deployment](https://shields.io)](https://onrender.com)

StrokeSight is a web-based decision-support prototype that evaluates statistical stroke probability based on clinical and lifestyle signals. Developed as part of an academic initiative in AI for Medical Intelligence, the platform demonstrates end-to-end machine learning lifecycle integration—spanning exploratory data analysis, class imbalance mitigation, pipeline optimization, and production-grade cloud deployment.

🔗 **Live Application URL:** [https://onrender.com](https://onrender.com)

---

## 🚀 Key Features
- **Interactive Risk Engine:** Clean interface designed for rapid entry of clinical and demographic data fields.
- **Risk Band Classification:** Real-time generation of mathematical probability scores grouped into clear risk bands.
- **Optimized Recall Profile:** Imbalanced training profiles were tuned to prioritize high sensitivity, ensuring fewer true high-risk cases are missed.
- **Responsive Clinical Design:** Structured using clear telemetry indicators and an intuitive UI optimized across diverse viewing profiles.

---

## 📊 Dataset & Predictive Performance
The underlying engine utilizes the standard Kaggle Stroke Prediction Dataset, containing **5,110 patient records** mapping clinical, demographic, and behavioral features.

### Feature Architecture
- **Clinical Signatures:** Age, Hypertension (Binary), Heart Disease (Binary), Average Blood Glucose Level (mg/dL), Body Mass Index (BMI).
- **Demographic Profiles:** Gender, Ever Married (Status), Work Type, Residence Type (Urban/Rural).
- **Behavioral Signatures:** Smoking Status.

### Model Evaluation & Metrics
Due to severe class imbalance (fewer stroke instances than non-stroke instances), model training prioritized sensitivity by leveraging `class_weight='balanced'`. While multiple algorithms were tested, **Logistic Regression** was chosen over Random Forest for deployment due to its superior clinical utility:

| Metric | Performance Value |
| :--- | :--- |
| **ROC-AUC Score** | 0.84 |
| **Sensitivity / Recall (Stroke Class)** | 0.80 |
| **Primary Indicators Discovered** | Age, Average Glucose Level, Hypertension |

---

## 🛠️ Technology Stack & Architecture
- **Backend Core:** Python, Flask, Gunicorn
- **Data Engineering:** Pandas, NumPy
- **Machine Learning & Pipeline Controls:** Scikit-learn (`StandardScaler`, `OneHotEncoder`, Stratified Splitting)
- **Serialization Handling:** Joblib (Saves preprocessing objects and model coefficients dynamically)
- **Frontend Presentation Layer:** HTML5, CSS3 (Custom responsive layout featuring responsive components)
- **Hosting Environment:** Render Cloud Platform API

---

## ⚙️ Local Configuration & Installation

1. Clone the project workspace repository:
   ```bash
   git clone https://github.com
   cd stroke-risk-app
   ```

2. Establish an isolated execution environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate
   ```

3. Initialize necessary runtime project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Execute the preprocessing pipeline script to train the model and serialize pipeline state files (`.pkl` outputs):
   ```bash
   python train_model.py
   ```

5. Launch the application interface server locally:
   ```bash
   python app.py
   ```
   Navigate to `http://127.0.0.1:5000` inside your web browser.

---

## ⚠️ Academic Project Disclaimer
StrokeSight is a data-science decision-support demonstration prototype built for academic purposes. It relies entirely on statistical trends found within historical training data and does not provide valid medical or clinical advice, diagnoses, or therapeutic validations. Always prioritize formal professional guidance for any clinical or health-related queries.
