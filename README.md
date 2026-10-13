# Severity Prediction Project

A complete, production-grade Machine Learning and Web Application system for **Incident Severity Prediction**, built with Python, Flask, Scikit-learn, HTML/CSS, Docker, and Docker Compose.

---

## 1. Project Overview

The **Severity Prediction Project** provides automated triage and severity assessment for emergency and road traffic incidents. When an accident occurs, dispatchers, emergency personnel, or autonomous vehicle monitoring stations need instantaneous, reliable risk categorization to allocate medical and emergency resources appropriately.

This system ingests observable crash characteristics (vehicle speed, weather, lighting, road surface condition, driver impairment factors, casualty counts) and outputs:
- Predicted Severity Level (`Minor`, `Moderate`, `Severe`, `Fatal`)
- Calibrated prediction confidence score (%)
- Full class probability distribution across all severity tiers
- Immediate triage guidance and operational dispatch protocols

---

## 2. Project Objectives

1. **End-to-End Machine Learning Lifecycle:** Cover data ingestion, cleaning, leak-free feature engineering, preprocessing, model benchmarking, and artifact packaging.
2. **Zero Training-Serving Skew:** Ensure that the exact same preprocessing pipelines, scalers, imputers, and encoders used during training are executed during real-time online inference.
3. **Class Imbalance Resilience:** Mitigate extreme class skew (rare fatal and severe events) through stratified sampling and cost-sensitive balanced weighting.
4. **Interactive Web Application:** Deliver an intuitive, responsive web portal and REST API powered by Flask.
5. **Containerized Portability:** Provide turnkey Docker and Docker Compose deployment for multi-environment reproducibility.

---

## 3. Dataset Information

The project dataset resides at:
```text
data/dataset.csv
```

### Dataset Attributes:
* **Accident_Index**: Unique incident identification key (e.g. `ACC-2026-0001`). Dropped during preprocessing to eliminate data leakage.
* **Driver_Age**: Continuous numerical feature (17 to 85 years).
* **Vehicle_Speed**: Continuous vehicle velocity in km/h at time of collision.
* **Speed_Limit**: Posted road speed limit (30, 50, 60, 80, 100, 120 km/h).
* **Weather_Conditions**: Nominal categorical (`Clear`, `Rainy`, `Snowy`, `Foggy`, `High_Winds`).
* **Road_Surface_Conditions**: Nominal categorical (`Dry`, `Wet`, `Ice`, `Flooded`).
* **Lighting_Conditions**: Nominal categorical (`Daylight`, `Darkness_Lights_Lit`, `Darkness_No_Lights`, `Dawn_Dusk`).
* **Vehicle_Type**: Categorical (`Sedan`, `SUV`, `Motorcycle`, `Van`, `Truck`, `Bus`).
* **Driver_Alcohol**: Binary indicator (`Yes`, `No`).
* **Seatbelt_Restraint**: Binary indicator (`Worn`, `Not_Worn`).
* **Traffic_Density**: Ordinal categorical (`Low`, `Medium`, `High`, `Congested`).
* **Urban_Rural**: Location environment (`Urban`, `Rural`).
* **Number_of_Vehicles**: Integer count of vehicles involved.
* **Number_of_Casualties**: Integer count of injured/affected individuals.
* **Accident_Severity** (Target): 4-tier categorization (`Minor`, `Moderate`, `Severe`, `Fatal`).

---

## 4. Dataset Analysis

Exploratory analysis on the dataset reveals:
- **Dimensions:** 2,208 total records, 15 columns.
- **Missing Values:** ~2% realistic missingness in `Driver_Age` (44 rows), `Weather_Conditions` (33 rows), and `Traffic_Density` (44 rows), handled via median and mode imputation.
- **Duplicate Rows:** 8 duplicate records detected and automatically purged during pipeline ingestion.
- **Class Imbalance:**
  - `Minor`: ~52.0%
  - `Moderate`: ~28.0%
  - `Severe`: ~14.0%
  - `Fatal`: ~6.0%
- **Target Determination:** If a custom dataset is uploaded with a different target column name (e.g., `Severity`, `injury_severity`, `target`), the system's smart detection engine automatically resolves the target column.

---

## 5. Project Structure

```text
severity_prediction_project/
│
├── app.py                      # Flask web application & REST API routes
├── requirements.txt            # Production dependencies
├── Dockerfile                  # Container build instructions
├── docker-compose.yml          # Multi-container orchestration config
├── README.md                   # Complete system documentation
├── .gitignore                  # Git exclusion rules
├── .dockerignore               # Docker build context exclusion rules
│
├── data/
│   └── dataset.csv             # Primary dataset
│
├── models/
│   ├── __init__.py             # Package exports
│   ├── preprocessing.py        # Leak-free cleaning, imputation, scaling, split
│   ├── feature_engineering.py  # Domain-specific interaction features
│   ├── eda.py                  # Statistical visual analysis script
│   ├── train_model.py          # Algorithm benchmarking & model serialization
│   ├── predict.py              # Production inference engine with confidence
│   └── model_utils.py          # Relative paths, artifact I/O, domain advice
│
├── templates/
│   ├── index.html              # Main assessment portal with dynamic form
│   ├── result.html             # Triage output, confidence gauges, advice
│   └── about.html              # Technical architecture & M1-M5 docs
│
├── static/
│   └── style.css               # Modern, responsive design system
│
├── artifacts/
│   ├── model.pkl               # Serialized champion model
│   ├── preprocessing.pkl       # Fitted DataPreprocessor pipeline
│   └── metadata.json           # Model parameters, evaluation metrics, schema
│
└── outputs/
    └── eda/                    # High-resolution charts & eda_summary.json
        ├── 01_target_distribution.png
        ├── 02_missing_values.png
        ├── 03_correlation_matrix.png
        ├── 04_numerical_distributions.png
        ├── 05_categorical_distributions.png
        ├── 06_outlier_boxplots.png
        ├── 07_feature_vs_target.png
        └── eda_summary.json
```

---

## 6. Preprocessing (`models/preprocessing.py`)

- **Data Leakage Prevention:** All scalers, imputers, and encoders are fit **exclusively** on training data splits (`X_train`). The holdout test set (`X_test`) and incoming serving requests are only transformed using the pre-fitted parameters.
- **Duplicate & ID Handling:** Drops exact duplicate rows and automatically strips non-predictive identifier columns (`Accident_Index`).
- **Missing Value Imputation:**
  - Numerical features: `SimpleImputer(strategy='median')`
  - Categorical features: `SimpleImputer(strategy='most_frequent')`
- **Feature Scaling:** `StandardScaler` standardizes continuous variables to zero mean and unit variance.
- **Categorical Encoding:** `OneHotEncoder(handle_unknown='ignore', sparse_output=False)` creates binary indicator features without introducing artificial ordinal bias.
- **Train/Test Splitting:** `prepare_train_test_split` enforces stratified splitting (80% train, 20% test) to preserve exact minority class proportions.

---

## 7. Feature Engineering (`models/feature_engineering.py`)

Domain-informed features computed identically across training and serving:
1. `Speed_Delta = Vehicle_Speed - Speed_Limit`: Direct magnitude of speed violation.
2. `Speed_Ratio = Vehicle_Speed / Speed_Limit`: Normalized velocity index.
3. `Excess_Speed_Flag`: Binary indicator (1 if speeding, 0 otherwise).
4. `Adverse_Weather_Flag`: Flags rainy, snowy, foggy, or high wind conditions.
5. `Hazardous_Road_Flag`: Flags wet, icy, or flooded road conditions.
6. `High_Risk_Lighting_Flag`: Flags unlit or pitch-dark nighttime conditions.
7. `Vulnerable_Vehicle_Flag`: Flags motorcycles and two-wheeled transport.
8. `Driver_Impairment_Risk`: Interaction flag active when alcohol is detected or seatbelts are unfastened.
9. `Casualties_Per_Vehicle = Number_of_Casualties / Number_of_Vehicles`: Victim density ratio.
10. `Environmental_Hazard_Index`: Composite sum of adverse weather, slippery surface, and darkness.

---

## 8. Exploratory Data Analysis (`models/eda.py`)

Run the standalone EDA module to generate visual artifacts in `outputs/eda/`:
```bash
python -m models.eda
```

Generated outputs include:
* `01_target_distribution.png`: Class frequencies and percentage breakdown.
* `02_missing_values.png`: Column-wise missing percentage plot.
* `03_correlation_matrix.png`: Pearson correlation heatmap of numerical features.
* `04_numerical_distributions.png`: Histograms with KDE curves.
* `05_categorical_distributions.png`: Categorical frequency counts.
* `06_outlier_boxplots.png`: Distribution spreads and outlier identification.
* `07_feature_vs_target.png`: Bivariate risk plots (Speed and Alcohol vs Severity).
* `eda_summary.json`: Machine-readable summary statistics.

---

## 9. Models Evaluated & Model Selection

Four candidate architectures were trained and benchmarked:
1. **Logistic Regression (L2 Regularized):** Linear baseline with softmax multiclass loss and balanced class weighting.
2. **Decision Tree Classifier:** Non-linear decision rules with depth-limited regularization (`max_depth=6`).
3. **Random Forest Classifier (Bagging Ensemble):** 120 decorrelated decision trees with balanced bootstrap sampling.
4. **Gradient Boosting Classifier (Boosting):** 100 sequential boosting stages minimizing multiclass log-loss.

### Model Evaluation Discipline:
Because rare classes (`Fatal` ~6%, `Severe` ~14%) carry critical real-world importance, **Macro F1** is the primary selection metric. Models maximizing raw accuracy while failing on minority classes are strictly penalized.

### Serialized Artifacts:
```text
artifacts/model.pkl
artifacts/preprocessing.pkl
artifacts/metadata.json
```

---

## 10. Curriculum Topics (M1 – M5)

### M1 — System Lifecycle & Training-Serving Parity
- **Lifecycle Pipeline:**
  ```text
  Raw Data → Cleaning & Deduplication → Feature Engineering → Stratified Split 
  → Preprocessing Pipeline Fit → Model Benchmarking → Artifact Serialization 
  → Production Endpoint Serving → Monitoring
  ```
- **Serving Path:**
  ```text
  HTTP Request → JSON/Form Parser → Schema Validation → engineer_features() 
  → preprocessor.transform() → model.predict_proba() → Triage Rules → HTML/JSON Response
  ```
- **Zero Skew:** All transformation parameters (medians, means, variances, one-hot categories) are loaded directly from `artifacts/preprocessing.pkl`.

### M2 — Supervised Learning & Linear Regularization
- In-depth implementation of `LogisticRegression` with balanced class weights.
- `StandardScaler` for zero-mean and unit-variance normalization.
- `OneHotEncoder(handle_unknown='ignore')` prevents inference failures on unseen categories.

### M3 — Tree-Based Ensembles & Feature Importance
- Bagging (Random Forest) vs. Boosting (Gradient Boosting).
- Multi-feature split entropy and gini impurity reduction.
- Feature importance analysis logged to `metadata.json`.

### M4 — Feature Engineering & Hazard Indices
- Domain-grounded composite indices without target leakage.
- Interaction features capturing environmental risk combinations.

### M5 — Model Evaluation & Metrics
- Stratified holdout evaluation preserving class ratios.
- Metrics reported: Accuracy, Precision (Macro/Weighted), Recall (Macro/Weighted), Macro F1, Weighted F1, Multiclass ROC-AUC (One-vs-Rest), and Confusion Matrices.

---

## 11. Installation & Local Setup

### Step 1: Clone or Navigate to the Project
```bash
cd severity_prediction_project
```

### Step 2: Create a Virtual Environment
```bash
python -m venv venv
```

### Step 3: Activate the Virtual Environment
- **Windows (PowerShell / Command Prompt):**
  ```bash
  venv\Scripts\activate
  ```
- **Linux / macOS:**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 12. Execution Commands

### Run Exploratory Data Analysis (EDA)
```bash
python -m models.eda
```

### Train and Evaluate Models
```bash
python -m models.train_model
```

### Start the Flask Application
```bash
python app.py
```

### Access Application in Browser
Open:
```text
http://localhost:5000
```

---

## 13. Docker & Docker Compose Instructions

### Build Docker Image
```bash
docker build -t severity-prediction-project .
```

### Run Docker Container
```bash
docker run -p 5000:5000 severity-prediction-project
```

### Launch with Docker Compose
```bash
docker compose up --build
```

### Stop Docker Compose Service
```bash
docker compose down
```

---

## 14. Programmatic REST API Usage

### Endpoint: `POST /api/predict`
**Request:**
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Driver_Age": 26,
    "Vehicle_Speed": 105,
    "Speed_Limit": 50,
    "Weather_Conditions": "Rainy",
    "Road_Surface_Conditions": "Wet",
    "Lighting_Conditions": "Darkness_No_Lights",
    "Vehicle_Type": "Motorcycle",
    "Driver_Alcohol": "Yes",
    "Seatbelt_Restraint": "Not_Worn",
    "Traffic_Density": "Medium",
    "Urban_Rural": "Rural",
    "Number_of_Vehicles": 2,
    "Number_of_Casualties": 2
  }'
```

**Response:**
```json
{
  "status": "success",
  "prediction": "Fatal",
  "confidence_percentage": 78.4,
  "class_probabilities": {
    "Fatal": 78.4,
    "Minor": 1.2,
    "Moderate": 4.8,
    "Severe": 15.6
  },
  "action_advice": {
    "urgency": "Immediate Emergency Response",
    "action": "Immediate dispatch of Advanced Life Support (ALS) paramedic units, heavy rescue extraction, and trauma center diversion.",
    "level": "Critical / Fatal"
  },
  "model_used": "Random Forest"
}
```

### Health Check Endpoint
```bash
curl http://localhost:5000/api/health
```

---

## 15. Troubleshooting

1. **Port 5000 Already in Use:**
   - Change port via environment variable:
     - Windows PowerShell: `$env:PORT=5050; python app.py`
     - Linux/macOS: `PORT=5050 python app.py`
2. **Missing Artifacts Warning:**
   - If running `app.py` for the first time without pre-existing `.pkl` files, the application automatically invokes `train_and_evaluate_models()` to generate all artifacts seamlessly.
3. **Custom Dataset Upload:**
   - Place any new CSV into `data/dataset.csv`.
   - Run `python -m models.train_model` to re-train the pipeline.
   - The web interface dynamically adapts to the new columns and categories!
