# 🚦 ASTRAM — Event-Driven Congestion Intelligence

ASTRAM is a **Streamlit-based traffic operations intelligence app** built for **event-driven congestion prediction and operational response planning**.

It combines:

* **Rule-based congestion impact scoring**
* **CatBoost models for road-closure probability and disruption duration**
* **H3 hotspot intelligence**
* **Operational recommendations** such as road closure planning, manpower deployment, barricading, and diversion strategy

The system is designed for traffic events such as **accidents, breakdowns, construction activity, public events, waterlogging, tree fall, and other disruptions** that can affect urban traffic flow.

---

# 📌 Features

## 1) Event Impact Prediction

Predicts the likely **impact level** of a traffic event:

* **Low**
* **Medium**
* **High**
* **Critical**

The prediction is based on:

* event type
* event cause
* priority
* corridor
* junction
* vehicle type
* event time
* description keywords
* hotspot intelligence

---

## 2) Hotspot Intelligence

The app uses **H3 spatial indexing** to identify whether an event falls inside a historically sensitive hotspot zone.

It displays:

* hotspot score
* hotspot level
* hotspot rank
* historical event count
* average historical impact
* average historical duration
* closure rate
* hotspot map with highlighted predicted hotspot cell

---

## 3) Road Closure & Duration Prediction

The app loads trained **CatBoost models** to predict:

* **road closure probability**
* **expected disruption duration**

---

## 4) Operational Recommendation Engine

Based on the predicted impact level, the app recommends:

* **Road Closure Plan**
* **Manpower Deployment**
* **Barricading Requirement**
* **Diversion Route Plan**

Example:

* **Low** → No closure + minimal field monitoring
* **Medium** → Temporary lane control + local diversion readiness
* **High** → Partial closure + corridor-level diversion
* **Critical** → Major diversion / full closure consideration + multi-team deployment

---

# 🧠 Project Workflow

This project follows a **3-stage pipeline**.

---

## **Step 1 — Feature Rebuild + Hotspot Generation**

Raw traffic event data is cleaned and transformed into engineered features used for modeling and app inference.

This stage also builds **H3 hotspot intelligence outputs**.

### Script

* `step1_rebuild_eda_h3_features.py`

### Step-1 responsibilities

* clean raw traffic event data
* engineer structured features
* derive temporal / categorical / description-based signals
* create hotspot features using H3 spatial indexing
* generate app-ready feature outputs

### Outputs generated in Step 1

Inside `rebuild_outputs/`

* `step1_rebuilt_features.csv`
* `h3_hotspot_summary.csv`

---

## **Step 2 — Model Training**

Using the Step-1 engineered features, CatBoost models are trained for:

* **Road Closure Prediction**
* **Disruption Duration Prediction**

### Script

* `step2_train_strong_models.py`

### Step-2 responsibilities

* load Step-1 rebuilt features
* train CatBoost classification model for closure prediction
* train CatBoost regression model for duration prediction
* save feature lists and categorical feature metadata
* export deployment-ready model artifacts

### Outputs generated in Step 2

Inside `final_safe_model_artifacts/`

* `closure_classifier_catboost.pkl`
* `closure_feature_columns.pkl`
* `closure_categorical_feature_names.pkl`
* `duration_regressor_catboost.pkl`
* `duration_feature_columns.pkl`
* `duration_categorical_feature_names.pkl`

---

## **Step 3 — Streamlit App**

The Streamlit app (`app.py`) loads the outputs generated in **Step 1** and **Step 2** and produces:

* congestion **impact level prediction**
* **hotspot intelligence**
* **road closure probability**
* **disruption duration**
* **operational recommendations**

---

# 🧠 Hybrid Prediction Architecture

The app uses a **hybrid architecture**:

## A) Rule-Based Impact Scoring

A calibrated scoring engine computes impact severity using:

* event cause severity
* operational priority
* peak-hour effect
* planned vs unplanned nature
* blockage / jam / overturned description flags
* heavy vehicle involvement
* corridor / junction criticality
* hotspot influence

## B) ML-Based Predictions

Two CatBoost models are loaded:

* **Closure model** → predicts probability of road closure
* **Duration model** → predicts disruption duration in minutes

## C) Operational Decision Layer

The outputs from the scoring system + models are combined into a final operational response recommendation.

---

# 📂 Project Structure

```bash
astram-congestion-app/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── step1_rebuild_eda_h3_features.py
├── step2_train_strong_models.py
│
├── final_safe_model_artifacts/
│   ├── closure_classifier_catboost.pkl
│   ├── closure_feature_columns.pkl
│   ├── closure_categorical_feature_names.pkl
│   ├── duration_regressor_catboost.pkl
│   ├── duration_feature_columns.pkl
│   └── duration_categorical_feature_names.pkl
│
└── rebuild_outputs/
    ├── h3_hotspot_summary.csv
    └── step1_rebuilt_features.csv
```

---

# 📁 Required Files

## 1) Application file

* `app.py`

## 2) Pipeline scripts

* `step1_rebuild_eda_h3_features.py`
* `step2_train_strong_models.py`

## 3) Model artifacts folder

`final_safe_model_artifacts/` must contain:

* `closure_classifier_catboost.pkl`
* `closure_feature_columns.pkl`
* `closure_categorical_feature_names.pkl`
* `duration_regressor_catboost.pkl`
* `duration_feature_columns.pkl`
* `duration_categorical_feature_names.pkl`

## 4) Data / hotspot files

`rebuild_outputs/` must contain:

* `h3_hotspot_summary.csv`
* `step1_rebuilt_features.csv`

---

# ⚙️ Installation

## Clone the repository

```bash
git clone https://github.com/Bhupanimounika22/astram-congestion-app.git
cd astram-congestion-app
```

Replace `YOUR_USERNAME` with your GitHub username.

---

## Create virtual environment (recommended)

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Mac / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ How to Run the Project

There are **two ways** to run this project.

---

# Option A — Run the app directly (if artifacts already exist)

If your repository already contains:

* `rebuild_outputs/step1_rebuilt_features.csv`
* `rebuild_outputs/h3_hotspot_summary.csv`
* `final_safe_model_artifacts/...`

then you can directly run:

```bash
streamlit run app.py
```

This is the easiest option for **demo / deployment**.

---

# Option B — Rebuild from scratch

If you want to regenerate the full pipeline:

## Step 1 — Rebuild features + hotspot outputs

```bash
python step1_rebuild_eda_h3_features.py
```

This generates:

* `rebuild_outputs/step1_rebuilt_features.csv`
* `rebuild_outputs/h3_hotspot_summary.csv`

## Step 2 — Train models

```bash
python step2_train_strong_models.py
```

This generates model artifacts inside:

* `final_safe_model_artifacts/`

## Step 3 — Run the app

```bash
streamlit run app.py
```

---

# 🧪 Example Use Cases

The app includes **example event scenarios** for quick testing:

* **Low** → minor planned maintenance / construction
* **Medium** → planned event with moderate slowdown
* **High** → accident with lane blockage and congestion
* **Critical** → major truck accident with overturned vehicle and severe congestion

These examples help validate the calibrated impact scoring logic.

---

# 🗺️ App Outputs

For each event, the app displays:

## Hotspot Intelligence

* hotspot score
* hotspot level
* hotspot rank
* H3 cell
* historical impact metrics

## Prediction Summary

* impact level
* impact score
* closure probability
* predicted duration

## Operational Recommendation

* road closure plan
* manpower deployment
* barricading requirement
* diversion route plan

## Explanation Panel

A **“Why this prediction?”** section explains the key factors that influenced the result.

---

# 🚀 Deployment on Streamlit Community Cloud

## 1) Push this project to GitHub

Make sure your repository contains:

* `app.py`
* `requirements.txt`
* `final_safe_model_artifacts/`
* `rebuild_outputs/`

If you want the repo to also support rebuilding from scratch, keep:

* `step1_rebuild_eda_h3_features.py`
* `step2_train_strong_models.py`

---

## 2) Deploy on Streamlit

Go to **Streamlit Community Cloud** and create a new app.

Set:

* **Repository** → your GitHub repository
* **Branch** → `main`
* **Main file path** → `app.py`

Then click **Deploy**.

---

# 📦 requirements.txt

Use the following dependencies in `requirements.txt`:

```txt
streamlit
pandas
numpy
joblib
folium
streamlit-folium
h3
catboost
scikit-learn
pyarrow
```

---

# ⚠️ Notes

## 1) Keep folder names unchanged

The app expects these exact paths:

* `final_safe_model_artifacts`
* `rebuild_outputs`

If you rename them, the app will fail to load files.

---

## 2) Step-1 and Step-2 outputs are required for app inference

`app.py` does **not** train models on the fly.
It only **loads pre-generated outputs** from:

* **Step 1** → `rebuild_outputs/`
* **Step 2** → `final_safe_model_artifacts/`

---

## 3) Large files

If your `.pkl` model files or `.csv` files are very large, GitHub upload may be slow or hit size limits.

---

## 4) Streamlit deployment performance

If rendering **all hotspot cells** makes the app slow, you can switch the map to show only nearby hotspots by changing:

```python
hotspot_map = draw_hotspot_map(latitude, longitude, result["hotspot"], show_all_hotspots=True)
```

to

```python
hotspot_map = draw_hotspot_map(latitude, longitude, result["hotspot"], show_all_hotspots=False)
```

---

# 📌 Future Improvements

Possible next improvements for the app:

* live traffic API integration
* weather-aware congestion adjustment
* police deployment optimization
* corridor-specific diversion recommendations
* incident timeline / escalation dashboard
* downloadable PDF operational report

---

# 👩‍💻 Author

Developed as part of an **event-driven congestion prediction and traffic operations intelligence project** for Bengaluru traffic scenarios.
