# 🚦 Smart Traffic Predictor — Nowcasting (Streamlit + XGBoost)

Predicts **traffic volume (vehicles/hour)** for the **current hour** (nowcasting) using time & weather features from the *Metro Interstate Traffic Volume* dataset (Kaggle).

> This project is part of a Smart City portfolio: it addresses urban congestion by learning traffic patterns and making them explorable in a simple dashboard.

---

## What’s included
- **Model**: XGBoost Regressor trained with time features and weather one-hots.
- **Leakage-safe split**: Train on the first 80%, test on the last 20% (chronological).
- **Artifact**: Saved with **model + feature schema + metrics** for reliable inference.
- **App**: Streamlit dashboard with timestamped charts, **Actual vs Predicted**, and **Peak Hours** summary.
- **Notebook (optional)**: EDA → training → metrics → feature importances.

---

## 🗂️ Project structure
```
smart-traffic/
├─ data/
│ ├─ raw/Metro_Interstate_Traffic_Volume.csv 
├─ models/
│ └─ model_xgb.joblib # trained artifact (model+schema+metrics)
├─ notebooks/
│ └─ 01_train_xgb.ipynb # optional: EDA + training
├─ outputs/ # metrics or sample predictions
├─ src/
│ ├─ app/app.py # Streamlit app (nowcasting)
│ └─ models/train_xgb.py # training script
├─ requirements.txt
└─ README.md
```


---

## 🚀 Quickstart

### 1) Create a virtual environment & install deps
```bash
# Windows (PowerShell/CMD)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Add the dataset

Download **`Metro_Interstate_Traffic_Volume.csv`** from Kaggle and place it at:

---

## Run the app

Just run the app:

```bash
streamlit run src/app/app.py
```

🧠 How it works (model)

Target: traffic_volume (vehicles/hour)

Features:

Time: hour, dayofweek, month

Weather & holiday: one-hot encoded (weather_main, weather_description, holiday)

Split: First 80% train, last 20% test (chronological to mimic real future)



