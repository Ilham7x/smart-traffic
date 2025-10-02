# 🚦 Smart Traffic Predictor — Nowcasting (Streamlit + XGBoost)

Predicts **traffic volume (vehicles/hour)** for the **current hour** (nowcasting) using time & weather features from the *Metro Interstate Traffic Volume* dataset (Kaggle).

> This is part of a Smart City project: it addresses urban congestion by learning traffic patterns and making them explorable in a simple dashboard.

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
---
## 2) Dataset
The CSV is included in this repo at:
```
data/raw/Metro_Interstate_Traffic_Volume.csv
```
---
## 3) Run the app
```bash
streamlit run src/app/app.py
```
---
## 4) When the app opens, click Upload and select the csv file:
```bash
Metro_Interstate_Traffic_Volume.csv
```
---

## 🧠 How it works (model)

**Target:** `traffic_volume` (vehicles/hour)

**Features:**
- **Time:** `hour`, `dayofweek`, `month`
- **Weather & holiday:** one-hot encoded (`weather_main`, `weather_description`, `holiday`)

**Split:** First **80%** train, last **20%** test 

---

## 📊 What you’ll see in the app

### **Predictions (first 200 points)**
Model-predicted **vehicles/hour** over time (**timestamp x-axis** with tooltips).

### **Actual vs Predicted (first 200)**
Overlay of **ground truth** vs **model prediction** for the **same hour**.  
- **Tight overlap** ⇒ good fit  
- **Gaps** ⇒ error (see **MAE/RMSE** at the top)

### **Peak hours summary**
- **Top 3 hours** by predicted volume (veh/hr)  
- **Average predicted volume** by **day-of-week**

> Units are **vehicles per hour**; data is **hourly**.

---

## ✅ Results / Metrics

Evaluation uses the **last 20%** of the timeline (chronological holdout).  
(Results can vary slightly by environment/seed.)

- **MAE:** ~`293` veh/hr  
- **RMSE:** ~`506` veh/hr

**Algorithm:**
```python
from xgboost import XGBRegressor

XGBRegressor(
  n_estimators=300,
  max_depth=6,
  learning_rate=0.1,
  subsample=0.8,
  colsample_bytree=0.8,
  random_state=42
)

```
---

## 📸 Screenshots

- Predictions chart
<img width="2540" height="716" alt="Screenshot 2025-10-02 211540" src="https://github.com/user-attachments/assets/bf0d3e84-1c77-43bc-93a1-7d69806fb726" />
 
- Actual vs Predicted chart
<img width="2533" height="640" alt="Screenshot 2025-10-02 211554" src="https://github.com/user-attachments/assets/d3a83cfc-128a-4ce6-ba14-73bcf9188a41" />

---

## 📌 Limitations

- **Nowcasting only** (predicts the same hour).  
- Single-corridor hourly dataset (no incident/route topology).  
- Assumes clean timestamps and consistent hourly cadence.

---

## 🔭 Future work

- **Next-hour forecasting** using lag features (`lag_1h..lag_4h`) and proper t+1 alignment.  
- **Live data** integration (Google/HERE) for real-time dashboards.  
- **Routing prototype** to suggest congestion-aware alternatives.  
- **Monitoring** for drift + scheduled retraining.

---







