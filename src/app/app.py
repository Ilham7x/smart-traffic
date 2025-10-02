import streamlit as st, pandas as pd, joblib
from pathlib import Path
from xgboost import XGBRegressor

st.set_page_config(page_title="Smart Traffic Predictor", layout="wide")
st.title("🚦 Smart Traffic Predictor")

# intro
st.markdown("""
This demo predicts **traffic volume (vehicles/hour)** using time & weather features.
- **Nowcasting**: predicts volume for the current hour in the data.
- **Forecast**: (if available) predicts **next hour** using lag features.
Upload the Kaggle CSV to see predictions and how they compare to ground truth.
""")

NOWCAST_PATH = Path("models/model_xgb.joblib")
FORECAST_PATH = Path("models/model_xgb_t+60.joblib")

modes = ["Nowcasting (current hour)"]
if FORECAST_PATH.exists():
    modes.append("Forecast (next hour)")
mode = st.radio("Mode", modes, index=0, help="Switch to next-hour forecast if the model file exists.")

MODEL_FILE = NOWCAST_PATH if mode.startswith("Nowcasting") else FORECAST_PATH
if not MODEL_FILE.exists():
    st.error(f"Model file missing: {MODEL_FILE}. Train and save it first.")
    st.stop()

art = joblib.load(MODEL_FILE)
if isinstance(art, dict):
    model = art["model"]
    columns = art["columns"]
    metrics = art.get("metrics", {})
else:
    model = art
    columns = None
    metrics = {}

# KPI strip
c1, c2, c3 = st.columns(3)
c1.metric("Mode", "Nowcast" if mode.startswith("Nowcasting") else "Next hour")
c2.metric("MAE (vehicles/hr)", f"{metrics.get('mae','-'):.0f}" if metrics else "-")
c3.metric("RMSE (vehicles/hr)", f"{metrics.get('rmse','-'):.0f}" if metrics else "-")
st.caption(f"Using `{MODEL_FILE.name}`. Lower MAE/RMSE is better.")

up = st.file_uploader("Upload Metro_Interstate_Traffic_Volume.csv", type=["csv"])

def featurize(df, expected_cols=None):
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df = pd.get_dummies(df, columns=["holiday","weather_main","weather_description"], drop_first=True)
    X = df.drop(columns=[c for c in ["traffic_volume","date_time"] if c in df.columns])
    if expected_cols is not None:
        X = X.reindex(columns=expected_cols, fill_value=0)  # align to training schema
    return df, X

if up:
    raw = pd.read_csv(up)
    df, X = featurize(raw, expected_cols=columns)
    preds = model.predict(X)

    st.success("Predictions (first 200 points):")
    st.caption("Y-axis = vehicles per hour. X-axis = time steps from the uploaded file (hourly).")
    st.line_chart(pd.DataFrame({"predicted_volume": preds[:200]}))

    if "traffic_volume" in df.columns and mode.startswith("Nowcasting"):
        st.markdown("### Actual vs Predicted (first 200)")
        st.caption("Blue = actual ground truth from CSV; Orange = model prediction for the same timestamps.")
        n = min(len(df), len(preds))
        comp = pd.DataFrame({
            "actual": df["traffic_volume"].iloc[:n].values[:200],
            "predicted": preds[:200]
        })
        st.line_chart(comp)

    with st.expander("How to read these charts"):
        st.markdown("""
- **Peaks** ≈ rush hours; **valleys** ≈ off-peak/night.  
- **Prediction chart** shows model output only; **Actual vs Predicted** overlays truth vs model.  
- **MAE/RMSE** at the top quantify typical error (vehicles/hour).  
- If you switch to **Forecast (next hour)**, predictions shift one hour ahead.
        """)
else:
    st.info("Upload the Kaggle CSV to view predictions and comparisons.")
