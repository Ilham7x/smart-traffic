import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(page_title="Smart Traffic Predictor", layout="wide")
st.title("🚦 Smart Traffic Predictor (nowcasting demo)")

MODEL_PATH = Path("models/model_xgb.joblib")
if not MODEL_PATH.exists():
    st.error("Model not found. Train it first: python src/models/train_xgb.py")
    st.stop()

model = joblib.load(MODEL_PATH)

st.subheader("Upload the same CSV you trained on (or a subset)")
uploaded = st.file_uploader("Metro_Interstate_Traffic_Volume.csv", type=["csv"])

def preprocess(df):
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["date_time"]).copy()
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    # one-hot 
    df = pd.get_dummies(df, columns=["holiday", "weather_main", "weather_description"], drop_first=True)
    return df

if uploaded:
    raw = pd.read_csv(uploaded)
    df = preprocess(raw)
    target = "traffic_volume"
    drop_cols = [c for c in ["traffic_volume","date_time"] if c in df.columns]
    X = df.drop(columns=drop_cols)

    for col in ["hour","dayofweek","month"]:
        if col not in X.columns:
            st.warning(f"Missing column {col}; creating a placeholder 0.")
            X[col] = 0

    # prediction
    try:
        yhat = model.predict(X)
        st.success("Predictions ready (showing first 200).")
        st.line_chart(pd.DataFrame({"predicted_traffic_volume": yhat[:200]}))
        if "traffic_volume" in raw.columns:
            st.write("Sample actual vs predicted (first 200):")
            show = pd.DataFrame({
                "actual": raw["traffic_volume"].iloc[:len(yhat)].values[:200],
                "pred": yhat[:200]
            })
            st.line_chart(show)
    except Exception as e:
        st.error(f"Could not predict. Tip: save training columns and reuse them in the app.\n{e}")
else:
    st.info("Upload the CSV to see predictions.")