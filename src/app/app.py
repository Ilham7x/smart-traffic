import streamlit as st, pandas as pd, joblib
import altair as alt
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
else:  # legacy raw model
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
    # original timestamps for plotting
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)

    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df_feat = pd.get_dummies(
        df.copy(),
        columns=["holiday","weather_main","weather_description"],
        drop_first=True
    )
    X = df_feat.drop(columns=[c for c in ["traffic_volume","date_time"] if c in df_feat.columns])
    if expected_cols is not None:
        X = X.reindex(columns=expected_cols, fill_value=0)  # align to training schema
    return df, X

if up:
    raw = pd.read_csv(up)
    df, X = featurize(raw, expected_cols=columns)
    preds = model.predict(X)

    # Chart 1: prediction over time
    st.success("Predictions (first 200 points):")
    st.caption("Y-axis = vehicles/hour. X-axis = actual timestamp from the dataset (hourly).")

    pred_df = pd.DataFrame({
        "timestamp": df["date_time"].iloc[:len(preds)],
        "predicted": preds
    }).iloc[:200]

    chart_pred = (
        alt.Chart(pred_df)
        .mark_line()
        .encode(
            x=alt.X("timestamp:T", title="Time"),
            y=alt.Y("predicted:Q", title="Predicted traffic volume (veh/hr)"),
            tooltip=[alt.Tooltip("timestamp:T"), alt.Tooltip("predicted:Q", format=".0f")]
        )
        .interactive()
    )
    st.altair_chart(chart_pred, use_container_width=True)

    # Chart 2: Actual vs Predicted with timestamps
    if "traffic_volume" in df.columns and mode.startswith("Nowcasting"):
        st.markdown("### Actual vs Predicted (first 200)")
        st.caption("Blue = actual ground truth from CSV; Orange = model prediction for the same timestamps.")

        n = min(len(df), len(preds))
        comp_df = pd.DataFrame({
            "timestamp": df["date_time"].iloc[:n].values[:200],
            "actual": df["traffic_volume"].iloc[:n].values[:200],
            "predicted": preds[:200]
        })
        comp_long = comp_df.melt("timestamp", var_name="series", value_name="volume")

        chart_comp = (
            alt.Chart(comp_long)
            .mark_line()
            .encode(
                x=alt.X("timestamp:T", title="Time"),
                y=alt.Y("volume:Q", title="Traffic volume (veh/hr)"),
                color=alt.Color(
                    "series:N",
                    title="Series",
                    scale=alt.Scale(
                        domain=["actual", "predicted"],         # order matters
                        range=["#1f77b4", "#ff7f0e"]            # Blue, Orange
                    )
                ),
                tooltip=[
                    alt.Tooltip("timestamp:T", title="Time"),
                    alt.Tooltip("series:N", title="Series"),
                    alt.Tooltip("volume:Q", title="Volume (veh/hr)", format=".0f"),
                ],
            )
            .interactive()
        )
        st.altair_chart(chart_comp, use_container_width=True)

    # peak hours summary 
    st.markdown("### Peak hours summary")
    tmp = df.iloc[:len(preds)].copy()
    tmp["predicted"] = preds

    # Top 3 hours of day (by predicted mean)
    hourly = (
        tmp.assign(hour=tmp["date_time"].dt.hour)
           .groupby("hour")
           .agg(predicted_mean=("predicted","mean"),
                actual_mean=("traffic_volume","mean"))
           .round(0)
    )
    top3 = hourly.sort_values("predicted_mean", ascending=False).head(3)

    # Day of week averages (predicted)
    dow = (
        tmp.assign(dow=tmp["date_time"].dt.day_name())
           .groupby("dow")["predicted"].mean()
           .reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
           .round(0)
           .reset_index(name="predicted_mean")
    )

    c1, c2 = st.columns(2)
    with c1:
        st.write("**Top 3 hours by predicted volume (veh/hr):**")
        st.table(top3)
    with c2:
        st.write("**Average predicted volume by day of week (veh/hr):**")
        st.table(dow)

    with st.expander("How to read these charts"):
        st.markdown("""
- **Peaks** ≈ rush hours; **valleys** ≈ off-peak/night.  
- The **time axis** uses real timestamps from the CSV—hover to see exact time.  
- **MAE/RMSE** at the top quantify typical error (vehicles/hour).  
- **Forecast** mode (if available) predicts the **next hour** using lag features.
        """)
else:
    st.info("Upload the Kaggle CSV to view predictions and comparisons.")
