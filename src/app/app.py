import streamlit as st, pandas as pd, joblib
import altair as alt
from pathlib import Path

SAMPLE_PATH = Path("../data/sample/metro_sample_500.csv")

st.set_page_config(page_title="Smart Traffic Predictor", layout="wide")
st.title("🚦 Smart Traffic Predictor")

# Intro 
st.markdown("""
This app predicts **traffic volume (vehicles/hour)** for the **same hour** (nowcasting),
using time & weather features. Upload the Kaggle CSV to see predictions and how they
compare to ground truth.
""")

# Model - nowcasting
MODEL_FILE = Path("models/model_xgb.joblib")
if not MODEL_FILE.exists():
    st.error("Nowcasting model missing at `models/model_xgb.joblib`. Train it first (run `python src/models/train_xgb.py`).")
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

# KPIs 
c1, c2, c3 = st.columns(3)
c1.metric("Mode", "Nowcasting")
c2.metric("MAE (vehicles/hr)", f"{metrics.get('mae','-'):.0f}" if metrics else "-")
c3.metric("RMSE (vehicles/hr)", f"{metrics.get('rmse','-'):.0f}" if metrics else "-")
st.caption(f"Using `{MODEL_FILE.name}`. Lower MAE/RMSE is better.")

# data input
up = st.file_uploader("Upload Metro_Interstate_Traffic_Volume.csv", type=["csv"])

def featurize(df, expected_cols=None):
    df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M", errors="coerce")
    df = df.dropna(subset=["date_time"]).sort_values("date_time").reset_index(drop=True)

    # Time features
    df["hour"] = df["date_time"].dt.hour
    df["dayofweek"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month

    # One-hot
    df_feat = pd.get_dummies(
        df.copy(),
        columns=["holiday", "weather_main", "weather_description"],
        drop_first=True
    )

    # feature matrix
    X = df_feat.drop(columns=[c for c in ["traffic_volume", "date_time"] if c in df_feat.columns])
    if expected_cols is not None:
        X = X.reindex(columns=expected_cols, fill_value=0)  # align to training schema
    return df, X

if up:
    raw = pd.read_csv(up)
    df, X = featurize(raw, expected_cols=columns)
    preds = model.predict(X)

    # Chart 1: predictions over time (first 200)
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

    # Chart 2: Actual vs Predicted (same hour, first 200) 
    if "traffic_volume" in df.columns:
        st.markdown("### Actual vs Predicted (first 200)")
        st.caption("**Blue = actual; Orange = predicted (same hour).**")

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
                        domain=["actual", "predicted"],  # order matters
                        range=["#1f77b4", "#ff7f0e"]      # Blue, Orange
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

    # Peak hours summary
    st.markdown("### Peak hours summary")
    tmp = df.iloc[:len(preds)].copy()
    tmp["predicted"] = preds

    hourly = (
        tmp.assign(hour=tmp["date_time"].dt.hour)
           .groupby("hour")
           .agg(predicted_mean=("predicted", "mean"),
                actual_mean=("traffic_volume", "mean"))
           .round(0)
    )
    top3 = hourly.sort_values("predicted_mean", ascending=False).head(3)

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
- **Nowcasting** compares model prediction vs actual for the **same hour**.
- The **time axis** uses real timestamps—hover to see exact date/hour.
- **MAE/RMSE** at the top quantify typical error (vehicles/hour).
        """)
else:
    st.info("Upload the Kaggle CSV to view predictions and comparisons.")
