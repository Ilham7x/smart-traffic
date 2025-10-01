import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
import numpy as np

CSV_PATH = Path("data/raw/Metro_Interstate_Traffic_Volume.csv")
MODEL_PATH = Path("models/model_xgb.joblib")

df = pd.read_csv(CSV_PATH)
df["date_time"] = pd.to_datetime(df["date_time"], format="%d-%m-%Y %H:%M")


# time-based features
df["hour"] = df["date_time"].dt.hour
df["dayofweek"] = df["date_time"].dt.dayofweek
df["month"] = df["date_time"].dt.month

# encode categorical features (simple one-hot encoding)
df = pd.get_dummies(df, columns=["holiday", "weather_main", "weather_description"], drop_first=True)

target = "traffic_volume"
X = df.drop(columns=[target, "date_time"])
y = df[target]

# train/test split 
split_idx = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# model training
model = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

Path("models").mkdir(exist_ok=True, parents=True)
joblib.dump(model, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")