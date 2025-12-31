import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


# PATHS

DATA_DIR = r"C:data"

PRICE_FILE = os.path.join(DATA_DIR, "stock_price_realtime.csv")
PRED_FILE = os.path.join(DATA_DIR, "stock_predictions.csv")
METRICS_FILE = os.path.join(DATA_DIR, "model_metrics.csv")


df = pd.read_csv(PRICE_FILE)

df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values(["company_code", "Datetime"])


# RSI FUNCTION

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# FEATURE ENGINEERING

df["return"] = df.groupby("company_code")["Close"].pct_change()
df["ma_5"] = (
    df.groupby("company_code")["Close"]
    .rolling(5).mean()
    .reset_index(0, drop=True)
)
df["ma_10"] = (
    df.groupby("company_code")["Close"]
    .rolling(10).mean()
    .reset_index(0, drop=True)
)
df["volatility"] = (
    df.groupby("company_code")["Close"]
    .rolling(10).std()
    .reset_index(0, drop=True)
)

#  RSI (14-period)
df["rsi_14"] = (
    df.groupby("company_code")["Close"]
      .transform(lambda x: compute_rsi(x, period=14))
)


df.dropna(inplace=True)

# TARGET VARIABLE (NEXT CLOSE)

df["target_close"] = df.groupby("company_code")["Close"].shift(-1)
df.dropna(inplace=True)


# MODEL TRAINING (COMPANY-WISE)

results = []
all_y_true = []
all_y_pred = []

features = [
    "Close",
    "Volume",
    "return",
    "ma_5",
    "ma_10",
    "volatility",
    "rsi_14"
]

for company in df["company_code"].unique():
    print(f" Processing {company}...")

    company_df = df[df["company_code"] == company]

    X = company_df[features]
    y = company_df["target_close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Collect global metrics
    all_y_true.extend(y_test.values)
    all_y_pred.extend(preds)

    mae = mean_absolute_error(y_test, preds)
    print(f"   MAE: {mae:.2f}")

    out_df = company_df.iloc[-len(preds):].copy()
    out_df["predicted_close"] = preds
    out_df["prediction_error"] = out_df["target_close"] - preds
    out_df["model"] = "RandomForest"

    results.append(out_df)

# SAVE PREDICTIONS FOR POWER BI

final_df = pd.concat(results, ignore_index=True)
final_df.to_csv(PRED_FILE, index=False)

print("\n Predictions saved")
print(f" {PRED_FILE}")

# MODEL METRICS (GLOBAL)

y_true = np.array(all_y_true)
y_pred = np.array(all_y_pred)

mae = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

metrics_df = pd.DataFrame([{
    "MAE": round(mae, 2),
    "RMSE": round(rmse, 2),
    "MAPE": round(mape, 2)
}])

metrics_df.to_csv(METRICS_FILE, index=False)

print("\n Model Metrics")
print(metrics_df)
print(f"{METRICS_FILE}")

