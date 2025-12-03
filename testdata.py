"""
Train TWO models:
    1. Linear Regression
    2. RandomForestRegressor

Using technical indicators:
    Open, High, Low, Close, Volume, SMA5, SMA20, RSI, MACD, Return

Saves:
    lin_model.pkl
    rf_model.pkl
    scaler.pkl

ADDED:
    • Prediction line graph
    • Comparison table (MAE, RMSE, R²)
"""

import pandas as pd
import numpy as np
import time
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

CSV_PATH = r"C:\Users\noosa\NVidia_stock_history.csv"

RF_PATH = "rf_model.pkl"
LIN_PATH = "lin_model.pkl"
SCALER_PATH = "scaler.pkl"


# ===== INDICATORS =====

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Return"] = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df


# ===== LOAD DATA =====

df = pd.read_csv(CSV_PATH)
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date")

df = add_indicators(df)
df["Target"] = df["Open"].shift(-1)
df = df.dropna().reset_index(drop=True)

features = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_5", "SMA_20", "RSI", "MACD", "Return"
]

X = df[features]
y = df["Target"]

train_size = int(len(df) * 0.7)
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]
X_test = X.iloc[train_size:]
y_test = y.iloc[train_size:]

print(f"Loaded {len(df)} rows. Training split: {train_size} rows.")


# ===== SCALING =====

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ===== LINEAR REGRESSION =====

print("\nTraining Linear Regression...")
start = time.time()
lin = LinearRegression()
lin.fit(X_train_scaled, y_train)
lin_time = time.time() - start

lin_preds = lin.predict(X_test_scaled)
lin_mae = mean_absolute_error(y_test, lin_preds)
lin_rmse = np.sqrt(mean_squared_error(y_test, lin_preds))
lin_r2 = r2_score(y_test, lin_preds)


# ===== RANDOM FOREST =====

print("\nTraining Random Forest...")
start = time.time()
rf = RandomForestRegressor(
    n_estimators=120,
    max_depth=15,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train_scaled, y_train)
rf_time = time.time() - start

rf_preds = rf.predict(X_test_scaled)
rf_mae = mean_absolute_error(y_test, rf_preds)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_r2 = r2_score(y_test, rf_preds)


# ===== PRINT RESULTS =====

print("\n===== LINEAR REGRESSION RESULTS =====")
print(f"Training Time: {lin_time:.2f}s")
print(f"MAE:  {lin_mae:.4f}")
print(f"RMSE: {lin_rmse:.4f}")
print(f"R²:   {lin_r2:.4f}")

print("\n===== RANDOM FOREST RESULTS =====")
print(f"Training Time: {rf_time:.2f}s")
print(f"MAE:  {rf_mae:.4f}")
print(f"RMSE: {rf_rmse:.4f}")
print(f"R²:   {rf_r2:.4f}")

print("\nFeature Importances (RF):")
for f, imp in sorted(zip(features, rf.feature_importances_), key=lambda x: x[1], reverse=True):
    print(f"  {f}: {imp:.4f}")


# ===== GRAPH: ACTUAL vs PREDICTIONS =====

plt.figure(figsize=(13, 6))
plt.plot(y_test.values, label="Actual", linewidth=2)
plt.plot(lin_preds, label="Linear Regression")
plt.plot(rf_preds, label="Random Forest")
plt.title("Prediction Comparison")
plt.xlabel("Test Samples")
plt.ylabel("Next Open Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ===== TABLE: MODEL COMPARISON =====

comparison_table = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [lin_mae, rf_mae],
    "RMSE": [lin_rmse, rf_rmse],
    "R²": [lin_r2, rf_r2],
    "Train Time (s)": [lin_time, rf_time]
})

print("\n===== MODEL COMPARISON TABLE =====")
print(comparison_table.to_string(index=False))


# ===== SAVE MODELS =====

joblib.dump(lin, LIN_PATH)
joblib.dump(rf, RF_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"\nSaved LinearRegression → {LIN_PATH}")
print(f"Saved RandomForest → {RF_PATH}")
print(f"Saved scaler → {SCALER_PATH}")
print("\nTraining complete with graphs and table.")

