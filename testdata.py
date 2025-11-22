
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv(r"C:\Users\noosa\NVidia_stock_history.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# ---------------------------
# 2. Add Technical Indicators (computed once)
# ---------------------------
def add_indicators(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Return'] = df['Close'].pct_change()

    # RSI 14
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    return df

df = add_indicators(df)
df = df.dropna().reset_index(drop=True)

# ---------------------------
# 3. Feature Setup
# ---------------------------
features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Return']

df['Target'] = df['Close'].shift(-1)
df = df.dropna().reset_index(drop=True)

X = df[features]
y = df['Target']

# ---------------------------
# Walk-Forward Parameters
# ---------------------------
train_size = int(len(df) * 0.7)
step = 5

# Make Random Forest the primary model and add informative prints
print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns")
print("Features used:", features)
print(f"Train size: {train_size} rows ({round(train_size/len(df)*100,1)}%)")

rf = RandomForestRegressor(
    n_estimators=60,
    max_depth=12,
    min_samples_leaf=3,
    n_jobs=-1,
    random_state=42
)

scaler = MinMaxScaler()
preds_rf = []
actuals_rf = []

# Initial train
X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

print(f"Starting initial training on {X_train.shape[0]} samples...")
start_time = time.time()
X_scaled = scaler.fit_transform(X_train)
rf.fit(X_scaled, y_train)
elapsed = time.time() - start_time
print(f"Initial training completed in {elapsed:.2f}s")

# Walk-forward prediction (no repeated re-fit to keep runtime reasonable)
last_log = 0
for t in range(train_size, len(df), step):
    X_test = X.iloc[t:t+step]
    X_test_scaled = scaler.transform(X_test)

    preds = rf.predict(X_test_scaled).tolist()
    preds_rf.extend(preds)
    actuals_rf.extend(y.iloc[t:t+step].tolist())

    # occasional progress print
    if ((t - train_size) // step) % 10 == 0:
        print(f"Processed up to row {min(t+step, len(df))} / {len(df)}")

# Eval Random Forest
mae_rf = mean_absolute_error(actuals_rf, preds_rf)
rmse_rf = np.sqrt(mean_squared_error(actuals_rf, preds_rf))
r2_rf = r2_score(actuals_rf, preds_rf)

print("\n===== RANDOM FOREST RESULTS =====")
print("MAE:", round(mae_rf, 4))
print("RMSE:", round(rmse_rf, 4))
print("R²:", round(r2_rf, 4))

# Show a small sample of predictions vs actuals
print("\nSample predictions vs actuals (first 10):")
for i, (p, a) in enumerate(zip(preds_rf[:10], actuals_rf[:10])):
    print(f"#{i+1}: pred={p:.2f}  actual={a:.2f}  diff={(p-a):.2f}")

# Feature importances
try:
    importances = rf.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature importances:")
    for f, imp in feat_imp:
        print(f"  {f}: {imp:.4f}")
except Exception:
    print("Could not retrieve feature importances from the model.")