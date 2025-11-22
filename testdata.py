
import pandas as pd
import numpy as np
import time
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

df = pd.read_csv(r"C:\Users\noosa\NVidia_stock_history.csv")

df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

def add_indicators(df):
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['Return'] = df['Close'].pct_change()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    return df

df = add_indicators(df)
df = df.dropna().reset_index(drop=True)

features = ['Open', 'High', 'Low', 'Close', 'Volume',
            'SMA_5', 'SMA_20', 'RSI', 'MACD', 'Return']

df['Target'] = df['Close'].shift(-1)
df = df.dropna().reset_index(drop=True)

X = df[features]
y = df['Target']

train_size = int(len(df) * 0.7)
step = 5

print(f"Dataset loaded: {len(df)} rows, {df.shape[1]} columns")
print("Features used:", features)
print(f"Train size: {train_size} rows ({round(train_size/len(df)*100,1)}%)")

rf = RandomForestRegressor(
    n_estimators=60,
    max_depth=12,
    min_samples_leaf=3,
    n_jobs=1,
    random_state=42
)

scaler = MinMaxScaler()
preds_rf = []
preds_lin = []
actuals_rf = []

X_train = X.iloc[:train_size]
y_train = y.iloc[:train_size]

print(f"Starting initial training on {X_train.shape[0]} samples...")
start_time = time.time()
X_scaled = scaler.fit_transform(X_train)
rf.fit(X_scaled, y_train)
rf_time = time.time() - start_time

start_time = time.time()
lin = LinearRegression()
lin.fit(X_scaled, y_train)
lin_time = time.time() - start_time

print(f"RandomForest initial training completed in {rf_time:.2f}s")
print(f"LinearRegression initial training completed in {lin_time:.2f}s")

last_log = 0
for t in range(train_size, len(df), step):
    X_test = X.iloc[t:t+step]
    X_test_scaled = scaler.transform(X_test)

    preds = rf.predict(X_test_scaled).tolist()
    preds_rf.extend(preds)

    preds_l = lin.predict(X_test_scaled).tolist()
    preds_lin.extend(preds_l)

    actuals_rf.extend(y.iloc[t:t+step].tolist())

    if ((t - train_size) // step) % 10 == 0:
        print(f"Processed up to row {min(t+step, len(df))} / {len(df)}")

mae_rf = mean_absolute_error(actuals_rf, preds_rf)
rmse_rf = np.sqrt(mean_squared_error(actuals_rf, preds_rf))
r2_rf = r2_score(actuals_rf, preds_rf)

mae_lin = mean_absolute_error(actuals_rf, preds_lin)
rmse_lin = np.sqrt(mean_squared_error(actuals_rf, preds_lin))
r2_lin = r2_score(actuals_rf, preds_lin)

print("\n===== RANDOM FOREST RESULTS =====")
print("MAE:", round(mae_rf, 4))
print("RMSE:", round(rmse_rf, 4))
print("R²:", round(r2_rf, 4))

print("\n===== LINEAR REGRESSION RESULTS =====")
print("MAE:", round(mae_lin, 4))
print("RMSE:", round(rmse_lin, 4))
print("R²:", round(r2_lin, 4))


print("\nSample predictions vs actuals (first 10) — RandomForest:")
for i, (p, a) in enumerate(zip(preds_rf[:10], actuals_rf[:10])):
    print(f"RF #{i+1}: pred={p:.2f}  actual={a:.2f}  diff={(p-a):.2f}")

print("\nSample predictions vs actuals (first 10) — LinearRegression:")
for i, (p, a) in enumerate(zip(preds_lin[:10], actuals_rf[:10])):
    print(f"LR #{i+1}: pred={p:.2f}  actual={a:.2f}  diff={(p-a):.2f}")

try:
    importances = rf.feature_importances_
    feat_imp = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
    print("\nFeature importances:")
    for f, imp in feat_imp:
        print(f"  {f}: {imp:.4f}")
except Exception:
    print("Could not retrieve feature importances from the model.")

try:
    import pandas as _pd
    rows = [
        {"model": "RandomForest", "MAE": round(mae_rf, 4), "RMSE": round(rmse_rf, 4), "R2": round(r2_rf, 4), "train_time_s": round(rf_time, 3)},
        {"model": "LinearRegression", "MAE": round(mae_lin, 4), "RMSE": round(rmse_lin, 4), "R2": round(r2_lin, 4), "train_time_s": round(lin_time, 3)},
    ]
    comp = _pd.DataFrame(rows)
    print("\n===== MODELS COMPARISON =====")
    print(comp.to_string(index=False))
except Exception:
    print("Could not build comparison table (pandas missing?).")