"""
Prediction system using technical-indicator model.

Loads:
    lin_model.pkl
    rf_model.pkl
    scaler.pkl

User calls:
    predict_next_open(..., model="linear")
    predict_next_open(..., model="rf")

Both models use the same feature pipeline.
    recommend_buy(yesterday_close, model="rf")
    evaluate_models(history_df)
    etc etc
    this is update [TEN] done by me :)
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error


RF_PATH = "rf_model.pkl"
LIN_PATH = "lin_model.pkl"
SCALER_PATH = "scaler.pkl"


# loading models + scaler
try:
    lin_model = joblib.load(LIN_PATH)
    rf_model = joblib.load(RF_PATH)
    scaler = joblib.load(SCALER_PATH)
    MODELS_READY = True
except Exception:
    lin_model = None
    rf_model = None
    scaler = None
    MODELS_READY = False


# compute indicators
def compute_indicators(df):
    df = df.copy()

    df["SMA_5"] = df["Close"].rolling(5).mean()
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["Return"] = df["Close"].pct_change()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    rs = gain.rolling(14).mean() / loss.rolling(14).mean().replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26

    return df


FEATURES = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_5", "SMA_20", "RSI", "MACD", "Return"
]


# feature prepping
def prepare_input(open_, high, low, close, volume, history_df):
    """
    history_df: must contain at least 20–26+ previous rows.
    Returns a DataFrame (1 row) with exactly the FEATURES columns.
    """
    df = history_df.copy()

    df.loc[len(df)] = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume
    }

    df = compute_indicators(df).dropna()
    row = df.iloc[-1]

    # return as a single-row DataFrame with the right columns (keeps feature names)
    return row[FEATURES].to_frame().T



# prediction api
def predict_next_open(open_, high, low, close, volume, history_df, model="rf"):
    if not MODELS_READY:
        return None

    X_df = prepare_input(open_, high, low, close, volume, history_df)  # DataFrame (1 row)
    X_scaled = scaler.transform(X_df)  # scaler expects feature names; passing DF avoids the warning

    if model == "linear":
        return float(lin_model.predict(X_scaled)[0])
    else:
        return float(rf_model.predict(X_scaled)[0])



def recommend(open_, high, low, close, volume, history_df, model="rf"):
    pred = predict_next_open(open_, high, low, close, volume, history_df, model)
    if pred is None:
        return {"recommend": "UNKNOWN", "reason": "Model not loaded"}

    delta = pred - close
    pct = (delta / close) * 100 if close != 0 else 0

    rec = "BUY" if pred > close else "NO BUY"

    return {
        "model_used": model,
        "recommend": rec,
        "predicted_next_open": pred,
        "delta": delta,
        "pct_change": pct,
        "explanation": f"{model.upper()} predicts {pred:.2f} vs close {close:.2f} ({pct:.2f}%)."
    }


# evaluation function
def evaluate_models(history_df):
    """
    Computes model R2 and RMSE based on historical data.
    Prediction target = next day's Open.
    """
    if not MODELS_READY:
        return None

    df = compute_indicators(history_df.copy()).dropna()

    # align features X(t) → next-open(t+1)
    df["Next_Open"] = df["Open"].shift(-1)
    df = df.dropna()

    X_df = df[FEATURES]        # DataFrame, keeps column names
    y = df["Next_Open"].to_numpy()

    X_scaled = scaler.transform(X_df)  # pass DataFrame to avoid feature-name warning

    # predictions
    pred_lin = lin_model.predict(X_scaled)
    pred_rf = rf_model.predict(X_scaled)
    pred_ens = (pred_lin + pred_rf) / 2

    # compute RMSE via sqrt of MSE (compatible with older sklearn versions)
    rmse_lin = float(np.sqrt(mean_squared_error(y, pred_lin)))
    rmse_rf = float(np.sqrt(mean_squared_error(y, pred_rf)))
    rmse_ens = float(np.sqrt(mean_squared_error(y, pred_ens)))

    return {
        "R2_linear": float(r2_score(y, pred_lin)),
        "R2_rf": float(r2_score(y, pred_rf)),
        "RMSE_linear": rmse_lin,
        "RMSE_rf": rmse_rf,
        "RMSE_ensemble": rmse_ens,
    }


# main program
if __name__ == "__main__":
    if not MODELS_READY:
        print("Models are not loaded. Ensure rf_model.pkl, lin_model.pkl, scaler.pkl exist.")
        exit()

    print("=== Stock Prediction System ===")
    print("Enter today's OHLCV data:")

    open_ = float(input("Open: "))
    high = float(input("High: "))
    low = float(input("Low: "))
    close = float(input("Close: "))
    volume = float(input("Volume: "))

    hist_path = input("Path to historical CSV (same one used for training): ")

    try:
        history_df = pd.read_csv(hist_path)
    except Exception as e:
        print("Could not load history file:", e)
        exit()

    # Prepare history (indicators)
    history_df = compute_indicators(history_df).dropna()

    # === Predictions ===
    pred_lin = predict_next_open(open_, high, low, close, volume, history_df, model="linear")
    pred_rf = predict_next_open(open_, high, low, close, volume, history_df, model="rf")
    ensemble_pred = (pred_lin + pred_rf) / 2

    print("\n=== Predictions ===")
    print(f"Linear Regression: {pred_lin:.4f}")
    print(f"Random Forest:     {pred_rf:.4f}")
    print(f"Ensemble (avg):    {ensemble_pred:.4f}")

    # === Model Evaluation ===
    print("\n=== Model Accuracy (Historical) ===")
    metrics = evaluate_models(pd.read_csv(hist_path))

    if metrics:
        rmse_lin = metrics["RMSE_linear"]
        rmse_rf = metrics["RMSE_rf"]

        print(f"Linear R²:        {metrics['R2_linear']:.4f}")
        print(f"RandomForest R²:  {metrics['R2_rf']:.4f}")
        print(f"Linear RMSE:      {rmse_lin:.4f}")
        print(f"RF RMSE:          {rmse_rf:.4f}")
        print(f"Ensemble RMSE:    {metrics['RMSE_ensemble']:.4f}")
    else:
        print("Could not compute metrics.")
        exit()

    # === Auto-select best model by RMSE ===
    print("\n=== Best Model Selection ===")
    best_model = "linear" if rmse_lin < rmse_rf else "rf"
    print(f"→ Best model based on RMSE: {best_model.upper()}")

    # === Final Recommendation ===
    print("\n=== Final Recommendation (Best Model) ===")
    rec = recommend(open_, high, low, close, volume, history_df, model=best_model)

    print(f"Model Used:       {rec['model_used']}")
    print(f"Recommendation:   {rec['recommend']}")
    print(f"Prediction:       {rec['predicted_next_open']:.4f}")
    print(f"Delta:            {rec['delta']:.4f}")
    print(f"Pct Change:       {rec['pct_change']:.4f}%")
    print(f"Explanation:      {rec['explanation']}")


# C:\Users\noosa\NVidia_stock_history.csv