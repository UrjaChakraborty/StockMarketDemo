
# SCRAPPED
# DONT DELETE YET THO IN CASE WE NEED IT LATER

'''
"""
Random Forest Stock Predictor Module

This module trains a RandomForestRegressor to map the previous day's
features (Open, High, Low, Close, Volume) to the next day's Open price.

Public API:
- predict_next_open(yesterday_close: float) -> float
- explain_model() -> str
- recommend_buy(yesterday_close: float) -> dict
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


DEFAULT_CSV = r"C:\Users\noosa\NVidia_stock_history.csv"


# ---------------------------------------------------------
#  TRAINING FUNCTION
# ---------------------------------------------------------

def _train_model(csv_path: str | Path = DEFAULT_CSV):

    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        print(f"Could not read CSV at {csv_path}: {e}")
        return None, None, None

    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    if any(col not in df.columns for col in required):
        print("CSV must contain OHLCV columns")
        return None, None, None

    # Ensure chronological order
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')

    # Shift target (predict next day Open)
    df['Next_Open'] = df['Open'].shift(-1)
    df = df.dropna()

    feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    X = df[feature_cols].to_numpy()
    y = df['Next_Open'].to_numpy()

    if len(df) < 50:
        print("Not enough data to train.")
        return None, None, None

    # ----- RANDOM FOREST MODEL -----
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)
    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))

    return model, rmse, feature_cols


# ---------------------------------------------------------
#  GLOBALS (lazy-load model)
# ---------------------------------------------------------

_GLOBAL_MODEL = None
_GLOBAL_RMSE = None
_GLOBAL_FEATURES = None


def _ensure_model():
    global _GLOBAL_MODEL, _GLOBAL_RMSE, _GLOBAL_FEATURES
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL, _GLOBAL_RMSE, _GLOBAL_FEATURES = _train_model(DEFAULT_CSV)


# ---------------------------------------------------------
#  PREDICTOR API
# ---------------------------------------------------------

def predict_next_open(yesterday_close: float) -> Optional[float]:
    """Predict next Open using Random Forest model."""
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return None

    # For prediction we need ALL features.  
    # We approximate using yesterday_close for all OHLC inputs.
    # (This keeps API simple but is still better than your old version.)
    x = np.array([[yesterday_close, yesterday_close, yesterday_close,
                   yesterday_close, 0]])  # volume unknown → assume 0

    pred = float(_GLOBAL_MODEL.predict(x)[0])
    return pred


def explain_model() -> str:
    """Return human-readable model summary."""
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return "Model not available."

    rmse = _GLOBAL_RMSE
    feat_importance = _GLOBAL_MODEL.feature_importances_

    fi_text = ", ".join(
        f"{name}: {imp:.3f}" for name, imp in zip(_GLOBAL_FEATURES, feat_importance)
    )

    return (
        f"RandomForestRegressor model\n"
        f"Features: {', '.join(_GLOBAL_FEATURES)}\n"
        f"Feature Importances: {fi_text}\n"
        f"RMSE={rmse:.4f}"
    )


def recommend_buy(yesterday_close: float):
    """Return dictionary with YES/NO recommendation."""
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return {"recommend": "UNKNOWN", "explanation": "Model unavailable"}

    pred = predict_next_open(yesterday_close)
    rmse = _GLOBAL_RMSE

    delta = pred - yesterday_close
    pct = (delta / yesterday_close) * 100 if yesterday_close > 0 else 0

    if pred > yesterday_close:
        decision = "YES"
        explanation = (
            f"Predicted open {pred:.4f} > yesterday close {yesterday_close:.4f}. "
            f"Expected gain {delta:.4f} ({pct:.2f}%)."
        )
    else:
        decision = "NO"
        explanation = (
            f"Predicted open {pred:.4f} is not higher than yesterday close {yesterday_close:.4f}. "
            f"Expected change {delta:.4f} ({pct:.2f}%)."
        )

    if abs(delta) <= rmse:
        explanation += f" Change ({delta:.4f}) is within RMSE ({rmse:.4f}) → Low confidence."

    explanation += f" Model RMSE={rmse:.4f}."

    return {
        "recommend": decision,
        "predicted_open": pred,
        "delta": delta,
        "pct_change": pct,
        "explanation": explanation,
    }


# ---------------------------------------------------------
#  CLI
# ---------------------------------------------------------

def _cli(argv):
    if len(argv) >= 2:
        try:
            y = float(argv[1])
        except ValueError:
            print("Usage: python algo_rf.py 482.25")
            return 2
    else:
        y = float(input("Enter yesterday's close: "))

    pred = predict_next_open(y)
    print(f"Predicted next Open: {pred:.4f}")
    print(explain_model())

    rec = recommend_buy(y)
    print(f"Recommendation: {rec['recommend']}")
    print(rec["explanation"])
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))

    '''

# SCRAPPED
# DONT DELETE YET THO IN CASE WE NEED IT LATER