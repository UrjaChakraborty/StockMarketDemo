"""
Simple predictor module

This module trains a tiny linear model that maps yesterday's `Close` price
to the next day's `Open` price using the same CSV dataset used by
`testdata.py` (default path: C:/Users/noosa/NVidia_stock_history.csv).

Public API:
- `predict_next_open(yesterday_close: float) -> float`

If run as a script you can pass a numeric value on the command line or
the script will prompt for one interactively.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


DEFAULT_CSV = r"C:\Users\noosa\NVidia_stock_history.csv"


def _train_model(csv_path: str | Path = DEFAULT_CSV) -> tuple[Optional[LinearRegression], Optional[float]]:
    """Train a linear regression mapping Close(t) -> Open(t+1).

    Returns (model, rmse) where model is a fitted LinearRegression or None on error.
    """
    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        print(f"Could not read CSV at {csv_path}: {e}")
        return None, None

    if 'Close' not in df.columns or 'Open' not in df.columns:
        print("CSV must contain 'Close' and 'Open' columns")
        return None, None

    df = df.sort_values('Date') if 'Date' in df.columns else df

    df['Next_Open'] = df['Open'].shift(-1)
    data = df[['Close', 'Next_Open']].dropna()
    if len(data) < 10:
        print("Not enough data to train the predictor")
        return None, None

    X = data['Close'].to_numpy().reshape(-1, 1)
    y = data['Next_Open'].to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    preds = model.predict(X)
    rmse = float(np.sqrt(mean_squared_error(y, preds)))
    return model, rmse


_GLOBAL_MODEL: Optional[LinearRegression] = None
_GLOBAL_RMSE: Optional[float] = None


def _ensure_model():
    global _GLOBAL_MODEL, _GLOBAL_RMSE
    if _GLOBAL_MODEL is None:
        _GLOBAL_MODEL, _GLOBAL_RMSE = _train_model(DEFAULT_CSV)


def predict_next_open(yesterday_close: float) -> Optional[float]:
    """Predict tomorrow's open price given yesterday's close.

    Returns predicted open as float, or None if model couldn't be trained.
    """
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return None

    x = np.array([[float(yesterday_close)]])
    pred = float(_GLOBAL_MODEL.predict(x)[0])
    return pred


def explain_model() -> str:
    """Return a short explanation of the trained linear model (coef, intercept, rmse).

    If model not available returns an explanation string describing the problem.
    """
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return "Model not available (CSV missing or insufficient data)."

    coef = float(_GLOBAL_MODEL.coef_[0])
    intercept = float(_GLOBAL_MODEL.intercept_)
    rmse = _GLOBAL_RMSE
    return f"next_open = {coef:.6f} * yesterday_close + {intercept:.6f}  (rmse={rmse:.4f})"


def recommend_buy(yesterday_close: float) -> dict:
    """Return a recommendation dict based on predicted next open vs yesterday_close.

    The dict contains:
      - `recommend` : 'YES' or 'NO' or 'UNKNOWN'
      - `predicted_open`: float or None
      - `delta`: predicted_open - yesterday_close
      - `pct_change`: percent change
      - `explanation`: human-readable explanation including RMSE for confidence.
    """
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return {"recommend": "UNKNOWN", "predicted_open": None, "delta": None, "pct_change": None,
                "explanation": "Model not available (CSV missing or insufficient data)."}

    pred = predict_next_open(yesterday_close)
    rmse = _GLOBAL_RMSE if _GLOBAL_RMSE is not None else 0.0
    delta = pred - float(yesterday_close)
    pct = (delta / float(yesterday_close)) * 100 if yesterday_close != 0 else float('inf')

    # Simple decision rule: buy if predicted open > yesterday close
    if pred > float(yesterday_close):
        recommend = "YES"
        explanation = (f"Predicted next open {pred:.4f} is higher than yesterday's close {yesterday_close:.4f}. "
                       f"Expected gain {delta:.4f} ({pct:.2f}%).")
    else:
        recommend = "NO"
        explanation = (f"Predicted next open {pred:.4f} is not higher than yesterday's close {yesterday_close:.4f}. "
                       f"Expected change {delta:.4f} ({pct:.2f}%).")

    # Add a confidence note when the expected change is within the model RMSE
    if abs(delta) <= rmse:
        explanation += f" Note: change ({delta:.4f}) is within model RMSE (~{rmse:.4f}), so prediction is uncertain."

    explanation += f" Model RMSE={rmse:.4f}."

    return {"recommend": recommend, "predicted_open": pred, "delta": delta, "pct_change": pct, "explanation": explanation}


def _cli(argv: list[str]) -> int:
    if len(argv) >= 2:
        try:
            y_close = float(argv[1])
        except ValueError:
            print("Provide a numeric yesterday close, e.g. 482.25")
            return 2
    else:
        try:
            v = input("Enter yesterday's Close price: ")
            y_close = float(v)
        except Exception:
            print("Invalid input — expected a number")
            return 2

    pred = predict_next_open(y_close)
    if pred is None:
        print("Prediction unavailable — could not train model from CSV")
        return 1

    print(f"Predicted next Open: {pred:.4f}")
    print(explain_model())

    rec = recommend_buy(y_close)
    if rec["recommend"] == "UNKNOWN":
        print("Recommendation: UNKNOWN - could not compute recommendation")
    else:
        print(f"Recommendation: {rec['recommend']}")
        print(rec['explanation'])
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
