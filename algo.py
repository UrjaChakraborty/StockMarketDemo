"""
new test adds rec feature and if yesterday and weekly trend inputs
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

# ----------- Helper: Encode trend strings into numeric dummies -----------

def _encode_trend(trend: str) -> tuple[int, int, int]:
    """
    Returns (is_higher, is_lower, is_same) one-hot encoded.
    """
    t = trend.lower().strip()
    if t == "higher":
        return (1, 0, 0)
    elif t == "lower":
        return (0, 1, 0)
    else:
        return (0, 0, 1)


# ------------------------- Training ---------------------------------------

def _train_model(csv_path: str | Path = DEFAULT_CSV) -> tuple[Optional[LinearRegression], Optional[float]]:
    """Train a linear regression mapping:
       (Close(t), yesterday_trend, weekly_trend) → Open(t+1)
    """
    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        print(f"Could not read CSV at {csv_path}: {e}")
        return None, None

    if 'Close' not in df.columns or 'Open' not in df.columns:
        print("CSV must contain 'Close' and 'Open' columns")
        return None, None

    if 'Date' in df.columns:
        df = df.sort_values('Date')

    # Next day's open (target)
    df['Next_Open'] = df['Open'].shift(-1)

    # Yesterday trend vs previous day's close
    df['Prev_Close'] = df['Close'].shift(1)
    df['Yesterday_Trend'] = np.where(df['Close'] > df['Prev_Close'], "higher",
                            np.where(df['Close'] < df['Prev_Close'], "lower", "same"))

    # Weekly trend (compare to close 7 days ago)
    df['Prev_Week_Close'] = df['Close'].shift(7)
    df['Weekly_Trend'] = np.where(df['Close'] > df['Prev_Week_Close'], "higher",
                          np.where(df['Close'] < df['Prev_Week_Close'], "lower", "same"))

    df = df.dropna(subset=['Next_Open'])

    if len(df) < 10:
        print("Not enough data to train the predictor")
        return None, None

    # Encode trend dummies
    yt_encoded = np.array([_encode_trend(t) for t in df['Yesterday_Trend']])
    wt_encoded = np.array([_encode_trend(t) for t in df['Weekly_Trend']])

    X = np.column_stack([
        df['Close'].to_numpy(),
        yt_encoded,
        wt_encoded
    ])

    y = df['Next_Open'].to_numpy()

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


# ------------------------- Public Prediction API --------------------------

def predict_next_open(yesterday_close: float,
                      yesterday_trend: str,
                      weekly_trend: str) -> Optional[float]:
    """
    Predict tomorrow's open using close + trends.
    """
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return None

    yt = np.array(_encode_trend(yesterday_trend))
    wt = np.array(_encode_trend(weekly_trend))

    x = np.concatenate([[float(yesterday_close)], yt, wt]).reshape(1, -1)
    pred = float(_GLOBAL_MODEL.predict(x)[0])
    return pred


def explain_model() -> str:
    """Explain model coefficients."""
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return "Model not available (CSV missing or insufficient data)."

    coef = _GLOBAL_MODEL.coef_
    intercept = float(_GLOBAL_MODEL.intercept_)
    rmse = _GLOBAL_RMSE

    labels = [
        "close_coef",
        "yesterday_higher", "yesterday_lower", "yesterday_same",
        "weekly_higher", "weekly_lower", "weekly_same"
    ]

    lines = [f"{lab}: {float(c):.6f}" for lab, c in zip(labels, coef)]
    lines.append(f"intercept: {intercept:.6f}")
    lines.append(f"RMSE: {rmse:.4f}")

    return "\n".join(lines)


def recommend_buy(yesterday_close: float,
                  yesterday_trend: str,
                  weekly_trend: str) -> dict:
    """Decision rule using trend-aware prediction."""
    _ensure_model()
    if _GLOBAL_MODEL is None:
        return {"recommend": "UNKNOWN", "predicted_open": None, "delta": None,
                "pct_change": None, "explanation": "Model not available."}

    pred = predict_next_open(yesterday_close, yesterday_trend, weekly_trend)
    rmse = _GLOBAL_RMSE or 0.0

    delta = pred - float(yesterday_close)
    pct = (delta / float(yesterday_close)) * 100 if yesterday_close != 0 else float('inf')

    if pred > yesterday_close:
        recommend = "YES"
        explanation = (
            f"Predicted next open {pred:.4f} is higher than yesterday's close {yesterday_close:.4f}. "
            f"(yesterday trend={yesterday_trend}, weekly trend={weekly_trend}) "
            f"Expected change {delta:.4f} ({pct:.2f}%)."
        )
    else:
        recommend = "NO"
        explanation = (
            f"Predicted next open {pred:.4f} is not higher than yesterday's close {yesterday_close:.4f}. "
            f"(yesterday trend={yesterday_trend}, weekly trend={weekly_trend}) "
            f"Expected change {delta:.4f} ({pct:.2f}%)."
        )

    if abs(delta) <= rmse:
        explanation += f" Note: change ({delta:.4f}) is within RMSE (~{rmse:.4f}), uncertainty is high."

    explanation += f" RMSE={rmse:.4f}."

    return {"recommend": recommend, "predicted_open": pred,
            "delta": delta, "pct_change": pct, "explanation": explanation}


# ----------------------------- CLI Option ---------------------------------

def _cli(argv: list[str]) -> int:
    try:
        y_close = float(input("Enter yesterday's Close price: "))
        yesterday_trend = input("Was yesterday higher, lower, or same? ").strip()
        weekly_trend = input("Was this week higher, lower, or same? ").strip()
    except Exception:
        print("Invalid input.")
        return 2

    pred = predict_next_open(y_close, yesterday_trend, weekly_trend)
    if pred is None:
        print("Prediction unavailable — could not train model.")
        return 1

    print(f"\nPredicted next Open: {pred:.4f}")
    print("\nModel details:")
    print(explain_model())

    rec = recommend_buy(y_close, yesterday_trend, weekly_trend)
    print(f"\nRecommendation: {rec['recommend']}")
    print(rec['explanation'])
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
