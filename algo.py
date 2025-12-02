"""
Enhanced predictor module

This module trains multiple linear models that use the last three days' open/close prices
and trend information to predict the next three days' close prices.

Features used:
- Last 3 days' open prices
- Last 3 days' close prices  
- Trend indicators (increase/decrease/same) for each day

Public API:
- `predict_next_three_closes(features: dict) -> dict`
- `recommend_buy_enhanced(features: dict) -> dict`

If run as a script you can input data interactively or provide via command line.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder


DEFAULT_CSV = r"C:\Users\noosa\NVidia_stock_history.csv"


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature matrix from historical data."""
    df = df.sort_values('Date') if 'Date' in df.columns else df
    
    # Create features for last 3 days
    for i in range(1, 4):  # Days -3, -2, -1
        df[f'Open_m{i}'] = df['Open'].shift(i)
        df[f'Close_m{i}'] = df['Close'].shift(i)
    
    # Create target variables for next 3 days
    df['Close_p1'] = df['Close'].shift(-1)
    df['Close_p2'] = df['Close'].shift(-2)
    df['Close_p3'] = df['Close'].shift(-3)
    
    # Add trend features (1: increase, 0: same, -1: decrease)
    for i in range(1, 4):
        df[f'Trend_m{i}'] = np.where(df[f'Close_m{i}'] > df[f'Open_m{i}'], 1,
                                    np.where(df[f'Close_m{i}'] < df[f'Open_m{i}'], -1, 0))
    
    # Drop rows with NaN values
    df_clean = df.dropna()
    
    return df_clean


def _train_models(csv_path: str | Path = DEFAULT_CSV) -> tuple[Optional[Dict[str, LinearRegression]], Optional[Dict[str, float]]]:
    """Train linear regression models for predicting next 3 days' close prices.
    
    Returns (models_dict, rmse_dict) where models_dict contains models for each day ahead.
    """
    try:
        df = pd.read_csv(str(csv_path))
    except Exception as e:
        print(f"Could not read CSV at {csv_path}: {e}")
        return None, None

    if 'Close' not in df.columns or 'Open' not in df.columns:
        print("CSV must contain 'Close' and 'Open' columns")
        return None, None

    # Prepare features
    df_features = _prepare_features(df)
    
    if len(df_features) < 10:
        print("Not enough data to train the predictor")
        return None, None
    
    # Define feature columns
    feature_cols = []
    for i in range(1, 4):  # Last 3 days
        feature_cols.extend([f'Open_m{i}', f'Close_m{i}', f'Trend_m{i}'])
    
    # Target columns
    target_cols = ['Close_p1', 'Close_p2', 'Close_p3']
    
    X = df_features[feature_cols].to_numpy()
    models = {}
    rmses = {}
    
    # Train separate model for each prediction horizon
    for idx, target_col in enumerate(target_cols):
        y = df_features[target_col].to_numpy()
        
        model = LinearRegression()
        model.fit(X, y)
        models[target_col] = model
        
        # Calculate RMSE for this model
        preds = model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        rmses[target_col] = rmse
    
    return models, rmses


_GLOBAL_MODELS: Optional[Dict[str, LinearRegression]] = None
_GLOBAL_RMSES: Optional[Dict[str, float]] = None


def _ensure_models():
    global _GLOBAL_MODELS, _GLOBAL_RMSES
    if _GLOBAL_MODELS is None:
        _GLOBAL_MODELS, _GLOBAL_RMSES = _train_models(DEFAULT_CSV)


def _prepare_input_features(opens: List[float], closes: List[float], trends: List[str]) -> Optional[np.ndarray]:
    """Prepare input feature vector from user inputs."""
    if len(opens) != 3 or len(closes) != 3 or len(trends) != 3:
        print("Error: Need exactly 3 values for opens, closes, and trends")
        return None
    
    # Convert trend strings to numeric
    trend_mapping = {'increase': 1, 'decrease': -1, 'same': 0}
    trends_numeric = []
    
    for trend in trends:
        trend_lower = trend.lower()
        if trend_lower in trend_mapping:
            trends_numeric.append(trend_mapping[trend_lower])
        else:
            print(f"Warning: Invalid trend '{trend}'. Using 'same' (0).")
            trends_numeric.append(0)
    
    # Prepare feature vector in correct order
    features = []
    for i in range(3):
        # Note: We're using day -3 as m1, day -2 as m2, day -1 as m3
        # Adjust indexing based on your preference
        features.extend([opens[i], closes[i], trends_numeric[i]])
    
    return np.array(features).reshape(1, -1)


def predict_next_three_closes(opens: List[float], closes: List[float], trends: List[str]) -> Optional[Dict[str, float]]:
    """Predict next 3 days' close prices given last 3 days' data.
    
    Args:
        opens: List of 3 open prices (most recent last)
        closes: List of 3 close prices (most recent last)
        trends: List of 3 trend indicators ('increase', 'decrease', or 'same')
    
    Returns:
        Dictionary with predicted close prices for days +1, +2, +3
    """
    _ensure_models()
    if _GLOBAL_MODELS is None:
        return None
    
    # Prepare input features
    X_input = _prepare_input_features(opens, closes, trends)
    if X_input is None:
        return None
    
    # Make predictions
    predictions = {}
    for target_col in ['Close_p1', 'Close_p2', 'Close_p3']:
        model = _GLOBAL_MODELS[target_col]
        pred = float(model.predict(X_input)[0])
        predictions[target_col] = pred
    
    return predictions


def explain_models() -> str:
    """Return explanation of all trained models."""
    _ensure_models()
    if _GLOBAL_MODELS is None:
        return "Models not available (CSV missing or insufficient data)."
    
    explanation = "Models for predicting next 3 days' close prices:\n"
    for target_col, model in _GLOBAL_MODELS.items():
        coef_str = ", ".join([f"{c:.6f}" for c in model.coef_])
        intercept = float(model.intercept_)
        rmse = _GLOBAL_RMSES[target_col] if _GLOBAL_RMSES else 0.0
        explanation += f"{target_col}: coefs=[{coef_str}], intercept={intercept:.6f}, RMSE={rmse:.4f}\n"
    
    explanation += "\nFeature order: [Open_m1, Close_m1, Trend_m1, Open_m2, Close_m2, Trend_m2, Open_m3, Close_m3, Trend_m3]"
    explanation += "\nTrend encoding: increase=1, decrease=-1, same=0"
    return explanation


def recommend_buy_enhanced(opens: List[float], closes: List[float], trends: List[str]) -> dict:
    """Return recommendation based on predicted next 3 days' close prices.
    
    The dict contains:
      - `recommend` : 'YES' or 'NO' or 'UNKNOWN'
      - `predictions`: dict of predicted close prices
      - `avg_predicted_change`: average predicted change over 3 days
      - `max_predicted_change`: maximum predicted change
      - `explanation`: human-readable explanation
    """
    _ensure_models()
    if _GLOBAL_MODELS is None:
        return {"recommend": "UNKNOWN", "predictions": None, "avg_predicted_change": None,
                "max_predicted_change": None, "explanation": "Model not available."}
    
    # Get predictions
    predictions = predict_next_three_closes(opens, closes, trends)
    if predictions is None:
        return {"recommend": "UNKNOWN", "predictions": None, "avg_predicted_change": None,
                "max_predicted_change": None, "explanation": "Could not prepare features."}
    
    last_close = closes[-1]  # Most recent close
    
    # Calculate changes
    changes = []
    change_details = []
    for i, (target_col, pred) in enumerate(predictions.items()):
        change = pred - last_close
        pct_change = (change / last_close) * 100 if last_close != 0 else float('inf')
        changes.append(change)
        change_details.append(f"Day+{i+1}: {pred:.4f} (Δ={change:+.4f}, {pct_change:+.2f}%)")
    
    avg_change = sum(changes) / len(changes)
    max_change = max(changes)
    
    # Simple decision rule: buy if average predicted change > 0
    if avg_change > 0:
        recommend = "YES"
        reason = f"Average predicted change over 3 days is positive (+{avg_change:.4f})"
    else:
        recommend = "NO"
        reason = f"Average predicted change over 3 days is negative ({avg_change:.4f})"
    
    # Add confidence based on RMSE
    avg_rmse = sum(_GLOBAL_RMSES.values()) / len(_GLOBAL_RMSES) if _GLOBAL_RMSES else 0.0
    confidence_note = ""
    if abs(avg_change) <= avg_rmse:
        confidence_note = f" Note: Average change ({avg_change:.4f}) is within average RMSE ({avg_rmse:.4f}), prediction uncertain."
    
    explanation = f"Recommendation: {recommend}\n"
    explanation += f"Last close: {last_close:.4f}\n"
    explanation += "Predicted closes:\n  " + "\n  ".join(change_details) + "\n"
    explanation += f"Average predicted change: {avg_change:.4f}\n"
    explanation += f"Maximum predicted change: {max_change:.4f}\n"
    explanation += reason + confidence_note
    
    return {
        "recommend": recommend,
        "predictions": predictions,
        "avg_predicted_change": avg_change,
        "max_predicted_change": max_change,
        "explanation": explanation
    }


def _cli(argv: list[str]) -> int:
    """Command line interface."""
    print("Enhanced Stock Predictor")
    print("=" * 50)
    
    # Check if we have command line arguments
    if len(argv) >= 7:
        # Try to parse command line arguments
        try:
            opens = [float(argv[i]) for i in range(1, 4)]
            closes = [float(argv[i]) for i in range(4, 7)]
            trends = argv[7:10]
        except (ValueError, IndexError):
            print("Command line usage: python script.py open1 open2 open3 close1 close2 close3 trend1 trend2 trend3")
            print("trend values: 'increase', 'decrease', or 'same'")
            return 2
    else:
        # Interactive input
        opens = []
        closes = []
        trends = []
        
        print("Enter last 3 days of data (most recent last):")
        for i in range(3):
            print(f"\nDay {i+1}:")
            try:
                open_val = float(input(f"  Open price: "))
                close_val = float(input(f"  Close price: "))
                trend_val = input(f"  Trend (increase/decrease/same): ").strip().lower()
                
                if trend_val not in ['increase', 'decrease', 'same']:
                    print(f"  Warning: Invalid trend '{trend_val}'. Using 'same'.")
                    trend_val = 'same'
                
                opens.append(open_val)
                closes.append(close_val)
                trends.append(trend_val)
            except ValueError:
                print("Invalid input — expected numbers for prices")
                return 2
    
    # Get predictions
    predictions = predict_next_three_closes(opens, closes, trends)
    
    if predictions is None:
        print("Prediction unavailable — could not train model from CSV or invalid input")
        return 1
    
    print("\n" + "=" * 50)
    print("PREDICTION RESULTS")
    print("=" * 50)
    
    last_close = closes[-1]
    print(f"\nBased on last close: {last_close:.4f}")
    print("\nPredicted close prices:")
    for i, (key, value) in enumerate(predictions.items()):
        change = value - last_close
        pct = (change / last_close) * 100
        print(f"  Day+{i+1}: {value:.4f} (Δ={change:+.4f}, {pct:+.2f}%)")
    
    print("\n" + explain_models())
    
    print("\n" + "=" * 50)
    print("RECOMMENDATION")
    print("=" * 50)
    
    rec = recommend_buy_enhanced(opens, closes, trends)
    print("\n" + rec['explanation'])
    
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))