# Stock Market Demo — Linear Regression

## Overview
A small demo project that predicts next-day stock closing prices using a linear regression model. Designed to illustrate data preparation, feature engineering, training, evaluation, and plotting for educational purposes.

## Features
- Load historical OHLCV CSV data
- Create lag features and simple technical indicators (moving averages)
- Train a scikit-learn LinearRegression model
- Evaluate with MSE and R²
- Visualize predictions vs. actuals

## Dataset (expected)
CSV with columns:
- Date (YYYY-MM-DD)
- Open, High, Low, Close, Volume

Example row:
`2023-01-03,100.5,102.0,99.8,101.2,1500000`

## Model & Features
- Target: next-day Close
- Example features:
    - Close_t (today's close)
    - Close_t-1, Close_t-2 (lags)
    - MA_5, MA_10 (moving averages)
- Model: sklearn.linear_model.LinearRegression
- Train/test split: time-series split (e.g., last 20% as test)

## Prerequisites
- Python 3.8+
- Install:
```
pip install pandas numpy scikit-learn matplotlib
```

## Usage
1. Place dataset at `data/stock.csv`.
2. Run training script:
```
python train.py --data data/stock.csv --output model.pkl
```
3. Run evaluation / plot:
```
python evaluate.py --model model.pkl --data data/stock.csv
```

## Evaluation
- Report Mean Squared Error (MSE) and R² on the test set.
- Plot predictions vs actual closing prices for visual inspection.

## Extending the demo
- Add ridge/lasso regression or time-series models (ARIMA, Prophet)
- Add more features: volume changes, RSI, MACD
- Use cross-validation with rolling windows
