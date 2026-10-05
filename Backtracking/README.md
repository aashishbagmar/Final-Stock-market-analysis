# Backtracking Module

This module provides backtesting capabilities to evaluate the accuracy of the AI Stock Analyst's prediction models.

## Files

- `backtest.py` - Core backtesting logic with rolling window evaluation (can also be run directly as a CLI)
- `run_backtest.py` - Command-line script to run backtests (single or multiple tickers)
- `__init__.py` - Package exports

## Usage

### Run from command line

Both `backtest.py` and `run_backtest.py` accept the same CLI arguments.

```bash
# Default: AAPL with 7-day horizon
python Backtracking/backtest.py
python Backtracking/run_backtest.py

# Custom ticker and horizon
python Backtracking/backtest.py AAPL 14
python Backtracking/run_backtest.py TSLA 30
python Backtracking/backtest.py "TCS.NS" 7

# Multiple tickers (comma-separated)
python Backtracking/backtest.py AAPL,MSFT,TSLA 7
python Backtracking/run_backtest.py AAPL,MSFT,TSLA 7

# Multiple tickers with explicit horizon and rolling windows
python Backtracking/backtest.py HDFCBANK.NS,TCS.NS 7 8
python Backtracking/run_backtest.py HDFCBANK.NS,TCS.NS 7 8

# Multiple quoted tickers also work
python Backtracking/backtest.py "HDFCBANK.NS" "TCS.NS" 7 8
```

### Use in Python

```python
from Backtracking.backtest import backtest_model, evaluate_prediction_accuracy, run_comprehensive_backtest

# Rolling window backtest (multiple train/test splits)
results = backtest_model("AAPL", horizon=7, n_windows=10)

# Evaluate recent prediction accuracy
accuracy = evaluate_prediction_accuracy("AAPL", horizon=7)

# Comprehensive backtest across multiple tickers/horizons
df = run_comprehensive_backtest(["AAPL", "MSFT", "TSLA"], horizons=[7, 14, 30], n_windows=5)
print(df)
```

## Backtest Methodology

### Rolling Window Backtest
- Splits historical data into multiple train/test windows
- Trains models on each training window
- Tests on subsequent test window
- Computes aggregate metrics across all windows

### Metrics Computed
- **Mean Absolute Error (MAE)** - Average prediction error in dollars
- **Median Absolute Error** - Median prediction error (robust to outliers)
- **Mean Error %** - Average percentage error
- **Directional Accuracy** - Percentage of correct direction predictions (up/down)
- **Min/Max Error** - Best and worst case errors

### Models Evaluated
1. **Linear Regression** - Baseline linear model
2. **Random Forest** - Ensemble tree-based model
3. **LSTM** - Deep learning sequence model
4. **Ensemble** - MAE-weighted combination of all three

## Output

The script saves results to `Backtracking/backtest_results.csv` with columns:
- `ticker` - Stock symbol
- `horizon` - Forecast horizon in days
- `n_windows` - Number of test windows
- `mean_absolute_error` - Mean $ error
- `median_absolute_error` - Median $ error
- `mean_error_pct` - Mean % error
- `directional_accuracy` - % correct direction
- `max_error` - Maximum $ error
- `min_error` - Minimum $ error

## Requirements

- yfinance
- scikit-learn
- pandas
- numpy
- tensorflow (for LSTM)
- requests

Install with:
```bash
pip install yfinance scikit-learn pandas numpy tensorflow requests
```
