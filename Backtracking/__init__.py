"""
Backtracking Package for AI Stock Analyst
Provides backtesting and model evaluation capabilities.
"""

from .backtest import (
    backtest_model,
    evaluate_prediction_accuracy,
    run_comprehensive_backtest,
    run_comprehensive_backtest_detailed,
    run_backtest,
    save_backtest_results,
    fetch_historical_data,
    fetch_backtest_data,
    parse_backtest_cli_args,
    engineer_features_uncached as engineer_features,
    train_linear_regression_uncached as train_linear_regression,
    train_random_forest_uncached as train_random_forest,
    train_lstm_uncached as train_lstm,
    ensemble as compute_ensemble,
)

__all__ = [
    "backtest_model",
    "evaluate_prediction_accuracy",
    "run_comprehensive_backtest",
    "run_comprehensive_backtest_detailed",
    "run_backtest",
    "save_backtest_results",
    "fetch_historical_data",
    "fetch_backtest_data",
    "parse_backtest_cli_args",
    "engineer_features",
    "train_linear_regression",
    "train_random_forest",
    "train_lstm",
    "compute_ensemble",
]
