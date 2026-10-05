#!/usr/bin/env python3
"""
Run backtesting for the AI Stock Analyst models.
Usage:
    python run_backtest.py [ticker[,ticker,...]] [horizon_days] [n_windows]

Examples:
    python run_backtest.py AAPL 7 5
    python run_backtest.py AAPL,MSFT,TSLA 7 5
    python run_backtest.py "AAPL, MSFT, TSLA" 14 10
    python run_backtest.py "HDFCBANK.NS" "TCS.NS" 7 8
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Backtracking.backtest import (
    backtest_model,
    evaluate_prediction_accuracy,
    run_comprehensive_backtest,
    run_comprehensive_backtest_detailed,
    save_backtest_results,
    parse_backtest_cli_args,
)


def main():
    tickers, horizon, n_windows = parse_backtest_cli_args(sys.argv[1:])
    output_path = os.path.join(os.path.dirname(__file__), "backtest_results.csv")

    print(f"Running backtest for {', '.join(tickers)} | horizon={horizon} | windows={n_windows}")
    print("=" * 60)

    if len(tickers) == 1:
        # Single-ticker backward-compatible path
        ticker = tickers[0]

        # 1. Rolling window backtest + save detailed results
        print(f"\n[1/3] Rolling Window Backtest — {ticker}")
        results = backtest_model(ticker, horizon=horizon, n_windows=n_windows)
        save_backtest_results(results, output_path)

        # 2. Recent prediction accuracy
        print("\n[2/3] Recent Prediction Accuracy")
        accuracy = evaluate_prediction_accuracy(ticker, horizon=horizon)
        print(
            f"  Directional accuracy: {accuracy.get('directional_accuracy')}%  |  "
            f"Mean error: {accuracy.get('mean_error_pct')}%  |  "
            f"Mean abs error: ${accuracy.get('mean_abs_error')}"
        )

        # 3. Comprehensive backtest across multiple tickers/horizons for summary table
        print("\n[3/3] Comprehensive Backtest (multiple tickers)")
        default_tickers = ["AAPL", "MSFT", "TSLA", "NVDA", "GOOGL", "AMZN"]
        extra = [t for t in tickers if t not in default_tickers]
        comp_tickers = default_tickers + extra
        df_results = run_comprehensive_backtest(comp_tickers, horizons=[horizon], n_windows=n_windows)
    else:
        # Multi-ticker path: comprehensive backtest across selected tickers
        print("\n[1/1] Comprehensive backtest across selected tickers")
        detailed = run_comprehensive_backtest_detailed(tickers, horizons=[horizon], n_windows=n_windows)
        df_results = run_comprehensive_backtest(detailed_df=detailed)

        # Save detailed per-window results for all requested tickers
        if not detailed.empty:
            try:
                detailed.to_csv(output_path, index=False)
                print(f"\nDetailed results ({len(detailed)} rows) saved to: {output_path}")
            except PermissionError:
                from datetime import datetime
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                alt_path = output_path.replace(".csv", f"_{ts}.csv")
                detailed.to_csv(alt_path, index=False)
                print(f"\nWarning: {output_path} is locked. Detailed results saved to {alt_path}")

    if not df_results.empty:
        print("\n" + "=" * 60)
        print("SUMMARY TABLE")
        print("=" * 60)
        print(df_results.to_string(index=False))

    print("\nDone!")


if __name__ == "__main__":
    main()
