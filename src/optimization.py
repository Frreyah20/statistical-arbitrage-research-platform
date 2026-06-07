from research_engine import backtest_pair
import pandas as pd

def optimize_parameters(
    stock1,
    stock2,
    prices
):

    results = []

    entry_values = [1.5, 2.0, 2.5, 3.0]
    exit_values = [0.25, 0.5, 0.75, 1.0]

    for entry in entry_values:

        for exit in exit_values:

            result = backtest_pair(
                stock1,
                stock2,
                prices,
                entry_threshold=entry,
                exit_threshold=exit
            )

            results.append({
                "Entry": entry,
                "Exit": exit,
                "Sharpe": result["Sharpe"],
                "PnL": result["Total PnL"],
                "Trades": result["Trades"]
            })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("Sharpe", ascending = False)

    return results_df