import pandas as pd
from cointegration import find_cointegrated_pairs
from research_engine import backtest_pair
from ranking import rank_pairs
from research_engine import backtest_all_pairs

def rolling_walkforward(
    prices,
    train_window=252,
    test_window=126
):

    results = []

    start = 0

    while (
        start + train_window + test_window
        <= len(prices)
    ):

        train = prices.iloc[
            start :
            start + train_window
        ]

        test = prices.iloc[
            start + train_window :
            start + train_window + test_window
        ]

        pairs = find_cointegrated_pairs(
            train
        )

        if len(pairs) == 0:
            start += test_window
            continue

        train_results = backtest_all_pairs(
            pairs,
            train
        )

        ranked = rank_pairs(
            train_results
        )

        best_pair = ranked.iloc[0]["Pair"]

        stock1, stock2 = best_pair.split("-")

        result = backtest_pair(
            stock1,
            stock2,
            test
        )
        print(
            f"{stock1}-{stock2}",
            "Trades:",
            result["Trades"]
        )

        result["Train End"] = train.index[-1]
        result["Test End"] = test.index[-1]

        results.append(result)

        start += test_window

    return pd.DataFrame(results)
