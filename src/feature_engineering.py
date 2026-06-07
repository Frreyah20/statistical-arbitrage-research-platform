from cointegration import test_cointegration, calculate_hedge_ratio, calculate_spread
from performance import calculate_half_life
import pandas as pd
from research_engine import backtest_pair
#Feature extractor
def build_pair_features(stock1, stock2, train_prices):
    series1 = train_prices[stock1]
    series2 = train_prices[stock2]

    correlation = series1.corr(series2)
    p_value = test_cointegration(series1, series2)
    beta = calculate_hedge_ratio(series1, series2)
    spread = calculate_spread(series1, series2, beta)
    half_life = calculate_half_life(spread)
    result = backtest_pair(stock1, stock2, train_prices)
    return {
        "Pair" : f"{stock1}-{stock2}",
        "Correlation": correlation,
        "Cointegeration PValue": p_value,
        "Beta": beta,
        "Half Life": half_life,
        "Train Sharpe" : result["Sharpe"],
        "Train Drawdown" : result["Max Drawdown"],
        "Train Trades" : result["Trades"]
    }

#Build Feature for every pair
def build_feature_dataset(pairs, train_prices):
    rows = []
    for stock1, stock2, _ in pairs:
        row = build_pair_features(stock1, stock2, train_prices)
        rows.append(row)
    return pd.DataFrame(rows)

def add_test_target(features_df, test_prices):
    test_sharpes = []
    for pair in features_df["Pair"]:
        stock1, stock2 = pair.split("-")
        result = backtest_pair(stock1, stock2, test_prices)
        test_sharpes.append(result["Sharpe"])
    features_df = features_df.copy()
    features_df["Test Sharpe"] = test_sharpes
    return features_df
    