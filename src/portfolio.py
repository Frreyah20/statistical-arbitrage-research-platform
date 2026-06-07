import pandas as pd

from research_engine import backtest_pair
from cointegration import calculate_hedge_ratio
from cointegration import calculate_spread
from signals import calculate_rolling_zscore
from signal_generator import generate_positions
from performance import (
    calculate_daily_spread_returns,
    calculate_strategy_returns
)


def get_pair_returns(stock1, stock2, prices):
    beta = calculate_hedge_ratio(
        prices[stock1],
        prices[stock2]
    )

    spread = calculate_spread(
        prices[stock1],
        prices[stock2],
        beta
    )

    zscore = calculate_rolling_zscore(
        spread,
        window=60
    )

    zscore = zscore.dropna()

    positions = generate_positions(zscore)

    spread_returns = calculate_daily_spread_returns(
        spread
    )

    spread_returns = spread_returns.loc[
        positions.index
    ]

    strategy_returns = calculate_strategy_returns(
        spread_returns,
        positions
    )

    return strategy_returns

def build_portfolio_returns(
    ranked_pairs,
    prices,
    top_n=5
):
    portfolio = pd.DataFrame()

    top_pairs = ranked_pairs.head(top_n)

    for pair in top_pairs["Pair"]:

        stock1, stock2 = pair.split("-")

        returns = get_pair_returns(
            stock1,
            stock2,
            prices
        )

        portfolio[pair] = returns

    portfolio = portfolio.fillna(0)

    portfolio["Portfolio"] = (
        portfolio.mean(axis=1)
    )

    return portfolio["Portfolio"]