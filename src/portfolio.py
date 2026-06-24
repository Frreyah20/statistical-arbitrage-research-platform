from research_engine import backtest_pair
from cointegration import calculate_hedge_ratio
from cointegration import calculate_spread
from signals import calculate_rolling_zscore
from signal_generator import generate_positions
from performance import calculate_daily_spread_returns,calculate_strategy_returns
import config
from transaction_costs import TransactionCostModel
import pandas as pd
import numpy as np

def calculate_pair_volatility(strategy_returns):
    return strategy_returns.std()

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

    cost_model = TransactionCostModel(
        commission_bps=config.COMMISSION_BPS,
        spread_bps=config.SPREAD_BPS,
        slippage_bps=config.SLIPPAGE_BPS
    )

    _, net_returns, _ = calculate_strategy_returns(
        spread_returns,
        positions,
        price1=prices[stock1].loc[positions.index],
        price2=prices[stock2].loc[positions.index],
        hedge_ratio=beta,
        cost_model=cost_model
    )

    return net_returns

def build_portfolio_returns(
    ranked_pairs,
    prices,
    top_n=5
):
    all_returns = []
    all_volatilities = []

    top_pairs = ranked_pairs.head(top_n)

    for pair in top_pairs["Pair"]:

        stock1, stock2 = pair.split("-")

        strategy_returns = get_pair_returns(
            stock1,
            stock2,
            prices
        )

        all_returns.append(strategy_returns)

        pair_volatility = calculate_pair_volatility(
            strategy_returns
        )

        all_volatilities.append(pair_volatility)

    returns_df = pd.concat(
        all_returns,
        axis=1
    )

    returns_df.columns = top_pairs["Pair"]

    inverse_vol = 1 / np.array(all_volatilities)

    weights = inverse_vol / inverse_vol.sum()

    print("\nPortfolio Weights:")
    for pair, weight in zip(
        top_pairs["Pair"],
        weights
    ):
        print(f"{pair}: {weight:.2%}")

    portfolio_returns = (
        returns_df * weights
    ).sum(axis=1)

    return portfolio_returns