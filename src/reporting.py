from cointegration import calculate_hedge_ratio, calculate_spread
from signals import calculate_rolling_zscore
from signal_generator import generate_positions
from performance import (
    calculate_daily_spread_returns,
    calculate_strategy_returns,
    build_equity_curve,
    calculate_drawdown
)

from plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_spread_series
)

def generate_pair_report(stock1, stock2, prices):

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
    ).dropna()

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

    equity_curve = build_equity_curve(
        strategy_returns
    )

    drawdown, _ = calculate_drawdown(
        equity_curve
    )

    print("Pair Max DD:", drawdown.min())
    print("Pair Equity Peak:", equity_curve.max())
    print("Pair Equity Trough:", equity_curve.min())

    pair_name = f"{stock1}_{stock2}"

    plot_spread_series(spread, filename=f"spread_{pair_name}.png")
    plot_equity_curve(equity_curve, filename=f"equity_{pair_name}.png")
    plot_drawdown(drawdown, filename=f"drawdown_{pair_name}.png")