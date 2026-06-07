import pandas as pd 
from cointegration import calculate_hedge_ratio, calculate_spread, test_stationarity
from data_loader import load_prices 
from signals import calculate_rolling_zscore
from signal_generator import generate_positions
from trade_log import extract_trades
from performance import calculate_daily_spread_returns, calculate_drawdown, calculate_half_life, calculate_performance_metrics, calculate_sharpe_ratio, calculate_strategy_returns, calculate_trade_spread, build_equity_curve

def backtest_pair(stock1, stock2, prices):
    beta = calculate_hedge_ratio(prices[stock1], prices[stock2])
    spread = calculate_spread(prices[stock1], prices[stock2], beta)
    half_life = calculate_half_life(spread)
    zscore = calculate_rolling_zscore(spread, window = 60)
    zscore = zscore.dropna()
    positions = generate_positions(zscore)
    spread_returns = calculate_daily_spread_returns(spread)
    spread_returns = spread_returns.loc[positions.index] #aligning indices, because rolliing spread removed first 59 rows
    strategy_returns = calculate_strategy_returns(spread_returns, positions)
    equity_curve = build_equity_curve(strategy_returns)
    sharpe = calculate_sharpe_ratio(strategy_returns)
    _, max_drawdown = calculate_drawdown(equity_curve)
    trades = extract_trades(positions)
    trades = calculate_trade_spread(trades, spread)
    metrics = calculate_performance_metrics(trades)
    return {
            "Pair" : f"{stock1}-{stock2}",
            "Sharpe" : sharpe,
            "Max Drawdown" : max_drawdown, 
            "Trades" : metrics["Number of Trades"],
            "Win Rate" : metrics["Win Rate"],
            "Total PnL" : metrics["Total PnL"],
            "Half Life" : half_life
    }

def backtest_all_pairs(pairs, prices):
    results = []
    for stock1, stock2, p_value in pairs:
        try:
            result = backtest_pair(stock1, stock2, prices)
            results.append(result)
        except Exception as e:
                print(f"Failed: {stock1}-{stock2}")
                print(e)
    return pd.DataFrame(results)