import pandas as pd 
from cointegration import calculate_hedge_ratio, calculate_spread, test_stationarity
from data_loader import load_prices 
from signals import calculate_rolling_zscore
from signal_generator import generate_positions
from trade_log import extract_trades
from performance import calculate_daily_spread_returns, calculate_drawdown, calculate_half_life, calculate_performance_metrics, calculate_sharpe_ratio, calculate_strategy_returns, calculate_trade_spread, build_equity_curve, calculate_cagr
import config
from transaction_costs import TransactionCostModel

def backtest_pair(stock1, stock2, prices, entry_threshold=2.0, exit_threshold=0.5):
    beta = calculate_hedge_ratio(prices[stock1], prices[stock2])
    spread = calculate_spread(prices[stock1], prices[stock2], beta)
    half_life = calculate_half_life(spread)
    zscore = calculate_rolling_zscore(spread, window = 60)
    zscore = zscore.dropna()
    positions = generate_positions(zscore, entry_threshold = entry_threshold, exit_threshold = exit_threshold)
    spread_returns = calculate_daily_spread_returns(spread)
    spread_returns = spread_returns.loc[positions.index] #aligning indices, because rolliing spread removed first 59 rows
    
    # Initialize cost model
    cost_model = TransactionCostModel(
        commission_bps=config.COMMISSION_BPS,
        spread_bps=config.SPREAD_BPS,
        slippage_bps=config.SLIPPAGE_BPS
    )
    
    gross_returns, net_returns, costs = calculate_strategy_returns(
        spread_returns, positions, 
        price1=prices[stock1].loc[positions.index], 
        price2=prices[stock2].loc[positions.index], 
        hedge_ratio=beta, 
        cost_model=cost_model
    )
    
    # Before costs
    gross_equity = build_equity_curve(gross_returns)
    gross_sharpe = calculate_sharpe_ratio(gross_returns)
    _, gross_max_dd = calculate_drawdown(gross_equity)
    gross_cagr = calculate_cagr(gross_equity)
    
    # After costs
    net_equity = build_equity_curve(net_returns)
    net_sharpe = calculate_sharpe_ratio(net_returns)
    _, net_max_dd = calculate_drawdown(net_equity)
    net_cagr = calculate_cagr(net_equity)
    
    trades = extract_trades(positions)
    # Using the true notional as required by the user
    trades = calculate_trade_spread(trades, prices[stock1], prices[stock2], beta, cost_model)
    metrics = calculate_performance_metrics(trades)
    
    return {
            "Pair" : f"{stock1}-{stock2}",
            "Sharpe (Before Costs)" : gross_sharpe,
            "Sharpe (After Costs)" : net_sharpe,
            "CAGR (Before Costs)" : gross_cagr,
            "CAGR (After Costs)" : net_cagr,
            "Max Drawdown (Before Costs)" : gross_max_dd,
            "Max Drawdown (After Costs)" : net_max_dd, 
            "Trades" : metrics["Number of Trades"],
            "Win Rate" : metrics["Win Rate"],
            "Total Gross PnL" : metrics["Total Gross PnL"],
            "Total Net PnL" : metrics["Total Net PnL"],
            "Total Cost": metrics["Total Cost"],
            "Cost vs Gross Profit %": metrics["Cost vs Gross Profit %"],
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