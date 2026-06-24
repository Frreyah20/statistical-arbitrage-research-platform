import pandas as pd 
import numpy as np 
import statsmodels.api as sm

def calculate_trade_spread(trades, price1, price2, hedge_ratio, cost_model):
    returns = []
    for _, trade in trades.iterrows():
        entry_date = trade["Entry Date"]
        exit_date = trade["Exit Date"]
        position = trade["Position Type"]
        
        # Gross PnL calculation
        entry_spread = price1.loc[entry_date] - hedge_ratio * price2.loc[entry_date]
        exit_spread = price1.loc[exit_date] - hedge_ratio * price2.loc[exit_date]
        gross_pnl = position * (exit_spread - entry_spread)
        
        # Notional calculation: abs(asset1) + abs(hedge_ratio * asset2)
        entry_notional = abs(price1.loc[entry_date]) + abs(hedge_ratio * price2.loc[entry_date])
        exit_notional = abs(price1.loc[exit_date]) + abs(hedge_ratio * price2.loc[exit_date])
        
        # Costs apply on entry and exit
        transaction_cost = cost_model.calculate_cost(entry_notional) + cost_model.calculate_cost(exit_notional)
        
        net_pnl = gross_pnl - transaction_cost
        
        returns.append((net_pnl, transaction_cost, gross_pnl))


    trades["Net PnL"] = [x[0] for x in returns]
    trades["Cost"] = [x[1] for x in returns]
    trades["Gross PnL"] = [x[2] for x in returns]
    # For backward compatibility, set PnL to Net PnL
    trades["PnL"] = trades["Net PnL"]
    return trades 

def calculate_performance_metrics(trades):
    total_net_pnl = trades["Net PnL"].sum() if "Net PnL" in trades.columns else trades["PnL"].sum()
    total_gross_pnl = trades["Gross PnL"].sum() if "Gross PnL" in trades.columns else total_net_pnl
    total_cost = trades["Cost"].sum() if "Cost" in trades.columns else 0
    num_trades = len(trades)
    winning_trades = (trades["Net PnL"] > 0).sum() if "Net PnL" in trades.columns else (trades["PnL"] > 0).sum()
    win_rate = ((winning_trades/num_trades) * 100 if num_trades > 0 else 0)
    average_net_pnl = trades["Net PnL"].mean() if "Net PnL" in trades.columns else trades["PnL"].mean()
    
    cost_percentage = (total_cost / total_gross_pnl * 100) if total_gross_pnl > 0 else 0

    metrics = {
        "Total Net PnL" : total_net_pnl,
        "Total Gross PnL": total_gross_pnl,
        "Number of Trades" : num_trades,
        "Win Rate" : win_rate,
        "Average PnL" : average_net_pnl,
        "Total Cost": total_cost,
        "Cost vs Gross Profit %": cost_percentage
    } 
    return metrics

def calculate_daily_spread_returns(spread):
    spread_returns = spread.diff()
    return spread_returns

def calculate_strategy_returns(spread_returns, positions, price1=None, price2=None, hedge_ratio=None, cost_model=None):
    positions = positions.shift(1) #to prevent signal lookahead bias
    gross_returns = positions * spread_returns
    
    trades = positions.diff().abs()
    
    costs = pd.Series(0.0, index=gross_returns.index)
    if cost_model is not None and price1 is not None and price2 is not None and hedge_ratio is not None:
        notional = abs(price1) + abs(hedge_ratio * price2)
        costs = trades * cost_model.calculate_cost(notional)
        
    net_returns = gross_returns - costs
    return gross_returns, net_returns, costs

def build_equity_curve(strategy_returns, initial_capital = 100):
    equity_curve = initial_capital + strategy_returns.cumsum()
    return equity_curve

def calculate_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max)/running_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def calculate_cagr(equity_curve):
    if len(equity_curve) < 2:
        return 0
    days = (equity_curve.index[-1] - equity_curve.index[0]).days
    if days == 0:
        return 0
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    # Handle negative returns properly if equity falls below 0 (should use initial cap in absolute terms, but here initial cap is 100)
    if total_return <= 0:
        return -1.0
    cagr = (total_return ** (365.25 / days)) - 1
    return cagr

def calculate_sharpe_ratio(strategy_returns):
    mean_return = strategy_returns.mean()
    std_returns = strategy_returns.std()
    if std_returns == 0 or np.isnan(std_returns):
        return 0
    sharpe = (mean_return/std_returns) * np.sqrt(252)
    return sharpe

def calculate_half_life(spread):
    lagged_spread = spread.shift(1)
    delta_spread = spread - lagged_spread
    lagged_spread = lagged_spread.dropna()
    delta_spread = delta_spread.dropna()
    model = sm.OLS(delta_spread, sm.add_constant(lagged_spread)).fit()
    beta = model.params.iloc[1]
    half_life = -np.log(2) /beta
    return half_life