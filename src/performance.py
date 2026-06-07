import pandas as pd 
import numpy as np 
import statsmodels.api as sm

def calculate_trade_spread(trades, spread):
    returns = []
    for _, trade in trades.iterrows():
        entry_date = trade["Entry Date"]
        exit_date = trade["Exit Date"]
        position = trade["Position Type"]
        
        entry_spread = spread.loc[entry_date] #loc is used to select rows/columns by there labels
        exit_spread = spread.loc[exit_date]

        pnl = position * (exit_spread - entry_spread)
        
        returns.append(pnl)


    trades["PnL"] = returns
    return trades 

def calculate_performance_metrics(trades):
    total_pnl = trades["PnL"].sum()
    num_trades = len(trades)
    winning_trades = (trades["PnL"] > 0).sum()
    win_rate = ((winning_trades/num_trades) * 100 if num_trades > 0 else 0)
    average_pnl = trades["PnL"].mean()
    metrics = {"Total PnL" : total_pnl,
                "Number of Trades" : num_trades,
                "Win Rate" : win_rate,
                "Average PnL" : average_pnl}
    return metrics

def calculate_daily_spread_returns(spread):
    spread_returns = spread.diff()
    return spread_returns

def calculate_strategy_returns(spread_returns, positions):
    positions = positions.shift(1) #to prevent signal lookahead bias
    strategy_returns = positions * spread_returns
    return strategy_returns

def build_equity_curve(strategy_returns, initial_capital = 100):
    equity_curve = initial_capital + strategy_returns.cumsum()
    return equity_curve

def calculate_drawdown(equity_curve):
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max)/running_max
    max_drawdown = drawdown.min()
    return drawdown, max_drawdown

def calculate_sharpe_ratio(strategy_returns):
    mean_return = strategy_returns.mean()
    std_returns = strategy_returns.std()
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