import pandas as pd

def calculate_returns(prices):
    returns = prices.pct_change()
    returns = returns.dropna()  # Remove the first row which will be NaN
    return returns 

def calculate_correlation_matrix(returns):
    correlation_matrix = returns.corr()
    return correlation_matrix