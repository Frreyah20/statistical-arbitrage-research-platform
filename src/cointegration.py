from statsmodels.tsa.stattools import coint  #Imports the Engle-Granger cointegration test.
from itertools import combinations
import statsmodels.api as sm 
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt 

def test_cointegration(series1, series2):
    score, p_value, _ = coint(series1, series2)

    return p_value 

def find_cointegrated_pairs(prices, significance = 0.05): 
    pairs = []
    for stock1, stock2 in combinations(prices.columns, 2):
        p_value  = test_cointegration(prices[stock1], prices[stock2])
        #print(stock1, stock2, p_value)
        if p_value < significance:
            pairs.append((stock1, stock2, p_value))
    pairs.sort(key = lambda x :x[2]) #sort by p_value, lowest first
    return pairs

def calculate_hedge_ratio(series1, series2):
    model = sm.OLS(series1, series2).fit() #goal: Find the β that minimizes total squared prediction error.
    beta = model.params.iloc[0]
    return beta 

def calculate_spread(series1, series2, beta):
    spread = series1 - beta*series2
    return spread  #spread is the regression error

def test_stationarity(series):
    result = adfuller(series)
    p_value = result[1]
    return p_value

def plot_spread(spread):
    plt.figure(figsize = (12, 6))
    plt.plot(spread)
    plt.axhline(spread.mean(), linestyle="--")
    plt.title("Visa-Mastercard Spread")
    plt.show()
     