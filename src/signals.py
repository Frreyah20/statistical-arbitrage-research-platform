'''
def calculate_zscore(spread):
    
    this created lookahead bias, because it using all data from 2022 to 2024 to make calculations for 2022
    mean = spread.mean()
    std = spread.std()
    zscore = (spread-mean)/std
    return zscore
'''

#correction: after every day T, compute mean and std of previous 60 days
def calculate_rolling_zscore(spread, window = 60):
    rolling_mean = spread.rolling(window).mean()
    rolling_std = spread.rolling(window).std()
    zscore = (spread - rolling_mean)/rolling_std
    return zscore