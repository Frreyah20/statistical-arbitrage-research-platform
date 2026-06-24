import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def calculate_hedge_ratio(series1, series2):
    model = sm.OLS(series1, series2).fit()
    return model.params.iloc[0]

def calculate_half_life(spread):
    lagged_spread = spread.shift(1)
    delta_spread = spread - lagged_spread
    lagged_spread = lagged_spread.dropna()
    delta_spread = delta_spread.dropna()
    if len(delta_spread) < 2: return np.nan
    model = sm.OLS(delta_spread, sm.add_constant(lagged_spread)).fit()
    beta = model.params.iloc[1] if len(model.params) > 1 else np.nan
    if beta >= 0: return np.nan
    half_life = -np.log(2) / beta
    return half_life

def rolling_hl(x):
    """Helper for rolling apply of half-life"""
    try:
        return calculate_half_life(pd.Series(x))
    except:
        return np.nan

def rolling_adf_stat(x):
    """Helper for rolling apply of ADF statistic"""
    try:
        return adfuller(x, maxlag=1)[0]
    except:
        return np.nan

def rolling_adf_pval(x):
    """Helper for rolling apply of ADF p-value"""
    try:
        return adfuller(x, maxlag=1)[1]
    except:
        return np.nan

def generate_pair_time_series_features(stock1, stock2, prices):
    df = pd.DataFrame(index=prices.index)
    df["pair"] = f"{stock1}-{stock2}"
    df["date"] = df.index

    # 5. Spread Construction
    series1 = prices[stock1]
    series2 = prices[stock2]
    beta = calculate_hedge_ratio(series1, series2)
    df["hedge_ratio"] = beta
    
    spread = series1 - beta * series2
    df["spread"] = spread

    # Spread Features
    df["spread_return"] = spread.diff()
    df["spread_momentum_5"] = spread.diff(5)
    df["spread_momentum_20"] = spread.diff(20)

    # Rolling Means and Stds
    spread_mean_20 = spread.rolling(20).mean()
    spread_mean_60 = spread.rolling(60).mean()
    spread_std_20 = spread.rolling(20).std()
    spread_std_60 = spread.rolling(60).std()

    df["spread_zscore_20"] = (spread - spread_mean_20) / spread_std_20
    df["spread_zscore_60"] = (spread - spread_mean_60) / spread_std_60
    
    # Additional Features
    df["spread_percentile_60"] = spread.rolling(60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    df["spread_percentile_120"] = spread.rolling(120).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    
    df["distance_from_mean_20"] = spread - spread_mean_20
    df["distance_from_mean_60"] = spread - spread_mean_60
    
    df["zscore_change"] = df["spread_zscore_60"].diff()

    # Volatility Features
    df["spread_volatility_20"] = df["spread_return"].rolling(20).std()
    df["spread_volatility_60"] = df["spread_return"].rolling(60).std()
    df["realized_volatility"] = df["spread_return"].rolling(20).std() * np.sqrt(252)

    # Market Regime Features
    # 3. Volatility Regime using trailing 252-day percentile of the 60-day volatility
    vol_60 = df["spread_volatility_60"]
    vol_rank = vol_60.rolling(252, min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=True)
    
    df["vol_regime_high"] = (vol_rank > 0.75).astype(int)
    df["vol_regime_low"] = (vol_rank < 0.25).astype(int)
    df["vol_regime_medium"] = ((vol_rank >= 0.25) & (vol_rank <= 0.75)).astype(int)
    
    # 4. Trend Regime
    df["trend_bullish"] = (spread_mean_20 > spread_mean_60).astype(int)
    df["trend_bearish"] = (spread_mean_20 < spread_mean_60).astype(int)

    # Mean Reversion Features
    # Note: Rolling half-life and ADF might be slow, using window=60 or 120
    df["rolling_half_life"] = spread.rolling(120).apply(rolling_hl, raw=False)
    df["rolling_mean_reversion_speed"] = np.where(df["rolling_half_life"] > 0, np.log(2) / df["rolling_half_life"], np.nan)
    df["distance_from_equilibrium"] = df["distance_from_mean_60"]

    # Relationship Features
    df["rolling_correlation_20"] = series1.rolling(20).corr(series2)
    df["rolling_correlation_60"] = series1.rolling(60).corr(series2)
    
    cov_60 = series1.rolling(60).cov(series2)
    var_60 = series2.rolling(60).var()
    df["rolling_beta"] = cov_60 / var_60

    # Stationarity Features
    df["rolling_adf_statistic"] = spread.rolling(120).apply(rolling_adf_stat, raw=False)
    df["rolling_adf_pvalue"] = spread.rolling(120).apply(rolling_adf_pval, raw=False)

    # Distribution Features
    df["rolling_skew"] = df["spread_return"].rolling(60).skew()
    df["rolling_kurtosis"] = df["spread_return"].rolling(60).kurt()

    # Target Construction (N=5 days)
    N = 5
    spread_fwd = spread.shift(-N)
    
    reverted = abs(spread_fwd) < abs(spread)
    
    # Avoid division by zero
    moved_pct = np.zeros_like(spread)
    valid_idx = (abs(spread) > 0)
    moved_pct[valid_idx] = (abs(spread[valid_idx]) - abs(spread_fwd[valid_idx])) / abs(spread[valid_idx])
    
    moved_25 = moved_pct >= 0.25
    
    target = (reverted & moved_25).astype(float)
    target[spread_fwd.isna()] = np.nan
    df["target"] = target
    
    return df

def build_complete_feature_dataset(pairs, prices):
    all_dfs = []
    print(f"Generating features for {len(pairs)} pairs...")
    for idx, (stock1, stock2, _) in enumerate(pairs):
        print(f"[{idx+1}/{len(pairs)}] Generating features for {stock1}-{stock2}...")
        try:
            pair_df = generate_pair_time_series_features(stock1, stock2, prices)
            all_dfs.append(pair_df)
        except Exception as e:
            print(f"Error on {stock1}-{stock2}: {e}")
            
    full_dataset = pd.concat(all_dfs, ignore_index=True)
    # Drop rows with NaN targets (the last N rows per pair) or NaN features (initial rolling windows)
    full_dataset = full_dataset.dropna().reset_index(drop=True)
    return full_dataset

def add_test_target(features_df, test_prices):
    # This function is no longer needed in the new time-series design since the target 
    # is generated intrinsically per pair on a continuous basis. We'll leave a stub to prevent import errors.
    pass