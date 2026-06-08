# Statistical Arbitrage Research Platform

## Architecture

```text
Market Data
    │
    ▼
Cointegration Screening
    │
    ▼
Pair Backtesting
    │
    ▼
Research Ranking
    │
    ├── Portfolio Construction
    │
    ├── Walk-Forward Validation
    │
    ├── Parameter Optimization
    │
    └── Machine Learning Ranking
    │
    ▼
Performance Analytics & Reporting
```

## Overview

A quantitative research platform for discovering, ranking, backtesting, and evaluating statistical arbitrage opportunities using pairs trading.

The project identifies cointegrated stock pairs, constructs mean-reverting spreads, generates trading signals using z-score deviations, evaluates performance through backtesting, ranks candidate pairs using multiple quantitative metrics, and applies machine learning to investigate which pair characteristics are most predictive of future performance.

---
## Sample Outputs

| Equity Curve | Drawdown |
|-------------|----------|
| ![](results/plots/research_portfolio_equity.png) | ![](results/plots/research_portfolio_drawdown.png) |

| Feature Importance |
|---------|-------------------|
| ![](results/plots/feature_importance.png) |

## Features

### Data Pipeline

- Historical price data collection
- Multi-asset universe support
- Train/Test data split for out-of-sample validation

### Statistical Arbitrage Research

- Correlation analysis
- Cointegration testing (Engle-Granger)
- Hedge ratio estimation using OLS regression
- Spread construction
- Stationarity testing (ADF Test)
- Half-life estimation of mean reversion

### Trading Strategy

- Rolling z-score calculation
- Mean reversion signal generation
- Long/Short spread positions
- Position management with entry/exit thresholds
- Look-ahead bias prevention through signal shifting

### Backtesting Engine

- Daily strategy return calculation
- Trade extraction and logging
- Equity curve construction
- Sharpe ratio calculation
- Maximum drawdown analysis
- Win rate and trade statistics

### Pair Ranking System

Pairs are ranked using a composite research score based on:

- Sharpe Ratio
- Maximum Drawdown
- Mean-Reversion Half-Life
- Trade Frequency

This enables systematic selection of the most promising statistical arbitrage candidates.

### Walk-Forward Validation

The strategy is evaluated using:

- In-sample (Training) period
- Out-of-sample (Testing) period

to assess robustness and reduce overfitting risk.

### Portfolio Construction

- Multi-pair portfolio creation
- Volatility-weighted allocation
- Portfolio-level Sharpe ratio analysis
- Portfolio-level drawdown analysis

### Parameter Optimization

Grid-search optimization of:

- Entry Z-Score Thresholds
- Exit Z-Score Thresholds

to identify improved signal configurations.

### Machine Learning Analysis

Feature engineering is performed on each pair:

- Correlation
- Cointegration P-Value
- Hedge Ratio
- Half-Life
- Spread Mean
- Spread Standard Deviation
- Z-Score Volatility
- Training Sharpe Ratio
- Training Drawdown
- Number of Trades

A Random Forest Regressor is trained to predict out-of-sample Sharpe ratios and analyze feature importance.

---

## Project Structure

```text
stat_arb_platform/
│
├── data/
│
├── results/
│   ├── plots/
│   ├── cointegrated_pairs.csv
│   ├── ranked_pairs.csv
│   ├── walkforward_results.csv
│   ├── rolling_walkforward.csv
│   ├── portfolio_returns.csv
│   ├── portfolio_equity.csv
│   ├── ml_dataset.csv
│   ├── ml_ranked_pairs.csv
│   ├── feature_importance.csv
│   ├── parameter_optimization.csv
│   └── project_summary.csv
│
├── src/
│   ├── data_loader.py
│   ├── dashboard.py
│   ├── correlation.py
│   ├── cointegration.py
│   ├── portfolio.py
│   ├── walkforward.py
│   ├── optimization.py
│   ├── signals.py
│   ├── signal_generator.py
│   ├── trade_log.py
│   ├── performance.py
│   ├── research_engine.py
│   ├── ranking.py
│   ├── feature_engineering.py
│   ├── ml_model.py
│   ├── reporting.py
│   ├── plots.py
│   └── main.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Research Workflow

```text
Price Data
    ↓
Correlation Analysis
    ↓
Cointegration Testing
    ↓
Spread Construction
    ↓
Stationarity Validation
    ↓
Signal Generation
    ↓
Backtesting
    ↓
Pair Ranking
    ↓
Walk-Forward Testing
    ↓
Feature Engineering
    ↓
Machine Learning Analysis
```

---

## Results

### Portfolio Performance

| Portfolio | Sharpe | Max Drawdown |
|------------|---------|---------|
| Research Portfolio | 2.73 | -6.44% |
| ML Portfolio | -0.02 | -29.63% |

### Walk-Forward Validation

Average Test Sharpe: **2.76**

Best Test Sharpe: **3.74**

Worst Test Sharpe: **1.88**

### Parameter Optimization

Best parameter set discovered:

| Entry Z-Score | Exit Z-Score | Sharpe |
|---------------|--------------|---------|
| 2.0 | 0.25 | 1.77 |

## Example Performance Metrics

For each pair the platform computes:

- Sharpe Ratio
- Maximum Drawdown
- Total PnL
- Win Rate
- Number of Trades
- Half-Life

Example output:

| Pair | Sharpe | Max Drawdown | Trades |
|--------|---------|-------------|---------|
| HD-WFC | 1.52 | -17.0% | 10 |
| MA-V | 1.47 | -8.4% | 8 |
| JPM-V | 0.91 | -9.9% | 10 |

---

## Machine Learning Results

A Random Forest Regressor was trained to predict future pair performance using statistical characteristics of each pair.

### Features Used

- Correlation
- Cointegration P-Value
- Hedge Ratio
- Half-Life
- Spread Mean
- Spread Standard Deviation
- Z-Score Volatility
- Training Sharpe Ratio
- Training Drawdown
- Number of Trades

### Observation

The ML-selected portfolio underperformed the research-ranked portfolio, indicating that simple statistical features alone were insufficient to reliably forecast future Sharpe ratios.

This result highlights the importance of rigorous out-of-sample validation and reflects a realistic quantitative research workflow.

---

## Generated Outputs

The platform automatically exports:

- Cointegrated Pairs
- Ranked Research Pairs
- Walk-Forward Results
- Rolling Walk-Forward Analysis
- Portfolio Returns
- Portfolio Equity Curves
- Feature Importance Rankings
- ML Predictions
- Parameter Optimization Results
- Experiment Summary Dashboard

All outputs are stored inside:

```text
results/
```

---

## Technologies Used

- Python
- Pandas
- NumPy
- Statsmodels
- Scikit-Learn
- Matplotlib

---

## Key Concepts Implemented

- Statistical Arbitrage
- Pairs Trading
- Cointegration
- Mean Reversion
- Ordinary Least Squares (OLS)
- Augmented Dickey-Fuller Test
- Walk-Forward Testing
- Feature Engineering
- Random Forest Regression
- Quantitative Research

---

## Future Improvements

- Dynamic Hedge Ratios using Rolling Regression
- Kalman Filter Spread Estimation
- Regime Detection Models
- Enhanced Transaction Cost Modeling
- Slippage Simulation
- Multi-Factor Pair Selection
- Bayesian Optimization
- XGBoost and LightGBM Models
- Live Paper Trading Deployment
- Real-Time Signal Monitoring Dashboard

---

## Disclaimer

This project is intended for research and educational purposes only and should not be considered financial advice.

```
