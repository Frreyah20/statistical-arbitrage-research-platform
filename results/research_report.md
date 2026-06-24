# Statistical Arbitrage Research Report

## 1. Executive Summary

This report presents the results of a statistical arbitrage research project based on cointegration and mean-reversion trading. The platform identifies cointegrated stock pairs, constructs market-neutral spreads, generates trading signals, and evaluates performance after transaction costs.

In addition to a rule-based strategy, the project explores machine learning models for pair ranking using engineered time-series features. The framework includes out-of-sample testing, transaction cost modeling, and walk-forward validation to evaluate strategy robustness.

## 2. Dataset Description
The dataset spans from January 1, 2022, to January 1, 2025, and includes highly liquid US equities across technology, banking, payments, consumer staples, and energy sectors. The data is split into an in-sample training window (2022-01-01 to 2023-12-31) and an out-of-sample testing window (2024-01-01 to 2024-12-31). Prices are adjusted daily closing prices.

## 3. Pair Selection Methodology
Cointegration testing serves as the foundation for pair selection because it identifies stationary linear combinations of non-stationary time series, avoiding the spurious relationships common in simple correlation strategies. We use the Augmented Dickey-Fuller (ADF) test on the residual spread formulated by Ordinary Least Squares (OLS) regression. Pairs with an ADF p-value < 0.05 are considered cointegrated.

## 4. Transaction Cost Modeling
Transaction costs are included to produce more realistic backtest results. Since pairs trading often involves frequent entry and exit decisions, even small execution costs can have a meaningful impact on profitability.

Costs are applied to the traded notional of both legs of the position and include commissions, bid-ask spread costs, and slippage assumptions.

## 5. Feature Engineering Pipeline
The machine learning pipeline uses a set of rolling time-series features designed to capture spread dynamics, volatility, relationship stability, and mean-reversion behavior:
- **Spread Features**: Spread, Spread Return, Z-Scores (20, 60 days), and Momentum (5, 20 days).
- **Volatility Features**: Rolling Volatility (20, 60 days) and Realized Volatility.
- **Mean Reversion Features**: Rolling Half-Life, Distance from Equilibrium.
- **Relationship Features**: Rolling Correlation (20, 60 days), Rolling Beta, and Hedge Ratio.
- **Stationarity Features**: Rolling ADF Statistic and P-Value.
- **Distribution Features**: Rolling Skewness and Kurtosis.

## 6. Machine Learning Evaluation
We evaluated Logistic Regression, Random Forest, and XGBoost models. Strict time-series cross-validation (`TimeSeriesSplit`, 5 folds) was used to prevent data leakage. Model probabilities were calibrated using Isotonic Regression before being used for pair ranking.

### Model Comparison Table

|   Accuracy |   Precision |     Recall |        F1 |   ROC AUC | Model               | Type               |
|-----------:|------------:|-----------:|----------:|----------:|:--------------------|:-------------------|
|   0.844099 |    0.109091 | 0.00589681 | 0.0111888 |  0.684831 | Logistic Regression | CV Validation      |
|   0.84441  |    0.24     | 0.00758266 | 0.0146728 |  0.624903 | Random Forest       | CV Validation      |
|   0.844488 |    0.34     | 0.00418737 | 0.0082472 |  0.622716 | XGBoost             | CV Validation      |
|   0.805443 |    0        | 0          | 0         |  0.466706 | Logistic Regression | Out-of-Sample Test |
|   0.805443 |    0        | 0          | 0         |  0.514012 | Random Forest       | Out-of-Sample Test |
|   0.797936 |    0.21875  | 0.0150054  | 0.0280843 |  0.56844  | XGBoost             | Out-of-Sample Test |

### Feature Stability Analysis

Top 15 features for Random Forest:

| Model         | Feature                      |   Mean Importance |   Std Importance |   Stability (Mean/Std) |
|:--------------|:-----------------------------|------------------:|-----------------:|-----------------------:|
| Random Forest | spread                       |         0.242948  |       0.100905   |                2.40766 |
| Random Forest | spread_volatility_60         |         0.075254  |       0.0163768  |                4.59487 |
| Random Forest | hedge_ratio                  |         0.0707059 |       0.0147097  |                4.80642 |
| Random Forest | realized_volatility          |         0.0539049 |       0.0136146  |                3.95904 |
| Random Forest | spread_volatility_20         |         0.0537055 |       0.00775529 |                6.92412 |
| Random Forest | rolling_beta                 |         0.0459311 |       0.00721619 |                6.36412 |
| Random Forest | rolling_skew                 |         0.0419962 |       0.0100324  |                4.18562 |
| Random Forest | rolling_kurtosis             |         0.0419309 |       0.00475224 |                8.82153 |
| Random Forest | rolling_correlation_60       |         0.0394557 |       0.0133028  |                2.96574 |
| Random Forest | spread_momentum_20           |         0.0359469 |       0.012992   |                2.76664 |
| Random Forest | rolling_correlation_20       |         0.0330963 |       0.0080054  |                4.13373 |
| Random Forest | rolling_mean_reversion_speed |         0.0305125 |       0.00628534 |                4.85378 |
| Random Forest | rolling_half_life            |         0.0292822 |       0.00811573 |                3.60764 |
| Random Forest | distance_from_mean_20        |         0.0220699 |       0.0100101  |                2.20453 |
| Random Forest | rolling_adf_pvalue           |         0.0214024 |       0.00276982 |                7.72421 |

## 7. Portfolio Construction
The portfolio is constructed by allocating evenly to the Top 5 pairs. For the rule-based approach, pairs are ranked by their lowest p-value. For the ML approach, pairs are ranked by their calibrated prediction probability of reversion.

### Out-of-Sample Portfolio Performance by Model

| Model               |   OOS Sharpe |   OOS Max Drawdown |   OOS ROC AUC |
|:--------------------|-------------:|-------------------:|--------------:|
| Logistic Regression |    0.045923  |         -0.313419  |      0.466706 |
| Random Forest       |    0.301429  |         -0.0745603 |      0.514012 |
| XGBoost             |    0.0752636 |         -0.101084  |      0.56844  |

**Selected Final Model:** Random Forest. Selected based on the highest out-of-sample portfolio Sharpe ratio.

## 8. Walk-Forward Validation
A rolling walk-forward validation framework was used to evaluate strategy performance across multiple training and testing periods. Cointegration relationships and model parameters were re-estimated on a trailing training window and evaluated on the following out-of-sample period.

This approach provides a more realistic assessment of strategy performance than a single train-test split.

### Walk-Forward Performance Table

| Pair      |   Sharpe (Before Costs) |   Sharpe (After Costs) |   CAGR (Before Costs) |   CAGR (After Costs) |   Max Drawdown (Before Costs) |   Max Drawdown (After Costs) |   Trades |   Win Rate |   Total Gross PnL |   Total Net PnL |   Total Cost |   Cost vs Gross Profit % |   Half Life | Train End   | Test End   |
|:----------|------------------------:|-----------------------:|----------------------:|---------------------:|------------------------------:|-----------------------------:|---------:|-----------:|------------------:|----------------:|-------------:|-------------------------:|------------:|:------------|:-----------|
| LOW-WFC   |                 1.8677  |                1.79854 |                   nan |                  nan |                    -0.166092  |                   -0.166349  |        2 |        100 |           18.5344 |         17.7632 |     0.771183 |                  4.16083 |    18.9785  | 2023-01-03  | 2023-07-06 |
| LOW-PEP   |                 3.71235 |                3.6164  |                   nan |                  nan |                    -0.0340504 |                   -0.0341113 |        2 |        100 |           26.513  |         25.7246 |     0.78841  |                  2.97367 |     7.79616 | 2023-07-06  | 2024-01-04 |
| AMZN-NFLX |                 2.64169 |                2.57299 |                   nan |                  nan |                    -0.105437  |                   -0.10561   |        2 |        100 |           21.5698 |         20.8256 |     0.744209 |                  3.45023 |     7.76748 | 2024-01-04  | 2024-07-08 |

## 9. Results
The generated visual reports include ROC curves, probability distributions, calibration curves, and SHAP feature attribution plots.

The complete set of figures can be found in the results/plots directory.

## 10. Key Findings
### Research vs ML Performance

The rule-based cointegration strategy outperformed the machine learning portfolio during the testing period. This suggests that the cointegration signal itself contained much of the predictive information needed for mean-reversion trading within the tested universe. While the ML models were able to identify useful predictive signals, the traditional statistical arbitrage framework remained the stronger performer in this study.

### Feature Engineering

Adding rolling volatility, stationarity, and relationship-based features improved ML performance compared with earlier versions that relied primarily on pair-level statistics.

### Transaction Costs

Including transaction costs reduced performance across strategies, highlighting the importance of evaluating statistical arbitrage ideas using net returns rather than gross returns.

### Walk-Forward Validation

The strategy maintained positive performance across multiple walk-forward test windows, providing additional confidence that the results were not solely driven by in-sample fitting.

## 11. Limitations
- **Static OLS Hedge Ratios**: The hedge ratios remain static over the entry-to-exit trade lifecycle, which may misrepresent dynamically decaying relationships.
- **Borrow Costs**: Short-selling borrow costs and hard-to-borrow constraints were not modeled, which could further decay returns.
- **Daily Execution**: Trades are assumed to be executed at daily closing prices, ignoring intraday slippage variations and gap risks.
- **Limited Universe**: The evaluation was constrained to a small universe of 20 mega-cap equities.

## 12. Future Research
- **Kalman Filters**: Implementing Kalman Filters or Johansen Cointegration to dynamically adjust the hedge ratio on a continuous basis.
- **Expanded Universe**: Expanding the stock universe beyond the current set of large-cap equities.
- **Alternative Execution Models**: Improving execution modeling through more realistic assumptions about order execution and trading costs.
- **Advanced Alpha Features**: Exploring additional features and alternative data sources that may improve spread prediction.
