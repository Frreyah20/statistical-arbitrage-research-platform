# Statistical Arbitrage Research Platform

A comprehensive Python-based quantitative research engine for statistical arbitrage, cointegration pair-trading, and machine learning-driven signal generation.

## Overview
This platform implements an end-to-end quantitative workflow designed to discover cointegrated stock pairs, engineer time-series features, train calibrated machine learning classifiers to predict spread convergence, and conduct rigorous walk-forward validation under realistic transaction costs.

## Core Frameworks

### 1. Feature Engineering (`feature_engineering.py`)
To capture the dynamic relationships between asset pairs, the platform generates rolling time-series features without look-ahead bias:
- **Spread & Convergence Features**: Z-scores (20-day, 60-day), momentum indicators, and distance from equilibrium.
- **Volatility Regimes**: Rolling realized volatility to contextualize the market environment.
- **Mean-Reversion Speed**: Rolling half-life and augmented Dickey-Fuller (ADF) statistics to quantify stationarity.
- **Distributional Skew**: Rolling skewness and kurtosis of the spread to identify asymmetric risks.

### 2. Machine Learning & Model Comparison (`ml_model.py`)
The pipeline natively compares three classification architectures:
- **Logistic Regression (Baseline)**: Scaled using `StandardScaler` to identify basic linear signals.
- **Random Forest**: Bounded tree ensembles to capture non-linear interactions while resisting financial noise.
- **XGBoost**: Gradient boosted trees for complex regime-based signal mapping.

**Evaluation Integrity**: 
- **Time-Series Cross Validation**: Evaluated strictly using `TimeSeriesSplit` (5 folds) to preserve chronological ordering.
- **Probability Calibration**: Uses `CalibratedClassifierCV` (Isotonic regression) to correct the probability spaces of tree-based models, yielding well-calibrated confidence scores.
- **Feature Stability Analysis**: Extracts `Mean` and `Std` of feature importances across time folds to detect transient vs. robust alphas.
- **Model Selection**: The framework builds out-of-sample portfolios for all three models and selects the final candidate dynamically based on a hierarchical heuristic: `OOS Sharpe > Max Drawdown > ROC-AUC`.

### 3. Transaction Cost Framework (`transaction_costs.py` & `reporting.py`)
To ensure backtests reflect realistic execution environments, the platform explicitly deducts execution costs on every portfolio turnover:
- **Traded Notional Calculation**: Computes the true traded value of both the long and short legs (incorporating the OLS hedge ratio).
- **Frictions Applied**: 
  - `Commission`: Applied as basis points per trade.
  - `Bid-Ask Spread`: Applied on entries and exits to simulate crossing the spread.
  - `Slippage`: Applied proportionally based on volatility to simulate market impact.

### 4. Walk-Forward Validation (`walkforward.py`)
Instead of a single train/test split, the platform tests parameter robustness through a rolling walk-forward protocol:
1. **Train Window (e.g., 252 days)**: Scans the universe, discovers cointegration, builds features, trains the ML classifier, and ranks the top pairs.
2. **Test Window (e.g., 126 days)**: Trades the selected pairs exclusively on unseen future data.
3. **Rolling Sequence**: The window advances chronologically, ensuring the strategy dynamically adapts to changing market correlations.

## Reporting & Artifacts
The pipeline automatically generates a markdown research report (`results/research_report.md`) containing model comparison tables, performance benchmarks, and feature stability rankings. 

Visualizations outputted to `results/plots/` include:
- `roc_curves.png`: True vs. False positive trade-offs.
- `prob_distributions.png`: Density of predicted confidence.
- `calibration_curves.png`: Model reliability mapping.
- `shap_summary.png`: Feature attribution and interpretability.
- `equity_comparison_[pair].png`: Gross vs. Net (post-cost) equity curves.
