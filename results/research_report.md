# Machine Learning Research Report

## 1. Model Evaluation Metrics

|   Accuracy |   Precision |     Recall |        F1 |   ROC AUC | Model               | Type               |
|-----------:|------------:|-----------:|----------:|----------:|:--------------------|:-------------------|
|   0.844099 |    0.109091 | 0.00589681 | 0.0111888 |  0.684831 | Logistic Regression | CV Validation      |
|   0.84441  |    0.24     | 0.00758266 | 0.0146728 |  0.624903 | Random Forest       | CV Validation      |
|   0.844488 |    0.34     | 0.00418737 | 0.0082472 |  0.622716 | XGBoost             | CV Validation      |
|   0.805443 |    0        | 0          | 0         |  0.466706 | Logistic Regression | Out-of-Sample Test |
|   0.805443 |    0        | 0          | 0         |  0.514012 | Random Forest       | Out-of-Sample Test |
|   0.797936 |    0.21875  | 0.0150054  | 0.0280843 |  0.56844  | XGBoost             | Out-of-Sample Test |

## 2. Portfolio Performance by Model (Out-of-Sample)

| Model               |   OOS Sharpe |   OOS Max Drawdown |   OOS ROC AUC |
|:--------------------|-------------:|-------------------:|--------------:|
| Logistic Regression |    0.045923  |         -0.313419  |      0.466706 |
| Random Forest       |    0.301429  |         -0.0745603 |      0.514012 |
| XGBoost             |    0.0752636 |         -0.101084  |      0.56844  |

## 3. Selected Final Model: Random Forest

The `Random Forest` was selected based on the highest out-of-sample portfolio Sharpe ratio, while also considering drawdown and ROC AUC.

## 4. Feature Stability Analysis

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

## 5. Visualizations Generated
- `results/plots/roc_curves.png`
- `results/plots/prob_distributions.png`
- `results/plots/calibration_curves.png`
- `results/plots/shap_summary.png` (for tree models)
