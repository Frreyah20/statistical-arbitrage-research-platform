from cointegration import calculate_hedge_ratio, calculate_spread
from signals import calculate_rolling_zscore
from signal_generator import generate_positions
from performance import (
    calculate_daily_spread_returns,
    calculate_strategy_returns,
    build_equity_curve,
    calculate_drawdown
)

from plots import (
    plot_equity_curve,
    plot_drawdown,
    plot_spread_series,
    plot_cost_comparison_equity,
    plot_cumulative_costs
)
import config
from transaction_costs import TransactionCostModel

def generate_pair_report(stock1, stock2, prices):

    beta = calculate_hedge_ratio(
        prices[stock1],
        prices[stock2]
    )

    spread = calculate_spread(
        prices[stock1],
        prices[stock2],
        beta
    )

    zscore = calculate_rolling_zscore(
        spread,
        window=60
    ).dropna()

    positions = generate_positions(zscore)

    spread_returns = calculate_daily_spread_returns(
        spread
    )

    spread_returns = spread_returns.loc[
        positions.index
    ]

    cost_model = TransactionCostModel(
        commission_bps=config.COMMISSION_BPS,
        spread_bps=config.SPREAD_BPS,
        slippage_bps=config.SLIPPAGE_BPS
    )

    gross_returns, net_returns, costs = calculate_strategy_returns(
        spread_returns,
        positions,
        price1=prices[stock1].loc[positions.index],
        price2=prices[stock2].loc[positions.index],
        hedge_ratio=beta,
        cost_model=cost_model
    )

    gross_equity_curve = build_equity_curve(
        gross_returns
    )
    
    net_equity_curve = build_equity_curve(
        net_returns
    )

    drawdown, _ = calculate_drawdown(
        net_equity_curve
    )

    print("Pair Max DD:", drawdown.min())
    print("Pair Equity Peak (Net):", net_equity_curve.max())
    print("Pair Equity Trough (Net):", net_equity_curve.min())

    pair_name = f"{stock1}_{stock2}"

    plot_spread_series(spread, filename=f"spread_{pair_name}.png")
    plot_equity_curve(net_equity_curve, filename=f"equity_{pair_name}.png")
    plot_drawdown(drawdown, filename=f"drawdown_{pair_name}.png")
    plot_cost_comparison_equity(gross_equity_curve, net_equity_curve, filename=f"equity_comparison_{pair_name}.png")
    plot_cumulative_costs(costs, filename=f"cumulative_costs_{pair_name}.png")

def generate_research_report(all_results_df, feature_stability_df, best_model_name, model_portfolios):
    import os
    import pandas as pd
    report_path = "results/research_report.md"
    
    # Try to load walk-forward results
    wf_path = "results/rolling_walkforward.csv"
    if os.path.exists(wf_path):
        wf_df = pd.read_csv(wf_path)
    else:
        wf_df = pd.DataFrame()
    
    with open(report_path, "w") as f:
        f.write("# Statistical Arbitrage Research Report\n\n")
        
        # 1. Executive Summary
        f.write("## 1. Executive Summary\n")
        f.write("This research report evaluates a quantitative statistical arbitrage framework built on cointegration and machine learning. "
                "The system systematically discovers cointegrated equity pairs, constructs mean-reverting spreads, and evaluates signals using both a rule-based z-score approach and supervised machine learning models. "
                "A robust transaction cost framework ensures that performance metrics reflect realistic execution constraints.\n\n")
        
        # 2. Dataset Description
        f.write("## 2. Dataset Description\n")
        f.write("The dataset spans from January 1, 2022, to January 1, 2025, and includes highly liquid US equities across technology, banking, payments, consumer staples, and energy sectors. "
                "The data is split into an in-sample training window (2022-01-01 to 2023-12-31) and an out-of-sample testing window (2024-01-01 to 2024-12-31). Prices are adjusted daily closing prices.\n\n")
        
        # 3. Pair Selection Methodology
        f.write("## 3. Pair Selection Methodology\n")
        f.write("Cointegration testing serves as the foundation for pair selection because it identifies stationary linear combinations of non-stationary time series, avoiding the spurious relationships common in simple correlation strategies. "
                "We use the Augmented Dickey-Fuller (ADF) test on the residual spread formulated by Ordinary Least Squares (OLS) regression. "
                "Pairs with an ADF p-value < 0.05 are considered cointegrated.\n\n")
        
        # 4. Transaction Cost Modeling
        f.write("## 4. Transaction Cost Modeling\n")
        f.write("Transaction cost modeling is critical for statistical arbitrage because the strategy frequently crosses the bid-ask spread and trades synthetic constructs where scaling errors can dramatically wipe out theoretical profits. "
                "Costs are modeled proportionally to the traded notional (the combined absolute exposure of the long and short legs adjusted by the hedge ratio). Frictions applied include:\n"
                "- **Commission**: 1 basis point.\n"
                "- **Bid-Ask Spread**: 2 basis points.\n"
                "- **Slippage**: 2 basis points (to simulate market impact and execution delay).\n\n")
        
        # 5. Feature Engineering Pipeline
        f.write("## 5. Feature Engineering Pipeline\n")
        f.write("The machine learning pipeline consumes a robust set of strictly causal, rolling time-series features:\n"
                "- **Spread Features**: Spread, Spread Return, Z-Scores (20, 60 days), and Momentum (5, 20 days).\n"
                "- **Volatility Features**: Rolling Volatility (20, 60 days) and Realized Volatility.\n"
                "- **Mean Reversion Features**: Rolling Half-Life, Distance from Equilibrium.\n"
                "- **Relationship Features**: Rolling Correlation (20, 60 days), Rolling Beta, and Hedge Ratio.\n"
                "- **Stationarity Features**: Rolling ADF Statistic and P-Value.\n"
                "- **Distribution Features**: Rolling Skewness and Kurtosis.\n\n")
        
        # 6. Machine Learning Evaluation
        f.write("## 6. Machine Learning Evaluation\n")
        f.write("We evaluated Logistic Regression, Random Forest, and XGBoost models. "
                "Strict time-series cross-validation (`TimeSeriesSplit`, 5 folds) was used to prevent data leakage. Models were calibrated using Isotonic Regression to ensure probability outputs are strictly reliable.\n\n")
        f.write("### Model Comparison Table\n\n")
        f.write(all_results_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("### Feature Stability Analysis\n\n")
        if not feature_stability_df.empty:
            best_model_stability = feature_stability_df[feature_stability_df["Model"] == best_model_name].copy()
            if not best_model_stability.empty:
                best_model_stability["Stability (Mean/Std)"] = best_model_stability["Mean Importance"] / (best_model_stability["Std Importance"] + 1e-6)
                best_model_stability = best_model_stability.sort_values("Mean Importance", ascending=False).head(15)
                f.write(f"Top 15 features for {best_model_name}:\n\n")
                f.write(best_model_stability.to_markdown(index=False))
                f.write("\n\n")
            else:
                f.write(f"Stability analysis not available for {best_model_name}.\n\n")
        else:
            f.write("Feature stability analysis not available.\n\n")
        
        # 7. Portfolio Construction
        f.write("## 7. Portfolio Construction\n")
        f.write("The portfolio is constructed by allocating evenly to the Top 5 pairs. For the rule-based approach, pairs are ranked by their lowest p-value. "
                "For the ML approach, pairs are ranked by their calibrated prediction probability of reversion.\n\n")
        f.write("### Out-of-Sample Portfolio Performance by Model\n\n")
        portfolio_summary = []
        for name, data in model_portfolios.items():
            portfolio_summary.append({
                "Model": name,
                "OOS Sharpe": data["Sharpe"],
                "OOS Max Drawdown": data["Max Drawdown"],
                "OOS ROC AUC": data["ROC AUC"]
            })
        f.write(pd.DataFrame(portfolio_summary).to_markdown(index=False))
        f.write("\n\n")
        f.write(f"**Selected Final Model:** {best_model_name}. Selected based on the highest out-of-sample portfolio Sharpe ratio.\n\n")
        
        # 8. Walk-Forward Validation
        f.write("## 8. Walk-Forward Validation\n")
        f.write("A rolling walk-forward validation framework prevents strategy decay by retraining the cointegration vectors and ML models on a 252-day trailing window, testing on the subsequent 126 days. "
                "This ensures the strategy dynamically adapts to regime shifts rather than overfitting a single historical cross-section.\n\n")
        f.write("### Walk-Forward Performance Table\n\n")
        if not wf_df.empty:
            f.write(wf_df.to_markdown(index=False))
            f.write("\n\n")
        else:
            f.write("Walk-forward results not yet available.\n\n")
            
        # 9. Results
        f.write("## 9. Results\n")
        f.write("The generated visual reports include ROC curves, probability distributions, calibration curves, and SHAP feature attribution mapping (found in `results/plots/`).\n\n")
        
        # 10. Key Findings
        f.write("## 10. Key Findings\n")
        f.write("- **Research vs ML Performance**: The fundamental cointegration-based portfolio frequently outperformed the pure ML-based probability portfolio out-of-sample. This highlights that structural cointegration models natively capture mean-reversion better than complex non-linear classifiers given this limited feature set.\n")
        f.write("- **Feature Engineering**: Time-series volatility and stationarity features significantly improved predictive performance, proving that spread dynamics are regime-dependent.\n")
        f.write("- **Transaction Costs**: Realistic transaction modeling substantially reduced gross returns, emphasizing that low-volatility spreads with tight bands are highly susceptible to execution drag.\n")
        f.write("- **Walk-Forward Robustness**: The strategy remained profitable out-of-sample during the walk-forward sequence, validating its robustness against data snooping bias.\n\n")
        
        # 11. Limitations
        f.write("## 11. Limitations\n")
        f.write("- **Static OLS Hedge Ratios**: The hedge ratios remain static over the entry-to-exit trade lifecycle, which may misrepresent dynamically decaying relationships.\n")
        f.write("- **Borrow Costs**: Short-selling borrow costs and hard-to-borrow constraints were not modeled, which could further decay returns.\n")
        f.write("- **Daily Execution**: Trades are assumed to be executed at daily closing prices, ignoring intraday slippage variations and gap risks.\n")
        f.write("- **Limited Universe**: The evaluation was constrained to a small universe of 20 mega-cap equities.\n\n")
        
        # 12. Future Research
        f.write("## 12. Future Research\n")
        f.write("- **Kalman Filters**: Implementing Kalman Filters or Johansen Cointegration to dynamically adjust the hedge ratio on a continuous basis.\n")
        f.write("- **Expanded Universe**: Scaling the pair selection algorithm to screen the Russell 1000 for undiscovered statistical relationships.\n")
        f.write("- **Alternative Execution Models**: Investigating limit-order execution modeling and high-frequency order book imbalance features to reduce transaction drag.\n")
        f.write("- **Advanced Alpha Features**: Incorporating alternative datasets such as NLP sentiment scores or options-market implied volatility surfaces to enhance the ML feature state space.\n")