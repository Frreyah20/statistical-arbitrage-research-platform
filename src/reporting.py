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
    report_path = "results/research_report.md"
    
    with open(report_path, "w") as f:
        f.write("# Machine Learning Research Report\n\n")
        
        f.write("## 1. Model Evaluation Metrics\n\n")
        f.write(all_results_df.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("## 2. Portfolio Performance by Model (Out-of-Sample)\n\n")
        portfolio_summary = []
        for name, data in model_portfolios.items():
            portfolio_summary.append({
                "Model": name,
                "OOS Sharpe": data["Sharpe"],
                "OOS Max Drawdown": data["Max Drawdown"],
                "OOS ROC AUC": data["ROC AUC"]
            })
        import pandas as pd
        f.write(pd.DataFrame(portfolio_summary).to_markdown(index=False))
        f.write("\n\n")
        
        f.write(f"## 3. Selected Final Model: {best_model_name}\n\n")
        f.write(f"The `{best_model_name}` was selected based on the highest out-of-sample portfolio Sharpe ratio, while also considering drawdown and ROC AUC.\n\n")
        
        f.write("## 4. Feature Stability Analysis\n\n")
        if not feature_stability_df.empty:
            # Show top 10 most stable features for the best model (highest mean/std ratio or just highest mean)
            best_model_stability = feature_stability_df[feature_stability_df["Model"] == best_model_name].copy()
            if not best_model_stability.empty:
                best_model_stability["Stability (Mean/Std)"] = best_model_stability["Mean Importance"] / (best_model_stability["Std Importance"] + 1e-6)
                best_model_stability = best_model_stability.sort_values("Mean Importance", ascending=False).head(15)
                f.write(f"Top 15 features for {best_model_name}:\n\n")
                f.write(best_model_stability.to_markdown(index=False))
            else:
                f.write(f"Stability analysis not available for {best_model_name}.\n")
        else:
            f.write("Feature stability analysis not available.\n")
        
        f.write("\n\n## 5. Visualizations Generated\n")
        f.write("- `results/plots/roc_curves.png`\n")
        f.write("- `results/plots/prob_distributions.png`\n")
        f.write("- `results/plots/calibration_curves.png`\n")
        f.write("- `results/plots/shap_summary.png` (for tree models)\n")