from data_loader import load_prices
from cointegration import find_cointegrated_pairs
from research_engine import backtest_pair, backtest_all_pairs
from ranking import rank_pairs
import pandas as pd 
from feature_engineering import build_complete_feature_dataset
from ml_model import train_model , predict_pairs
from plots import plot_feature_importance
from reporting import generate_pair_report
from portfolio import build_portfolio_returns
from performance import calculate_sharpe_ratio, calculate_drawdown
from walkforward import rolling_walkforward
from optimization import optimize_parameters
from dashboard import create_experiment_summary
from plots import plot_equity_curve, plot_drawdown

TICKERS = [
    #technology
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    #banks
    "JPM",
    "BAC",
    "GS",
    "WFC",
    #payment companies
    "V",
    "MA",
    #beverages
    "KO",
    "PEP",
    #energy
    "XOM",
    "CVX",
    "HD",
    "LOW",
    "DIS",
    "NFLX"
]

def main():

    #Load Data

    prices = load_prices(
        tickers=TICKERS,
        start_date="2022-01-01",
        end_date="2025-01-01"
    )

    train_prices = prices.loc["2022-01-01":"2023-12-31"]
    test_prices = prices.loc["2024-01-01":"2024-12-31"]

    #Cointegration Search

    pairs = find_cointegrated_pairs(train_prices)
    print(f"Number of cointegrated pairs found: {len(pairs)}")
    pairs_df = pd.DataFrame(pairs, columns=["Stock1", "Stock2", "PValue"])
    pairs_df.to_csv("results/cointegrated_pairs.csv", index = False)
    
    #Pair Backtesting + ranking

    results_train = backtest_all_pairs(pairs,train_prices)
    ranked_train = rank_pairs(results_train)
    print("\nTop Research pairs:")
    print(ranked_train.head(10))
    ranked_train.to_csv("results/ranked_pairs.csv", index=False)

    # ML Dataset
    features_df = build_complete_feature_dataset(pairs, prices)
    features_df.to_csv("results/features.csv", index=False)
    
    train_data = features_df[features_df["date"] <= "2023-12-31"]
    test_data = features_df[features_df["date"] >= "2024-01-01"]
    
    train_data.to_csv("results/ml_train_dataset.csv", index=False)
    test_data.to_csv("results/ml_test_dataset.csv", index=False)

    # ML Model
    model, score, importance_df = train_model(train_data, test_data)
    print("\nModel ROC-AUC / Accuracy:")
    print(score)
    importance_df.to_csv("results/feature_importance.csv", index=False)

    # ML Ranking
    predictions_df = predict_pairs(model, test_data)
    # Aggregate predictions to get a single score per pair
    pair_scores = predictions_df.groupby("pair")["Predicted_Probability"].mean().reset_index()
    pair_scores.rename(columns={"pair": "Pair"}, inplace=True)
    predicted_pairs = pair_scores.sort_values("Predicted_Probability", ascending=False)
    
    print("\nTop ML Prediction Pairs: ")
    print(predicted_pairs.head(10))
    
    # --- Advanced ML Evaluation & Reporting (Non-Intrusive) ---
    import os
    if os.path.exists("results/test_probabilities.csv"):
        from plots import plot_roc_curves, plot_probability_distributions, plot_calibration_curves
        from reporting import generate_research_report
        
        test_probs_df = pd.read_csv("results/test_probabilities.csv")
        y_test_df = pd.read_csv("results/y_test.csv")["target"]
        test_probs_dict = {col: test_probs_df[col].values for col in test_probs_df.columns}
        
        plot_roc_curves(y_test_df, test_probs_dict)
        plot_probability_distributions(test_probs_dict)
        plot_calibration_curves(y_test_df, test_probs_dict)
        
        model_portfolios = {}
        all_results_df = pd.read_csv("results/model_results.csv")
        
        for model_name in test_probs_df.columns:
            temp_df = test_data.copy()
            temp_df["Predicted_Probability"] = test_probs_df[model_name].values
            p_scores = temp_df.groupby("pair")["Predicted_Probability"].mean().reset_index()
            p_scores.rename(columns={"pair": "Pair"}, inplace=True)
            p_scores = p_scores.sort_values("Predicted_Probability", ascending=False)
            
            try:
                m_returns = build_portfolio_returns(p_scores, test_prices, top_n=5)
                m_equity = (100 + m_returns.cumsum())
                m_sharpe = calculate_sharpe_ratio(m_returns)
                _, m_max_drawdown = calculate_drawdown(m_equity)
            except:
                m_sharpe = -999
                m_max_drawdown = -999
                
            model_roc_auc = all_results_df[(all_results_df["Model"] == model_name) & (all_results_df["Type"] == "Out-of-Sample Test")]["ROC AUC"].values[0]
            
            model_portfolios[model_name] = {
                "Sharpe": m_sharpe,
                "Max Drawdown": m_max_drawdown,
                "ROC AUC": model_roc_auc,
                "Predicted Pairs": p_scores
            }
            
        best_model_name = sorted(model_portfolios.keys(), key=lambda k: (model_portfolios[k]["Sharpe"], model_portfolios[k]["Max Drawdown"], model_portfolios[k]["ROC AUC"]), reverse=True)[0]
        print(f"\n[Advanced Eval] Best Portfolio Model: {best_model_name} (OOS Sharpe: {model_portfolios[best_model_name]['Sharpe']:.4f})")
        
        # Override predicted_pairs for downstream pipeline compatibility
        predicted_pairs = model_portfolios[best_model_name]["Predicted Pairs"]
    # ------------------------------------------------------------
    
    predicted_pairs.to_csv("results/predicted_pairs.csv", index=False)

    #Research Porfolio
    portfolio_returns = build_portfolio_returns(ranked_train,train_prices,top_n=5)
    portfolio_equity = (100 + portfolio_returns.cumsum())
    portfolio_drawdown, portfolio_max_dd = (calculate_drawdown(portfolio_equity))
    plot_equity_curve(portfolio_equity, "research_portfolio_equity.png")
    plot_drawdown(portfolio_drawdown, "research_portfolio_drawdown.png")
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
    print("\nPortfolio Results")
    print(f"Portfolio Sharpe: {portfolio_sharpe:.4f}")
    print(f"Portfolio Max Drawdown: {portfolio_max_dd:.4f}")
    portfolio_returns.to_csv("results/portfolio_returns.csv")
    pd.DataFrame({"Portfolio Equity": portfolio_equity}).to_csv("results/portfolio_equity.csv")
    print("Portfolio Peak:", portfolio_equity.max())
    print("Portfolio Trough:", portfolio_equity.min())
    print("Portfolio Max DD:", portfolio_max_dd)
 

    #ML Portfolio
    ml_returns = build_portfolio_returns(predicted_pairs, train_prices, top_n = 5)
    ml_equity = (100 + ml_returns.cumsum())
    ml_sharpe = calculate_sharpe_ratio(ml_returns)
    _, ml_max_drawdown = calculate_drawdown(ml_equity)
    print("\nML Portfolio")
    print(f"Sharpe: {ml_sharpe:.4f}")
    print(f"Max Drawdown: {ml_max_drawdown:.4f}")

    #Portfolio Comparison
    print("\nPortfolio Comparison") 
    print(f"Research Sharpe: {portfolio_sharpe:.4f}")
    print(f"ML Sharpe: {ml_sharpe:.4f}")

    #Experiment Summary
    summary = create_experiment_summary(
        portfolio_sharpe,
        portfolio_max_dd,
        ml_sharpe,
        ml_max_drawdown
    )
    print("\nExperiment Summary:")
    print(summary)

    #Rolling Walk Forward
    rolling_results = rolling_walkforward(prices)
    print("\nRolling Walk-Forward Results:")
    print(rolling_results)
    rolling_results.to_csv("results/rolling_walkforward.csv",index=False)   
    print("\nRolling Walk-Forward Summary")
    print("Average Test Sharpe:", rolling_results["Sharpe (After Costs)"].mean())
    print("Best Test Sharpe:",rolling_results["Sharpe (After Costs)"].max())
    print("Worst Test Sharpe:",rolling_results["Sharpe (After Costs)"].min())
    
    #Parameter Optimization
    optimization_results = optimize_parameters("HD","WFC",train_prices)
    print("\nParameter Optimization Results:")
    print(optimization_results.head(10))
    optimization_results.to_csv("results/parameter_optimization.csv",index=False)

    #Automated Reports
    for pair in ranked_train.head(3)["Pair"]:
        stock1, stock2 = pair.split("-")
        print(f"\n Generating Report for: {pair}")
        generate_pair_report(stock1, stock2, prices)
        
        
    #Plots
    plot_feature_importance(importance_df)
    
    # Generate Research Report
    if os.path.exists("results/test_probabilities.csv"):
        feature_stability_df = pd.read_csv("results/feature_stability.csv") if os.path.exists("results/feature_stability.csv") else pd.DataFrame()
        generate_research_report(all_results_df, feature_stability_df, best_model_name, model_portfolios)

    print("\nResults saved to results/")
    print("Plots saved to results/plots/")
    

if __name__ == "__main__": # Only run main() when this file is executed directly.
    main()

    

