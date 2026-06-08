from data_loader import load_prices
from cointegration import find_cointegrated_pairs
from research_engine import backtest_pair, backtest_all_pairs
from ranking import rank_pairs
import pandas as pd 
from feature_engineering import build_feature_dataset, add_test_target
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

    #ML Dataset
    features_df = build_feature_dataset(pairs, train_prices)
    print(features_df.head())
    ml_dataset = add_test_target(features_df, test_prices)
    print(ml_dataset.head())
    ml_dataset.to_csv("results/ml_dataset.csv", index=False)

    #ML Model
    model, score, importance_df = train_model(ml_dataset)
    print("\nModel R2:")
    print(score)
    importance_df.to_csv("results/feature_importance.csv", index=False)

    #ML Ranking
    predicted_pairs = predict_pairs(model, features_df)
    print("\nTop ML Prediction Pairs: ")
    print(predicted_pairs[["Pair", "Predicted Sharpe"]].head(10))
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
    print("Average Test Sharpe:", rolling_results["Sharpe"].mean())
    print("Best Test Sharpe:",rolling_results["Sharpe"].max())
    print("Worst Test Sharpe:",rolling_results["Sharpe"].min())
    
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
    print("\nResults saved to results/")
    print("Plots saved to results/plots/")
    

if __name__ == "__main__": # Only run main() when this file is executed directly.
    main()

    

