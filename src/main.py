from data_loader import load_prices
from cointegration import find_cointegrated_pairs
from research_engine import backtest_pair, backtest_all_pairs
from ranking import rank_pairs
import pandas as pd 
from feature_engineering import build_feature_dataset, add_test_target
from ml_model import train_model 
from plots import plot_feature_importance
from reporting import generate_pair_report
from portfolio import build_portfolio_returns
from performance import calculate_sharpe_ratio, calculate_drawdown

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
    prices = load_prices(
        tickers=TICKERS,
        start_date="2022-01-01",
        end_date="2025-01-01"
    )

    train_prices = prices.loc["2022-01-01":"2023-12-31"]
    test_prices = prices.loc["2024-01-01":"2024-12-31"]

    pairs = find_cointegrated_pairs(train_prices)
    print(f"Number of cointegrated pairs found: {len(pairs)}")
    pairs_df = pd.DataFrame(pairs, columns=["Stock1", "Stock2", "PValue"])
    pairs_df.to_csv("results/cointegrated_pairs.csv", index = False)
    
    results_train = backtest_all_pairs(pairs,train_prices)
    ranked_train = rank_pairs(results_train)
    print("\nTop Research pairs:")
    print(ranked_train.head(10))
    ranked_train.to_csv("results/ranked_pairs.csv", index=False)

    portfolio_returns = build_portfolio_returns(ranked_train,train_prices,top_n=5)
    portfolio_equity = (100 + portfolio_returns.cumsum())
    portfolio_sharpe = calculate_sharpe_ratio(portfolio_returns)
    portfolio_drawdown, portfolio_max_dd = (calculate_drawdown(portfolio_equity))
    print("\nPortfolio Results")
    print(f"Portfolio Sharpe: {portfolio_sharpe:.4f}")
    print(f"Portfolio Max Drawdown: {portfolio_max_dd:.4f}")
    portfolio_returns.to_csv("results/portfolio_returns.csv")
    pd.DataFrame({"Portfolio Equity": portfolio_equity}).to_csv("results/portfolio_equity.csv")

    top_pairs = ranked_train.head(5)
    test_results = []
    for pair in top_pairs["Pair"]:
        stock1, stock2 = pair.split("-")
        result = backtest_pair(stock1, stock2, test_prices)
        test_results.append(result)
    test_results = pd.DataFrame(test_results)
    print("\nOut-of-Sample Results:")
    print(test_results)
    test_results.to_csv("results/walkforward_results.csv", index=False)

    features_df = build_feature_dataset(pairs, train_prices)
    print(features_df.head())

    ml_dataset = add_test_target(features_df, test_prices)
    print(ml_dataset.head())
    ml_dataset.to_csv("results/ml_dataset.csv", index=False)

    model, score, importance_df = train_model(ml_dataset)
    print("\nModel R2:")
    print(score)
    importance_df.to_csv("results/feature_importance.csv", index=False)


    for pair in ranked_train.head(3)["Pair"]:
        stock1, stock2 = pair.split("-")
        print(f"\n Generating Report for: {pair}")
        generate_pair_report(stock1, stock2, prices)
        
    plot_feature_importance(importance_df)

    print("\nResults saved to results/")
    print("Plots saved to results/plots/")



if __name__ == "__main__": # Only run main() when this file is executed directly.
    main()

    

