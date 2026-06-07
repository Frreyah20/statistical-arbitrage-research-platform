import matplotlib.pyplot as plt

def plot_equity_curve(equity_curve):
    plt.figure(figsize = (12, 6))
    plt.plot(equity_curve)
    plt.title("StrategyEquity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig("results/plots/equity_curve.png")
    plt.close()

def plot_drawdown(drawdown):
    plt.figure(figsize = (12, 6))
    plt.plot(drawdown)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.savefig("results/plots/drawdown.png")
    plt.close()
    
def plot_spread_series(spread):
    plt.figure(figsize=(12,6))
    plt.plot(spread)
    plt.axhline(spread.mean(), linestyle = "--")
    plt.title("Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.savefig("results/plots/spread_series.png")
    plt.close()

def plot_feature_importance(importance_df):
    plt.figure(figsize = (10,6))
    plt.bar(importance_df["Feature"], importance_df["Importance"])
    plt.xticks(rotation = 45)
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig("results/plots/feature_importance.png")
    plt.close()