def rank_pairs(results_df):
    ranked_df = results_df.copy()
    ranked_df["Sharpe Rank"] = ranked_df["Sharpe (After Costs)"].rank(ascending = False)
    ranked_df["Drawdown Rank"] = ranked_df["Max Drawdown (After Costs)"].abs().rank(ascending=True)
    ranked_df["Half-Life Rank"] = ranked_df["Half Life"].rank(ascending = True)
    ranked_df["Trades Rank"] = ranked_df["Trades"].rank(ascending = False)
    ranked_df["Research Score"] = ranked_df["Sharpe Rank"] + ranked_df["Drawdown Rank"] + ranked_df["Half-Life Rank"] + ranked_df["Trades Rank"]
    ranked_df = ranked_df.sort_values("Research Score")
    return ranked_df