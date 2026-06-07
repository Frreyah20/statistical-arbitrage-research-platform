import pandas as pd


def create_experiment_summary(
    research_sharpe,
    research_dd,
    ml_sharpe,
    ml_dd,
):
    summary = pd.DataFrame(
        {
            "Strategy": [
                "Research Portfolio",
                "ML Portfolio"
            ],
            "Sharpe": [
                research_sharpe,
                ml_sharpe
            ],
            "Max Drawdown": [
                research_dd,
                ml_dd
            ]
        }
    )

    summary.to_csv(
        "results/project_summary.csv",
        index=False
    )

    return summary