import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

def plot_equity_curve(equity_curve, filename="equity_curve.png"):
    plt.figure(figsize = (12, 6))
    plt.plot(equity_curve)
    plt.title("StrategyEquity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_cost_comparison_equity(gross_equity, net_equity, filename="cost_comparison_equity.png"):
    plt.figure(figsize = (12, 6))
    plt.plot(gross_equity, label="Before Costs", linestyle="--")
    plt.plot(net_equity, label="After Costs")
    plt.title("Equity Curve: Before vs After Transaction Costs")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_cumulative_costs(costs_series, filename="cumulative_costs.png"):
    plt.figure(figsize = (12, 6))
    plt.plot(costs_series.cumsum(), color='red')
    plt.title("Cumulative Transaction Costs")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Cost")
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_drawdown(drawdown, filename="drawdown.png"):
    plt.figure(figsize = (12, 6))
    plt.plot(drawdown)
    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()
    
def plot_spread_series(spread, filename="spread_series.png"):
    plt.figure(figsize=(12,6))
    plt.plot(spread)
    plt.axhline(spread.mean(), linestyle = "--")
    plt.title("Spread")
    plt.xlabel("Date")
    plt.ylabel("Spread")
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_feature_importance(importance_df, filename="feature_importance.png"):
    plt.figure(figsize = (10,6))
    plt.bar(importance_df["Feature"].head(15), importance_df["Importance"].head(15))
    plt.xticks(rotation = 45, ha='right')
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_roc_curves(y_true, test_probs_dict, filename="roc_curves.png"):
    plt.figure(figsize=(10, 8))
    for model_name, y_prob in test_probs_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_probability_distributions(test_probs_dict, filename="prob_distributions.png"):
    plt.figure(figsize=(10, 6))
    for model_name, y_prob in test_probs_dict.items():
        sns.kdeplot(y_prob, label=model_name, fill=True)
    
    plt.title('Prediction Probability Distribution')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.xlim([0, 1])
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_calibration_curves(y_true, test_probs_dict, filename="calibration_curves.png"):
    plt.figure(figsize=(10, 8))
    for model_name, y_prob in test_probs_dict.items():
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o', label=model_name)
        
    plt.plot([0, 1], [0, 1], 'k--', label="Perfectly calibrated")
    plt.title('Calibration Curves')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"results/plots/{filename}")
    plt.close()

def plot_shap_summary_plot(shap_values, X, filename="shap_summary.png"):
    if not HAS_SHAP or shap_values is None:
        return
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(f"results/plots/{filename}")
    plt.close()