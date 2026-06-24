from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd
import numpy as np

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

def evaluate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['Accuracy'] = accuracy_score(y_true, y_pred)
    metrics['Precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['Recall'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['F1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics['ROC AUC'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['ROC AUC'] = np.nan
    return metrics

def train_model(train_data, test_data):
    # Drop non-feature columns
    drop_cols = ["pair", "date", "target"]
    x_train = train_data.drop(columns=[col for col in drop_cols if col in train_data.columns]).fillna(0)
    y_train = train_data["target"]
    
    x_test = test_data.drop(columns=[col for col in drop_cols if col in test_data.columns]).fillna(0)
    y_test = test_data["target"]

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=42)),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, eval_metric='logloss')

    tscv = TimeSeriesSplit(n_splits=5)
    
    cv_results = []
    feature_importances_folds = {name: [] for name in models if name != "Logistic Regression"}
    calibrated_models = {}
    test_metrics = []
    
    print("\n--- Starting Time-Series Cross-Validation ---")
    
    for name, base_model in models.items():
        fold_metrics = []
        for train_idx, val_idx in tscv.split(x_train):
            X_fold_train, X_fold_val = x_train.iloc[train_idx], x_train.iloc[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            base_model.fit(X_fold_train, y_fold_train)
            if hasattr(base_model, "feature_importances_"):
                feature_importances_folds[name].append(base_model.feature_importances_)
                
            calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method='isotonic')
            calibrated.fit(X_fold_val, y_fold_val)
            
            y_fold_pred = calibrated.predict(X_fold_val)
            y_fold_prob = calibrated.predict_proba(X_fold_val)[:, 1]
            fold_metrics.append(evaluate_metrics(y_fold_val, y_fold_pred, y_fold_prob))

        avg_cv_metrics = pd.DataFrame(fold_metrics).mean().to_dict()
        avg_cv_metrics["Model"] = name
        avg_cv_metrics["Type"] = "CV Validation"
        cv_results.append(avg_cv_metrics)
        
        train_split_idx = int(len(x_train) * 0.8)
        X_train_sub, X_val_sub = x_train.iloc[:train_split_idx], x_train.iloc[train_split_idx:]
        y_train_sub, y_val_sub = y_train.iloc[:train_split_idx], y_train.iloc[train_split_idx:]
        
        base_model.fit(X_train_sub, y_train_sub)
        final_calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method='isotonic')
        final_calibrated.fit(X_val_sub, y_val_sub)
        calibrated_models[name] = final_calibrated
        
        y_test_pred = final_calibrated.predict(x_test)
        y_test_prob = final_calibrated.predict_proba(x_test)[:, 1]
        
        t_metrics = evaluate_metrics(y_test, y_test_pred, y_test_prob)
        t_metrics["Model"] = name
        t_metrics["Type"] = "Out-of-Sample Test"
        test_metrics.append(t_metrics)

    all_results_df = pd.concat([pd.DataFrame(cv_results), pd.DataFrame(test_metrics)], ignore_index=True)
    all_results_df.to_csv("results/model_results.csv", index=False)
    
    stability_dfs = []
    for name, importances in feature_importances_folds.items():
        if len(importances) == 0: continue
        imp_array = np.array(importances)
        stab_df = pd.DataFrame({
            "Model": name,
            "Feature": x_train.columns,
            "Mean Importance": imp_array.mean(axis=0),
            "Std Importance": imp_array.std(axis=0)
        })
        stability_dfs.append(stab_df)
    
    if stability_dfs:
        feature_stability_df = pd.concat(stability_dfs, ignore_index=True)
        feature_stability_df.to_csv("results/feature_stability.csv", index=False)

    best_model_row = all_results_df[all_results_df["Type"] == "Out-of-Sample Test"].sort_values("ROC AUC", ascending=False).iloc[0]
    best_model_name = best_model_row["Model"]
    best_score = best_model_row["ROC AUC"]
    best_model = calibrated_models[best_model_name]
    best_model.model_name_ = best_model_name # Tag model for SHAP later
    
    test_probs = {}
    for name, model in calibrated_models.items():
        test_probs[name] = model.predict_proba(x_test)[:, 1]
    pd.DataFrame(test_probs).to_csv("results/test_probabilities.csv", index=False)
    y_test.to_csv("results/y_test.csv", index=False)

    try:
        import shap
        HAS_SHAP = True
    except ImportError:
        HAS_SHAP = False
        
    if HAS_SHAP and best_model_name in ["Random Forest", "XGBoost"]:
        try:
            base_estimator = best_model.calibrated_classifiers_[0].estimator
            if hasattr(base_estimator, "estimator"):
                actual_estimator = base_estimator.estimator
            else:
                actual_estimator = base_estimator
            explainer = shap.TreeExplainer(actual_estimator)
            shap_values = explainer.shap_values(x_test)
            from plots import plot_shap_summary_plot
            plot_shap_summary_plot(shap_values, x_test, "shap_summary.png")
        except Exception as e:
            print(f"SHAP Error: {e}")

    if best_model_name in feature_importances_folds and len(feature_importances_folds[best_model_name]) > 0:
        base_estimator = best_model.calibrated_classifiers_[0].estimator
        if hasattr(base_estimator, "estimator"):
            actual_estimator = base_estimator.estimator
        else:
            actual_estimator = base_estimator
        importance_df = pd.DataFrame({"Feature": x_train.columns, "Importance": actual_estimator.feature_importances_})
    else:
        importance_df = pd.DataFrame({"Feature": x_train.columns, "Importance": np.zeros(len(x_train.columns))})
    
    importance_df = importance_df.sort_values("Importance", ascending=False)
    
    return best_model, best_score, importance_df

def predict_pairs(model, features_df):
    drop_cols = ["pair", "date", "target"]
    X = features_df.drop(columns=[col for col in drop_cols if col in features_df.columns]).fillna(0)
    
    predictions = model.predict_proba(X)[:, 1]
        
    if "pair" in features_df.columns and "date" in features_df.columns:
        results = features_df[["pair", "date"]].copy()
    else:
        results = features_df.copy()
        
    results["Predicted_Probability"] = predictions
    return results.sort_values("Predicted_Probability", ascending=False)