"""
app.py — Telco Customer Churn Pipeline
Based on EDA.ipynb

Functions:
    load_and_preprocess(filepath)   → Cleaned DataFrame
    encode_features(df)             → Encoded DataFrame
    fix_multicollinearity(df)       → VIF-cleaned DataFrame
    compute_vif(df)                 → VIF scores DataFrame
    train_models(df, threshold)     → Trained models dict + split data
    tune_xgboost(X_train, X_test, y_train, y_test, threshold, n_trials)  → best_params, study
    train_final_xgboost(X_train, X_test, y_train, y_test, best_params, threshold) → model, metrics
    log_to_mlflow(model, best_params, metrics)  → MLflow run
    run_pipeline(filepath)          → End-to-end helper
"""

import pandas as pd
import numpy as np
import time
import os
import joblib
import warnings

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────
# 1. LOAD & BASIC CLEANING
# ─────────────────────────────────────────────────────────────

def load_and_preprocess(filepath: str) -> pd.DataFrame:
    """
    Load the Telco CSV, cast TotalCharges to numeric,
    drop customerID, and convert bool columns to int.

    Parameters
    ----------
    filepath : str
        Path to Telco-Customer-Churn.csv

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame (no encoding yet).
    """
    df = pd.read_csv(filepath)

    # Fix TotalCharges (whitespace strings → NaN)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop ID column — not a feature
    df = df.drop("customerID", axis=1)

    # bool → int (0/1) if any exist
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[load_and_preprocess] Shape: {df.shape}  |  Missing: {df.isnull().sum().sum()}")
    return df


# ─────────────────────────────────────────────────────────────
# 2. ENCODING
# ─────────────────────────────────────────────────────────────

def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Binary-encode 2-category columns and one-hot encode
    multi-category columns.

    Parameters
    ----------
    df : pd.DataFrame
        Output of load_and_preprocess().

    Returns
    -------
    pd.DataFrame
        Fully encoded DataFrame.
    """
    df = df.copy()

    # --- Binary encoding ---
    binary_cols = [
        "gender", "Partner", "Dependents",
        "PhoneService", "PaperlessBilling",
    ]
    if "Churn" in df.columns:
        binary_cols.append("Churn")

    df[binary_cols] = df[binary_cols].replace(
        {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    )

    # --- One-hot encoding ---
    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # bool → int (produced by get_dummies in older pandas)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    print(f"[encode_features] Shape after encoding: {df.shape}")
    return df


# ─────────────────────────────────────────────────────────────
# 3. MULTICOLLINEARITY FIX
# ─────────────────────────────────────────────────────────────

def fix_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the redundant 'No internet service' and
    'No phone service' dummy columns into single binary flags.

    Parameters
    ----------
    df : pd.DataFrame
        Output of encode_features().

    Returns
    -------
    pd.DataFrame
        DataFrame with multicollinearity reduced.
    """
    df = df.copy()

    # Collapse all "No internet service" dummies into one flag
    no_inet_cols = [c for c in df.columns if "No internet service" in c]
    if no_inet_cols:
        df["No_internet_service"] = df[no_inet_cols].any(axis=1).astype(int)
        df = df.drop(columns=no_inet_cols)

    # Collapse "No phone service" dummy
    if "MultipleLines_No phone service" in df.columns:
        df["No_phone_service"] = df["MultipleLines_No phone service"].astype(int)
        df = df.drop(columns=["MultipleLines_No phone service"])

    print(f"[fix_multicollinearity] Shape after VIF fix: {df.shape}")
    return df


def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor for all features
    (excludes the Churn target column).

    Parameters
    ----------
    df : pd.DataFrame
        Encoded + multicollinearity-fixed DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ['feature', 'VIF'] sorted descending.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    X = df.drop(columns=["Churn"], errors="ignore")
    X = X.replace([np.inf, -np.inf], np.nan).dropna()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i) for i in range(X.shape[1])
    ]
    return vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# 4. CORRELATION HEATMAP
# ─────────────────────────────────────────────────────────────

def plot_churn_correlation(df: pd.DataFrame, save_path: str = None):
    """
    Plot a heatmap of feature correlations with the Churn column.

    Parameters
    ----------
    df  : Encoded DataFrame containing a 'Churn' column.
    save_path : Optional file path to save the figure (e.g. 'corr.png').
    """
    import seaborn as sns
    import matplotlib.pyplot as plt

    corr_matrix = df.corr(numeric_only=True)
    churn_corr = corr_matrix[["Churn"]].sort_values(by="Churn", ascending=False)

    plt.figure(figsize=(4, 12))
    sns.heatmap(churn_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Correlation of Features with Churn")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"[plot_churn_correlation] Saved to {save_path}")
    else:
        plt.show()


# ─────────────────────────────────────────────────────────────
# 5. TRAIN / EVALUATE BASELINE MODELS
# ─────────────────────────────────────────────────────────────

def train_models(
    df: pd.DataFrame,
    threshold: float = 0.30,
    test_size: float = 0.20,
    random_state: int = 42,
) -> dict:
    """
    Train RandomForest, LightGBM, and XGBoost classifiers
    and print classification reports for each.

    Parameters
    ----------
    df           : Fully preprocessed DataFrame (with Churn column).
    threshold    : Probability threshold for positive class (default 0.30).
    test_size    : Fraction of data for testing.
    random_state : Reproducibility seed.

    Returns
    -------
    dict with keys:
        'rf'       → fitted RandomForestClassifier
        'lgbm'     → fitted LGBMClassifier
        'xgb'      → fitted XGBClassifier
        'X_train', 'X_test', 'y_train', 'y_test'
    """
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import classification_report

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    results = {"X_train": X_train, "X_test": X_test,
               "y_train": y_train, "y_test": y_test}

    # ── RandomForest ──
    print("\n── RandomForest ──")
    rf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced",
        random_state=random_state, n_jobs=-1
    )
    t0 = time.time()
    rf.fit(X_train, y_train)
    print(f"  Train: {time.time()-t0:.1f}s")
    proba = rf.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    results["rf"] = rf

    # ── LightGBM ──
    print("\n── LightGBM ──")
    lgbm = LGBMClassifier(
        n_estimators=500, learning_rate=0.05, class_weight="balanced",
        random_state=random_state, n_jobs=-1
    )
    t0 = time.time()
    lgbm.fit(X_train, y_train)
    print(f"  Train: {time.time()-t0:.1f}s")
    proba = lgbm.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    results["lgbm"] = lgbm

    # ── XGBoost ──
    print("\n── XGBoost ──")
    xgb = XGBClassifier(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, eval_metric="logloss",
        random_state=random_state, n_jobs=-1
    )
    t0 = time.time()
    xgb.fit(X_train, y_train)
    print(f"  Train: {time.time()-t0:.1f}s")
    proba = xgb.predict_proba(X_test)[:, 1]
    y_pred = (proba >= threshold).astype(int)
    print(classification_report(y_test, y_pred, digits=3))
    results["xgb"] = xgb

    return results


# ─────────────────────────────────────────────────────────────
# 6. HYPERPARAMETER TUNING (OPTUNA)
# ─────────────────────────────────────────────────────────────

def tune_xgboost(
    X_train, X_test, y_train, y_test,
    threshold: float = 0.30,
    n_trials: int = 30,
):
    """
    Run Optuna hyperparameter search on XGBoost optimising recall
    for the churn (positive) class.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : Split data arrays/DataFrames.
    threshold  : Probability cut-off for positive-class prediction.
    n_trials   : Number of Optuna trials (default 30).

    Returns
    -------
    best_params : dict   — best hyperparameters found
    study       : optuna.Study object (inspect with study.trials_dataframe())
    """
    import optuna
    from xgboost import XGBClassifier
    from sklearn.metrics import recall_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 300, 800),
            "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":  trial.suggest_int("min_child_weight", 1, 10),
            "gamma":             trial.suggest_float("gamma", 0, 5),
            "reg_alpha":         trial.suggest_float("reg_alpha", 0, 5),
            "reg_lambda":        trial.suggest_float("reg_lambda", 0, 5),
            "random_state":      42,
            "n_jobs":            -1,
            "scale_pos_weight":  scale_pos_weight,
            "eval_metric":       "logloss",
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        y_pred = (proba >= threshold).astype(int)
        return recall_score(y_test, y_pred, pos_label=1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"[tune_xgboost] Best recall: {study.best_value:.3f}")
    print(f"[tune_xgboost] Best params: {study.best_params}")
    return study.best_params, study


# ─────────────────────────────────────────────────────────────
# 7. TRAIN FINAL XGBOOST (TUNED)
# ─────────────────────────────────────────────────────────────

def train_final_xgboost(
    X_train, X_test, y_train, y_test,
    best_params: dict,
    threshold: float = 0.30,
):
    """
    Train the final XGBoost model with tuned hyperparameters
    and return the model + evaluation metrics.

    Parameters
    ----------
    X_train, X_test, y_train, y_test : Split data.
    best_params : dict from tune_xgboost().
    threshold   : Probability cut-off (default 0.30).

    Returns
    -------
    model   : Fitted XGBClassifier
    metrics : dict with precision, recall, f1, roc_auc, train_time, pred_time
    """
    from xgboost import XGBClassifier
    from sklearn.metrics import (
        classification_report, precision_score,
        recall_score, f1_score, roc_auc_score,
    )

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    params = {
        **best_params,
        "random_state":     42,
        "n_jobs":           -1,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric":      "logloss",
    }

    model = XGBClassifier(**params)

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    t0 = time.time()
    proba = model.predict_proba(X_test)[:, 1]
    pred_time = time.time() - t0

    y_pred = (proba >= threshold).astype(int)
    print(classification_report(y_test, y_pred, digits=3))

    metrics = {
        "precision":  precision_score(y_test, y_pred, pos_label=1),
        "recall":     recall_score(y_test, y_pred, pos_label=1),
        "f1":         f1_score(y_test, y_pred, pos_label=1),
        "roc_auc":    roc_auc_score(y_test, proba),
        "train_time": train_time,
        "pred_time":  pred_time,
    }
    return model, metrics


# ─────────────────────────────────────────────────────────────
# 8. MLFLOW LOGGING
# ─────────────────────────────────────────────────────────────

def log_to_mlflow(model, best_params: dict, metrics: dict):
    """
    Log a trained XGBoost model, its hyperparameters, and
    evaluation metrics to MLflow.

    Parameters
    ----------
    model       : Fitted XGBClassifier.
    best_params : dict of hyperparameters used.
    metrics     : dict from train_final_xgboost().
    """
    import mlflow
    import mlflow.xgboost

    project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
    mlflow.set_tracking_uri(f"file://{project_root}/mlruns")
    mlflow.set_experiment("Telco Churn - XGBoost")

    with mlflow.start_run():
        mlflow.log_params(best_params)
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
        mlflow.xgboost.log_model(model, "model")
        print("[log_to_mlflow] Run logged successfully.")


# ─────────────────────────────────────────────────────────────
# 9. PREDICT ON NEW DATA
# ─────────────────────────────────────────────────────────────

def predict_churn(model, raw_df: pd.DataFrame, threshold: float = 0.30) -> pd.DataFrame:
    """
    Run the full preprocessing pipeline on new raw data
    and return churn probability + binary prediction.

    Parameters
    ----------
    model     : Trained XGBClassifier (or any sklearn-compatible model).
    raw_df    : Raw customer DataFrame (same schema as training CSV,
                without the customerID column).
    threshold : Probability cut-off (default 0.30).

    Returns
    -------
    pd.DataFrame with columns ['churn_proba', 'churn_pred'].
    """
    df = encode_features(raw_df.copy())
    df = fix_multicollinearity(df)

    # Drop target if accidentally present
    X = df.drop(columns=["Churn"], errors="ignore")

    proba = model.predict_proba(X)[:, 1]
    predictions = pd.DataFrame({
        "churn_proba": proba,
        "churn_pred":  (proba >= threshold).astype(int),
    })
    return predictions


# ─────────────────────────────────────────────────────────────
# 10. END-TO-END PIPELINE HELPER
# ─────────────────────────────────────────────────────────────

def run_pipeline(
    filepath: str,
    threshold: float = 0.30,
    n_optuna_trials: int = 30,
    log_mlflow: bool = False,
):
    """
    Convenience function that runs the entire pipeline end-to-end:
      load → encode → fix multicollinearity → train baselines
      → Optuna tuning → final XGBoost → (optional) MLflow logging.

    Parameters
    ----------
    filepath        : Path to Telco-Customer-Churn.csv
    threshold       : Prediction threshold (default 0.30).
    n_optuna_trials : Optuna search budget (default 30).
    log_mlflow      : Whether to log the final model to MLflow.

    Returns
    -------
    dict with 'model', 'metrics', 'best_params', 'study', 'df'
    """
    print("=" * 55)
    print("  Telco Churn Pipeline")
    print("=" * 55)

    df = load_and_preprocess(filepath)
    df = encode_features(df)
    df = fix_multicollinearity(df)

    print("\n[Step 1/4] Baseline models")
    split = train_models(df, threshold=threshold)
    X_train = split["X_train"]
    X_test  = split["X_test"]
    y_train = split["y_train"]
    y_test  = split["y_test"]

    print(f"\n[Step 2/4] Optuna tuning ({n_optuna_trials} trials)")
    best_params, study = tune_xgboost(
        X_train, X_test, y_train, y_test,
        threshold=threshold,
        n_trials=n_optuna_trials,
    )

    print("\n[Step 3/4] Final XGBoost with tuned params")
    model, metrics = train_final_xgboost(
        X_train, X_test, y_train, y_test,
        best_params=best_params,
        threshold=threshold,
    )

    feature_cols = list(X_train.columns)
    model_bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "threshold": threshold,
        "best_params": best_params,
    }
    joblib.dump(model_bundle, "churn_model.joblib")
    print("[save_model] Model saved as churn_model.joblib")

    if log_mlflow:
        print("\n[Step 4/4] Logging to MLflow")
        log_to_mlflow(model, best_params, metrics)
    else:
        print("\n[Step 4/4] MLflow logging skipped (log_mlflow=False)")

    print("\n✅ Pipeline complete.")
    print(f"   Recall: {metrics['recall']:.3f}  |  ROC-AUC: {metrics['roc_auc']:.3f}")
    return {
        "model":       model,
        "metrics":     metrics,
        "best_params": best_params,
        "study":       study,
        "df":          df,
        "feature_cols": feature_cols,
    }


# ─────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Telco Churn Pipeline")
    parser.add_argument("--data",      default="Telco-Customer-Churn.csv",
                        help="Path to the CSV file")
    parser.add_argument("--threshold", type=float, default=0.30,
                        help="Prediction threshold (default 0.30)")
    parser.add_argument("--trials",    type=int,   default=30,
                        help="Optuna trials (default 30)")
    parser.add_argument("--mlflow",    action="store_true",
                        help="Log final model to MLflow")
    args = parser.parse_args()

    run_pipeline(
        filepath=args.data,
        threshold=args.threshold,
        n_optuna_trials=args.trials,
        log_mlflow=args.mlflow,
    )
