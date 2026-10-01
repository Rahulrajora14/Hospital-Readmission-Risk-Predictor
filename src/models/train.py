"""
Step 5 - Train, tune and compare models.

Flow:
  clean_data.csv
     -> split into train (70%) / validation (15%) / test (15%)
     -> train Logistic Regression  (simple baseline)
     -> train Random Forest        (non-linear, bagging)
     -> tune + train XGBoost       (gradient boosting, tuned with Optuna)
     -> pick a decision threshold per model on VALIDATION data
     -> choose the best model by VALIDATION PR-AUC
     -> report final scores on the untouched TEST set
     -> save the best model + metadata for the app
"""
import json
import warnings

import joblib
import numpy as np
import sklearn
import xgboost
import optuna
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src import config
from src.features.engineer import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.models import evaluate

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Data split
# ---------------------------------------------------------------------------
def load_splits():
    """
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    Stratified = every split keeps the same ~9% positive rate.
    Fixed random_state = identical split every time (reproducible).
    """
    df = pd.read_csv(config.CLEAN_CSV)
    X, y = df[ALL_FEATURES], df[config.TARGET]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    val_share = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_share, stratify=y_temp, random_state=config.RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Preprocessing + models
# ---------------------------------------------------------------------------
def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """
    Numbers  -> optionally standardised (needed for Logistic Regression,
                useless for tree models, which only care about order).
    Text     -> one-hot encoded. handle_unknown='ignore' means a category
                never seen in training will not crash the model later.
    """
    numeric = StandardScaler() if scale_numeric else "passthrough"
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=30)
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_logistic_regression() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=True)),
        ("model", LogisticRegression(max_iter=2000, C=0.5)),
    ])


def make_random_forest() -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=config.RANDOM_STATE,
        )),
    ])


def make_xgboost(params: dict) -> Pipeline:
    return Pipeline([
        ("preprocess", build_preprocessor(scale_numeric=False)),
        ("model", XGBClassifier(
            **params, eval_metric="aucpr", tree_method="hist",
            n_jobs=-1, random_state=config.RANDOM_STATE,
        )),
    ])


# ---------------------------------------------------------------------------
# Hyperparameter tuning
# ---------------------------------------------------------------------------
def tune_xgboost(X_train, y_train, X_val, y_val, n_trials=config.N_TUNING_TRIALS) -> dict:
    """
    Optuna tries different XGBoost settings. Each trial trains on TRAIN and is
    scored on VALIDATION (PR-AUC). Optuna learns from earlier trials which
    regions of the search space look promising (smarter than grid search).
    """
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
        }
        model = make_xgboost(params).fit(X_train, y_train)
        return average_precision_score(y_val, model.predict_proba(X_val)[:, 1])

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    print(f"  Best validation PR-AUC from tuning: {study.best_value:.4f}")
    print(f"  Best params: {study.best_params}")
    return study.best_params


# ---------------------------------------------------------------------------
# Optional experiment tracking
# ---------------------------------------------------------------------------
def log_to_mlflow(name: str, params: dict, val_metrics: dict, test_metrics: dict) -> None:
    """Log one model run to MLflow (skipped quietly if mlflow is not installed)."""
    if not config.USE_MLFLOW:
        return
    try:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{config.MLFLOW_DB.as_posix()}")
        mlflow.set_experiment("hospital-readmission")
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: v for k, v in params.items() if np.isscalar(v)})
            mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
    except Exception as err:  # tracking must never break training
        print(f"    (MLflow logging skipped: {err.__class__.__name__})")


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------
def train_and_compare():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()
    print(f"  Train {len(X_train):,} | Validation {len(X_val):,} | Test {len(X_test):,}")

    print(f"\n  Tuning XGBoost with Optuna ({config.N_TUNING_TRIALS} trials)...")
    best_xgb_params = tune_xgboost(X_train, y_train, X_val, y_val)

    candidates = {
        "Logistic Regression": (make_logistic_regression(), {"C": 0.5}),
        "Random Forest": (make_random_forest(), {"n_estimators": 300, "max_depth": 12}),
        "XGBoost": (make_xgboost(best_xgb_params), best_xgb_params),
    }

    rows, fitted, test_probas = [], {}, {}
    for name, (pipeline, params) in candidates.items():
        print(f"\n  Training {name}...")
        pipeline.fit(X_train, y_train)

        val_proba = pipeline.predict_proba(X_val)[:, 1]
        threshold = evaluate.find_best_threshold(y_val, val_proba)
        val_metrics = evaluate.compute_metrics(y_val, val_proba, threshold)

        test_proba = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate.compute_metrics(y_test, test_proba, threshold)

        print(f"    validation: {val_metrics}")
        print(f"    test:       {test_metrics}")
        log_to_mlflow(name, params, val_metrics, test_metrics)

        fitted[name] = (pipeline, threshold)
        test_probas[name] = test_proba
        rows.append({"model": name, "val_pr_auc": val_metrics["pr_auc"],
                     **{f"test_{k}": v for k, v in test_metrics.items()}})

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.COMPARISON_PATH, index=False)

    # Model selection uses VALIDATION score - the test set stays untouched
    best_name = comparison.loc[comparison["val_pr_auc"].idxmax(), "model"]
    best_pipeline, best_threshold = fitted[best_name]
    print(f"\n  Best model (by validation PR-AUC): {best_name}")
    print("\n" + comparison.to_string(index=False))

    # Plots
    evaluate.plot_roc_curves(y_test, test_probas)
    evaluate.plot_pr_curves(y_test, test_probas)
    evaluate.plot_confusion_matrix(y_test, test_probas[best_name], best_threshold, best_name)

    # Save everything the app and the explain step need
    joblib.dump(best_pipeline, config.MODEL_PATH)
    best_row = comparison[comparison["model"] == best_name].iloc[0].to_dict()
    metadata = {
        "model_name": best_name,
        "threshold": best_threshold,
        "test_metrics": {k.replace("test_", ""): v for k, v in best_row.items() if k.startswith("test_")},
        "positive_rate": round(float(y_train.mean()), 4),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_ranges": {c: [int(X_train[c].min()), int(X_train[c].max()), int(X_train[c].median())]
                           for c in NUMERIC_FEATURES},
        "category_options": {c: sorted(X_train[c].astype(str).unique().tolist())
                             for c in CATEGORICAL_FEATURES},
        "category_defaults": {c: str(X_train[c].mode()[0]) for c in CATEGORICAL_FEATURES},
        "xgb_params": best_xgb_params,
        # A pickled model is only reliable with the library versions it was saved with.
        # The app checks these and asks you to retrain if yours differ.
        "library_versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    config.METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    config.METRICS_PATH.write_text(json.dumps(comparison.to_dict(orient="records"), indent=2))

    # A small sample of test patients for the demo app
    config.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    sample = X_test.copy()
    sample[config.TARGET] = y_test.values
    sample.sample(300, random_state=config.RANDOM_STATE).to_csv(config.SAMPLE_PATIENTS_CSV, index=False)

    print(f"\n  Saved model -> {config.MODEL_PATH}")
    return comparison


if __name__ == "__main__":
    train_and_compare()
