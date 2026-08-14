"""Cross-validated LightGBM tuning with a final untouched holdout evaluation."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from lightgbm import LGBMClassifier
from utils.data import engineer_features, feature_columns, load_raw
from utils.modeling import choose_threshold, evaluate, make_preprocessor, save_bundle

def main():
    df = engineer_features(load_raw("data/raw/cell2celltrain.csv").drop_duplicates())
    columns = feature_columns(df); X, y = df[columns], df["Churn"].astype(int)
    # The final 20% is never used for model selection or threshold selection.
    X_development, X_test, y_development, y_test = train_test_split(X, y, test_size=.20, stratify=y, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_development, y_development, test_size=.20, stratify=y_development, random_state=42)
    pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", LGBMClassifier(class_weight="balanced", random_state=42, n_jobs=1, verbosity=-1))])
    search = RandomizedSearchCV(
        pipeline,
        param_distributions={
            "model__n_estimators": randint(250, 800), "model__learning_rate": uniform(.015, .085),
            "model__num_leaves": randint(15, 64), "model__min_child_samples": randint(20, 120),
            "model__subsample": uniform(.65, .35), "model__colsample_bytree": uniform(.65, .35),
            "model__reg_alpha": uniform(0, .5), "model__reg_lambda": uniform(.1, 2.5),
        },
        n_iter=12, scoring="average_precision", cv=StratifiedKFold(3, shuffle=True, random_state=42),
        n_jobs=-1, verbose=1, random_state=42, refit=True,
    )
    search.fit(X_train, y_train)
    validation_probability = search.best_estimator_.predict_proba(X_valid)[:, 1]
    threshold = choose_threshold(y_valid, validation_probability)
    test_probability = search.best_estimator_.predict_proba(X_test)[:, 1]
    metrics = evaluate(y_test, test_probability, "lightgbm_tuned", threshold["threshold"])
    accuracy_threshold = max([round(x, 2) for x in np.arange(.05, .96, .01)], key=lambda t: accuracy_score(y_valid, validation_probability >= t))
    accuracy_pred = test_probability >= accuracy_threshold
    conservative_metrics = {"threshold": accuracy_threshold, "accuracy": round(float(accuracy_score(y_test, accuracy_pred)), 5), "recall": round(float(recall_score(y_test, accuracy_pred)), 5), "precision": round(float(precision_score(y_test, accuracy_pred, zero_division=0)), 5), "f1": round(float(f1_score(y_test, accuracy_pred, zero_division=0)), 5)}
    save_bundle("models/champion.joblib", search.best_estimator_, columns, threshold, metrics)
    result = {"best_cv_pr_auc": round(float(search.best_score_), 5), "best_params": search.best_params_, "validation_threshold": threshold, "untouched_test": metrics.__dict__, "conservative_accuracy_mode": conservative_metrics, "overfit_gap_pr_auc": round(float(search.best_score_ - metrics.pr_auc), 5)}
    Path("reports/tuning_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(search.cv_results_).sort_values("rank_test_score").to_csv("reports/lightgbm_tuning_trials.csv", index=False)
    print(json.dumps(result, indent=2))
if __name__ == "__main__": main()
