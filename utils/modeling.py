from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.data import feature_columns


@dataclass
class Metrics:
    model: str
    pr_auc: float
    roc_auc: float
    recall: float
    precision: float
    f1: float
    accuracy: float


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=np.number).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=10))]), categorical),
    ])


def candidate_models() -> dict[str, Any]:
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear"),
        "random_forest": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=-1, random_state=42),
    }
    try:
        from lightgbm import LGBMClassifier
        models["lightgbm"] = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, class_weight="balanced", random_state=42, n_jobs=-1, verbosity=-1)
    except ImportError:
        # Kept explicit so a missing optional package never silently changes a production run.
        pass
    return models


def evaluate(y: pd.Series, probability: np.ndarray, name: str, threshold: float = 0.5) -> Metrics:
    prediction = (probability >= threshold).astype(int)
    return Metrics(name, float(average_precision_score(y, probability)), float(roc_auc_score(y, probability)), float(recall_score(y, prediction)), float(precision_score(y, prediction, zero_division=0)), float(f1_score(y, prediction, zero_division=0)), float(accuracy_score(y, prediction)))


def train_candidates(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series) -> tuple[dict[str, Pipeline], list[Metrics]]:
    fitted, results = {}, []
    for name, estimator in candidate_models().items():
        pipeline = Pipeline([("preprocessor", make_preprocessor(X_train)), ("model", estimator)])
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results.append(evaluate(y_valid, pipeline.predict_proba(X_valid)[:, 1], name))
    return fitted, results


def choose_threshold(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    choices = [(t, f1_score(y, probability >= t)) for t in np.arange(0.15, 0.81, 0.05)]
    threshold, score = max(choices, key=lambda x: x[1])
    return {"threshold": round(float(threshold), 2), "validation_f1": round(float(score), 4)}


def save_bundle(path: str | Path, pipeline: Pipeline, columns: list[str], threshold: dict[str, float], metrics: Metrics) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "feature_columns": columns, "threshold": threshold, "metrics": asdict(metrics)}, path)


def explain_linear(bundle: dict, row: pd.DataFrame, top_n: int = 5) -> list[dict[str, Any]]:
    """Return local directional contributions for linear or tree champion models."""
    pipe = bundle["pipeline"]
    model = pipe.named_steps["model"]
    transformed = pipe.named_steps["preprocessor"].transform(row)
    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    if hasattr(model, "coef_"):
        values = transformed.toarray().ravel() if hasattr(transformed, "toarray") else np.asarray(transformed).ravel()
        contributions = values * model.coef_.ravel()
    else:
        # LightGBM's native pred_contrib output is TreeSHAP-compatible and avoids
        # shipping the large optional SHAP package with the production API image.
        try:
            native_values = model.booster_.predict(transformed, pred_contrib=True)
            if hasattr(native_values, "toarray"): native_values = native_values.toarray()
            contributions = np.asarray(native_values).ravel()[:-1]
        except Exception:
            values = transformed.toarray().ravel() if hasattr(transformed, "toarray") else np.asarray(transformed).ravel()
            contributions = values * np.asarray(getattr(model, "feature_importances_", np.zeros(len(names))))
    ranked = sorted(zip(names, contributions), key=lambda x: abs(x[1]), reverse=True)[:top_n]
    return [{"feature": n.replace("numeric__", "").replace("categorical__", ""), "impact": round(float(v), 4), "direction": "increases risk" if v > 0 else "reduces risk"} for n, v in ranked]
