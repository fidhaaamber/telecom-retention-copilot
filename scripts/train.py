from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from utils.data import checksum, engineer_features, feature_columns, load_raw
from utils.modeling import choose_threshold, save_bundle, train_candidates

RAW = Path("data/raw/cell2celltrain.csv")


def main() -> None:
    raw = load_raw(RAW).drop_duplicates()
    engineered = engineer_features(raw)
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    engineered.to_parquet("data/processed/cell2cell_cleaned.parquet", index=False)
    columns = feature_columns(engineered)
    X, y = engineered[columns], engineered["Churn"].astype(int)
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models, results = train_candidates(X_train, y_train, X_valid, y_valid)
    champion_metrics = max(results, key=lambda m: m.pr_auc)
    champion = models[champion_metrics.model]
    threshold = choose_threshold(y_valid, champion.predict_proba(X_valid)[:, 1])
    save_bundle("models/champion.joblib", champion, columns, threshold, champion_metrics)
    Path("reports").mkdir(exist_ok=True)
    pd.DataFrame([r.__dict__ for r in results]).to_csv("reports/model_comparison.csv", index=False)
    quality = {"rows": len(raw), "columns": len(raw.columns), "duplicates_removed": len(load_raw(RAW)) - len(raw), "missing_cells": int(raw.isna().sum().sum()), "sha256": checksum(RAW)}
    Path("reports/data_quality.json").write_text(json.dumps(quality, indent=2), encoding="utf-8")
    scored = engineered[["CustomerID", "Churn"]].copy()
    scored["churn_probability"] = champion.predict_proba(X)[:, 1]
    scored["risk_band"] = pd.cut(scored.churn_probability, [-.01, .3, .6, 1], labels=["Low", "Medium", "High"])
    scored.sort_values("churn_probability", ascending=False).to_csv("reports/scored_customers.csv", index=False)
    Path("reports/model_card.md").write_text(
        f"# Champion model card\n\nChampion: **{champion_metrics.model}** selected by validation PR-AUC.\n\n"
        f"PR-AUC: {champion_metrics.pr_auc:.4f}; ROC-AUC: {champion_metrics.roc_auc:.4f}; Accuracy: {champion_metrics.accuracy:.4f}; F1 threshold: {threshold['threshold']:.2f}.\n\n"
        "Use: prioritise analyst review only. Limitations: historical associations may encode bias, and predictions do not establish causality. "
        "Excluded leakage fields: CustomerID, RetentionCalls, MadeCallToRetentionTeam, RetentionOffersAccepted.\n", encoding="utf-8")
    print(json.dumps({"champion": champion_metrics.__dict__, "threshold": threshold, **quality}, indent=2))


if __name__ == "__main__":
    main()
