"""Optional SHAP report for the champion model; run after `python -m scripts.train`."""
from pathlib import Path
import joblib
import pandas as pd
from utils.data import engineer_features, load_raw

def main():
    try:
        import shap
    except ImportError as exc:
        raise SystemExit("Install requirements.txt to generate SHAP reports.") from exc
    bundle = joblib.load("models/champion.joblib")
    df = engineer_features(load_raw("data/raw/cell2celltrain.csv").head(500))
    X = df[bundle["feature_columns"]]
    pipe = bundle["pipeline"]; transformed = pipe.named_steps["preprocessor"].transform(X)
    model = pipe.named_steps["model"]
    explainer = shap.Explainer(model, transformed, feature_names=pipe.named_steps["preprocessor"].get_feature_names_out())
    values = explainer(transformed)
    importance = pd.DataFrame({"feature": values.feature_names, "mean_abs_shap": abs(values.values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    Path("reports").mkdir(exist_ok=True); importance.to_csv("reports/shap_global_importance.csv", index=False)
    print("Wrote reports/shap_global_importance.csv")
if __name__ == "__main__": main()
