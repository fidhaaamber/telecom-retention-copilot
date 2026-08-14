"""Generate reproducible profiling, EDA figures, and data dictionary artifacts."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
from utils.data import load_raw

RAW = Path("data/raw/cell2celltrain.csv")
OUT = Path("reports/eda_charts")

def chart(data, column, kind="hist"):
    fig, ax = plt.subplots(figsize=(8, 4))
    if kind == "bar": data.plot(kind="bar", ax=ax)
    else:
        for label, group in data.groupby("Churn"):
            pd.to_numeric(group[column], errors="coerce").dropna().clip(upper=pd.to_numeric(data[column], errors="coerce").quantile(.99)).plot(kind="hist", bins=35, alpha=.55, label=str(label), ax=ax)
        ax.legend(title="Churn")
    ax.set_title(column); fig.tight_layout(); fig.savefig(OUT / f"{column}.png", dpi=150); plt.close(fig)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_raw(RAW)
    dictionary = pd.DataFrame({"field": df.columns, "dtype": df.dtypes.astype(str).values, "missing_count": df.isna().sum().values, "unique_values": df.nunique(dropna=True).values})
    dictionary.to_csv("reports/data_dictionary.csv", index=False)
    chart(df.groupby("Churn").size(), "churn_distribution", "bar")
    for col in ["MonthlyRevenue", "MonthlyMinutes", "OverageMinutes", "DroppedCalls", "BlockedCalls", "CustomerCareCalls", "MonthsInService", "CurrentEquipmentDays", "UniqueSubs", "PercChangeMinutes"]:
        if col in df: chart(df, col)
    try:
        from ydata_profiling import ProfileReport
        ProfileReport(df, title="Cell2Cell Data Profile", minimal=True).to_file("reports/data_profile.html")
    except ImportError:
        Path("reports/data_profile_status.txt").write_text("Install ydata-profiling and rerun this script to generate the HTML profile.")
    Path("reports/eda_findings.md").write_text("# EDA findings\n\nRun `python -m scripts.profile_data` to reproduce ten charts and the data dictionary. Interpret churn gaps only after validating missingness, outliers, and potential leakage in the generated artifacts.\n")
    print(json.dumps({"charts": len(list(OUT.glob('*.png'))), "rows": len(df)}, indent=2))
if __name__ == "__main__": main()
