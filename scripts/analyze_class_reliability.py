import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "reports/subset_reliability_predictions.csv"
)

# Class-wise reliability statistics
summary = (
    df.groupby("true_class")
    .agg(
        total_samples=("is_correct", "count"),
        accuracy=("is_correct", "mean"),
        supported=("decision", lambda x: (x == "Supported").sum()),
        provisional=("decision", lambda x: (x == "Provisional").sum()),
        abstain=("decision", lambda x: (x == "Abstain").sum())
    )
    .reset_index()
)

for col in ["accuracy", "supported", "provisional", "abstain"]:
    if col == "accuracy":
        summary[col] = (summary[col] * 100).round(2)

summary["supported_rate"] = (
    summary["supported"] / summary["total_samples"] * 100
).round(2)

summary["provisional_rate"] = (
    summary["provisional"] / summary["total_samples"] * 100
).round(2)

summary["abstain_rate"] = (
    summary["abstain"] / summary["total_samples"] * 100
).round(2)

summary = summary.sort_values("accuracy")

output = ROOT / "reports/subset_class_reliability.csv"
summary.to_csv(output, index=False)

print("\nCLASS-WISE RELIABILITY ANALYSIS")
print(summary.to_string(index=False))

print("\nSaved:", output)
