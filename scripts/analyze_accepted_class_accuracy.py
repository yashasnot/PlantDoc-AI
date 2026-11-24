import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "reports/subset_reliability_predictions.csv"
)

df["accepted"] = df["decision"].isin(
    ["Supported", "Provisional"]
)

rows = []

for disease, group in df.groupby("true_class"):
    accepted = group[group["accepted"]]

    rows.append({
        "class": disease,
        "total": len(group),
        "accepted": len(accepted),
        "coverage_pct": round(
            len(accepted) / len(group) * 100, 2
        ),
        "accepted_accuracy_pct": round(
            accepted["is_correct"].mean() * 100, 2
        ) if len(accepted) else None,
        "rejected": len(group) - len(accepted)
    })

result = pd.DataFrame(rows)

result = result.sort_values(
    "accepted_accuracy_pct",
    ascending=True
)

output = ROOT / "reports/subset_accepted_class_accuracy.csv"
result.to_csv(output, index=False)

print("\nCLASS-WISE ACCEPTED PERFORMANCE")
print(result.to_string(index=False))

print("\nSaved:", output)
