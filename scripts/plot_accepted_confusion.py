import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "reports/subset_reliability_predictions.csv"
)

accepted = df[
    df["decision"].isin(["Supported", "Provisional"])
].copy()

classes = sorted(df["true_class"].unique())

accepted["predicted_display"] = accepted["predicted_class"].where(
    accepted["predicted_class"].isin(classes),
    "Outside 15-class subset"
)

columns = classes + ["Outside 15-class subset"]

cm = pd.crosstab(
    accepted["true_class"],
    accepted["predicted_display"]
).reindex(
    index=classes,
    columns=columns,
    fill_value=0
)

fig, ax = plt.subplots(figsize=(19, 14))

im = ax.imshow(cm.values, cmap="Blues", aspect="auto")

ax.set_xticks(range(len(columns)))
ax.set_xticklabels(columns, rotation=90, fontsize=8)

ax.set_yticks(range(len(classes)))
ax.set_yticklabels(classes, fontsize=8)

for i in range(len(classes)):
    for j in range(len(columns)):
        value = cm.iloc[i, j]
        if value > 0:
            ax.text(j, i, str(value), ha="center", va="center",
                    fontsize=7)

ax.set_title("Accepted Predictions - Corrected Confusion Matrix")
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")

fig.colorbar(im, ax=ax, label="Sample Count")
plt.tight_layout()

output = ROOT / "reports/subset_accepted_confusion_matrix.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print("Saved:", output)
print("Matrix total:", int(cm.values.sum()))
print("Accepted samples:", len(accepted))
print("Outside-subset predictions:", int(
    cm["Outside 15-class subset"].sum()
))
