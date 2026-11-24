import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

df = pd.read_csv(
    ROOT / "reports/subset_reliability_predictions.csv"
)

accepted = df[
    df["decision"].isin(["Supported", "Provisional"])
].copy()

errors = accepted[~accepted["is_correct"]].copy()

print("\nACCEPTED PREDICTION ERROR ANALYSIS")
print("Accepted samples:", len(accepted))
print("Incorrect accepted predictions:", len(errors))

print("\nErrors by reliability decision:")
print(
    errors["decision"].value_counts()
)

print("\nTop true-to-predicted error pairs:")
pairs = (
    errors.groupby(["true_class", "predicted_class"])
    .size()
    .reset_index(name="error_count")
    .sort_values("error_count", ascending=False)
)

print(pairs.head(20).to_string(index=False))

output = ROOT / "reports/subset_accepted_errors.csv"
errors.to_csv(output, index=False)

pairs_output = ROOT / "reports/subset_accepted_error_pairs.csv"
pairs.to_csv(pairs_output, index=False)

print("\nSaved:", output)
print("Saved:", pairs_output)
