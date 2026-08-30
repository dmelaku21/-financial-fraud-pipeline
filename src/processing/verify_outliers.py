from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.parquet"
)


df = pd.read_parquet(INPUT)

Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)
IQR = Q3 - Q1

lower = max(0, Q1 - 1.5 * IQR)
upper = Q3 + 1.5 * IQR

calculated_flag = (
    (df["Amount"] < lower) |
    (df["Amount"] > upper)
).astype(int)

print("=" * 60)
print("IQR OUTLIER VERIFICATION")
print("=" * 60)

print(f"Q1              : {Q1:.4f}")
print(f"Q3              : {Q3:.4f}")
print(f"IQR             : {IQR:.4f}")
print(f"Lower boundary  : {lower:.4f}")
print(f"Upper boundary  : {upper:.4f}")

print("\nOutlier distribution:")
print(calculated_flag.value_counts().sort_index())

# Verify existing feature
if "amount_outlier_flag" in df.columns:
    matches = (
        calculated_flag
        == df["amount_outlier_flag"]
    ).all()

    print(
        f"\nExisting flag matches calculation: {matches}"
    )

print("\nOutlier handling: FLAGGED, NOT DELETED")
print("=" * 60)