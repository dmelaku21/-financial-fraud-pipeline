from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_transactions.parquet"
)

REPORT_DIR = PROJECT_ROOT / "reports"

CSV_OUTPUT = REPORT_DIR / "class_balance.csv"
PNG_OUTPUT = REPORT_DIR / "class_balance.png"


# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

df = pd.read_parquet(INPUT)

print("=" * 60)
print("CLASS IMBALANCE ANALYSIS")
print("=" * 60)

print(f"Input : {INPUT}")
print(f"Rows  : {len(df):,}")


# ------------------------------------------------------------
# 2. Calculate class distribution
# ------------------------------------------------------------

class_counts = df["Class"].value_counts().sort_index()

legitimate_count = class_counts.get(0, 0)
fraud_count = class_counts.get(1, 0)

total = len(df)

legitimate_percentage = legitimate_count / total * 100
fraud_percentage = fraud_count / total * 100


# ------------------------------------------------------------
# 3. Create report
# ------------------------------------------------------------

balance_report = pd.DataFrame({
    "class": [0, 1],
    "count": [
        legitimate_count,
        fraud_count
    ],
    "percentage": [
        legitimate_percentage,
        fraud_percentage
    ]
})

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

balance_report.to_csv(
    CSV_OUTPUT,
    index=False
)


# ------------------------------------------------------------
# 4. Print results
# ------------------------------------------------------------

print("\nClass distribution:")
print(balance_report.to_string(index=False))

print(f"\nLegitimate transactions: {legitimate_count:,}")
print(f"Fraud transactions     : {fraud_count:,}")
print(f"Fraud percentage       : {fraud_percentage:.4f}%")

print("\nCSV report saved:")
print(CSV_OUTPUT)


# ------------------------------------------------------------
# 5. Create visualization
# ------------------------------------------------------------

labels = [
    "Legitimate (0)",
    "Fraud (1)"
]

counts = [
    legitimate_count,
    fraud_count
]

plt.figure(figsize=(8, 5))

plt.bar(labels, counts)

plt.title("Transaction Class Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    PNG_OUTPUT,
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nVisualization saved:")
print(PNG_OUTPUT)

print("\n" + "=" * 60)
print("CLASS IMBALANCE ANALYSIS COMPLETE")
print("=" * 60)
