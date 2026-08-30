from pathlib import Path
import json

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW = PROJECT_ROOT / "data/raw/kaggle/creditcard.csv"
CLEAN = PROJECT_ROOT / "data/processed/clean_transactions.parquet"
ML_READY = PROJECT_ROOT / "data/ml_ready/fraud_ml_ready.parquet"

REPORTS = PROJECT_ROOT / "reports"
REPORTS.mkdir(parents=True, exist_ok=True)


print("=" * 60)
print("FINAL REPORT GENERATION")
print("=" * 60)


# ------------------------------------------------------------
# 1. Dataset profile
# ------------------------------------------------------------

raw = pd.read_csv(RAW)

profile = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "missing_values",
        "duplicate_rows",
        "fraud_count",
        "legitimate_count",
        "fraud_percentage",
    ],
    "value": [
        len(raw),
        len(raw.columns),
        int(raw.isna().sum().sum()),
        int(raw.duplicated().sum()),
        int((raw["Class"] == 1).sum()),
        int((raw["Class"] == 0).sum()),
        float((raw["Class"] == 1).mean() * 100),
    ],
})

profile.to_csv(
    REPORTS / "dataset_profile.csv",
    index=False,
)

print("dataset_profile.csv: CREATED")


# ------------------------------------------------------------
# 2. Class balance
# ------------------------------------------------------------

ml = pd.read_parquet(ML_READY)

class_counts = ml["Class"].value_counts().sort_index()

class_balance = pd.DataFrame({
    "class": class_counts.index,
    "count": class_counts.values,
})

class_balance["percentage"] = (
    class_balance["count"]
    / len(ml)
    * 100
)

class_balance.to_csv(
    REPORTS / "class_balance.csv",
    index=False,
)

print("class_balance.csv: CREATED")


# ------------------------------------------------------------
# 3. Class balance visualization
# ------------------------------------------------------------

plt.figure(figsize=(7, 5))

plt.bar(
    ["Legitimate", "Fraud"],
    [
        class_counts.get(0, 0),
        class_counts.get(1, 0),
    ],
)

plt.title("Class Distribution")
plt.xlabel("Transaction Class")
plt.ylabel("Number of Transactions")

plt.tight_layout()

plt.savefig(
    REPORTS / "class_balance.png",
    dpi=300,
)

plt.close()

print("class_balance.png: CREATED")


# ------------------------------------------------------------
# 4. Feature summary
# ------------------------------------------------------------

feature_summary = ml.describe().T.reset_index()

feature_summary = feature_summary.rename(
    columns={"index": "feature"}
)

feature_summary.to_csv(
    REPORTS / "feature_summary.csv",
    index=False,
)

print("feature_summary.csv: CREATED")


# ------------------------------------------------------------
# 5. Data quality report
# ------------------------------------------------------------

clean = pd.read_parquet(CLEAN)

quality_report = {
    "dataset": "clean_transactions.parquet",
    "rows": int(len(clean)),
    "columns": int(len(clean.columns)),
    "missing_values": int(
        clean.isna().sum().sum()
    ),
    "duplicate_rows": int(
        clean.duplicated().sum()
    ),
    "negative_amounts": int(
        (clean["Amount"] < 0).sum()
    ),
    "invalid_class_values": int(
        (~clean["Class"].isin([0, 1])).sum()
    ),
}

with open(
    REPORTS / "data_quality_report.json",
    "w",
) as f:
    json.dump(
        quality_report,
        f,
        indent=4,
    )

print("data_quality_report.json: CREATED")


# ------------------------------------------------------------
# 6. Processing summary
# ------------------------------------------------------------

processing_summary = {
    "raw_rows": int(len(raw)),
    "clean_rows": int(len(clean)),
    "rows_removed": int(
        len(raw) - len(clean)
    ),
    "ml_ready_rows": int(len(ml)),
    "ml_ready_features": int(
        len(ml.columns) - 1
    ),
    "target_column": "Class",
    "output_dataset": (
        "data/ml_ready/fraud_ml_ready.parquet"
    ),
}

with open(
    REPORTS / "processing_summary.json",
    "w",
) as f:
    json.dump(
        processing_summary,
        f,
        indent=4,
    )

print("processing_summary.json: CREATED")


# ------------------------------------------------------------
# 7. Amount distribution
# ------------------------------------------------------------

plt.figure(figsize=(8, 5))

plt.hist(
    ml["amount_log"],
    bins=50,
)

plt.title(
    "Distribution of Log-Transformed Transaction Amount"
)

plt.xlabel("amount_log")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    REPORTS / "amount_distribution.png",
    dpi=300,
)

plt.close()

print("amount_distribution.png: CREATED")


print("\n" + "=" * 60)
print("REPORT GENERATION COMPLETE")
print("=" * 60)
