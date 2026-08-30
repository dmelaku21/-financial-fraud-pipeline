from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_transactions.parquet"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.parquet"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)

print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")

df = pd.read_parquet(INPUT)

print(f"\nInput rows    : {len(df):,}")
print(f"Input columns : {len(df.columns)}")


# ============================================================
# VALIDATE INPUT
# ============================================================

required_columns = [
    "Time",
    "Amount",
    "Class",
    "timestamp",
]

missing = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )

print("Input validation: PASSED")


# ============================================================
# SORT TRANSACTIONS CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    by=["Time"],
    kind="mergesort"
).reset_index(drop=True)


# ============================================================
# 1. AMOUNT LOG TRANSFORMATION
# ============================================================

df["amount_log"] = np.log1p(df["Amount"])


# ============================================================
# 2–5. TEMPORAL FEATURES
# ============================================================

df["hour_of_day"] = df["timestamp"].dt.hour

df["day_of_week"] = df["timestamp"].dt.dayofweek

df["is_weekend"] = (
    df["day_of_week"] >= 5
).astype(int)

df["is_night"] = (
    (df["hour_of_day"] < 6)
    | (df["hour_of_day"] >= 22)
).astype(int)


# ============================================================
# 6. AMOUNT Z-SCORE
# ============================================================

amount_mean = df["Amount"].mean()
amount_std = df["Amount"].std()

if amount_std == 0:
    df["amount_zscore"] = 0.0
else:
    df["amount_zscore"] = (
        (df["Amount"] - amount_mean)
        / amount_std
    )


# ============================================================
# 7. IQR OUTLIER FLAG
# ============================================================

Q1 = df["Amount"].quantile(0.25)
Q3 = df["Amount"].quantile(0.75)

IQR = Q3 - Q1

lower = max(
    0.0,
    Q1 - 1.5 * IQR
)

upper = Q3 + 1.5 * IQR

df["amount_outlier_flag"] = (
    (df["Amount"] < lower)
    | (df["Amount"] > upper)
).astype(int)


# ============================================================
# 8–10. TRANSACTION VELOCITY FEATURES
# ============================================================
#
# IMPORTANT:
# The dataset has no customer/card identifier.
#
# Therefore these are ordered transaction-window features,
# NOT customer/card-level velocity features.
#
# Time is measured in elapsed seconds.
# ============================================================

time_seconds = df["Time"]

df["transactions_last_1h"] = (
    time_seconds
    .rolling(window=3600, min_periods=1)
    .count()
    - 1
)

df["transactions_last_6h"] = (
    time_seconds
    .rolling(window=21600, min_periods=1)
    .count()
    - 1
)

df["transactions_last_24h"] = (
    time_seconds
    .rolling(window=86400, min_periods=1)
    .count()
    - 1
)


# ============================================================
# 11. TIME SINCE PREVIOUS TRANSACTION
# ============================================================

df["time_since_previous_transaction"] = (
    df["Time"].diff()
)

df["time_since_previous_transaction"] = (
    df["time_since_previous_transaction"]
    .fillna(0)
)


# ============================================================
# 12. AMOUNT CHANGE RATIO
# ============================================================

previous_amount = df["Amount"].shift(1)

df["amount_change_ratio"] = (
    (df["Amount"] - previous_amount).abs()
    / previous_amount.replace(0, np.nan)
)

df["amount_change_ratio"] = (
    df["amount_change_ratio"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# FEATURE VALIDATION
# ============================================================

feature_columns = [
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "amount_zscore",
    "amount_outlier_flag",
    "transactions_last_1h",
    "transactions_last_6h",
    "transactions_last_24h",
    "time_since_previous_transaction",
    "amount_change_ratio",
]

target_column = "Class"

required_output_columns = (
    feature_columns + [target_column]
)

print("\nGenerated features:")

for number, column in enumerate(
    feature_columns,
    start=1
):
    print(f"{number:2}. {column}")


# ============================================================
# CHECK MISSING VALUES
# ============================================================

feature_missing = (
    df[required_output_columns]
    .isna()
    .sum()
)

if feature_missing.sum() > 0:

    print("\nMissing feature values:")

    print(
        feature_missing[
            feature_missing > 0
        ]
    )

    raise ValueError(
        "Missing values detected in final features."
    )

print("\nFeature missing-value check: PASSED")


# ============================================================
# TARGET VALIDATION
# ============================================================

invalid_classes = (
    ~df[target_column].isin([0, 1])
)

if invalid_classes.any():

    raise ValueError(
        "Invalid values found in Class."
    )

print("Target validation: PASSED")


# ============================================================
# BUILD ML FEATURE DATASET
# ============================================================

ml_df = df[
    required_output_columns
].copy()


# ============================================================
# SAVE
# ============================================================

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_parquet(
    OUTPUT,
    index=False
)


# ============================================================
# OUTPUT VERIFICATION
# ============================================================

verification = pd.read_parquet(
    OUTPUT
)

print("\nOutput verification:")
print(
    f"Rows    : {len(verification):,}"
)

print(
    f"Columns : {len(verification.columns)}"
)

print("\nFinal columns:")

for number, column in enumerate(
    verification.columns,
    start=1
):
    print(f"{number:2}. {column}")


if len(verification) != len(df):
    raise ValueError(
        "Row count changed during feature engineering."
    )

if list(verification.columns) != required_output_columns:
    raise ValueError(
        "Final feature columns do not match expected schema."
    )

if verification.isna().sum().sum() != 0:
    raise ValueError(
        "Final ML dataset contains missing values."
    )

print("\nRow-count validation : PASSED")
print("Column validation    : PASSED")
print("Missing-value check  : PASSED")
print("Target validation    : PASSED")

print("\nClass distribution:")

print(
    verification["Class"]
    .value_counts()
)

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 60)
