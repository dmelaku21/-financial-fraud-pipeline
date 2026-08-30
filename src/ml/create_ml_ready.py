from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_features.parquet"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "ml_ready"
    / "fraud_ml_ready.parquet"
)


# ------------------------------------------------------------
# 1. Load processed dataset
# ------------------------------------------------------------

print("=" * 60)
print("ML-READY DATASET CREATION")
print("=" * 60)

print(f"Input : {INPUT}")
print(f"Output: {OUTPUT}")

df = pd.read_parquet(INPUT)

print(f"\nInput rows    : {len(df):,}")
print(f"Input columns : {len(df.columns)}")


# ------------------------------------------------------------
# 2. Define ML features
# ------------------------------------------------------------

feature_columns = [
    "amount_log",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
    "is_night",
    "amount_outlier_flag",
    "transactions_last_1h",
    "transactions_last_6h",
    "transactions_last_24h",
    "time_since_previous_transaction",
    "amount_change_ratio",
]

target_column = "Class"


# ------------------------------------------------------------
# 3. Validate required columns
# ------------------------------------------------------------

required_columns = feature_columns + [target_column]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required ML columns: "
        + str(missing_columns)
    )


# ------------------------------------------------------------
# 4. Select ML-ready columns
# ------------------------------------------------------------

ml_df = df[required_columns].copy()


# ------------------------------------------------------------
# 5. Validate target
# ------------------------------------------------------------

invalid_target = ~ml_df[target_column].isin([0, 1])

if invalid_target.any():
    raise ValueError(
        "Invalid values found in Class target."
    )


# ------------------------------------------------------------
# 6. Validate missing values
# ------------------------------------------------------------

missing_values = ml_df.isna().sum()

if missing_values.sum() > 0:
    print("\nMissing values:")
    print(
        missing_values[
            missing_values > 0
        ]
    )

    raise ValueError(
        "ML-ready dataset contains missing values."
    )


# ------------------------------------------------------------
# 7. Validate dataset
# ------------------------------------------------------------

if len(ml_df) == 0:
    raise ValueError(
        "ML-ready dataset is empty."
    )


# ------------------------------------------------------------
# 8. Save ML-ready dataset
# ------------------------------------------------------------

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

ml_df.to_parquet(
    OUTPUT,
    index=False
)


# ------------------------------------------------------------
# 9. Verify output
# ------------------------------------------------------------

verification = pd.read_parquet(OUTPUT)

print("\nML feature columns:")
for i, column in enumerate(
    feature_columns,
    start=1
):
    print(f"{i:2}. {column}")

print(f"\nTarget: {target_column}")

print("\nOutput verification:")
print(f"Rows    : {len(verification):,}")
print(f"Columns : {len(verification.columns)}")

if len(verification) != len(df):
    raise ValueError(
        "Row count changed during ML dataset creation."
    )

if list(verification.columns) != required_columns:
    raise ValueError(
        "Output columns do not match expected ML schema."
    )

print("Row-count validation : PASSED")
print("Column validation    : PASSED")
print("Missing-value check  : PASSED")
print("Target validation    : PASSED")

print("\n" + "=" * 60)
print("ML-READY DATASET COMPLETE")
print("=" * 60)
