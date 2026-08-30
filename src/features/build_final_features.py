from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_velocity_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "final_features.parquet"
)


# ============================================================
# Final feature engineering
# ============================================================

def build_final_features() -> None:
    """
    Build the final domain-relevant feature set.

    Important:
    The public dataset does not contain customer/card IDs.
    Therefore, sequential and velocity features represent
    activity in the observed transaction stream rather than
    customer-level behavior.
    """

    print("=" * 60)
    print("FINAL FEATURE ENGINEERING")
    print("=" * 60)

    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    # --------------------------------------------------------
    # 2. Load dataset
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    original_rows = len(df)
    original_columns = len(df.columns)

    print("\nInput dataset:")
    print(f"Rows    : {original_rows:,}")
    print(f"Columns : {original_columns:,}")

    # --------------------------------------------------------
    # 3. Validate required columns
    # --------------------------------------------------------

    required_columns = [
        "Amount",
        "amount_log",
        "timestamp",
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
        "transactions_last_1h",
        "transactions_last_6h",
        "transactions_last_24h",
        "Class",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    print("\nRequired-column validation: PASSED")

    # --------------------------------------------------------
    # 4. Convert timestamp and sort chronologically
    # --------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print("Timestamp validation       : PASSED")
    print("Chronological ordering     : PASSED")

    # --------------------------------------------------------
    # 5. Amount Z-score
    # --------------------------------------------------------
    #
    # Important:
    # This is descriptive feature engineering.
    # Later, ML preprocessing should calculate scaling
    # parameters using the training data only.
    #
    # --------------------------------------------------------

    amount_mean = df["Amount"].mean()
    amount_std = df["Amount"].std()

    if amount_std == 0:
        raise ValueError(
            "Amount standard deviation is zero."
        )

    df["amount_zscore"] = (
        (df["Amount"] - amount_mean)
        / amount_std
    )

    # --------------------------------------------------------
    # 6. Amount outlier flag
    # --------------------------------------------------------

    q1 = df["Amount"].quantile(0.25)
    q3 = df["Amount"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = max(
        0,
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    df["amount_outlier_flag"] = (
        (df["Amount"] < lower_bound)
        | (df["Amount"] > upper_bound)
    ).astype("int8")

    print("\nAmount feature engineering:")
    print(f"Mean              : {amount_mean:.4f}")
    print(f"Standard deviation : {amount_std:.4f}")
    print(f"Q1                : {q1:.4f}")
    print(f"Q3                : {q3:.4f}")
    print(f"IQR               : {iqr:.4f}")
    print(f"Lower boundary    : {lower_bound:.4f}")
    print(f"Upper boundary    : {upper_bound:.4f}")
    print(
        "Outlier records   : "
        f"{df['amount_outlier_flag'].sum():,}"
    )

    # --------------------------------------------------------
    # 7. Time since previous transaction
    # --------------------------------------------------------

    time_difference = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
    )

    # First transaction has no previous transaction.
    # Set its value to zero.
    df["time_since_previous_transaction"] = (
        time_difference
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # 8. Amount change ratio
    # --------------------------------------------------------
    #
    # Formula:
    #
    # |current amount - previous amount|
    # ----------------------------------
    #          previous amount
    #
    # Zero previous amounts are handled safely.
    # --------------------------------------------------------

    previous_amount = df["Amount"].shift(1)

    denominator = previous_amount.abs()

    numerator = (
        df["Amount"] - previous_amount
    ).abs()

    df["amount_change_ratio"] = np.where(
        denominator > 0,
        numerator / denominator,
        0.0
    )

    # --------------------------------------------------------
    # 9. Validate new features
    # --------------------------------------------------------

    new_features = [
        "amount_zscore",
        "amount_outlier_flag",
        "time_since_previous_transaction",
        "amount_change_ratio",
    ]

    print("\nNew feature validation:")

    for feature in new_features:

        missing = df[feature].isna().sum()

        if missing > 0:
            raise ValueError(
                f"{feature} contains "
                f"{missing:,} missing values."
            )

        infinite = np.isinf(
            df[feature].to_numpy()
        ).sum()

        if infinite > 0:
            raise ValueError(
                f"{feature} contains "
                f"{infinite:,} infinite values."
            )

        print(
            f"{feature}: PASSED"
        )

    # --------------------------------------------------------
    # 10. Validate requested feature set
    # --------------------------------------------------------

    requested_features = [
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

    missing_final_features = [
        feature
        for feature in requested_features
        if feature not in df.columns
    ]

    if missing_final_features:
        raise ValueError(
            "Missing final features: "
            + ", ".join(missing_final_features)
        )

    print(
        f"\nRequested feature count: "
        f"{len(requested_features)}"
    )

    print(
        "Minimum 10-feature requirement: PASSED"
    )

    # --------------------------------------------------------
    # 11. Display feature summary
    # --------------------------------------------------------

    print("\nFinal feature set:")

    for number, feature in enumerate(
        requested_features,
        start=1
    ):
        print(
            f"{number:2d}. {feature}"
        )

    # --------------------------------------------------------
    # 12. Display sample
    # --------------------------------------------------------

    print("\nFeature sample:")

    print(
        df[
            [
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
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 13. Feature statistics
    # --------------------------------------------------------

    print("\nFeature statistics:")

    print(
        df[requested_features]
        .describe()
        .T[
            [
                "count",
                "mean",
                "std",
                "min",
                "max",
            ]
        ]
    )

    # --------------------------------------------------------
    # 14. Save final dataset
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow"
    )

    # --------------------------------------------------------
    # 15. Verify output
    # --------------------------------------------------------

    verification_df = pd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    print("\nOutput verification:")
    print(
        f"Rows    : {len(verification_df):,}"
    )
    print(
        f"Columns : {len(verification_df.columns):,}"
    )

    if len(verification_df) != original_rows:
        raise ValueError(
            "Row count changed during final feature engineering."
        )

    if not all(
        feature in verification_df.columns
        for feature in requested_features
    ):
        raise ValueError(
            "One or more final features are missing."
        )

    if verification_df[
        requested_features
    ].isna().any().any():

        raise ValueError(
            "Final feature set contains missing values."
        )

    print("Row-count validation : PASSED")
    print("Feature validation    : PASSED")
    print("Output validation     : PASSED")

    print("\n" + "=" * 60)
    print("FINAL FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    build_final_features()