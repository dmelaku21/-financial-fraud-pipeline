from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_date.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_transactions.parquet"
)


# ============================================================
# Required columns
# ============================================================

REQUIRED_COLUMNS = [
    "Time",
    "Amount",
    "Class",
    "timestamp",
    "transaction_date",
]


# ============================================================
# Data cleaning
# ============================================================

def clean_transactions() -> None:
    """Perform data-quality checks and create a clean dataset."""

    print("=" * 60)
    print("TRANSACTION DATA CLEANING")
    print("=" * 60)

    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # 1. Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_PATH}"
        )

    # --------------------------------------------------------
    # 2. Load data
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    original_rows = len(df)

    print("\nInitial dataset:")
    print(f"Rows    : {len(df):,}")
    print(f"Columns : {len(df.columns):,}")

    # --------------------------------------------------------
    # 3. Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("\nRequired-column validation: PASSED")

    # --------------------------------------------------------
    # 4. Missing-value check
    # --------------------------------------------------------

    missing_values = df.isna().sum()
    total_missing = missing_values.sum()

    print("\nMissing-value check:")
    print(f"Total missing values: {total_missing:,}")

    if total_missing > 0:
        print("\nColumns containing missing values:")
        print(
            missing_values[
                missing_values > 0
            ]
        )

        raise ValueError(
            "Missing values detected. "
            "Review before continuing."
        )

    print("Missing-value validation: PASSED")

    # --------------------------------------------------------
    # 5. Duplicate check
    # --------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    print("\nDuplicate check:")
    print(f"Duplicate rows: {duplicate_count:,}")

    # --------------------------------------------------------
    # 6. Remove exact duplicate records
    # --------------------------------------------------------

    if duplicate_count > 0:

        print(
            "\nRemoving exact duplicate rows "
            "from the processed dataset."
        )

        df = df.drop_duplicates().copy()

    else:

        print(
            "\nNo duplicate rows found."
        )

    print(
        f"Rows after duplicate handling: {len(df):,}"
    )

    # --------------------------------------------------------
    # 7. Data type conversion
    # --------------------------------------------------------

    print("\nData type conversion:")

    df["Time"] = pd.to_numeric(
        df["Time"],
        errors="coerce"
    )

    df["Amount"] = pd.to_numeric(
        df["Amount"],
        errors="coerce"
    )

    df["Class"] = pd.to_numeric(
        df["Class"],
        errors="coerce"
    )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    df["transaction_date"] = pd.to_datetime(
        df["transaction_date"],
        errors="coerce"
    ).dt.date

    # Validate conversion
    conversion_missing = df[
        ["Time", "Amount", "Class", "timestamp", "transaction_date"]
    ].isna().sum().sum()

    if conversion_missing > 0:
        raise ValueError(
            "Invalid values detected during type conversion."
        )

    print("Type conversion: PASSED")

    # --------------------------------------------------------
    # 8. Amount validation
    # --------------------------------------------------------

    print("\nAmount validation:")

    negative_amounts = (
        df["Amount"] < 0
    ).sum()

    print(
        f"Negative amounts: {negative_amounts:,}"
    )

    if negative_amounts > 0:
        raise ValueError(
            "Negative transaction amounts detected."
        )

    print("Amount validation: PASSED")

    # --------------------------------------------------------
    # 9. Target validation
    # --------------------------------------------------------

    print("\nTarget validation:")

    invalid_classes = ~df["Class"].isin([0, 1])

    invalid_class_count = invalid_classes.sum()

    print(
        f"Invalid Class values: {invalid_class_count:,}"
    )

    if invalid_class_count > 0:
        raise ValueError(
            "Target column contains values other than 0 and 1."
        )

    print("Target validation: PASSED")

    # --------------------------------------------------------
    # 10. Timestamp validation
    # --------------------------------------------------------

    print("\nTimestamp validation:")

    invalid_timestamp = df["timestamp"].isna().sum()

    print(
        f"Invalid timestamps: {invalid_timestamp:,}"
    )

    if invalid_timestamp > 0:
        raise ValueError(
            "Invalid timestamps detected."
        )

    print("Timestamp validation: PASSED")

    # --------------------------------------------------------
    # 11. IQR-based amount outlier detection
    # --------------------------------------------------------

    print("\nAmount outlier analysis:")

    q1 = df["Amount"].quantile(0.25)
    q3 = df["Amount"].quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - (1.5 * iqr)
    upper_bound = q3 + (1.5 * iqr)

    # Amount cannot be negative, so use zero as the
    # practical lower boundary.
    lower_bound = max(0, lower_bound)

    df["amount_outlier"] = (
        (df["Amount"] < lower_bound)
        | (df["Amount"] > upper_bound)
    )

    outlier_count = df["amount_outlier"].sum()

    outlier_percentage = (
        outlier_count / len(df) * 100
    )

    print(f"Q1              : {q1:.4f}")
    print(f"Q3              : {q3:.4f}")
    print(f"IQR             : {iqr:.4f}")
    print(f"Lower boundary  : {lower_bound:.4f}")
    print(f"Upper boundary  : {upper_bound:.4f}")
    print(f"Amount outliers : {outlier_count:,}")
    print(
        f"Outlier percent : {outlier_percentage:.4f}%"
    )

    print(
        "\nOutlier handling: FLAGGED, NOT DELETED"
    )

    # --------------------------------------------------------
    # 12. Class distribution
    # --------------------------------------------------------

    print("\nClass distribution after cleaning:")

    class_counts = df["Class"].value_counts().sort_index()

    print(class_counts)

    class_percentages = (
        df["Class"]
        .value_counts(normalize=True)
        .sort_index()
        * 100
    )

    print("\nClass percentages:")

    for class_value, percentage in class_percentages.items():

        label = (
            "Legitimate"
            if class_value == 0
            else "Fraud"
        )

        print(
            f"Class {class_value} ({label}): "
            f"{percentage:.4f}%"
        )

    # --------------------------------------------------------
    # 13. Final validation
    # --------------------------------------------------------

    print("\nFinal validation:")

    final_missing = df.isna().sum().sum()

    if final_missing > 0:
        raise ValueError(
            "Final dataset contains missing values."
        )

    if df.empty:
        raise ValueError(
            "Clean dataset is empty."
        )

    print("Missing values : PASSED")
    print("Dataset non-empty: PASSED")

    # --------------------------------------------------------
    # 14. Save clean dataset
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
    # 15. Output verification
    # --------------------------------------------------------

    verification_df = pd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    print("\nOutput verification:")
    print(
        f"Original rows : {original_rows:,}"
    )
    print(
        f"Final rows    : {len(verification_df):,}"
    )
    print(
        f"Final columns : {len(verification_df.columns):,}"
    )

    print(
        f"Rows removed  : "
        f"{original_rows - len(verification_df):,}"
    )

    if verification_df.empty:
        raise ValueError(
            "Output dataset is empty."
        )

    print("Output validation: PASSED")

    # --------------------------------------------------------
    # Complete
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA CLEANING COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    clean_transactions()