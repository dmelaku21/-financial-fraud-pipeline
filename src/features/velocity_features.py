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
    / "transactions_with_temporal_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_velocity_features.parquet"
)


# ============================================================
# Velocity feature engineering
# ============================================================

def create_velocity_features() -> None:
    """
    Create dataset-level rolling transaction velocity features.

    IMPORTANT:
    The dataset does not contain a customer/card identifier.
    Therefore these features represent transaction activity
    across the observed dataset, not customer-level velocity.
    """

    print("=" * 60)
    print("VELOCITY FEATURE ENGINEERING")
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
    # 3. Validate timestamp
    # --------------------------------------------------------

    if "timestamp" not in df.columns:
        raise ValueError(
            "Required column 'timestamp' is missing."
        )

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError(
            "Invalid timestamps detected."
        )

    # --------------------------------------------------------
    # 4. Sort chronologically
    # --------------------------------------------------------

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    print("\nChronological ordering: PASSED")

    # --------------------------------------------------------
    # 5. Create transaction event indicator
    # --------------------------------------------------------

    df["_transaction_event"] = 1

    # --------------------------------------------------------
    # 6. Create rolling velocity features
    # --------------------------------------------------------

    timestamp_indexed = df.set_index(
        "timestamp"
    )

    transactions_last_1h = (
        timestamp_indexed[
            "_transaction_event"
        ]
        .rolling("1h", closed="left")
        .sum()
        .fillna(0)
        .astype("int64")
    )

    transactions_last_6h = (
        timestamp_indexed[
            "_transaction_event"
        ]
        .rolling("6h", closed="left")
        .sum()
        .fillna(0)
        .astype("int64")
    )

    transactions_last_24h = (
        timestamp_indexed[
            "_transaction_event"
        ]
        .rolling("24h", closed="left")
        .sum()
        .fillna(0)
        .astype("int64")
    )

    # --------------------------------------------------------
    # 7. Restore original row order
    # --------------------------------------------------------

    timestamp_indexed[
        "transactions_last_1h"
    ] = transactions_last_1h

    timestamp_indexed[
        "transactions_last_6h"
    ] = transactions_last_6h

    timestamp_indexed[
        "transactions_last_24h"
    ] = transactions_last_24h

    df = (
        timestamp_indexed
        .reset_index()
    )

    # --------------------------------------------------------
    # 8. Remove temporary column
    # --------------------------------------------------------

    df = df.drop(
        columns=["_transaction_event"]
    )

    # --------------------------------------------------------
    # 9. Validate velocity features
    # --------------------------------------------------------

    velocity_columns = [
        "transactions_last_1h",
        "transactions_last_6h",
        "transactions_last_24h",
    ]

    print("\nVelocity feature validation:")

    for column in velocity_columns:

        if df[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )

        if (df[column] < 0).any():
            raise ValueError(
                f"{column} contains negative values."
            )

        print(
            f"{column}: PASSED"
        )

    # --------------------------------------------------------
    # 10. Display statistics
    # --------------------------------------------------------

    print("\nVelocity statistics:")

    print(
        df[
            velocity_columns
        ].describe()
    )

    print("\nSample velocity features:")

    print(
        df[
            [
                "timestamp",
                "transactions_last_1h",
                "transactions_last_6h",
                "transactions_last_24h",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 11. Save output
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
    # 12. Output verification
    # --------------------------------------------------------

    verification_df = pd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    expected_columns = (
        original_columns + 3
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
            "Row count changed during velocity engineering."
        )

    if len(verification_df.columns) != expected_columns:
        raise ValueError(
            "Unexpected number of output columns."
        )

    print("Row-count validation : PASSED")
    print("Column validation     : PASSED")
    print("Velocity validation   : PASSED")

    print("\n" + "=" * 60)
    print("VELOCITY FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    create_velocity_features()