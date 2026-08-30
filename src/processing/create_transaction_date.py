from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "batch_transactions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_date.parquet"
)

# IMPORTANT:
# The original dataset does not contain a real calendar date.
# This is an engineered project reference date.
BASE_DATE = pd.Timestamp("2025-01-01")


# ============================================================
# Create transaction date
# ============================================================

def create_transaction_date() -> None:
    """Create a reference timestamp and transaction date."""

    print("=" * 60)
    print("TRANSACTION DATE ENGINEERING")
    print("=" * 60)

    print(f"Input     : {INPUT_PATH}")
    print(f"Output    : {OUTPUT_PATH}")
    print(f"Base date : {BASE_DATE}")

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    # --------------------------------------------------------
    # Load staging data
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nLoaded rows    : {len(df):,}")
    print(f"Loaded columns : {len(df.columns):,}")

    if "Time" not in df.columns:
        raise ValueError(
            "Required column 'Time' is missing."
        )

    # --------------------------------------------------------
    # Create reference timestamp
    # --------------------------------------------------------

    df["timestamp"] = (
        BASE_DATE
        + pd.to_timedelta(df["Time"], unit="s")
    )

    # --------------------------------------------------------
    # Create transaction date
    # --------------------------------------------------------

    df["transaction_date"] = (
        df["timestamp"].dt.date
    )

    # --------------------------------------------------------
    # Validate generated fields
    # --------------------------------------------------------

    if df["timestamp"].isna().any():
        raise ValueError(
            "Generated timestamp contains missing values."
        )

    if df["transaction_date"].isna().any():
        raise ValueError(
            "Generated transaction_date contains missing values."
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\nGenerated columns:")
    print("  timestamp")
    print("  transaction_date")

    print("\nTimestamp range:")
    print(f"  Start: {df['timestamp'].min()}")
    print(f"  End  : {df['timestamp'].max()}")

    print("\nTransaction dates:")
    print(df["transaction_date"].value_counts().sort_index())

    # --------------------------------------------------------
    # Create processed directory
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save processed data
    # --------------------------------------------------------

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow"
    )

    print(f"\nSaved processed dataset:")
    print(f"  {OUTPUT_PATH}")

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    verification_df = pd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    print("\nOutput verification:")
    print(f"  Rows    : {len(verification_df):,}")
    print(f"  Columns : {len(verification_df.columns):,}")

    if len(verification_df) != len(df):
        raise ValueError(
            "Row count changed during date engineering."
        )

    if "timestamp" not in verification_df.columns:
        raise ValueError(
            "timestamp column missing from output."
        )

    if "transaction_date" not in verification_df.columns:
        raise ValueError(
            "transaction_date column missing from output."
        )

    print("  Validation: PASSED")

    print("\n" + "=" * 60)
    print("TRANSACTION DATE ENGINEERING COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    create_transaction_date()