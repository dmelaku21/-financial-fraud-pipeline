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
    / "clean_transactions.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_amount_features.parquet"
)


# ============================================================
# Amount transformation
# ============================================================

def transform_amount() -> None:
    """Create a log-transformed transaction amount feature."""

    print("=" * 60)
    print("AMOUNT TRANSFORMATION")
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
    # 2. Load clean dataset
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    original_rows = len(df)
    original_columns = len(df.columns)

    print("\nInput dataset:")
    print(f"Rows    : {original_rows:,}")
    print(f"Columns : {original_columns:,}")

    # --------------------------------------------------------
    # 3. Validate Amount column
    # --------------------------------------------------------

    if "Amount" not in df.columns:
        raise ValueError(
            "Required column 'Amount' is missing."
        )

    if df["Amount"].isna().any():
        raise ValueError(
            "Amount contains missing values."
        )

    if (df["Amount"] < 0).any():
        raise ValueError(
            "Amount contains negative values."
        )

    # --------------------------------------------------------
    # 4. Preserve original Amount
    # --------------------------------------------------------

    original_amount = df["Amount"].copy()

    # --------------------------------------------------------
    # 5. Apply log1p transformation
    # --------------------------------------------------------

    df["amount_log"] = np.log1p(
        df["Amount"]
    )

    # --------------------------------------------------------
    # 6. Validate transformed feature
    # --------------------------------------------------------

    if df["amount_log"].isna().any():
        raise ValueError(
            "amount_log contains missing values."
        )

    if not np.isfinite(
        df["amount_log"]
    ).all():
        raise ValueError(
            "amount_log contains infinite values."
        )

    # Verify original Amount was not changed
    if not df["Amount"].equals(original_amount):
        raise ValueError(
            "Original Amount column was modified."
        )

    # --------------------------------------------------------
    # 7. Display transformation statistics
    # --------------------------------------------------------

    print("\nOriginal Amount statistics:")
    print(df["Amount"].describe())

    print("\nTransformed amount_log statistics:")
    print(df["amount_log"].describe())

    print("\nSample transformation:")

    sample = df[
        ["Amount", "amount_log"]
    ].head(10)

    print(sample.to_string(index=False))

    # --------------------------------------------------------
    # 8. Create output directory
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # 9. Save transformed dataset
    # --------------------------------------------------------

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow"
    )

    # --------------------------------------------------------
    # 10. Verify output
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
            "Row count changed during transformation."
        )

    if len(verification_df.columns) != (
        original_columns + 1
    ):
        raise ValueError(
            "Unexpected number of output columns."
        )

    if "Amount" not in verification_df.columns:
        raise ValueError(
            "Original Amount column missing."
        )

    if "amount_log" not in verification_df.columns:
        raise ValueError(
            "amount_log column missing."
        )

    print("Row-count validation : PASSED")
    print("Column validation     : PASSED")
    print("Amount preservation   : PASSED")
    print("Transformation        : PASSED")

    print("\n" + "=" * 60)
    print("AMOUNT TRANSFORMATION COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    transform_amount()