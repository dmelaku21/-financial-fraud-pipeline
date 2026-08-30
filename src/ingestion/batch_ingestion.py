from pathlib import Path
import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "kaggle" / "creditcard.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "staging" / "batch_transactions.parquet"


# ============================================================
# Batch ingestion
# ============================================================

def ingest_batch() -> None:
    """Load the raw CSV dataset and store it as staging Parquet."""

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    print("=" * 60)
    print("BATCH INGESTION")
    print("=" * 60)

    print(f"Input : {INPUT_PATH}")
    print(f"Output: {OUTPUT_PATH}")

    # --------------------------------------------------------
    # Read raw CSV
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_PATH)

    print(f"\nLoaded transactions: {len(df):,}")
    print(f"Loaded columns     : {len(df.columns):,}")

    # --------------------------------------------------------
    # Basic ingestion validation
    # --------------------------------------------------------

    if df.empty:
        raise ValueError("The input dataset is empty.")

    if "Class" not in df.columns:
        raise ValueError(
            "Required target column 'Class' is missing."
        )

    print("\nInput validation: PASSED")

    # --------------------------------------------------------
    # Create staging directory
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Write Parquet
    # --------------------------------------------------------

    df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow"
    )

    print("\nParquet file created successfully.")

    # --------------------------------------------------------
    # Verify output
    # --------------------------------------------------------

    staging_df = pd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    print("\nOutput verification:")
    print(f"Rows    : {len(staging_df):,}")
    print(f"Columns : {len(staging_df.columns):,}")

    if staging_df.shape != df.shape:
        raise ValueError(
            "Validation failed: input and output shapes differ."
        )

    if staging_df.columns.tolist() != df.columns.tolist():
        raise ValueError(
            "Validation failed: input and output columns differ."
        )

    print("\nOutput validation: PASSED")

    print("\n" + "=" * 60)
    print("BATCH INGESTION COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    ingest_batch()