from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


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

DATABASE_DIR = PROJECT_ROOT / "database"
DATABASE_PATH = DATABASE_DIR / "fraud.db"

TABLE_NAME = "transactions_raw"


# ============================================================
# SQLite storage
# ============================================================

def store_in_sqlite() -> None:
    """Store staging transactions in SQLite."""

    print("=" * 60)
    print("SQLITE STORAGE")
    print("=" * 60)

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input Parquet file not found: {INPUT_PATH}"
        )

    print(f"Input database source: {INPUT_PATH}")
    print(f"SQLite database     : {DATABASE_PATH}")
    print(f"Table               : {TABLE_NAME}")

    # --------------------------------------------------------
    # Load Parquet
    # --------------------------------------------------------

    df = pd.read_parquet(INPUT_PATH)

    print(f"\nLoaded transactions: {len(df):,}")
    print(f"Loaded columns     : {len(df.columns):,}")

    if df.empty:
        raise ValueError("The input dataset is empty.")

    # --------------------------------------------------------
    # Create database directory
    # --------------------------------------------------------

    DATABASE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Create SQLite engine
    # --------------------------------------------------------

    database_url = f"sqlite:///{DATABASE_PATH}"

    engine = create_engine(database_url)

    # --------------------------------------------------------
    # Store data
    # --------------------------------------------------------

    df.to_sql(
        TABLE_NAME,
        engine,
        if_exists="replace",
        index=False
    )

    print("\nData successfully stored in SQLite.")

    # --------------------------------------------------------
    # Verify database
    # --------------------------------------------------------

    with engine.connect() as connection:

        result = connection.exec_driver_sql(
            f"SELECT COUNT(*) FROM {TABLE_NAME}"
        )

        row_count = result.scalar()

    print("\nDatabase verification:")
    print(f"SQLite rows: {row_count:,}")

    if row_count != len(df):
        raise ValueError(
            "SQLite validation failed: row counts do not match."
        )

    print("Row-count validation: PASSED")

    print("\n" + "=" * 60)
    print("SQLITE STORAGE COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    store_in_sqlite()