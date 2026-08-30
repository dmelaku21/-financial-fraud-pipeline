import json
import time
from pathlib import Path

import pandas as pd


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "kaggle"
    / "creditcard.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "streaming"
    / "transactions.jsonl"
)

DEFAULT_LIMIT = 1000
DEFAULT_DELAY = 0.01


# ============================================================
# Simulated streaming
# ============================================================

def simulate_stream(
    limit: int = DEFAULT_LIMIT,
    delay: float = DEFAULT_DELAY
) -> None:
    """
    Simulate transaction events arriving sequentially.

    The raw CSV is read without modification. Selected
    transactions are converted into JSON objects and written
    as newline-delimited JSON (JSONL).
    """

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    if limit <= 0:
        raise ValueError("Limit must be greater than zero.")

    if delay < 0:
        raise ValueError("Delay cannot be negative.")

    print("=" * 60)
    print("SIMULATED STREAMING INGESTION")
    print("=" * 60)

    print(f"Input       : {INPUT_PATH}")
    print(f"Output      : {OUTPUT_PATH}")
    print(f"Event limit : {limit:,}")
    print(f"Event delay : {delay} seconds")

    # --------------------------------------------------------
    # Load selected transactions
    # --------------------------------------------------------

    df = pd.read_csv(INPUT_PATH, nrows=limit)

    if df.empty:
        raise ValueError("No transactions were loaded.")

    print(f"\nLoaded transactions: {len(df):,}")

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Generate streaming events
    # --------------------------------------------------------

    generated = 0

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        for _, row in df.iterrows():

            # Convert Pandas/NumPy values to JSON-safe values
            event = {}

            for column, value in row.items():

                if pd.isna(value):
                    event[column] = None
                else:
                    event[column] = value.item() if hasattr(
                        value, "item"
                    ) else value

            # Write one JSON event per line
            file.write(
                json.dumps(event, separators=(",", ":"))
                + "\n"
            )

            file.flush()

            generated += 1

            # Simulate arrival delay
            if delay > 0:
                time.sleep(delay)

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("STREAM SIMULATION COMPLETE")
    print("=" * 60)

    print(f"Generated events: {generated:,}")
    print(f"Output file     : {OUTPUT_PATH}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    simulate_stream()