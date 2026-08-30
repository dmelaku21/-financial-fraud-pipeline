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
    / "transactions_with_amount_features.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_temporal_features.parquet"
)


# ============================================================
# Temporal feature engineering
# ============================================================

def create_temporal_features() -> None:
    """Create time-based transaction features."""

    print("=" * 60)
    print("TEMPORAL FEATURE ENGINEERING")
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

    invalid_timestamps = df["timestamp"].isna().sum()

    if invalid_timestamps > 0:
        raise ValueError(
            f"Invalid timestamps detected: "
            f"{invalid_timestamps:,}"
        )

    print("\nTimestamp validation: PASSED")

    # --------------------------------------------------------
    # 4. Create temporal features
    # --------------------------------------------------------

    df["hour_of_day"] = df["timestamp"].dt.hour

    df["day_of_week"] = (
        df["timestamp"].dt.dayofweek
    )

    df["is_weekend"] = (
        df["day_of_week"] >= 5
    ).astype("int8")

    df["is_night"] = (
        (df["hour_of_day"] < 6)
        | (df["hour_of_day"] >= 22)
    ).astype("int8")

    # --------------------------------------------------------
    # 5. Validate feature values
    # --------------------------------------------------------

    if not df["hour_of_day"].between(
        0, 23
    ).all():
        raise ValueError(
            "hour_of_day contains invalid values."
        )

    if not df["day_of_week"].between(
        0, 6
    ).all():
        raise ValueError(
            "day_of_week contains invalid values."
        )

    if not df["is_weekend"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "is_weekend contains invalid values."
        )

    if not df["is_night"].isin(
        [0, 1]
    ).all():
        raise ValueError(
            "is_night contains invalid values."
        )

    print("Temporal feature validation: PASSED")

    # --------------------------------------------------------
    # 6. Display feature distributions
    # --------------------------------------------------------

    print("\nHour-of-day distribution:")
    print(
        df["hour_of_day"]
        .value_counts()
        .sort_index()
    )

    print("\nDay-of-week distribution:")
    print(
        df["day_of_week"]
        .value_counts()
        .sort_index()
    )

    print("\nWeekend distribution:")
    print(
        df["is_weekend"]
        .value_counts()
        .sort_index()
    )

    print("\nNight-time distribution:")
    print(
        df["is_night"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # 7. Display sample
    # --------------------------------------------------------

    print("\nSample temporal features:")

    print(
        df[
            [
                "timestamp",
                "hour_of_day",
                "day_of_week",
                "is_weekend",
                "is_night",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # 8. Save dataset
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
    # 9. Verify output
    # --------------------------------------------------------

    verification_df = pd.read_parquet(
        OUTPUT_PATH,
        engine="pyarrow"
    )

    expected_columns = (
        original_columns + 4
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
            "Row count changed during feature engineering."
        )

    if len(verification_df.columns) != expected_columns:
        raise ValueError(
            "Unexpected number of output columns."
        )

    required_features = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night",
    ]

    for feature in required_features:

        if feature not in verification_df.columns:
            raise ValueError(
                f"Missing output feature: {feature}"
            )

    if verification_df[
        required_features
    ].isna().any().any():

        raise ValueError(
            "Temporal features contain missing values."
        )

    print("Row-count validation : PASSED")
    print("Column validation     : PASSED")
    print("Feature validation    : PASSED")

    print("\n" + "=" * 60)
    print("TEMPORAL FEATURE ENGINEERING COMPLETE")
    print("=" * 60)


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    create_temporal_features()