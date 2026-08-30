from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    lag,
    count,
)
from pyspark.sql.window import Window


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    PROJECT_ROOT
    / "data"
    / "staging"
    / "batch_transactions.parquet"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "spark_transactions.parquet"
)


spark = (
    SparkSession.builder
    .appName("FinancialFraudPipeline")
    .master("local[*]")
    .getOrCreate()
)

try:
    print("=" * 60)
    print("PYSPARK WINDOW PROCESSING")
    print("=" * 60)

    print(f"Input : {INPUT}")
    print(f"Output: {OUTPUT}")

    # --------------------------------------------------------
    # 1. Read Parquet
    # --------------------------------------------------------

    df = spark.read.parquet(str(INPUT))

    row_count = df.count()

    print(f"\nRows loaded: {row_count:,}")

    # --------------------------------------------------------
    # 2. Validate input
    # --------------------------------------------------------

    required_columns = [
        "Time",
        "Amount",
        "Class",
    ]

    missing = [
        c for c in required_columns
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )

    print("Input validation: PASSED")

    # --------------------------------------------------------
    # 3. Basic transformation
    # --------------------------------------------------------

    df = df.withColumn(
        "amount_positive",
        (col("Amount") > 0).cast("int")
    )

    # --------------------------------------------------------
    # 4. Spark Window
    # --------------------------------------------------------

    window = (
        Window
        .orderBy("Time")
    )

    # Previous transaction time
    df = df.withColumn(
        "previous_time",
        lag("Time").over(window)
    )

    # Time since previous observed transaction
    df = df.withColumn(
        "time_since_previous_transaction",
        (
            col("Time")
            - col("previous_time")
        )
    )

    # First transaction has no previous transaction
    df = df.fillna({
        "time_since_previous_transaction": 0.0
    })

    # --------------------------------------------------------
    # 5. Rolling transaction count - 1 hour
    # --------------------------------------------------------
    #
    # Time is measured in seconds.
    # 1 hour = 3,600 seconds.
    #
    # --------------------------------------------------------

    rolling_window = (
        Window
        .orderBy(col("Time"))
        .rangeBetween(-3600, 0)
    )

    df = df.withColumn(
        "transactions_last_1h",
        count("*").over(rolling_window)
    )

    print("Window processing: PASSED")

    # --------------------------------------------------------
    # 6. Display sample
    # --------------------------------------------------------

    print("\nWindow feature sample:")

    df.select(
        "Time",
        "Amount",
        "previous_time",
        "time_since_previous_transaction",
        "transactions_last_1h",
    ).show(10, truncate=False)

    # --------------------------------------------------------
    # 7. Save output
    # --------------------------------------------------------

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.write.mode("overwrite").parquet(
        str(OUTPUT)
    )

    # --------------------------------------------------------
    # 8. Verify output
    # --------------------------------------------------------

    verification = spark.read.parquet(
        str(OUTPUT)
    )

    output_count = verification.count()

    print("\nOutput verification:")
    print(f"Rows: {output_count:,}")

    if output_count != row_count:
        raise ValueError(
            "Row count changed during Spark processing."
        )

    # Verify new features
    window_features = [
        "previous_time",
        "time_since_previous_transaction",
        "transactions_last_1h",
    ]

    missing_features = [
        feature
        for feature in window_features
        if feature not in verification.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing window features: {missing_features}"
        )

    print("Row-count validation : PASSED")
    print("Window-feature validation: PASSED")
    print("Output validation     : PASSED")

    print("\n" + "=" * 60)
    print("PYSPARK WINDOW PROCESSING COMPLETE")
    print("=" * 60)

finally:
    spark.stop()