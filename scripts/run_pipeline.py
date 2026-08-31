"""
Financial Fraud Detection Pipeline
==================================

Reproducible pipeline entry point.

Pipeline flow:
1. Batch ingestion
2. Transaction date creation
3. Data cleaning
4. Amount transformation
5. Temporal feature engineering
6. Velocity feature engineering
7. Final feature engineering
8. SQLite storage
"""
from pathlib import Path
import sys

# Add project root to Python import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.batch_ingestion import ingest_batch
from src.processing.create_transaction_date import create_transaction_date
from src.processing.clean_transactions import clean_transactions
from src.features.amount_transformation import transform_amount
from src.features.temporal_features import create_temporal_features
from src.features.velocity_features import create_velocity_features
from src.features.build_final_features import build_final_features
from src.storage.sqlite_storage import store_in_sqlite


def run_pipeline() -> None:
    """Execute the complete financial fraud detection pipeline."""

    print("=" * 70)
    print("FINANCIAL FRAUD DETECTION PIPELINE")
    print("=" * 70)

    print("\n[1/8] Batch ingestion...")
    ingest_batch()

    print("\n[2/8] Creating transaction dates...")
    create_transaction_date()

    print("\n[3/8] Cleaning transactions...")
    clean_transactions()

    print("\n[4/8] Transforming transaction amounts...")
    transform_amount()

    print("\n[5/8] Creating temporal features...")
    create_temporal_features()

    print("\n[6/8] Creating velocity features...")
    create_velocity_features()

    print("\n[7/8] Building final features...")
    build_final_features()

    print("\n[8/8] Storing transactions in SQLite...")
    store_in_sqlite()

    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
