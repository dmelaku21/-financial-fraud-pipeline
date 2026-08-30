from pathlib import Path

import pandas as pd
import great_expectations as gx


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "transactions_with_amount_features.parquet"
)


print("=" * 60)
print("GREAT EXPECTATIONS VALIDATION")
print("=" * 60)

print(f"Dataset: {DATA_PATH}")

# ------------------------------------------------------------
# 1. Check dataset exists
# ------------------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}"
    )

# ------------------------------------------------------------
# 2. Load dataset
# ------------------------------------------------------------

df = pd.read_parquet(DATA_PATH)

print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")

# ------------------------------------------------------------
# 3. Create / reuse GX Data Context
# ------------------------------------------------------------

context = gx.get_context(mode="file")

# Reuse datasource if it already exists
datasource_name = "fraud_pandas"

if datasource_name in context.data_sources.all():
    datasource = context.data_sources.get(datasource_name)
    print("\nExisting datasource reused.")
else:
    datasource = context.data_sources.add_pandas(
        name=datasource_name
    )
    print("\nDatasource created.")

# ------------------------------------------------------------
# 4. Create / reuse data asset
# ------------------------------------------------------------

asset_name = "fraud_transactions"

try:
    data_asset = datasource.get_asset(asset_name)
    print("Existing data asset reused.")
except Exception:
    data_asset = datasource.add_dataframe_asset(
        name=asset_name
    )
    print("Data asset created.")

# ------------------------------------------------------------
# 5. Create batch definition
# ------------------------------------------------------------

batch_definition_name = "whole_dataset"

try:
    batch_definition = data_asset.get_batch_definition(
        batch_definition_name
    )
    print("Existing batch definition reused.")
except Exception:
    batch_definition = (
        data_asset.add_batch_definition_whole_dataframe(
            batch_definition_name
        )
    )
    print("Batch definition created.")

# ------------------------------------------------------------
# 6. Create batch
# ------------------------------------------------------------

batch = batch_definition.get_batch(
    batch_parameters={
        "dataframe": df
    }
)

print("\nDataset loaded successfully.")
print("Great Expectations batch created successfully.")

print("\n" + "=" * 60)
print("VALIDATION INPUT READY")
print("=" * 60)