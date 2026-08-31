from pathlib import Path
import pandas as pd


DATASET = Path("data/raw/kaggle/creditcard.csv")


def test_dataset_exists():
    assert DATASET.exists(), f"Dataset not found: {DATASET}"


def test_required_columns_exist():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, nrows=5)

    required_columns = {"Time", "Amount", "Class"}
    assert required_columns.issubset(df.columns)
