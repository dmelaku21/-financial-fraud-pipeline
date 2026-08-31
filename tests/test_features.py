from pathlib import Path
import pandas as pd


DATASET = Path("data/raw/kaggle/creditcard.csv")


def test_target_column_exists():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, nrows=5)

    assert "Class" in df.columns


def test_amount_feature_exists():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, nrows=5)

    assert "Amount" in df.columns
