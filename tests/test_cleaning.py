from pathlib import Path
import pandas as pd


DATASET = Path("data/raw/kaggle/creditcard.csv")


def test_amount_is_numeric():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, usecols=["Amount"])

    assert pd.api.types.is_numeric_dtype(df["Amount"])


def test_no_missing_amount_values():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, usecols=["Amount"])

    assert df["Amount"].notna().all()
