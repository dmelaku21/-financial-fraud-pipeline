from pathlib import Path
import pandas as pd


DATASET = Path("data/raw/kaggle/creditcard.csv")


def test_target_has_valid_values():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, usecols=["Class"])

    assert set(df["Class"].dropna().unique()).issubset({0, 1})


def test_dataset_has_records():
    assert DATASET.exists(), "Dataset is missing"

    df = pd.read_csv(DATASET, usecols=["Class"])

    assert len(df) > 0
