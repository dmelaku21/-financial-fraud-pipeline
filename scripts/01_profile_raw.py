import pandas as pd
from pathlib import Path

# ============================================================
# 1. Configuration
# ============================================================

INPUT = Path("data/raw/kaggle/creditcard.csv")

# ============================================================
# 2. Validate input
# ============================================================

if not INPUT.exists():
    raise FileNotFoundError(f"Dataset not found: {INPUT}")

# ============================================================
# 3. Load raw dataset
# ============================================================

df = pd.read_csv(INPUT)

# ============================================================
# 4. Dataset profile
# ============================================================

print("=" * 60)
print("RAW DATASET PROFILE")
print("=" * 60)

print(f"Dataset: {INPUT}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns):,}")

# ------------------------------------------------------------
# Columns
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("COLUMNS")
print("=" * 60)

for i, column in enumerate(df.columns, start=1):
    print(f"{i:2}. {column}")

# ------------------------------------------------------------
# Data types
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)

print(df.dtypes)

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)

missing = df.isna().sum()

print(missing)

print(f"\nTotal missing values: {missing.sum():,}")

# ------------------------------------------------------------
# Duplicate rows
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DUPLICATE ROWS")
print("=" * 60)

duplicates = df.duplicated().sum()

print(f"Duplicate rows: {duplicates:,}")

# ------------------------------------------------------------
# Class distribution
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

class_counts = df["Class"].value_counts().sort_index()

print(class_counts)

# ------------------------------------------------------------
# Class percentages
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS PERCENTAGE")
print("=" * 60)

class_percentages = (
    df["Class"]
    .value_counts(normalize=True)
    .sort_index()
    .mul(100)
)

print(class_percentages.round(4))

# ------------------------------------------------------------
# Amount statistics
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("AMOUNT STATISTICS")
print("=" * 60)

print(df["Amount"].describe())

# ------------------------------------------------------------
# Time statistics
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TIME STATISTICS")
print("=" * 60)

print(df["Time"].describe())

# ------------------------------------------------------------
# Target validation
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("TARGET VALIDATION")
print("=" * 60)

print("Target column: Class")
print(f"Unique target values: {sorted(df['Class'].unique())}")

# ------------------------------------------------------------
# Dataset memory usage
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MEMORY USAGE")
print("=" * 60)

memory_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)

print(f"DataFrame memory usage: {memory_mb:.2f} MB")

print("\n" + "=" * 60)
print("PROFILE COMPLETE")
print("=" * 60)
