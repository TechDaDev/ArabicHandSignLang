import pandas as pd
import os

path = "./dataset/Arabic Sign Language Letters Dataset.csv"

if not os.path.exists(path):
    print(f"Error: File not found at {path}")
else:
    df = pd.read_csv(path)

    print("--- Step 1: Load and Inspect ---")
    print("Shape:", df.shape)
    print("Columns (first 10):", df.columns[:10].tolist())
    print("Columns (last 10):", df.columns[-10:].tolist())
    print(df.head(3))

    print("\n--- Step 2: Labels and Class Balance ---")
    print("Unique letters:", df["letter"].nunique())
    print(df["letter"].value_counts().head(20))
    print("Missing values:", df.isna().sum().sum())

    print("\n--- Step 3: Sanity Checks ---")
    # Ensure all feature columns are numeric
    feature_cols = [c for c in df.columns if c != "letter"]
    print(df[feature_cols].dtypes.value_counts())

    # Basic range check
    print(df[feature_cols].describe().T[["min","max"]].head(12))
