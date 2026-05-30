from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "finished_data.csv"
TARGET_COLUMN = "churn"


def load_data(data_path=DEFAULT_DATA_PATH, target_column=TARGET_COLUMN):
    df = pd.read_csv(data_path)

    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' was not found in {data_path}.")

    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].astype(int)

    bool_columns = X.select_dtypes(include=["bool"]).columns
    X[bool_columns] = X[bool_columns].astype(int)

    return X, y
