import pandas as pd

from src.data import load_data


def test_load_data_splits_features_and_target(tmp_path):
    data_path = tmp_path / "finished_data.csv"
    pd.DataFrame(
        {
            "tenure": [1, 12],
            "monthlycharges": [29.85, 75.5],
            "is_month_to_month": [True, False],
            "churn": [0, 1],
        }
    ).to_csv(data_path, index=False)

    X, y = load_data(data_path)

    assert "churn" not in X.columns
    assert list(y) == [0, 1]
    assert X["is_month_to_month"].tolist() == [1, 0]
