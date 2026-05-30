import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


def build_baselines():
    return {
        "all_non_churn_baseline": DummyClassifier(strategy="constant", constant=0),
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression_no_smote": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
    }


def build_candidate_models(y_train):
    models = {
        "logistic_regression": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(max_iter=1000, random_state=42),
                ),
            ]
        ),
        "random_forest": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                ("clf", RandomForestClassifier(random_state=42)),
            ]
        ),
        "gradient_boosting": ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                ("clf", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }

    if XGBClassifier is not None:
        positive_count = y_train.sum()
        negative_count = len(y_train) - positive_count
        scale_pos_weight = negative_count / positive_count if positive_count else 1

        models["xgboost"] = ImbPipeline(
            [
                ("smote", SMOTE(random_state=42)),
                (
                    "clf",
                    XGBClassifier(
                        random_state=42,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    else:
        scale_pos_weight = 1

    param_grids = {
        "logistic_regression": {
            "clf__C": [0.1, 1.0, 10.0],
            "clf__class_weight": [None, "balanced"],
        },
        "random_forest": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [10, 20, None],
            "clf__min_samples_split": [2, 5],
            "clf__class_weight": [None, "balanced"],
        },
        "gradient_boosting": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [2, 3],
        },
    }

    if XGBClassifier is not None:
        param_grids["xgboost"] = {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.03, 0.1],
            "clf__max_depth": [3, 5],
            "clf__scale_pos_weight": [1, scale_pos_weight],
        }

    return models, param_grids


def train_candidate_models(X_train, y_train):
    models, param_grids = build_candidate_models(y_train)
    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining {name}...")

        grid = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring="f1",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )

        grid.fit(X_train, y_train)
        trained_models[name] = grid.best_estimator_

        print(f"Best parameters for {name}: {grid.best_params_}")
        print(f"Best CV F1 for {name}: {grid.best_score_:.4f}")

    return trained_models
