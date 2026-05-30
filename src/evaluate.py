import json

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


DISPLAY_NAMES = {
    "all_non_churn_baseline": "All non-churn baseline",
    "dummy_baseline": "Dummy baseline",
    "logistic_regression_no_smote": "Logistic Regression no SMOTE",
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
}


def get_positive_class_scores(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)

        if proba.shape[1] == 1:
            return proba[:, 0]

        classes = getattr(model, "classes_", None)
        if classes is None and hasattr(model, "named_steps"):
            classes = model.named_steps["clf"].classes_

        class_index = list(classes).index(1)
        return proba[:, class_index]

    return model.decision_function(X)


def find_best_threshold(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    if len(thresholds) == 0:
        return 0.5, 0.0

    f1_scores = 2 * precision[:-1] * recall[:-1] / (
        precision[:-1] + recall[:-1] + 1e-9
    )
    best_idx = f1_scores.argmax()

    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def score_predictions(y_true, y_pred, y_proba):
    return {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_proba)),
    }


def evaluate_baseline_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = get_positive_class_scores(model, X_test)
    scores = score_predictions(y_test, y_pred, y_proba)

    return {
        "model_name": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "threshold": 0.5,
        "validation_f1": None,
        "y_pred": y_pred,
        "y_proba": y_proba,
        **scores,
    }


def evaluate_threshold_model(name, model, X_val, y_val, X_test, y_test):
    val_proba = get_positive_class_scores(model, X_val)
    threshold, validation_f1 = find_best_threshold(y_val, val_proba)

    test_proba = get_positive_class_scores(model, X_test)
    test_pred = (test_proba >= threshold).astype(int)
    scores = score_predictions(y_test, test_pred, test_proba)

    return {
        "model_name": name,
        "display_name": DISPLAY_NAMES.get(name, name),
        "threshold": threshold,
        "validation_f1": validation_f1,
        "y_pred": test_pred,
        "y_proba": test_proba,
        **scores,
    }


def build_model_comparison(results):
    rows = []

    for result in results:
        rows.append(
            {
                "Model": result["display_name"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1": result["f1"],
                "PR AUC": result["pr_auc"],
                "Threshold": result["threshold"],
                "Validation F1": result["validation_f1"],
            }
        )

    return pd.DataFrame(rows)


def save_model_comparison(comparison, path):
    comparison.to_csv(path, index=False)


def save_metrics_json(best_result, path):
    payload = {
        "best_model": best_result["model_name"],
        "threshold": round(best_result["threshold"], 4),
        "precision": round(best_result["precision"], 4),
        "recall": round(best_result["recall"], 4),
        "f1": round(best_result["f1"], 4),
        "pr_auc": round(best_result["pr_auc"], 4),
    }

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def plot_confusion_matrix(y_true, y_pred, path, title):
    matrix = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["Non-churn", "Churn"],
        yticklabels=["Non-churn", "Churn"],
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def plot_pr_curves(results, y_true, path):
    plt.figure(figsize=(8, 6))

    for result in results:
        precision, recall, _ = precision_recall_curve(y_true, result["y_proba"])
        plt.plot(
            recall,
            precision,
            label=f"{result['display_name']} (AP={result['pr_auc']:.3f})",
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
