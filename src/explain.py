import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_final_estimator(model):
    if hasattr(model, "named_steps") and "clf" in model.named_steps:
        return model.named_steps["clf"]

    return model


def get_feature_importance(model, feature_names):
    estimator = get_final_estimator(model)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        importances = abs(estimator.coef_).ravel()
    else:
        return None

    return (
        pd.DataFrame({"feature": list(feature_names), "importance": importances})
        .sort_values("importance", ascending=False)
        .head(20)
    )


def plot_feature_importance(model_name, model, feature_names, path):
    feature_importance = get_feature_importance(model, feature_names)

    if feature_importance is None:
        print(f"Feature importance is not available for {model_name}.")
        return False

    plt.figure(figsize=(8, 7))
    sns.barplot(
        data=feature_importance,
        x="importance",
        y="feature",
        color="#2f6f8f",
    )
    plt.title(f"Feature Importance: {model_name}")
    plt.xlabel("Importance")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return True


def plot_shap_summary(model_name, model, X_test, path, max_rows=500):
    try:
        import shap
    except ImportError:
        print("SHAP is not installed. Skipping SHAP summary plot.")
        return False

    estimator = get_final_estimator(model)

    if not hasattr(estimator, "feature_importances_"):
        print(f"SHAP summary is skipped for {model_name}: tree explainer target not found.")
        return False

    X_sample = X_test.sample(
        n=min(max_rows, len(X_test)),
        random_state=42,
    )

    try:
        explainer = shap.Explainer(estimator, X_sample)
        shap_values = explainer(X_sample)

        plt.figure()
        shap.summary_plot(shap_values, X_sample, show=False, max_display=20)
        plt.tight_layout()
        plt.savefig(path, dpi=160, bbox_inches="tight")
        plt.close()
        return True
    except Exception as exc:
        print(f"Could not create SHAP summary for {model_name}: {exc}")
        return False
