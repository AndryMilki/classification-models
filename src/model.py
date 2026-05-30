from pathlib import Path

import joblib

try:
    from .data import load_data
    from .evaluate import (
        build_model_comparison,
        evaluate_baseline_model,
        evaluate_threshold_model,
        plot_confusion_matrix,
        plot_pr_curves,
        save_metrics_json,
        save_model_comparison,
    )
    from .explain import plot_feature_importance, plot_shap_summary
    from .preprocessing import make_train_val_test_split
    from .train import build_baselines, train_candidate_models
except ImportError:
    from data import load_data
    from evaluate import (
        build_model_comparison,
        evaluate_baseline_model,
        evaluate_threshold_model,
        plot_confusion_matrix,
        plot_pr_curves,
        save_metrics_json,
        save_model_comparison,
    )
    from explain import plot_feature_importance, plot_shap_summary
    from preprocessing import make_train_val_test_split
    from train import build_baselines, train_candidate_models


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"


def save_best_model(model_name, model, threshold, feature_columns):
    MODELS_DIR.mkdir(exist_ok=True)

    joblib.dump(model, MODELS_DIR / "best_model.joblib")
    joblib.dump(
        {
            "model_name": model_name,
            "threshold": threshold,
            "feature_columns": feature_columns,
        },
        MODELS_DIR / "model_metadata.joblib",
    )


def main():
    REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test = make_train_val_test_split(X, y)

    baselines = build_baselines()
    trained_models = train_candidate_models(X_train, y_train)

    baseline_results = [
        evaluate_baseline_model(name, model, X_train, y_train, X_test, y_test)
        for name, model in baselines.items()
    ]

    model_results = [
        evaluate_threshold_model(name, model, X_val, y_val, X_test, y_test)
        for name, model in trained_models.items()
    ]

    all_results = baseline_results + model_results
    comparison = build_model_comparison(all_results)
    save_model_comparison(comparison, REPORTS_DIR / "model_comparison.csv")

    candidate_results = [
        result for result in model_results if result["model_name"] in trained_models
    ]
    best_result = max(candidate_results, key=lambda result: result["validation_f1"])
    best_name = best_result["model_name"]
    best_model = trained_models[best_name]

    save_metrics_json(best_result, REPORTS_DIR / "metrics.json")
    save_best_model(
        best_name,
        best_model,
        best_result["threshold"],
        list(X.columns),
    )

    plot_confusion_matrix(
        y_test,
        best_result["y_pred"],
        FIGURES_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix: {best_result['display_name']}",
    )
    plot_confusion_matrix(
        y_test,
        best_result["y_pred"],
        REPORTS_DIR / "confusion_matrix.png",
        title=f"Confusion Matrix: {best_result['display_name']}",
    )

    plot_pr_curves(all_results, y_test, FIGURES_DIR / "pr_curve.png")
    plot_pr_curves(all_results, y_test, REPORTS_DIR / "pr_curve.png")

    plot_feature_importance(
        best_name,
        best_model,
        X.columns,
        FIGURES_DIR / "feature_importance.png",
    )
    plot_feature_importance(
        best_name,
        best_model,
        X.columns,
        REPORTS_DIR / "feature_importance.png",
    )

    shap_saved = plot_shap_summary(
        best_name,
        best_model,
        X_test,
        FIGURES_DIR / "shap_summary.png",
    )

    if not shap_saved:
        tree_candidates = sorted(
            (
                result
                for result in candidate_results
                if result["model_name"] != best_name
            ),
            key=lambda result: result["validation_f1"],
            reverse=True,
        )

        for result in tree_candidates:
            if plot_shap_summary(
                result["model_name"],
                trained_models[result["model_name"]],
                X_test,
                FIGURES_DIR / "shap_summary.png",
            ):
                break

    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model: {best_result['display_name']}")
    print(f"Validation threshold: {best_result['threshold']:.4f}")
    print(f"Test F1: {best_result['f1']:.4f}")
    print(f"Reports saved to: {REPORTS_DIR}")
    print(f"Best model saved to: {MODELS_DIR / 'best_model.joblib'}")


if __name__ == "__main__":
    main()
