# Customer Churn Classification Models

## Project Overview

This project predicts customer churn using several supervised machine learning models.

The goal is to compare classical ML classifiers, handle class imbalance, tune hyperparameters, optimize the decision threshold, and evaluate models using business-relevant classification metrics.

## Dataset

The dataset contains customer-level telecom features such as contract type, tenure, monthly charges, internet service, payment method, and churn label.

Target variable:

- `churn`: 1 if the customer left, 0 otherwise.

## Models

The following models are trained and compared:

- All non-churn baseline
- DummyClassifier baseline
- Logistic Regression without SMOTE baseline
- Logistic Regression
- Random Forest
- Gradient Boosting
- XGBoost

## Methodology

- Data preprocessing
- Train/validation/test split with stratification
- SMOTE for class imbalance
- Hyperparameter tuning with GridSearchCV
- Model comparison using F1-score
- Precision, recall, PR AUC, confusion matrix
- Threshold optimization on the validation set only
- Final model evaluation on the test set only
- Feature importance analysis
- SHAP summary analysis
- Best model saved with joblib

## Results

Running `python src/model.py` creates:

- `reports/metrics.json`
- `reports/model_comparison.csv`
- `reports/figures/confusion_matrix.png`
- `reports/figures/pr_curve.png`
- `reports/figures/feature_importance.png`
- `reports/figures/shap_summary.png`

| Model | Precision | Recall | F1 | PR AUC |
|---|---:|---:|---:|---:|
| All non-churn baseline | 0.000 | 0.000 | 0.000 | 0.265 |
| Dummy baseline | 0.000 | 0.000 | 0.000 | 0.265 |
| Logistic Regression no SMOTE | 0.603 | 0.511 | 0.553 | 0.579 |
| Logistic Regression | 0.514 | 0.707 | 0.595 | 0.566 |
| Random Forest | 0.508 | 0.714 | 0.593 | 0.580 |
| Gradient Boosting | 0.516 | 0.679 | 0.586 | 0.586 |
| XGBoost | 0.522 | 0.689 | 0.594 | 0.587 |

## Top Churn Drivers

1. contract type
2. tenure
3. monthly charges
4. payment method
5. internet service

## How to Run

```bash
pip install -r requirements.txt
python src/model.py
```

## FastAPI Prediction Endpoint

Start the API:

```bash
uvicorn src.predict:app --reload
```

Example request:

```json
{
  "tenure": 12,
  "monthlycharges": 75.5,
  "contract": "Month-to-month",
  "internetservice": "Fiber optic",
  "paymentmethod": "Electronic check"
}
```

Example response:

```json
{
  "churn_probability": 0.73,
  "prediction": 1,
  "threshold": 0.4431,
  "model": "logistic_regression"
}
```

## Project Structure

```text
classification-models/
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- pyproject.toml
+-- Dockerfile
+-- data/
|   +-- raw/
|   |   +-- raw_data.csv
|   +-- processed/
|       +-- finished_data.csv
+-- notebooks/
|   +-- 01_churn_eda_modeling.ipynb
+-- src/
|   +-- data.py
|   +-- preprocessing.py
|   +-- train.py
|   +-- evaluate.py
|   +-- explain.py
|   +-- predict.py
|   +-- model.py
+-- reports/
|   +-- metrics.json
|   +-- model_comparison.csv
|   +-- figures/
|       +-- confusion_matrix.png
|       +-- pr_curve.png
|       +-- feature_importance.png
|       +-- shap_summary.png
+-- models/
|   +-- best_model.joblib
+-- tests/
    +-- test_data.py
    +-- test_model.py
```

## Key Finding

The best-performing model was Logistic Regression with an optimized validation threshold of 0.4431.

The most important churn drivers were monthly charges, contract type, internet service, tenure, phone service, online security, and tech support.
