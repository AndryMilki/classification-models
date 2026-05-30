from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.joblib"
METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.joblib"


class ChurnInput(BaseModel):
    tenure: float = Field(..., ge=0)
    monthlycharges: float = Field(..., ge=0)
    contract: str = "Month-to-month"
    internetservice: str = "Fiber optic"
    paymentmethod: str = "Electronic check"
    gender: int = 0
    seniorcitizen: int = 0
    partner: int = 0
    dependents: int = 0
    phoneservice: int = 1
    multiplelines: int = 0
    onlinesecurity: int = 0
    onlinebackup: int = 0
    deviceprotection: int = 0
    techsupport: int = 0
    streamingtv: int = 0
    streamingmovies: int = 0
    paperlessbilling: int = 1


app = FastAPI(title="Customer Churn Classification API")


def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    return model, metadata


def normalize(value):
    return value.strip().lower()


def build_feature_row(payload, feature_columns):
    row = {feature: 0 for feature in feature_columns}

    direct_features = [
        "gender",
        "seniorcitizen",
        "partner",
        "dependents",
        "tenure",
        "phoneservice",
        "multiplelines",
        "onlinesecurity",
        "onlinebackup",
        "deviceprotection",
        "techsupport",
        "streamingtv",
        "streamingmovies",
        "paperlessbilling",
        "monthlycharges",
    ]

    for feature in direct_features:
        if feature in row:
            row[feature] = getattr(payload, feature)

    contract = normalize(payload.contract)
    internet_service = normalize(payload.internetservice)
    payment_method = normalize(payload.paymentmethod)

    if "is_month_to_month" in row:
        row["is_month_to_month"] = int(contract == "month-to-month")
    if "contract_One year" in row:
        row["contract_One year"] = int(contract == "one year")
    if "contract_Two year" in row:
        row["contract_Two year"] = int(contract == "two year")

    if "internetservice_Fiber optic" in row:
        row["internetservice_Fiber optic"] = int(internet_service == "fiber optic")
    if "internetservice_No" in row:
        row["internetservice_No"] = int(internet_service == "no")

    if "paymentmethod_Credit card (automatic)" in row:
        row["paymentmethod_Credit card (automatic)"] = int(
            payment_method == "credit card (automatic)"
        )
    if "paymentmethod_Electronic check" in row:
        row["paymentmethod_Electronic check"] = int(payment_method == "electronic check")
    if "paymentmethod_Mailed check" in row:
        row["paymentmethod_Mailed check"] = int(payment_method == "mailed check")

    if "high_risk" in row:
        row["high_risk"] = int(
            contract == "month-to-month" and payment_method == "electronic check"
        )

    return pd.DataFrame([row], columns=feature_columns)


@app.post("/predict")
def predict(payload: ChurnInput):
    model, metadata = load_artifacts()
    feature_columns = metadata["feature_columns"]
    threshold = metadata["threshold"]

    X = build_feature_row(payload, feature_columns)
    churn_probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(churn_probability >= threshold)

    return {
        "churn_probability": round(churn_probability, 4),
        "prediction": prediction,
        "threshold": round(threshold, 4),
        "model": metadata["model_name"],
    }
