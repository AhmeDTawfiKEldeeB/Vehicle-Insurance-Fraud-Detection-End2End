from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel, Field

from src.predict import FraudPredictor
from src.risk import classify_risk


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1, description="List of claim records.")


class PredictionResult(BaseModel):
    risk_level: str
    risk_label: str
    probability_score: float
    explanation: str
    fraud_probability: float
    fraud_prediction: int


class PredictResponse(BaseModel):
    predictions: list[PredictionResult]
    threshold: float


app = FastAPI(title="Vehicle Insurance Fraud API", version="1.0.0")


@lru_cache
def get_predictor() -> FraudPredictor:
    return FraudPredictor()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
def model_metrics() -> dict[str, Any]:
    try:
        return get_predictor().metrics
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        predictor = get_predictor()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    df = pd.DataFrame(payload.records)
    preds = predictor.predict_dataframe(df)

    result = []
    for _, row in preds.iterrows():
        prob = float(row["fraud_probability"])
        risk_result = classify_risk(prob)
        result.append(
            {
                "risk_level": str(risk_result["risk_level"]),
                "risk_label": str(risk_result["risk_label"]),
                "probability_score": float(risk_result["probability_score"]),
                "explanation": str(risk_result["explanation"]),
                "fraud_probability": prob,
                "fraud_prediction": int(row["fraud_prediction"]),
            }
        )

    return PredictResponse(predictions=result, threshold=float(predictor.threshold))
