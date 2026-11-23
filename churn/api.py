"""Serve the trained churn model.

Usage: uvicorn churn.api:app   (after python -m churn.train)
"""

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException

app = FastAPI(title="Churn Prediction API", version="1.0.0")
_bundle = None


def bundle():
    global _bundle
    if _bundle is None:
        _bundle = joblib.load("churn-model.joblib")
    return _bundle


@app.get("/api/health")
def health():
    return {"status": "ok", "model": bundle()["model_name"]}


@app.post("/api/predict")
def predict(customer: dict):
    features = bundle()["features"]
    missing = [f for f in features if f not in customer]
    if missing:
        raise HTTPException(status_code=422, detail=f"missing features: {missing}")
    frame = pd.DataFrame([{f: customer[f] for f in features}])
    proba = float(bundle()["pipeline"].predict_proba(frame)[:, 1][0])
    return {"churn_probability": round(proba, 4), "churn": proba >= 0.5, "model": bundle()["model_name"]}
