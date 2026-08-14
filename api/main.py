from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from api.copilot import recommend
from api.schemas import CustomerRecord, PredictionResponse, RecommendationResponse
from utils.data import engineer_features
from utils.modeling import explain_linear

app = FastAPI(title="Telecom Retention Copilot API", version="1.0.0")
BUNDLE_PATH = Path("models/champion.joblib")
def bundle():
    if not BUNDLE_PATH.exists(): raise HTTPException(503, "Model unavailable. Run: python scripts/train.py")
    return joblib.load(BUNDLE_PATH)
def score(customer):
    try:
        b = bundle(); features = engineer_features(pd.DataFrame([customer]))
        for col in b["feature_columns"]:
            if col not in features: features[col] = None
        features = features[b["feature_columns"]]
        probability = float(b["pipeline"].predict_proba(features)[0, 1]); threshold = b["threshold"]["threshold"]
        band = "High" if probability >= .60 else "Medium" if probability >= .30 else "Low"
        return PredictionResponse(churn_probability=round(probability, 5), risk_band=band, decision_threshold=threshold, drivers=explain_linear(b, features))
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500, detail=f"Scoring error: {str(e)} | {traceback.format_exc()}")
@app.get("/health")
def health(): return {"status": "ok", "model_loaded": BUNDLE_PATH.exists()}
@app.get("/")
def root():
    return {"service": "Telecom Retention Copilot API", "status": "ok", "health": "/health", "documentation": "/docs"}
@app.post("/predict", response_model=PredictionResponse)
def predict(request: CustomerRecord): return score(request.customer)
@app.post("/predict/batch", response_model=list[PredictionResponse])
def predict_batch(requests: list[CustomerRecord]):
    if not requests or len(requests) > 1000: raise HTTPException(422, "Provide 1 to 1000 customers.")
    return [score(r.customer) for r in requests]
@app.post("/explain", response_model=PredictionResponse)
def explain(request: CustomerRecord): return score(request.customer)
@app.post("/recommend", response_model=RecommendationResponse)
async def recommendation(request: CustomerRecord):
    result = score(request.customer)
    return await recommend(request.customer, result.churn_probability, result.drivers)
