# src/api/main.py

import os
import sys
from src.agents.market_analyst_agent import generate_market_commentary
import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

# Ensure stdout handles UTF-8
sys.stdout.reconfigure(encoding='utf-8')

from config import (
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES
)
from src.explainability import get_tree_explainer, explain_single_prediction
from src.api.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ExplainResponse,
    FeatureAttribution,
    FullAnalysisResponse
)

# Global model store for high-throughput serving
models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Loads models and explainers into memory at startup."""
    print("🚀 Initializing Mudra Production Inference Engine...")
    try:
        reg_model = joblib.load(REGRESSOR_MODEL_PATH)
        clf_model = joblib.load(CLASSIFIER_MODEL_PATH)

        # Force CPU device for sub-millisecond single-row inference
        reg_model.set_params(device="cpu")
        clf_model.set_params(device="cpu")

        models["regressor"] = reg_model
        models["classifier"] = clf_model
        models["explainer"] = get_tree_explainer(clf_model)
        print("✅ Models and TreeSHAP explainer loaded successfully.")
    except Exception as e:
        print(f"🚨 Failed to load model artifacts: {e}")
        raise RuntimeError(f"Model initialization failure: {e}")

    yield
    print("🛑 Shutting down Mudra Inference Engine...")
    models.clear()


app = FastAPI(
    title="Mudra Quantitative BTC Trading Engine API",
    description="High-performance async REST API serving tuned XGBoost models, TreeSHAP attributions, and Gemini Agentic RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# Allow Cross-Origin Requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    """Endpoint for load balancers and container health probes."""
    return HealthResponse(
        status="ok",
        model_version="optuna_tuned_xgboost_v1"
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    """Generates numeric return and directional movement predictions."""
    # 1. Validate required features
    missing_reg = [f for f in REGRESSOR_FEATURES if f not in req.features]
    missing_clf = [f for f in CLASSIFIER_FEATURES if f not in req.features]
    if missing_reg or missing_clf:
        missing_all = sorted(list(set(missing_reg + missing_clf)))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required feature fields: {missing_all}"
        )

    # 2. Build aligned DataFrames
    input_reg = pd.DataFrame([{f: req.features[f] for f in REGRESSOR_FEATURES}], columns=REGRESSOR_FEATURES)
    input_clf = pd.DataFrame([{f: req.features[f] for f in CLASSIFIER_FEATURES}], columns=CLASSIFIER_FEATURES)

    # 3. Model Inference
    pred_return = float(models["regressor"].predict(input_reg)[0])
    pred_price = float(req.current_price * (1.0 + pred_return))
    move_code = int(models["classifier"].predict(input_clf)[0])
    move_text = "Upward 📈" if move_code == 1 else "Downward 📉"

    return PredictResponse(
        current_price=req.current_price,
        predicted_price=round(pred_price, 2),
        predicted_return=round(pred_return, 6),
        directional_movement=move_text,
        movement_code=move_code
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(req: PredictRequest):
    """Computes local TreeSHAP feature attributions for a single market tick."""
    missing_clf = [f for f in CLASSIFIER_FEATURES if f not in req.features]
    if missing_clf:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required classifier features for SHAP: {missing_clf}"
        )

    input_clf = pd.DataFrame([{f: req.features[f] for f in CLASSIFIER_FEATURES}], columns=CLASSIFIER_FEATURES)
    base_val, top_pos, top_neg, fig = explain_single_prediction(models["explainer"], input_clf, top_n=4)

    # Convert to schema
    bullish = [FeatureAttribution(feature=d["feature"], shap_value=round(d["shap"], 4), actual_value=round(d["value"], 4)) for d in top_pos]
    bearish = [FeatureAttribution(feature=d["feature"], shap_value=round(d["shap"], 4), actual_value=round(d["value"], 4)) for d in top_neg]

    return ExplainResponse(
        base_value=round(float(base_val), 4),
        top_bullish_drivers=bullish,
        top_bearish_drivers=bearish
    )


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Agentic AI"])
async def analyze(req: PredictRequest):
    """Fuses numeric predictions, TreeSHAP attributions, and Gemini Agentic commentary."""
    pred_res = await predict(req)
    exp_res = await explain(req)

    # Convert drivers back to dict format expected by agent
    pos_drivers = [{"feature": d.feature, "shap": d.shap_value, "value": d.actual_value} for d in exp_res.top_bullish_drivers]
    neg_drivers = [{"feature": d.feature, "shap": d.shap_value, "value": d.actual_value} for d in exp_res.top_bearish_drivers]

    report = generate_market_commentary(
        current_price=pred_res.current_price,
        pred_price=pred_res.predicted_price,
        pred_return=pred_res.predicted_return,
        pred_move_text=pred_res.directional_movement,
        top_pos_drivers=pos_drivers,
        top_neg_drivers=neg_drivers
    )

    return FullAnalysisResponse(
        prediction=pred_res,
        explanation=exp_res,
        analyst_commentary=report
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
