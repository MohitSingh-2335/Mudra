from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

from config import (
    ACTIVE_MODEL_VERSION,
    REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256,
    REGRESSOR_FEATURES, CLASSIFIER_FEATURES
)
from src.model_loader import load_verified_model
from src.feature_engineering import validate_inference_features, predict_with_validation
from src.explainability import get_tree_explainer, explain_single_prediction
from src.agents.market_analyst_agent import MarketAnalystReport, generate_market_commentary

logger = logging.getLogger("src.api.main")
__all__ = [
    "app", "lifespan", "models", "ACTIVE_MODEL_VERSION",
    "HealthResponse", "PredictRequest", "FeatureAttribution",
    "PredictResponse", "ExplainResponse", "FullAnalysisResponse",
    "AgentTradeRequest", "AgentTradeResponse",
    "ReconciliationSummaryResponse", "PortfolioSummaryResponse",
]
models = {}


# pydantic schemas for api endpoints
class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str = "ok"
    model_version: str = ACTIVE_MODEL_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class PredictRequest(BaseModel):
    current_price: float = Field(..., gt=0)
    features: Dict[str, float]


class FeatureAttribution(BaseModel):
    feature: str
    shap_value: float
    actual_value: float


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    current_price: float
    predicted_price: float
    predicted_return: float
    directional_movement: str
    movement_code: int
    model_version: str = ACTIVE_MODEL_VERSION
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ExplainResponse(BaseModel):
    base_value: float
    top_bullish_drivers: List[FeatureAttribution]
    top_bearish_drivers: List[FeatureAttribution]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FullAnalysisResponse(BaseModel):
    prediction: PredictResponse
    explanation: ExplainResponse
    analyst_commentary: MarketAnalystReport


class AgentTradeRequest(BaseModel):
    symbol: str = "BTCUSDT"
    current_price: Optional[float] = None
    market_data: Optional[Dict[str, Any]] = None
    engineered_features: Optional[Dict[str, float]] = None
    ml_forecast: Optional[Dict[str, Any]] = None


class AgentTradeResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", protected_namespaces=())
    symbol: str
    current_price: float
    trade_action: str
    conviction_score: int
    executive_summary: str
    strategist_thesis: str
    risk_approved: bool
    veto_reason: Optional[str] = None
    recommended_position_pct: float
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    risk_critic_critique: str
    circuit_breaker_active: bool
    tools_called: List[str]
    rag_context: str
    timestamp: str
    final_decision: Optional[str] = None
    risk_verdict: Optional[str] = None


class ReconciliationSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    total_reconciled: int = 0
    directional_accuracy_pct: float = 0.0
    mae_usd: float = 0.0
    mape_pct: float = 0.0
    information_coefficient_ic: float = 0.0
    black_swan_count: int = 0
    tag_breakdown: Dict[str, int] = Field(default_factory=dict)
    model_version: Optional[str] = None
    query_failed: bool = False
    error: Optional[str] = None


class PortfolioSummaryResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cash_balance: float = 10000.0
    btc_balance: float = 0.0
    current_btc_price: float = 0.0
    btc_position_value_usd: float = 0.0
    total_equity_usd: float = 10000.0
    cumulative_return_pct: float = 0.0
    benchmark_btc_equity: float = 10000.0
    alpha_usd: float = 0.0
    alpha_pct: float = 0.0
    realized_pnl_usd: float = 0.0
    total_fees_usd: float = 0.0
    drawdown_pct: float = 0.0
    total_trades: int = 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # loads verified models and SHAP tree explainer on startup
    reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    reg.set_params(device="cpu")
    clf.set_params(device="cpu")
    models["regressor"] = reg
    models["classifier"] = clf
    models["explainer"] = get_tree_explainer(clf)
    yield
    models.clear()


app = FastAPI(title="Mudra Quantitative Trading API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check():
    return HealthResponse(status="ok", model_version=ACTIVE_MODEL_VERSION)


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(req: PredictRequest):
    # validates features and runs regression and direction classification
    missing = sorted(list(set([f for f in REGRESSOR_FEATURES if f not in req.features] + [f for f in CLASSIFIER_FEATURES if f not in req.features])))
    if missing:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Missing required feature fields: {missing}")

    try:
        input_reg = validate_inference_features(req.features, model_type="regressor")
        input_clf = validate_inference_features(req.features, model_type="classifier")
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Feature validation error: {val_err}")

    pred_return = float(predict_with_validation(models["regressor"], input_reg, model_type="regressor")[0])
    pred_price = float(req.current_price * (1.0 + pred_return))
    move_code = int(predict_with_validation(models["classifier"], input_clf, model_type="classifier")[0])

    return PredictResponse(
        current_price=req.current_price, predicted_price=round(pred_price, 2),
        predicted_return=round(pred_return, 6), directional_movement="Upward 📈" if move_code == 1 else "Downward 📉",
        movement_code=move_code, model_version=ACTIVE_MODEL_VERSION,
    )


@app.post("/explain", response_model=ExplainResponse, tags=["Explainability"])
async def explain(req: PredictRequest):
    # computes SHAP feature attributions for classifier prediction
    missing_clf = [f for f in CLASSIFIER_FEATURES if f not in req.features]
    if missing_clf:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Missing required classifier features for SHAP: {missing_clf}")

    try:
        input_clf = validate_inference_features(req.features, model_type="classifier")
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Feature validation error: {val_err}")

    base_val, top_pos, top_neg, _ = explain_single_prediction(models["explainer"], input_clf, top_n=4)
    bullish = [FeatureAttribution(feature=d["feature"], shap_value=round(d["shap"], 4), actual_value=round(d["value"], 4)) for d in top_pos]
    bearish = [FeatureAttribution(feature=d["feature"], shap_value=round(d["shap"], 4), actual_value=round(d["value"], 4)) for d in top_neg]

    return ExplainResponse(base_value=round(float(base_val), 4), top_bullish_drivers=bullish, top_bearish_drivers=bearish)


@app.post("/analyze", response_model=FullAnalysisResponse, tags=["Agentic AI"])
async def analyze(req: PredictRequest):
    # runs prediction, explanation, and Gemini market commentary
    pred_res = await predict(req)
    exp_res = await explain(req)
    pos_drivers = [{"feature": d.feature, "shap": d.shap_value, "value": d.actual_value} for d in exp_res.top_bullish_drivers]
    neg_drivers = [{"feature": d.feature, "shap": d.shap_value, "value": d.actual_value} for d in exp_res.top_bearish_drivers]

    report = generate_market_commentary(
        current_price=pred_res.current_price, pred_price=pred_res.predicted_price,
        pred_return=pred_res.predicted_return, pred_move_text=pred_res.directional_movement,
        top_pos_drivers=pos_drivers, top_neg_drivers=neg_drivers
    )
    return FullAnalysisResponse(prediction=pred_res, explanation=exp_res, analyst_commentary=report)


@app.post("/agents/trade", response_model=AgentTradeResponse, tags=["Multi-Agent LangGraph"])
async def run_agent_trade(req: AgentTradeRequest):
    # runs LangGraph multi-agent trade decision pipeline
    from src.agents.trading_graph import run_trading_graph
    try:
        state = run_trading_graph(
            symbol=req.symbol, current_price=req.current_price, market_data=req.market_data,
            engineered_features=req.engineered_features, ml_forecast=req.ml_forecast
        )
        return AgentTradeResponse(**state)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Multi-agent pipeline failed: {str(e)}")


@app.get("/reconciliation/summary", response_model=ReconciliationSummaryResponse, tags=["Observability & Database"])
async def get_reconciliation_metrics(lookback_hours: int = 168, model_version: Optional[str] = None):
    # fetches database reconciliation KPIs for model governance
    from src.database.queries import get_reconciliation_kpis
    try:
        return get_reconciliation_kpis(lookback_hours=lookback_hours, model_version=model_version)
    except Exception as e:
        logger.error("Failed to fetch reconciliation metrics: %s", type(e).__name__)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database query failed")


@app.get("/portfolio/summary", response_model=PortfolioSummaryResponse, tags=["Observability & Database"])
async def get_portfolio_metrics(current_price: float = 83000.0):
    # fetches live paper portfolio balance and performance statistics
    from src.database.queries import get_portfolio_kpis
    try:
        return get_portfolio_kpis(current_price=current_price)
    except Exception as e:
        logger.error("Failed to fetch portfolio metrics: %s", type(e).__name__)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Database query failed")
