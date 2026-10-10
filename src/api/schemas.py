# src/api/schemas.py

from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from src.agents.market_analyst_agent import MarketAnalystReport


class HealthResponse(BaseModel):
    """Health check status response."""
    model_config = ConfigDict(protected_namespaces=())
    status: str = Field(default="ok", examples=["ok"])
    model_version: str = Field(default="optuna_xgboost_v1")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PredictRequest(BaseModel):
    """
    Inbound prediction request.
    Requires current price and a dictionary mapping feature names to numerical values.
    """
    current_price: float = Field(..., gt=0, description="Latest BTC/USDT price in USD", examples=[62904.50])
    features: Dict[str, float] = Field(..., description="Mapping of feature names to float values")


class FeatureAttribution(BaseModel):
    """Local TreeSHAP feature contribution."""
    feature: str
    shap_value: float
    actual_value: float


class PredictResponse(BaseModel):
    """Response containing numeric forecast and directional classification."""
    current_price: float
    predicted_price: float
    predicted_return: float
    directional_movement: str
    movement_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ExplainResponse(BaseModel):
    """Local TreeSHAP attributions for a given prediction."""
    base_value: float
    top_bullish_drivers: List[FeatureAttribution]
    top_bearish_drivers: List[FeatureAttribution]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class FullAnalysisResponse(BaseModel):
    """Complete quantitative analysis fusing ML predictions, SHAP, and Gemini AI agent."""
    prediction: PredictResponse
    explanation: ExplainResponse
    analyst_commentary: MarketAnalystReport
