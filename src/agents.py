# src/agents.py
"""
Mudra Multi-Agent Intelligence Layer & Episodic RAG Memory.
Combines LangGraph Council (Analyst -> Strategist -> Risk Critic) with ChromaDB episodic memory.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import json
import logging
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict, Union

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

# Linux / Streamlit Cloud SQLite upgrade hook for ChromaDB
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

# Silence ChromaDB telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)

try:
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    from chromadb.config import Settings
    HAS_CHROMADB = True
except Exception:
    chromadb = None
    HAS_CHROMADB = False
    Documents = Any  # type: ignore
    Embeddings = Any  # type: ignore
    class EmbeddingFunction:  # type: ignore
        pass
    class Settings:  # type: ignore
        def __init__(self, *args, **kwargs):
            pass

from config import (
    ACTIVE_MODEL_VERSION,
    BTCUSDT_1H_CSV,
    CHROMADB_DIR,
    CHROMA_COLLECTION_NAME,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_MODEL_SHA256,
    FEATURED_BTC_DATA_PATH,
    LIVE_FETCH_LIMIT,
    MIN_HURDLE_PCT,
    REGRESSOR_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
    ROUND_TRIP_FRICTION_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.feature_engineering import (
    extract_features_for_inference,
    predict_proba_with_validation,
    predict_with_validation,
    validate_inference_features,
)
from src.model_loader import load_verified_model

load_dotenv(".env")
ALLOWED_TRADE_ACTIONS = ("HOLD", "BUY", "SELL")
GEMINI_MODEL_CANDIDATES = ("gemini-2.5-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-flash-latest")

# ---------------------------------------------------------------------------
# 1. Episodic RAG Memory (ChromaDB)
# ---------------------------------------------------------------------------

def _read_raw_gemini_api_key() -> Optional[str]:
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(key_name)
        if val and val.strip() and val.strip() != "your_gemini_api_key_here":
            return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec in ("Google", "google", "gemini", "Gemini"):
                if sec in st.secrets and "api_key" in st.secrets[sec]:
                    v = str(st.secrets[sec]["api_key"]).strip()
                    if v and v != "your_gemini_api_key_here":
                        return v
            for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if var in st.secrets and str(st.secrets[var]).strip() != "your_gemini_api_key_here":
                    return str(st.secrets[var]).strip()
    except Exception:
        pass
    return None


def _get_gemini_api_key() -> Optional[str]:
    return None if os.getenv("PYTEST_CURRENT_TEST") else _read_raw_gemini_api_key()


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_gemini_api_key()
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
            except Exception:
                pass

    def __call__(self, input: Documents) -> Embeddings:
        if not self.api_key:
            return [[0.0] * 768 for _ in input]
        try:
            import google.generativeai as genai
            return genai.embed_content(model="models/text-embedding-004", content=input, output_dimensionality=768)["embedding"]
        except Exception:
            return [[0.0] * 768 for _ in input]


FOUNDATIONAL_KNOWLEDGE = [
    {"id": "taker_ratio", "doc": "High Taker Buy Ratio signals aggressive buying; low ratios indicate seller dominance.", "category": "microstructure"},
    {"id": "momentum_return", "doc": "Positive 6-hour mean return with expanding volatility confirms trend continuation.", "category": "trend"},
    {"id": "macro_liquidity", "doc": "Fed rate cuts and QE expand liquidity, serving as historical macro tailwinds for Bitcoin.", "category": "macro"},
    {"id": "etf_flows", "doc": "Sustained net Spot Bitcoin ETF inflows absorb miner issuance, establishing support floors.", "category": "liquidity"},
    {"id": "fear_greed", "doc": "Extreme Fear (<25) often marks local bottoms; Extreme Greed (>75) indicates euphoric liquidation risk.", "category": "sentiment"},
    {"id": "onchain_activity", "doc": "Surges in on-chain transaction volume and miner hashrate precede major volatility expansions.", "category": "onchain"},
    {"id": "rsi_reversion", "doc": "Hourly RSI >70 indicates overbought conditions with mean-reversion pullback risk.", "category": "technical"},
    {"id": "halving_shock", "doc": "Halvings cut block subsidies by 50%, historically catalyzing bull markets over 6-18 months.", "category": "supply"},
]


class MarketKnowledgeRAG:
    def __init__(self, db_path: str = CHROMADB_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.has_chroma = False
        if HAS_CHROMADB and chromadb is not None:
            try:
                Path(db_path).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False, is_persistent=True))
                self.embedding_fn = GeminiEmbeddingFunction()
                self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)
                if self.collection.count() == 0:
                    self.collection.add(
                        ids=[item["id"] for item in FOUNDATIONAL_KNOWLEDGE],
                        documents=[item["doc"] for item in FOUNDATIONAL_KNOWLEDGE],
                        metadatas=[{"category": item["category"]} for item in FOUNDATIONAL_KNOWLEDGE],
                    )
                self.has_chroma = True
            except Exception:
                self.has_chroma = False

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        if self.has_chroma:
            try:
                self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata or {}])
            except Exception:
                pass

    def retrieve_market_context(self, query: str, n_results: int = 3) -> List[str]:
        if self.has_chroma:
            try:
                results = self.collection.query(query_texts=[query], n_results=n_results)
                docs = results.get("documents")
                if docs and len(docs) > 0:
                    return docs[0]
            except Exception:
                pass
        # Fallback to foundational knowledge in-memory
        q_lower = query.lower()
        matches = [item["doc"] for item in FOUNDATIONAL_KNOWLEDGE if any(w in item["doc"].lower() for w in q_lower.split() if len(w) > 3)]
        return matches[:n_results] if matches else [FOUNDATIONAL_KNOWLEDGE[0]["doc"]]


_rag_instance: Optional[MarketKnowledgeRAG] = None


def get_rag_agent() -> MarketKnowledgeRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MarketKnowledgeRAG()
    return _rag_instance


def get_rag_collection():
    try:
        return get_rag_agent().collection
    except Exception:
        return None


def initialize_knowledge_base() -> MarketKnowledgeRAG:
    return get_rag_agent()


def retrieve_rag_context(query: str, n_results: int = 3) -> List[str]:
    return get_rag_agent().retrieve_market_context(query=query, n_results=n_results)


def index_reconciled_outcome_memory(
    tick_id: int, timestamp_iso: str, market_regime: str, direction_predicted: str,
    direction_realized: str, direction_correct: bool, predicted_dollar_move: float,
    actual_dollar_move: float, absolute_error_usd: float, error_attribution_tag: str,
    plain_english_notes: str,
) -> bool:
    col = get_rag_collection()
    if col is None:
        return False
    doc_id = f"reconciled_tick_{int(tick_id)}"
    doc_text = f"[{timestamp_iso}] Regime: {market_regime} | Pred: {direction_predicted} (${predicted_dollar_move:+,.2f}) | Realized: {direction_realized} (${actual_dollar_move:+,.2f}) | Correct: {direction_correct} | Tag: {error_attribution_tag} | Notes: {plain_english_notes}"
    meta = {
        "type": "episodic_outcome", "tick_id": int(tick_id), "regime": str(market_regime or "UNKNOWN"),
        "direction_predicted": str(direction_predicted or "UP"), "direction_realized": str(direction_realized or "UP"),
        "direction_correct": bool(direction_correct), "error_attribution_tag": str(error_attribution_tag or "ACCURATE"),
        "absolute_error_usd": float(absolute_error_usd or 0.0),
    }
    try:
        col.upsert(ids=[doc_id], documents=[doc_text], metadatas=[meta])
        return True
    except Exception:
        return False


def query_similar_error_memories(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    col = get_rag_collection()
    if col is None:
        return []
    try:
        results = col.query(query_texts=[query_text], n_results=n_results, where={"type": "episodic_outcome"})
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# 2. QUANTITATIVE TOOLS (Candles, Technical Signals, XGBoost Inference)
# ---------------------------------------------------------------------------

def fetch_historical_candles(symbol: str = "BTCUSDT", limit: int = LIVE_FETCH_LIMIT) -> pd.DataFrame:
    try:
        url = f"https://data-api.binance.vision/api/v3/klines?symbol={symbol}&interval=1h&limit={limit + 2}"
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            now_ms = datetime.now(timezone.utc).timestamp() * 1000.0
            closed = [c for c in resp.json() if float(c[6]) <= now_ms]
            if len(closed) >= 35:
                return pd.DataFrame([{
                    "timestamp": pd.to_datetime(c[0], unit="ms", utc=True),
                    "open": float(c[1]), "high": float(c[2]), "low": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]), "number_of_trades": int(c[8]),
                    "taker_buy_ratio": round(float(c[9]) / float(c[5]), 4) if float(c[5]) > 0 else 0.50,
                } for c in closed[-limit:]])
    except Exception:
        pass

    try:
        cb_symbol = "BTC-USD" if "BTC" in symbol else symbol
        resp = requests.get(f"https://api.exchange.coinbase.com/products/{cb_symbol}/candles?granularity=3600", headers={"User-Agent": "Mudra-Quant/1.0"}, timeout=5)
        if resp.status_code == 200:
            now_sec = datetime.now(timezone.utc).timestamp()
            closed = [c for c in resp.json() if (float(c[0]) + 3600.0) <= now_sec]
            if len(closed) >= 35:
                return pd.DataFrame([{
                    "timestamp": pd.to_datetime(c[0], unit="s", utc=True),
                    "low": float(c[1]), "high": float(c[2]), "open": float(c[3]), "close": float(c[4]),
                    "volume": float(c[5]), "number_of_trades": None, "taker_buy_ratio": 0.50,
                } for c in reversed(closed[:limit])])
    except Exception:
        pass

    path = FEATURED_BTC_DATA_PATH if os.path.exists(FEATURED_BTC_DATA_PATH) else BTCUSDT_1H_CSV
    if os.path.exists(path):
        from src.data_preprocessing import load_and_clean_data
        df_local = pd.read_csv(path) if path == FEATURED_BTC_DATA_PATH else load_and_clean_data(path)
        if len(df_local) >= 35:
            df_local["timestamp"] = pd.to_datetime(df_local["timestamp"], utc=True)
            return df_local.tail(limit).reset_index(drop=True)

    raise ValueError(f"Unable to fetch historical candlesticks for {symbol} from any market source.")


@tool
def get_live_market_data(symbol: str = "BTCUSDT") -> Dict[str, Any]:
    """Fetches latest confirmed candlestick and order-flow telemetry."""
    df_candles = fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT)
    latest = df_candles.iloc[-1]
    raw_trades = latest.get("number_of_trades")
    return {
        "symbol": symbol, "close": float(latest["close"]), "open": float(latest["open"]),
        "high": float(latest["high"]), "low": float(latest["low"]), "volume": float(latest["volume"]),
        "taker_buy_ratio": float(latest.get("taker_buy_ratio", 0.50)),
        "number_of_trades": int(raw_trades) if pd.notna(raw_trades) and raw_trades is not None else None,
        "timestamp": latest.get("timestamp"), "source": "Market Data Feed with HistoricalParity",
        "status": "online", "candles": df_candles.to_dict(orient="records"),
    }


@tool
def calculate_technical_signals(
    close: Optional[float] = None, high: Optional[float] = None, low: Optional[float] = None,
    volume: Optional[float] = None, taker_buy_ratio: Optional[float] = None, open_price: Optional[float] = None,
    timestamp: Optional[Union[str, datetime]] = None, candles: Optional[Union[pd.DataFrame, List[Dict[str, Any]]]] = None,
    symbol: str = "BTCUSDT"
) -> Dict[str, Any]:
    """Computes technical indicators with strict training parity."""
    candles_df = pd.DataFrame(candles) if isinstance(candles, list) else (candles.copy() if candles is not None else fetch_historical_candles(symbol=symbol, limit=LIVE_FETCH_LIMIT))
    if not candles_df.empty:
        idx = candles_df.index[-1]
        if close and close > 0:
            candles_df.at[idx, "close"] = float(close)
        if high and high > 0:
            candles_df.at[idx, "high"] = float(high)
        if low and low > 0:
            candles_df.at[idx, "low"] = float(low)
        if volume is not None and volume >= 0:
            candles_df.at[idx, "volume"] = float(volume)
        if open_price and open_price > 0:
            candles_df.at[idx, "open"] = float(open_price)
        if taker_buy_ratio is not None:
            candles_df.at[idx, "taker_buy_ratio"] = float(taker_buy_ratio)
        elif "taker_buy_ratio" not in candles_df.columns:
            candles_df["taker_buy_ratio"] = 0.50
        if timestamp:
            candles_df.at[idx, "timestamp"] = pd.to_datetime(timestamp)
    return extract_features_for_inference(candles_df)


@tool
def get_model_return_forecast(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Runs dual inference using verified XGBoost regressor and classifier."""
    if not features or not isinstance(features, dict):
        raise ValueError("Invalid features dictionary passed to get_model_return_forecast.")

    df_reg = validate_inference_features(features, model_type="regressor")
    df_clf = validate_inference_features(features, model_type="classifier")
    close = float(features.get("close", features.get("close_price", features.get("current_price", 65000.0))))

    model_reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
    if hasattr(model_reg, "set_params"):
        try:
            model_reg.set_params(device="cpu")
        except Exception:
            pass
    raw_pred_return = float(predict_with_validation(model_reg, df_reg, model_type="regressor")[0])

    model_clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
    if hasattr(model_clf, "set_params"):
        try:
            model_clf.set_params(device="cpu")
        except Exception:
            pass
    move_code = int(predict_with_validation(model_clf, df_clf, model_type="classifier")[0])

    prob_up, prob_down = 0.50, 0.50
    if hasattr(model_clf, "predict_proba"):
        probs = predict_proba_with_validation(model_clf, df_clf)[0]
        prob_down, prob_up = float(probs[0]), float(probs[1])

    expected_return_pct = raw_pred_return * 100.0
    pred_dollar_move = close * raw_pred_return
    clears_hurdle = abs(expected_return_pct) >= MIN_HURDLE_PCT

    return {
        "expected_return_pct": round(expected_return_pct, 4),
        "predicted_dollar_move": round(pred_dollar_move, 2),
        "predicted_target_price": round(close + pred_dollar_move, 2),
        "clears_hurdle": clears_hurdle,
        "hurdle_threshold_pct": round(MIN_HURDLE_PCT, 4),
        "direction": "UP" if move_code == 1 else "DOWN",
        "direction_badge": "SIDEWAYS / CHOP" if not clears_hurdle else ("BULLISH" if move_code == 1 else "BEARISH"),
        "direction_probability_up": round(prob_up * 100.0, 2),
        "direction_probability_down": round(prob_down * 100.0, 2),
        "confidence": round(max(prob_up, prob_down), 4),
        "model_version": ACTIVE_MODEL_VERSION,
        "status": "success",
    }


@tool
def query_macro_sentiment_rag(query: str = "Bitcoin macroeconomic market sentiment and regime") -> str:
    """Queries ChromaDB vector store for market regime and error memory."""
    try:
        rag = get_rag_agent()
        docs = rag.retrieve_market_context(query, n_results=2) or []
        err_memories = [m["document"] for m in query_similar_error_memories(query_text=query, n_results=2) if "document" in m]
        parts: List[str] = []
        if docs:
            parts.append(" | ".join(docs))
        if err_memories:
            parts.append("Recent Error Memory: " + " ; ".join(err_memories))
        return " || ".join(parts) if parts else "Market regime: Standard spot BTC liquidity."
    except Exception as e:
        return f"Macro context unavailable (ChromaDB offline): {str(e)}"


def fetch_mcp_context(limit: int = 3) -> Dict[str, Any]:
    # fetches live derivatives funding pulse, recent news, and error memories from mcp tools
    try:
        from src.mcp_server import (
            get_live_btc_derivatives_pulse,
            get_recent_btc_news_headlines,
            get_recent_prediction_error_memories,
        )
        pulse = get_live_btc_derivatives_pulse()
        errs = get_recent_prediction_error_memories(limit=limit)
        news = get_recent_btc_news_headlines(hours_back=2, max_items=5)
        headlines = news.get("headlines", []) if isinstance(news, dict) else []
        return {
            "status": "success",
            "funding_rate_8h_pct": float(pulse.get("funding_rate_8h_pct", 0.01)),
            "derivatives_regime": str(pulse.get("derivatives_regime", "Neutral Funding")),
            "futures_basis_usd": float(pulse.get("futures_basis_usd", 0.0)),
            "recent_error_memories": errs if isinstance(errs, list) else [],
            "news_headlines": headlines,
        }
    except Exception:
        return {
            "status": "fallback",
            "funding_rate_8h_pct": 0.0100,
            "derivatives_regime": "Neutral Funding (Offline Fallback)",
            "futures_basis_usd": 0.0,
            "recent_error_memories": [],
            "news_headlines": [],
        }


def generate_hourly_postmortem_with_gemini(
    tick_features: Dict[str, Any],
    mcp_context: Dict[str, Any],
    model_outcomes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    # generates grounded 1-line post-mortem explanation per model using gemini
    model_versions = [str(m.get("model_version", "unknown")) for m in model_outcomes]

    api_key = _get_gemini_api_key()
    if not api_key or os.getenv("PYTEST_CURRENT_TEST"):
        return {
            "gemini_status": "UNAVAILABLE",
            "error_detail": "Gemini API key missing or test mode active",
            "market_summary": "Gemini unavailable (offline or missing key)",
            "per_model": {
                mv: {
                    "root_cause_driver": "GEMINI_UNAVAILABLE",
                    "diagnosis_summary": "Gemini API unavailable (check quota/logs) — manual review needed.",
                }
                for mv in model_versions
            },
        }

    try:
        headlines = mcp_context.get("news_headlines", []) if isinstance(mcp_context, dict) else []
        news_str = "\n".join(f"- {h}" for h in headlines) if headlines else "No breaking news reported in last 2 hours."

        models_str = "\n".join([
            f"- Model {m.get('model_version')}: Predicted ${float(m.get('predicted_price', 0)):,.2f} ({m.get('direction_predicted')}, {float(m.get('predicted_return_pct', 0)):+.2f}%) | "
            f"Actual ${float(m.get('actual_price', 0)):,.2f} ({m.get('direction_realized')}, {float(m.get('actual_return_pct', 0)):+.2f}%) | "
            f"Result: {'CORRECT' if m.get('direction_correct') else 'WRONG'} | Error: ${float(m.get('error_usd', 0)):,.2f}"
            for m in model_outcomes
        ])

        prompt = f"""You are Mudra's Lead Quantitative Post-Mortem Analyst.
Analyze the hourly Bitcoin forecast outcome across registered models.
Ground all statements strictly in the provided data. Do not hallucinate external events, prices, or numbers.

Market Candlestick Telemetry:
- Start Price: ${float(tick_features.get('close', 0)):,.2f}
- High/Low Volatility Spread: {float(tick_features.get('volatility_range_pct', 0)):.2f}%
- Taker Buy Ratio: {float(tick_features.get('taker_buy_ratio', 0.5)):.2%}
- RSI Estimate: {float(tick_features.get('rsi_estimate', 50)):.1f}

Recent News Context (Last 2 Hours):
{news_str}

Model Predictions vs Actual Outcomes:
{models_str}

Respond with strictly valid JSON matching this schema:
{{
  "market_summary": "1-2 sentence factual recap of price move and relevant news or order flow",
  "per_model": {{
    "<model_version>": {{
      "root_cause_driver": "TOP_FEATURE_OR_NEWS_OR_REGIME",
      "diagnosis_summary": "1 concise sentence explaining error or success citing specific numbers"
    }}
  }}
}}"""

        raw_resp = generate_with_gemini_fallback(api_key, prompt)
        parsed = json.loads(_clean_json_response(raw_resp))

        per_model = {}
        raw_per_model = parsed.get("per_model", {})
        for mv in model_versions:
            m_data = raw_per_model.get(mv, {})
            per_model[mv] = {
                "root_cause_driver": str(m_data.get("root_cause_driver", "TELEMETRY_DRIVER")),
                "diagnosis_summary": str(m_data.get("diagnosis_summary", "Reconciliation completed.")),
            }

        return {
            "gemini_status": "OK",
            "error_detail": None,
            "market_summary": str(parsed.get("market_summary", "Hourly reconciliation completed.")),
            "per_model": per_model,
        }
    except Exception as exc:
        return {
            "gemini_status": "UNAVAILABLE",
            "error_detail": str(exc),
            "market_summary": "Gemini post-mortem generation failed.",
            "per_model": {
                mv: {
                    "root_cause_driver": "GEMINI_UNAVAILABLE",
                    "diagnosis_summary": "Gemini API unavailable (check quota/logs) — manual review needed.",
                }
                for mv in model_versions
            },
        }


def get_multi_model_consensus(features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    # computes direction consensus across loaded models without mutating forecast return
    if not features or not isinstance(features, dict):
        return {"summary": "Consensus unavailable", "up_count": 0, "down_count": 0, "total": 0}
    try:
        from src.model_loader import get_registered_models
        models = get_registered_models()
        votes: Dict[str, str] = {}
        for ver, info in models.items():
            req_f = info.get("features", [])
            if req_f and all(k in features for k in req_f):
                inp = pd.DataFrame([{k: features[k] for k in req_f}])
                if info.get("regressor") is not None:
                    pred_r = float(info["regressor"].predict(inp)[0])
                    votes[ver] = "UP" if pred_r >= 0 else "DOWN"
        if not votes:
            return {"summary": "Model consensus unavailable", "up_count": 0, "down_count": 0, "total": 0}
        up = sum(1 for v in votes.values() if v == "UP")
        down = sum(1 for v in votes.values() if v == "DOWN")
        detail_str = ", ".join(f"{v}: {d}" for v, d in sorted(votes.items()))
        summary = f"{up}/{len(votes)} Models Bullish ({detail_str})"
        return {"summary": summary, "up_count": up, "down_count": down, "total": len(votes), "votes": votes}
    except Exception:
        return {"summary": "Model consensus offline", "up_count": 0, "down_count": 0, "total": 0}


# ---------------------------------------------------------------------------
# 3. LANGGRAPH MULTI-AGENT COUNCIL
# ---------------------------------------------------------------------------

class AgentState(TypedDict, total=False):
    symbol: str
    current_price: float
    timestamp: str
    market_data: Dict[str, Any]
    engineered_features: Dict[str, Any]
    ml_forecast: Dict[str, Any]
    rag_context: str
    tools_called: List[str]
    mcp_context: Optional[Dict[str, Any]]
    model_consensus: Optional[str]
    strategist_thesis: Optional[str]
    trade_action: Optional[Literal["BUY", "SELL", "HOLD"]]
    conviction_score: Optional[int]
    executive_summary: Optional[str]
    risk_critic_critique: Optional[str]
    risk_approved: Optional[bool]
    veto_reason: Optional[str]
    recommended_position_pct: Optional[float]
    stop_loss_price: Optional[float]
    take_profit_price: Optional[float]
    circuit_breaker_active: bool
    error: Optional[str]
    final_decision: Optional[str]
    risk_verdict: Optional[str]
    strategist_action: Optional[str]
    strategist_conviction: Optional[int]
    kelly_fraction: Optional[float]
    veto_reasons: Optional[List[str]]
    thesis: Optional[str]


def create_initial_agent_state(
    symbol: str = "BTCUSDT",
    current_price: float = 0.0,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None
) -> AgentState:
    return {
        "symbol": symbol,
        "current_price": float(current_price),
        "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
        "market_data": market_data or {},
        "engineered_features": engineered_features or {},
        "ml_forecast": ml_forecast or {},
        "rag_context": "",
        "tools_called": [],
        "mcp_context": {},
        "model_consensus": "",
        "strategist_thesis": None,
        "trade_action": None,
        "conviction_score": None,
        "executive_summary": None,
        "risk_critic_critique": None,
        "risk_approved": None,
        "veto_reason": None,
        "recommended_position_pct": 0.0,
        "stop_loss_price": None,
        "take_profit_price": None,
        "circuit_breaker_active": False,
        "error": None,
    }


class MarketAnalystReport(BaseModel):
    trade_thesis: str = Field(description="STRONG BULLISH, MODERATE BULLISH, NEUTRAL, MODERATE BEARISH, STRONG BEARISH")
    conviction_score: int = Field(description="Conviction rating 1-10")
    executive_summary: str = Field(description="Strategic executive summary")
    quantitative_driver_analysis: str = Field(description="Breakdown of top SHAP features")
    macro_market_context: str = Field(description="Retrieved domain knowledge")
    key_risks: List[str] = Field(description="Risk factors that invalidate thesis")


def get_gemini_status() -> Dict[str, Any]:
    raw_key = _read_raw_gemini_api_key()
    preferred = os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_CANDIDATES[0])
    masked = f"{raw_key[:4]}...{raw_key[-4:]}" if raw_key and len(raw_key) > 8 else ("Configured" if raw_key else None)
    return {
        "configured": bool(raw_key), "masked_key": masked, "preferred_model": preferred,
        "fallback_models": list(GEMINI_MODEL_CANDIDATES),
        "label": f"Gemini AI: Active ({preferred})" if raw_key else "Gemini AI: Rule Engine Fallback (No API Key)",
    }


def generate_with_gemini_fallback(api_key: str, prompt: str, system_instruction: Optional[str] = None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    custom_model = os.getenv("GEMINI_MODEL_NAME")
    candidates = (custom_model,) + tuple(m for m in GEMINI_MODEL_CANDIDATES if m != custom_model) if custom_model else GEMINI_MODEL_CANDIDATES

    last_exc: Optional[Exception] = None
    for idx, model_name in enumerate(candidates):
        try:
            if system_instruction and idx == 0:
                try:
                    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
                except TypeError:
                    model = genai.GenerativeModel(model_name)
            else:
                model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text_out = getattr(resp, "text", None)
            if text_out and str(text_out).strip():
                return str(text_out).strip()
        except AssertionError:
            raise
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_exc}")


def _clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1).strip() if match else text


def _generate_fallback_report(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]], rag_context: List[str]
) -> MarketAnalystReport:
    is_bullish = "Upward" in pred_move_text or pred_return > 0
    thesis = "MODERATE BULLISH" if is_bullish else "MODERATE BEARISH"
    conviction = 7 if abs(pred_return) > 0.002 else 6
    pos_summary = ", ".join([f"{d['feature']} (+{d['shap']:.3f})" for d in top_pos_drivers[:2]]) or "steady order book flow"
    neg_summary = ", ".join([f"{d['feature']} ({d['shap']:.3f})" for d in top_neg_drivers[:2]]) or "minor overhead resistance"

    return MarketAnalystReport(
        trade_thesis=thesis, conviction_score=conviction,
        executive_summary=f"Mudra ensemble forecasts {thesis.lower()} toward ${pred_price:,.2f} ({pred_return:+.2%}).",
        quantitative_driver_analysis=f"Upside catalysts: {pos_summary}. Downward drag: {neg_summary}.",
        macro_market_context=f"Market principles: {rag_context[0] if rag_context else 'Order flow imbalance dominates intraday.'}",
        key_risks=["Volatility expansion causing stop-out.", "Rapid mean-reversion at local resistance."],
    )


def generate_market_commentary(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]],
    market_snapshot: Optional[Dict[str, Any]] = None
) -> MarketAnalystReport:
    rag = get_rag_agent()
    query_terms = [pred_move_text] + ([top_pos_drivers[0].get("feature", "")] if top_pos_drivers else []) + ([top_neg_drivers[0].get("feature", "")] if top_neg_drivers else [])
    rag_docs = rag.retrieve_market_context(" ".join(query_terms).strip(), n_results=2)

    api_key = _get_gemini_api_key()
    if not api_key:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)

    try:
        pos_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_pos_drivers]) or "None"
        neg_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_neg_drivers]) or "None"
        rag_context_str = "\n".join([f"- {d}" for d in rag_docs])
        prompt = (
            f"You are Mudra AI's Lead Quantitative Strategist.\n"
            f"Current BTC: ${current_price:,.2f}, Predicted: ${pred_price:,.2f} ({pred_return:+.4%}, {pred_move_text})\n"
            f"Bullish drivers: {pos_str}\nBearish drivers: {neg_str}\nRAG context: {rag_context_str}\n"
            f"Return strictly JSON with keys: trade_thesis, conviction_score, executive_summary, quantitative_driver_analysis, macro_market_context, key_risks."
        )
        raw_text = generate_with_gemini_fallback(api_key, prompt)
        clean_text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        report_data = json.loads(match.group(1).strip() if match else clean_text)

        valid_theses = {"STRONG BULLISH", "MODERATE BULLISH", "NEUTRAL", "MODERATE BEARISH", "STRONG BEARISH"}
        raw_thesis = str(report_data.get("trade_thesis", "NEUTRAL")).upper().strip()
        report_data["trade_thesis"] = raw_thesis if raw_thesis in valid_theses else "NEUTRAL"
        report_data["conviction_score"] = max(1, min(10, int(report_data.get("conviction_score", 5))))
        return MarketAnalystReport(**report_data)
    except Exception:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)


def strategist_node(state: AgentState) -> AgentState:
    state["tools_called"] = state.get("tools_called", [])
    symbol = state.get("symbol", "BTCUSDT")

    if not state.get("market_data") or state.get("current_price", 0.0) <= 0.0:
        market_res = get_live_market_data.invoke({"symbol": symbol})
        state["market_data"] = market_res
        state["current_price"] = float(market_res.get("close", 65000.0))
        state["tools_called"].append("get_live_market_data")
    else:
        market_res = state["market_data"]

    if not state.get("engineered_features"):
        tech_input = {
            "close": state["current_price"], "high": float(market_res.get("high", state["current_price"] * 1.01)),
            "low": float(market_res.get("low", state["current_price"] * 0.99)), "volume": float(market_res.get("volume", 1500.0)),
        }
        if isinstance(market_res, dict) and market_res.get("taker_buy_ratio") is not None:
            tech_input["taker_buy_ratio"] = float(market_res["taker_buy_ratio"])
        if isinstance(market_res, dict) and market_res.get("candles") is not None:
            tech_input["candles"] = market_res["candles"]
        state["engineered_features"] = calculate_technical_signals.invoke(tech_input)
        state["tools_called"].append("calculate_technical_signals")

    if not state.get("ml_forecast"):
        state["ml_forecast"] = get_model_return_forecast.invoke({"features": {**state.get("engineered_features", {}), "close": state["current_price"]}})
        state["tools_called"].append("get_model_return_forecast")

    if not state.get("rag_context"):
        vol_spread = state.get("engineered_features", {}).get("volatility_range_pct", 1.0)
        taker_ratio = state.get("engineered_features", {}).get("taker_buy_ratio", 0.50)
        clears_hurdle = state.get("ml_forecast", {}).get("clears_hurdle", False)

        if vol_spread > 3.8:
            query = f"{symbol} extreme volatility liquidation flush cascade risk"
        elif clears_hurdle and taker_ratio >= 0.52:
            query = f"{symbol} bullish momentum spot ETF inflows liquidity expansion"
        elif taker_ratio < 0.48:
            query = f"{symbol} distribution selling pressure retail fear capitulation"
        else:
            query = f"{symbol} sideways chop mean reversion range regime"

        state["rag_context"] = query_macro_sentiment_rag.invoke({"query": query})
        state["tools_called"].append("query_macro_sentiment_rag")

    if not state.get("mcp_context"):
        state["mcp_context"] = fetch_mcp_context(limit=3)
        state["tools_called"].append("fetch_mcp_context")

    if not state.get("model_consensus"):
        mc = get_multi_model_consensus(state.get("engineered_features", {}))
        state["model_consensus"] = mc.get("summary", "Consensus unavailable")

    if not state.get("timestamp"):
        state["timestamp"] = datetime.now(timezone.utc).isoformat()

    pred_return = state["ml_forecast"].get("expected_return_pct", 0.0)
    pred_move = state["ml_forecast"].get("predicted_dollar_move", 0.0)
    clears_hurdle = state["ml_forecast"].get("clears_hurdle", False)
    taker_ratio = state["market_data"].get("taker_buy_ratio", 0.50)
    direction = state["ml_forecast"].get("direction", "SIDEWAYS")
    direction_badge = state["ml_forecast"].get("direction_badge", "NEUTRAL")
    prob_up = state["ml_forecast"].get("direction_probability_up", 50.0)

    api_key = _get_gemini_api_key()
    if api_key:
        try:
            mcp_data = state.get("mcp_context") or {}
            fund_rate = mcp_data.get("funding_rate_8h_pct", 0.01)
            deriv_regime = mcp_data.get("derivatives_regime", "Neutral")
            consensus = state.get("model_consensus", "N/A")
            prompt = f"""You are the Lead Quantitative Strategist at Mudra. Return strict JSON:
- Symbol: {symbol} | Price: ${state['current_price']:,.2f}
- XGBoost Return: {pred_return:+.4f}% (${pred_move:+,.2f}) | Direction: {direction} ({direction_badge}) | Up Prob: {prob_up:.1f}%
- Multi-Model Consensus: {consensus}
- Derivatives Pulse: Funding {fund_rate:.4f}% ({deriv_regime})
- Fee Hurdle: {'CLEARED' if clears_hurdle else 'SUB-HURDLE'} | Taker Ratio: {taker_ratio:.4f}
- Context: {state['rag_context'][:450]}
JSON Keys: trade_action (BUY/SELL/HOLD), conviction_score (1-10), executive_summary, strategist_thesis"""

            parsed = json.loads(_clean_json_response(generate_with_gemini_fallback(api_key, prompt)))
            action = str(parsed.get("trade_action", "HOLD")).strip().upper()
            state["trade_action"] = action if action in ("BUY", "SELL", "HOLD") else "HOLD"
            state["conviction_score"] = max(1, min(10, int(parsed.get("conviction_score", 5))))
            state["executive_summary"] = str(parsed.get("executive_summary", ""))
            state["strategist_thesis"] = str(parsed.get("strategist_thesis", ""))
            state["circuit_breaker_active"] = False
            return state
        except Exception:
            pass

    state["circuit_breaker_active"] = True
    if clears_hurdle and (pred_return > 0 or direction == "UP") and taker_ratio >= 0.50:
        action, conviction = "BUY", min(9, int(5 + (pred_return * 20) + (taker_ratio - 0.5) * 10))
        summary = f"Bullish setup: Return {pred_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% estimated round-trip hurdle ({direction_badge}, Up: {prob_up:.1f}%) with taker buying {taker_ratio:.2%}."
    elif clears_hurdle and (pred_return < 0 or direction == "DOWN") and taker_ratio < 0.50:
        action, conviction = "SELL", min(9, int(5 + abs(pred_return * 20)))
        summary = f"Bearish momentum: Return {pred_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% estimated round-trip hurdle ({direction_badge}) with taker selling {taker_ratio:.2%}."
    else:
        action, conviction = "HOLD", 4
        summary = f"Chop / sub-hurdle: Expected move ({pred_return:+.4f}%) does not justify {ROUND_TRIP_FRICTION_PCT:.2f}% friction ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Cash preserved."

    consensus_txt = state.get("model_consensus", "N/A")
    mcp_txt = state.get("mcp_context", {}).get("derivatives_regime", "Neutral")
    state["trade_action"] = action
    state["conviction_score"] = conviction
    state["executive_summary"] = summary
    state["strategist_thesis"] = f"Strategist Rule Signal: {action} ({conviction}/10). XGBoost: ${pred_move:+,.2f} ({pred_return:+.4f}%). Consensus: {consensus_txt}. Derivatives: {mcp_txt}. Taker: {taker_ratio:.4f}. Hurdle: {clears_hurdle}."
    return state


def _parse_float(val: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tuple[bool, float]:
    if val is None or isinstance(val, bool):
        return False, 0.0
    try:
        f = float(val)
        if not math.isfinite(f):
            return False, 0.0
        if min_val is not None and f < min_val:
            return False, 0.0
        if max_val is not None and f > max_val:
            return False, 0.0
        return True, f
    except (TypeError, ValueError, OverflowError):
        return False, 0.0


def risk_critic_node(state: AgentState) -> AgentState:
    raw_action = state.get("trade_action") if isinstance(state, dict) else None
    action = raw_action.strip().upper() if isinstance(raw_action, str) else ""

    if action not in ALLOWED_TRADE_ACTIONS:
        state["risk_approved"] = False
        state["veto_reason"] = f"Unsupported Trade Action: Action ({raw_action!r}) is not in allowed set ('HOLD', 'BUY', 'SELL')"
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Proposed trade_action ({raw_action!r}) is invalid or unsupported."
        return state

    state["trade_action"] = action

    if action == "HOLD":
        state["risk_approved"] = True
        state["veto_reason"] = None
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["risk_critic_critique"] = (
            f"RISK CRITIC CONCURS: Capital preservation authorized. "
            f"Market setup lacks sufficient directional momentum to clear estimated {ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction. "
            f"Recommended allocation: 0.0% (Cash)."
        )
        return state

    ml_forecast = state.get("ml_forecast") if isinstance(state.get("ml_forecast"), dict) else {}
    has_valid_return, expected_return = _parse_float(ml_forecast.get("expected_return_pct"))
    raw_clears = ml_forecast.get("clears_hurdle")
    has_valid_flag = isinstance(raw_clears, bool)
    caller_clears = raw_clears is True
    dir_match = has_valid_return and ((action == "BUY" and expected_return > 0.0) or (action == "SELL" and expected_return < 0.0))
    numeric_clears = has_valid_return and (abs(expected_return) >= MIN_HURDLE_PCT)

    if not (numeric_clears and dir_match and has_valid_flag and caller_clears):
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        if not has_valid_return:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Missing or non-finite forecasted return fails {MIN_HURDLE_PCT:.2f}% break-even threshold"
        elif not dir_match:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Directional Hurdle Mismatch ({action} with return {expected_return:+.4f}%)"
        elif not numeric_clears:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Forecasted return < {MIN_HURDLE_PCT:.2f}% break-even threshold"
        else:
            state["veto_reason"] = f"Sub-Hurdle Fee Drag: Upstream clears_hurdle contract did not authorize active trade"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: {action} order fails empirical friction hurdle ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side)."
        return state

    features = state.get("engineered_features") if isinstance(state.get("engineered_features"), dict) else {}
    has_valid_vol, vol_spread_pct = _parse_float(features.get("volatility_range_pct"), min_val=0.0)
    if not has_valid_vol or vol_spread_pct > 3.8:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Volatility Spike Anomaly: Single-hour spread ({vol_spread_pct:.2f}%) > 3.8% limit" if has_valid_vol else "Volatility Spike Anomaly: Invalid spread"
        state["risk_critic_critique"] = f"VETOED BY RISK CRITIC: Excessive volatility risk ({vol_spread_pct:.2f}%)."
        return state

    has_valid_conv, conviction_raw = _parse_float(state.get("conviction_score"), min_val=1.0, max_val=10.0)
    conviction = int(conviction_raw) if has_valid_conv else 0
    if not has_valid_conv or conviction < 4:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = f"Low Conviction Rating: Conviction ({conviction}/10) < 4 minimum threshold"
        state["risk_critic_critique"] = "VETOED BY RISK CRITIC: Conviction below threshold."
        return state

    has_valid_price, current_price = _parse_float(state.get("current_price"), min_val=1e-6)
    if not has_valid_price:
        state["risk_approved"] = False
        state["recommended_position_pct"] = 0.0
        state["stop_loss_price"] = None
        state["take_profit_price"] = None
        state["veto_reason"] = "Invalid Market Price: Invalid current_price"
        state["risk_critic_critique"] = "VETOED BY RISK CRITIC: Invalid market price."
        return state

    state["risk_approved"] = True
    state["veto_reason"] = None
    position_pct = max(0.05, min(0.20, 0.10 + ((conviction - 5) * 0.02)))
    state["recommended_position_pct"] = round(position_pct, 4)

    stop_dist = current_price * 0.015
    take_dist = current_price * 0.025
    if action == "BUY":
        state["stop_loss_price"] = round(current_price - stop_dist, 2)
        state["take_profit_price"] = round(current_price + take_dist, 2)
    elif action == "SELL":
        state["stop_loss_price"] = round(current_price + stop_dist, 2)
        state["take_profit_price"] = round(current_price - take_dist, 2)

    api_key = _get_gemini_api_key()
    if api_key and not state.get("circuit_breaker_active", False):
        try:
            prompt = f"Chief Risk Officer: {action} order, conviction {conviction}/10, price ${current_price:,.2f}. Position: {position_pct:.1%}. Authorize concisely."
            state["risk_critic_critique"] = generate_with_gemini_fallback(api_key, prompt)
            return state
        except Exception:
            pass

    state["risk_critic_critique"] = (
        f"TRADE AUTHORIZED BY RISK CRITIC: {action} order cleared all empirical risk gates. "
        f"Expected return {expected_return:+.4f}% clears {MIN_HURDLE_PCT:.2f}% hurdle. "
        f"Position sized at {position_pct:.1%} with stop-loss ${state['stop_loss_price']:,.2f} and target ${state['take_profit_price']:,.2f}."
    )
    return state


def build_trading_graph() -> Any:
    workflow = StateGraph(AgentState)
    workflow.add_node("strategist", strategist_node)
    workflow.add_node("risk_critic", risk_critic_node)
    workflow.set_entry_point("strategist")
    workflow.add_edge("strategist", "risk_critic")
    workflow.add_edge("risk_critic", END)
    return workflow.compile()


trading_graph_app = build_trading_graph()


def run_trading_graph(
    symbol: str = "BTCUSDT",
    current_price: Optional[float] = None,
    market_data: Optional[Dict[str, Any]] = None,
    engineered_features: Optional[Dict[str, Any]] = None,
    ml_forecast: Optional[Dict[str, Any]] = None,
    timestamp: Optional[str] = None,
    graph_app: Optional[Any] = None
) -> AgentState:
    initial_state = create_initial_agent_state(
        symbol=symbol,
        current_price=float(current_price or 0.0),
        market_data=market_data,
        engineered_features=engineered_features,
        ml_forecast=ml_forecast,
        timestamp=timestamp
    )
    runner = graph_app or trading_graph_app
    final_state: AgentState = runner.invoke(initial_state)

    raw_action = final_state.get("trade_action") or "HOLD"
    is_approved = bool(final_state.get("risk_approved", False))
    veto_reason = final_state.get("veto_reason")

    final_state["final_decision"] = raw_action if is_approved else "HOLD"
    final_state["risk_verdict"] = "APPROVED" if is_approved else "VETOED"
    final_state["strategist_action"] = raw_action
    final_state["strategist_conviction"] = final_state.get("conviction_score", 5)
    final_state["kelly_fraction"] = float(final_state.get("recommended_position_pct", 0.0))
    final_state["veto_reasons"] = [veto_reason] if veto_reason else []
    final_state["thesis"] = (
        final_state.get("strategist_thesis")
        or final_state.get("executive_summary")
        or "Thesis unavailable."
    )
    if "mcp_context" not in final_state:
        final_state["mcp_context"] = initial_state.get("mcp_context", {})
    if "model_consensus" not in final_state:
        final_state["model_consensus"] = initial_state.get("model_consensus", "")
    return final_state
