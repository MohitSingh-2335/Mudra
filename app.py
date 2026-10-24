# app.py
"""
Mudra — Institutional Quantitative Cryptocurrency Intelligence Terminal.
Real-Time Multi-Agent Deliberation, Delayed Ground Truth Reconciliation,
and Autonomous $10,000 Forward Paper Trading Ledger on Neon Cloud PostgreSQL.
"""
import os
import re
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Guard stdout reconfiguration against Windows CRT pipe corruption
if hasattr(sys.stdout, "reconfigure") and sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Tuple, Dict, Any, Optional, List, Callable
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
import requests
import plotly.graph_objects as go

from config import (
    APP_TITLE,
    APP_SUBTITLE,
    FEATURE_DISPLAY_NAMES,
    ACTIVE_MODEL_VERSION,
    FEATURE_SCHEMA_VERSION,
    MODEL_MANIFEST_PATH,
    REGRESSOR_MODEL_SHA256,
    CLASSIFIER_MODEL_SHA256,
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    FEATURED_BTC_DATA_PATH,
    TAKER_FEE_PCT,
    SLIPPAGE_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
    ROUND_TRIP_FRICTION_PCT,
    MIN_HURDLE_PCT,
)
from src.model_loader import load_verified_model
from src.explainability import get_tree_explainer, explain_single_prediction
from src.agents.trading_graph import run_trading_graph
from src.engine.sync import sync_hourly_quant_cycle

from src.database.queries import (
    get_latest_market_ticks,
    get_reconciliation_history,
    get_reconciliation_kpis,
    get_portfolio_history,
    get_portfolio_kpis,
    get_latest_council_audit,
)
from src.database.connection import test_connection
from src.ui.charts import (
    create_tradingview_candlestick,
    create_alpha_equity_curve,
    create_reconciliation_scatter,
)

# Reference centralized feature display naming dictionary
FEATURE_PLAIN_NAMES: Dict[str, str] = FEATURE_DISPLAY_NAMES


def get_feature_display_name(feat_name: str) -> str:
    """Returns a descriptive plain-English label or clean title-cased name for any model feature."""
    return FEATURE_DISPLAY_NAMES.get(feat_name, str(feat_name).replace("_", " ").title())


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Quantitative Terminal Styling
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric {
        background-color: #1A1F2C;
        border: 1px solid #2B313E;
        border-radius: 8px;
        padding: 12px 14px;
        min-height: 115px;
    }
    /* Metric label styling to prevent truncation and ellipsis */
    [data-testid="stMetricLabel"] {
        white-space: normal !important;
    }
    [data-testid="stMetricLabel"] > div,
    [data-testid="stMetricLabel"] p {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        line-height: 1.25 !important;
        font-size: 0.85rem !important;
        color: #A0AEC0 !important;
    }
    /* Metric value sizing and wrapping to fit badges without ellipsis */
    [data-testid="stMetricValue"] {
        font-size: 1.35rem !important;
        line-height: 1.25 !important;
    }
    [data-testid="stMetricValue"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
    }
    /* Metric delta subtext styling */
    [data-testid="stMetricDelta"] {
        font-size: 0.80rem !important;
        line-height: 1.2 !important;
    }
    [data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { background-color: #161B26; border-radius: 6px; padding: 8px 18px; color: #A0AEC0; }
    .stTabs [aria-selected="true"] { background-color: #2962FF !important; color: #FFFFFF !important; }
    div[data-testid="stExpander"] { border: 1px solid #2B313E; border-radius: 8px; background-color: #131722; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Cached Model & Resource Loading ---
@st.cache_resource
def load_models():
    """Loads production XGBoost regressor and TreeSHAP explainer with cryptographic hash verification."""
    try:
        reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
        clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
        reg.set_params(device="cpu")
        clf.set_params(device="cpu")
        explainer = get_tree_explainer(clf)
        return reg, clf, explainer
    except Exception as exc:
        st.error(f"[ERROR] Failed to load ML artifacts: {exc}")
        return None, None, None


reg_model, clf_model, explainer_clf = load_models()


def format_signed_usd(val: Optional[float], decimals: int = 2) -> str:
    """Format signed dollar amounts (+-$123.45) cleanly to avoid '$-' or '$+' in UI."""
    if val is None:
        return "Unavailable"
    sign = "+" if val > 0 else ("-" if val < 0 else "")
    return f"{sign}${abs(val):,.{decimals}f}"


def check_feed_staleness(df: pd.DataFrame, max_lag_hours: float = 3.0) -> Tuple[bool, float]:
    """
    Evaluates whether the market candle DataFrame is a stale fallback or offline replay.
    Returns (is_stale, stale_age_hours).
    """
    if df is None or df.empty:
        return True, 0.0
    latest_candle_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    stale_age_hours = max(0.0, (datetime.now(timezone.utc) - latest_candle_ts).total_seconds() / 3600.0)
    if stale_age_hours > max_lag_hours or df.attrs.get("source") == "local_fallback":
        return True, stale_age_hours
    return False, stale_age_hours


def generate_live_forecast(
    live_df: pd.DataFrame,
    reg_model: Any,
    clf_model: Any,
    is_stale: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Generates next-hour regression and classification forecast strictly when the market feed
    is fresh and confirmed closed. When is_stale is True, skips inference to prevent
    presenting stale replay data as live predictions.
    """
    if is_stale or live_df is None or live_df.empty or reg_model is None:
        return None

    if len(live_df) < 35:
        return None

    from src.feature_engineering import (
        extract_features_for_inference,
        validate_inference_features,
        predict_with_validation,
        predict_proba_with_validation,
    )
    try:
        sample_input = extract_features_for_inference(live_df)
    except Exception:
        return None

    recent_candle = live_df.iloc[-1]
    current_price = float(recent_candle["close"])
    candle_range_usd = float(recent_candle["high"] - recent_candle["low"])
    hurdle_dollar_min = current_price * (MIN_HURDLE_PCT / 100.0)
    spread_pct = (candle_range_usd / current_price) * 100.0 if current_price > 0 else 0.0
    taker = float(recent_candle.get("taker_buy_ratio", 0.5))

    # 1. Regressor Output (Raw decimal fraction: 0.0035 -> +0.35%)
    try:
        raw_pred_ret = float(predict_with_validation(reg_model, sample_input, model_type="regressor")[0])
    except Exception:
        raw_pred_ret = 0.0

    pred_ret_pct = raw_pred_ret * 100.0
    pred_dollar_move = current_price * raw_pred_ret
    pred_target = current_price + pred_dollar_move
    cleared_hurdle = abs(pred_ret_pct) >= MIN_HURDLE_PCT
    hurdle_badge_text = f"Cleared {MIN_HURDLE_PCT:.2f}% Hurdle [PASS]" if cleared_hurdle else "Below Fee Hurdle [SKIP]"

    # 2. Classifier Output (Direction: 1 = UP, 0 = DOWN)
    prob_up = 50.0
    prob_down = 50.0
    direction_label = "UP" if pred_ret_pct >= 0 else "DOWN"
    if clf_model:
        try:
            clf_df = validate_inference_features(sample_input, model_type="classifier")
            move_code = int(predict_with_validation(clf_model, clf_df, model_type="classifier")[0])
            direction_label = "UP" if move_code == 1 else "DOWN"
            if hasattr(clf_model, "predict_proba"):
                probs = predict_proba_with_validation(clf_model, clf_df)[0]
                prob_down = float(probs[0]) * 100.0
                prob_up = float(probs[1]) * 100.0
        except Exception:
            pass

    if not cleared_hurdle:
        direction_badge = "SIDEWAYS (CHOP)"
    elif direction_label == "UP":
        direction_badge = "BULLISH (UP)"
    else:
        direction_badge = "BEARISH (DOWN)"

    return {
        "raw_pred_ret": raw_pred_ret,
        "pred_ret_pct": pred_ret_pct,
        "pred_dollar_move": pred_dollar_move,
        "pred_target": pred_target,
        "cleared_hurdle": cleared_hurdle,
        "direction_label": direction_label,
        "direction_badge": direction_badge,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "taker": taker,
        "spread_pct": spread_pct,
        "candle_range_usd": candle_range_usd,
        "hurdle_dollar_min": hurdle_dollar_min,
        "hurdle_badge_text": hurdle_badge_text,
        "current_price": current_price,
    }


# Alias to centralized feature display dictionary for backward compatibility
FEATURE_HUMAN_NAMES: Dict[str, str] = FEATURE_DISPLAY_NAMES


def get_human_feature_label(feat_name: str) -> str:
    """Returns a descriptive plain-English label alongside the technical feature code."""
    desc = FEATURE_DISPLAY_NAMES.get(feat_name)
    if desc:
        return f"{feat_name} ({desc})"
    return feat_name


def generate_live_shap_attribution(
    live_df: pd.DataFrame,
    explainer_clf: Any,
    is_stale: bool = False,
    top_n: int = 4,
) -> Optional[Tuple[float, list, list, Any]]:
    """
    Computes TreeSHAP factor attributions strictly when the market feed is fresh.
    When is_stale is True, skips calculation to prevent presenting stale explanations as live.
    """
    if is_stale or live_df is None or live_df.empty or explainer_clf is None:
        return None

    if len(live_df) < 35:
        return None

    from src.feature_engineering import extract_features_for_inference, validate_inference_features
    try:
        sample_input = extract_features_for_inference(live_df)
    except Exception:
        return None

    try:
        clf_input_df = validate_inference_features(sample_input, model_type="classifier")
        return explain_single_prediction(explainer_clf, clf_input_df, top_n=top_n)
    except Exception:
        return None


def build_council_scenario_payload(
    scenario_selected: str,
    current_price: float,
    live_forecast: Optional[Dict[str, Any]] = None,
    live_df: Optional[pd.DataFrame] = None,
):
    """
    Constructs market_data, engineered_features, and ml_forecast dictionaries for Tab 2
    Multi-Agent Council stress/simulation scenarios.
    When scenario_selected is 'live' and live_forecast is provided, uses real live model outputs.
    Enforces strict internal mathematical consistency across all generated fields:
      - high = current_price * (1.0 + volatility_spread / 2.0)
      - low  = current_price * (1.0 - volatility_spread / 2.0)
      - volatility_range_pct = round(((high - low) / current_price) * 100.0, 3)
      - predicted_dollar_move = round(current_price * (expected_return_pct / 100.0), 2)
      - clears_hurdle = bool(abs(expected_return_pct) >= MIN_HURDLE_PCT)
    """
    current_price = float(current_price)
    if scenario_selected == "live" and live_forecast:
        expected_return_pct = float(live_forecast.get("pred_ret_pct", 0.08))
        c_dir = live_forecast.get("direction_label", "UP")
        confidence = (
            float(live_forecast.get("prob_up", 50.0)) / 100.0
            if c_dir == "UP"
            else float(live_forecast.get("prob_down", 50.0)) / 100.0
        )
        volatility_spread = max(0.002, float(live_forecast.get("spread_pct", 1.8)) / 100.0)
        taker_buy_ratio = float(live_forecast.get("taker", 0.51))
        order_flow_bias = (
            "BUYER_AGGRESSION"
            if taker_buy_ratio > 0.52
            else ("SELLER_AGGRESSION" if taker_buy_ratio < 0.48 else "NEUTRAL")
        )
        volume = (
            float(live_df["volume"].iloc[-1])
            if live_df is not None and not live_df.empty and "volume" in live_df.columns
            else 1200.0
        )
        rsi_estimate = (
            float(live_df["rsi"].iloc[-1])
            if live_df is not None and not live_df.empty and "rsi" in live_df.columns
            else 50.0
        )
    elif scenario_selected == "bullish":
        volatility_spread = 0.015
        volume = 5000.0
        taker_buy_ratio = 0.65
        order_flow_bias = "STRONG_BUYER_AGGRESSION"
        rsi_estimate = 62.0
        expected_return_pct = 0.42
        confidence = 0.88
    elif scenario_selected == "vol_spike":
        volatility_spread = 0.052
        volume = 12000.0
        taker_buy_ratio = 0.35
        order_flow_bias = "EXTREME_LIQUIDATION"
        rsi_estimate = 28.0
        expected_return_pct = -0.20
        confidence = 0.60
    elif scenario_selected == "chop":
        volatility_spread = 0.004
        volume = 400.0
        taker_buy_ratio = 0.49
        order_flow_bias = "FLAT_CHOP"
        rsi_estimate = 50.0
        expected_return_pct = 0.02
        confidence = 0.45
    else:
        volatility_spread = 0.018
        volume = 1200.0
        taker_buy_ratio = 0.51
        order_flow_bias = "NEUTRAL"
        rsi_estimate = 51.0
        expected_return_pct = 0.08
        confidence = 0.65

    half_spread = volatility_spread / 2.0
    high_price = current_price * (1.0 + half_spread)
    low_price = current_price * (1.0 - half_spread)
    volatility_range_pct = (
        round(((high_price - low_price) / current_price) * 100.0, 3)
        if current_price > 0.0
        else 0.0
    )
    predicted_dollar_move = round(current_price * (expected_return_pct / 100.0), 2)

    m_data = {
        "close": current_price,
        "high": high_price,
        "low": low_price,
        "volume": volume,
        "taker_buy_ratio": taker_buy_ratio,
    }
    t_data = {
        "volatility_spread": volatility_spread,
        "volatility_range_pct": volatility_range_pct,
        "order_flow_bias": order_flow_bias,
        "rsi_estimate": rsi_estimate,
    }

    clears_hurdle = bool(abs(expected_return_pct) >= MIN_HURDLE_PCT)
    if not clears_hurdle:
        direction = "SIDEWAYS"
        direction_badge = "SIDEWAYS [CHOP]"
    elif expected_return_pct >= 0.0:
        direction = "UP"
        direction_badge = "BULLISH [UP]"
    else:
        direction = "DOWN"
        direction_badge = "BEARISH [DOWN]"

    f_data = {
        "expected_return_pct": expected_return_pct,
        "predicted_dollar_move": predicted_dollar_move,
        "clears_hurdle": clears_hurdle,
        "confidence": confidence,
        "direction": direction,
        "direction_badge": direction_badge,
    }
    return m_data, t_data, f_data


_EXPECTED_MOVE_PCT_RE = re.compile(
    r"expected\s+(?:return|move)\s*\(\s*([+-]?\d+(?:\.\d+)?)\s*%\s*\)"
)
_BELOW_THRESHOLD_PCT_RE = re.compile(
    r"(?:below|<|fails?\s+to\s+clear(?:\s+the|\s+estimated)?|failed\s+to\s+clear(?:\s+the|\s+estimated)?|does\s+not\s+justify(?:\s+estimated)?|lacks\s+sufficient\s+directional\s+momentum\s+to\s+clear(?:\s+estimated)?)\s*(\d+(?:\.\d+)?)\s*%"
)


def format_portfolio_rationale(row) -> str:
    """
    Formats the plain-English Strategy Rationale column for the Paper Portfolio table,
    referencing canonical MIN_HURDLE_PCT dynamically and matching only explicit
    sub-hurdle / below-hurdle notes without hardcoded threshold strings or overmatching
    ambiguous hurdle or friction text.
    """
    act = str(row.get("action", "")).upper()
    notes = str(row.get("notes", "") or "")
    if act == "BUY":
        amt = float(row.get("btc_amount", 0.0) or 0.0)
        price = float(row.get("execution_price", 0.0) or 0.0)
        val = amt * price
        return f"Position Opened: Allocated 10% Kelly fraction (~${val:,.0f}) on approved bullish signal."
    elif act == "SELL":
        pnl = float(row.get("realized_pnl_usd", 0.0) or 0.0)
        return f"Position Closed: Sold inventory at market (Realized PnL: ${pnl:+,.2f})."
    elif act == "HOLD":
        notes_lower = notes.lower()
        move_match = _EXPECTED_MOVE_PCT_RE.search(notes_lower)
        effective_hurdle_pct = float(MIN_HURDLE_PCT)
        bound_matches = [
            float(m) for m in _BELOW_THRESHOLD_PCT_RE.findall(notes_lower)
        ]
        has_explicit_below_phrase = (
            "sub-hurdle" in notes_lower
            or "below hurdle" in notes_lower
            or "below fee hurdle" in notes_lower
            or "below the fee hurdle" in notes_lower
            or "fails to clear" in notes_lower
            or "failed to clear" in notes_lower
            or "fee drag" in notes_lower
            or "does not justify" in notes_lower
            or "lacks sufficient directional momentum to clear" in notes_lower
        )

        if move_match is not None:
            move_mag_pct = abs(float(move_match.group(1)))
            is_below_hurdle = has_explicit_below_phrase and (
                move_mag_pct < float(MIN_HURDLE_PCT)
            )
        elif bound_matches:
            is_below_hurdle = any(
                bound <= effective_hurdle_pct for bound in bound_matches
            )
        else:
            is_below_hurdle = has_explicit_below_phrase

        if is_below_hurdle:
            return (
                f"Capital Preserved: Expected move is below {MIN_HURDLE_PCT:.2f}% "
                "round-trip fee hurdle."
            )
        elif "veto" in notes_lower:
            return "Risk Veto: Trade prevented by Risk Critic to protect capital."
        return "Capital Preserved: Sideways chop; preserving cash until high-conviction setup."
    elif "DEPOSIT" in act:
        return "Portfolio Initialized: Starting bankroll of $10,000.00 USDT."
    return notes if notes else "Execution logged."


def build_model_pedigree_markdown() -> str:
    """
    Constructs the rendered Markdown table for Tab 5 Active Production Model Pedigree,
    preserving historical 0.30% friction and +33.55 pp alpha provenance while formatting
    current canonical ROUND_TRIP_FRICTION_PCT and TOTAL_FRICTION_PER_SIDE_PCT dynamically.
    """
    return (
        "| Parameter | Active Value | Rationale |\n"
        "| :--- | :--- | :--- |\n"
        "| **Model Architecture** | `XGBRegressor (CPU)` | Sub-millisecond single-row latency |\n"
        f"| **Model Checkpoint** | `{ACTIVE_MODEL_VERSION}` | Optuna 50 Bayesian trial champion |\n"
        "| **Cross-Validation RMSE**| `$396.22` | Chronological walk-forward split |\n"
        "| **Directional Accuracy**| `53.17%` | Out-of-sample test set |\n"
        f"| **Historical Backtest Friction** | `0.30% (Historical)` | Legacy Phase 1 report assumption (7,000h study); current canonical friction is `{ROUND_TRIP_FRICTION_PCT:.2f}%` (`{TOTAL_FRICTION_PER_SIDE_PCT:.2f}%` per side) |\n"
        f"| **Net Strategy Alpha (Historical)** | `+33.55 pp (Historical)` | Measured under the historical `0.30%` friction assumption; not evidence of equivalent performance under current canonical `{ROUND_TRIP_FRICTION_PCT:.2f}%` friction |"
    )


from src.agents.tools import fetch_historical_candles

# --- Cached Data Fetchers (TTL = 60s) ---
@st.cache_data(ttl=60)
def fetch_live_market_candles(symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Streams the latest 100 confirmed closed hourly candles reusing shared tools fetcher."""
    try:
        return fetch_historical_candles(symbol=symbol, limit=100)
    except Exception as exc:
        empty_df = pd.DataFrame()
        empty_df.attrs["error"] = f"{type(exc).__name__}: {exc}"
        return empty_df


@st.cache_data(ttl=60)
def fetch_cloud_database_state(current_price: Optional[float]):
    """Pulls track record, portfolio equity, and reconciliation KPIs directly from Neon."""
    fallback_state = {
        "db_ok": False,
        "db_ver": "Offline",
        "db_lat": 0.0,
        "db_error": "Neon database is unavailable.",
        "ticks_df": pd.DataFrame(),
        "rec_df": pd.DataFrame(),
        "rec_kpis": {},
        "port_df": pd.DataFrame(),
        "port_kpis": {},
        "latest_audit": None,
    }
    try:
        db_ok, db_ver, db_lat = test_connection()
        if not db_ok:
            return fallback_state

        ticks_df = get_latest_market_ticks(limit=500)
        rec_df = get_reconciliation_history(limit=500)
        rec_kpis = get_reconciliation_kpis(lookback_hours=168)
        port_df = get_portfolio_history(limit=500)
        port_kpis = get_portfolio_kpis(current_price=current_price) if current_price is not None else {}
        latest_audit = get_latest_council_audit()

        return {
            "db_ok": db_ok,
            "db_ver": db_ver,
            "db_lat": db_lat,
            "db_error": None,
            "ticks_df": ticks_df,
            "rec_df": rec_df,
            "rec_kpis": rec_kpis,
            "port_df": port_df,
            "port_kpis": port_kpis,
            "latest_audit": latest_audit,
        }
    except Exception as exc:
        fallback_state["db_error"] = f"{type(exc).__name__}: {exc}"
        return fallback_state


# =====================================================================================
# SIDEBAR CONTROLS & STATUS
# =====================================================================================
st.sidebar.title("Mudra BTC Terminal")
st.sidebar.caption("Hourly Bitcoin Price Prediction & Paper Trading")

# On-Demand Hourly Quant Deliberation & Sync Trigger
if st.sidebar.button(
    "Run Hourly Prediction Now",
    use_container_width=True,
    help="Fetches the latest confirmed 1-hour Bitcoin candle from Binance, updates past forecast evaluations in Neon, and runs the AI trading council.",
):
    with st.spinner("Streaming candle, calculating signals, and deliberating..."):
        try:
            synced, sync_msg, _ = sync_hourly_quant_cycle(force=True)
            if synced:
                st.sidebar.success(sync_msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.info(sync_msg)
        except Exception as exc:
            st.sidebar.error(f"Sync error: {exc}")

if st.sidebar.button(
    "Refresh Data",
    use_container_width=True,
    help="Clears the 60-second local dashboard cache and reloads the latest market data and portfolio metrics from the database.",
):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

# Fetch live market data (strictly confirmed closed hourly candles)
live_market_df = fetch_live_market_candles()
current_price = (
    float(live_market_df["close"].iloc[-1])
    if not live_market_df.empty
    else None
)

# Assess feed staleness / offline replay status
is_stale_fallback, stale_age_hours = check_feed_staleness(live_market_df)

# Fetch Neon cloud DB state
db_state = fetch_cloud_database_state(current_price=current_price)

if db_state["db_ok"]:
    st.sidebar.success(f"Database: Connected ({db_state['db_lat']:.0f}ms)")
else:
    st.sidebar.error("Database: Offline")
    if db_state.get("db_error"):
        st.sidebar.caption(db_state["db_error"])

if is_stale_fallback:
    source_label = (
        f"Stale local replay ({stale_age_hours:.0f}h old)"
        if not live_market_df.empty
        else "Unavailable"
    )
    st.sidebar.warning(f"Market Data: {source_label}")
elif not live_market_df.empty:
    st.sidebar.success("Market Data: Live (1h Confirmed Candles)")
else:
    st.sidebar.error("Market Data: Offline")
    if live_market_df.attrs.get("error"):
        st.sidebar.caption(live_market_df.attrs["error"])

try:
    from src.agents.market_analyst_agent import get_gemini_status
    _gem_info = get_gemini_status()
    if _gem_info["configured"]:
        st.sidebar.success(_gem_info["label"])
    else:
        st.sidebar.info(_gem_info["label"])
except Exception:
    pass

with st.sidebar.expander("System Info", expanded=False):
    st.markdown(
        f"""
        - Single Source of Truth: Neon Cloud PostgreSQL
        - Return Forecasting: {len(REGRESSOR_FEATURES)}-feature XGBoost Regressor (`{ACTIVE_MODEL_VERSION}`)
        - Direction Classification: {len(CLASSIFIER_FEATURES)}-feature XGBoost Classifier (`{ACTIVE_MODEL_VERSION}`)
        - Multi-Agent Deliberation: LangGraph 2-Node Council (Lead Strategist + Risk Critic)
        - Forward Ledger: $10,000 Paper Portfolio ({ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction)
        """
    )


# =====================================================================================
# TOP EXECUTIVE KPI BANNER
# =====================================================================================
st.title(APP_TITLE)

st.caption(APP_SUBTITLE)

# Top Metrics Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# Calculate 24h price change
price_24h_ago = (
    float(live_market_df["close"].iloc[-24])
    if len(live_market_df) >= 24
    else None
)
delta_24h_usd = (
    current_price - price_24h_ago
    if current_price is not None and price_24h_ago is not None
    else None
)

with kpi1:
    spot_label = "Bitcoin Price" if not is_stale_fallback else "Bitcoin Price (Replay)"
    st.metric(
        spot_label,
        (
            f"${current_price:,.2f}"
            if current_price is not None and not is_stale_fallback
            else "Unavailable"
        ),
        (
            f"{format_signed_usd(delta_24h_usd)} (24h)"
            if delta_24h_usd is not None and not is_stale_fallback
            else "Fresh candle required"
        ),
        help="Latest confirmed 1-hour candle close price in USD; delta measures 24-hour price change in Dollars.",
    )

with kpi2:
    audit = db_state["latest_audit"]
    action_badge = audit.get("final_decision") if audit else None
    verdict = audit.get("risk_verdict") if audit else None
    st.metric(
        "AI Trade Decision",
        action_badge or "Unavailable",
        (
            "Risk Check: Passed" if verdict == "APPROVED"
            else "Risk Check: Blocked" if verdict
            else "Database audit required"
        ),
        delta_color="off",
        help="Latest hourly trading decision (BUY, SELL, or HOLD) from the 2-Node Multi-Agent Council. Risk Critic confirms whether safety rules were APPROVED or VETOED.",
    )

with kpi3:
    rec_kpis = db_state["rec_kpis"]
    win_rate_value = rec_kpis.get("directional_accuracy_pct")
    if win_rate_value is None:
        win_rate_value = rec_kpis.get("win_rate_pct")
    total_eval_value = rec_kpis.get("total_reconciled")
    if total_eval_value is None:
        total_eval_value = rec_kpis.get("total_evaluated")
    lifetime_eval_value = rec_kpis.get("total_lifetime_evaluated", total_eval_value)
    st.metric(
        "Direction Accuracy (7d)",
        f"{float(win_rate_value):.1f}%" if win_rate_value is not None else "Unavailable",
        (
            f"{int(total_eval_value)}h rolling ({int(lifetime_eval_value)} total)"
            if total_eval_value is not None and lifetime_eval_value is not None
            else "Database required"
        ),
        delta_color="off",
        help=(
            "Percentage of evaluated 1-hour predictions over the last 7 days "
            "(168 hours lookback) where the forecasted direction matched the "
            "realized market close."
        ),
    )

with kpi4:
    port = db_state["port_kpis"]
    equity = port.get("total_equity_usd")
    st.metric(
        "Paper Account Value",
        f"${float(equity):,.2f}" if equity is not None else "Unavailable",
        "Database required" if equity is None else f"{format_signed_usd(float(equity) - 10000.0)} vs $10k start",
        help="Current marked-to-market value of the simulated $10,000 paper-trading account (Cash + Bitcoin holdings) and profit/loss in Dollars.",
    )

with kpi5:
    alpha = port.get("alpha_usd")
    st.metric(
        "vs. Holding Bitcoin",
        format_signed_usd(float(alpha)) if alpha is not None else "Unavailable",
        (
            "Ahead of Buy & Hold" if float(alpha) >= 0 else "Behind Buy & Hold"
        ) if alpha is not None else "Database required",
        help="Dollar difference between the simulated paper portfolio and a simple Bitcoin Buy & Hold benchmark started at the exact same time ($10k base).",
    )

st.markdown("---")


# =====================================================================================
# 3 evidence-first dashboard sections
# =====================================================================================
forecast_result = None
tab1, tab2, tab3 = st.tabs([
    "Overview",
    "Model Evidence",
    "Paper Portfolio",
])


# -------------------------------------------------------------------------------------
# TAB 1: OVERVIEW
# -------------------------------------------------------------------------------------
with tab1:
    st.subheader("Live Bitcoin (BTC/USDT) 1-Hour Chart")
    st.caption("Live Bitcoin spot market and next-hour machine learning price forecast.")

    # Live Real-Time Binance Candlestick Widget via TradingView
    components.html(
        """
        <div class="tradingview-widget-container" style="height:540px;width:100%;margin:0;padding:0;">
          <div id="tradingview_mudra" style="height:100%;width:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({
            "autosize": true,
            "symbol": "BINANCE:BTCUSDT",
            "interval": "60",
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#0E1117",
            "enable_publishing": false,
            "hide_side_toolbar": false,
            "allow_symbol_change": false,
            "container_id": "tradingview_mudra",
            "backgroundColor": "#0E1117",
            "gridColor": "rgba(43, 49, 62, 0.4)",
            "details": false,
            "hotlist": false,
            "calendar": false
          });
          </script>
        </div>
        """,
        height=550,
    )

    with st.expander("Python Technical Chart (EMA 20/50 & Volume)", expanded=False):
        if is_stale_fallback:
            st.info(f"Technical chart displaying historical local data replay ({stale_age_hours:.0f}h lag).")
        candlestick_fig = create_tradingview_candlestick(live_market_df)
        st.plotly_chart(candlestick_fig, use_container_width=True)

    # Next-Hour Forecast Panel
    st.markdown("### Next-Hour Model Prediction")
    if is_stale_fallback:
        st.warning(
            f"Live market feed unavailable (offline replay active, {stale_age_hours:.0f}h lag). "
            "Next-hour live model forecast inference is paused to prevent misleading predictions."
        )
    else:
        forecast_result = generate_live_forecast(
            live_df=live_market_df,
            reg_model=reg_model,
            clf_model=clf_model,
            is_stale=is_stale_fallback,
        )
        if forecast_result:
            f_col1, f_col2, f_col3, f_col4, f_col5 = st.columns(5)
            with f_col1:
                conf_pct = forecast_result["prob_up"] if forecast_result["direction_label"] == "UP" else forecast_result["prob_down"]
                st.metric(
                    "Predicted Direction (1h)",
                    forecast_result["direction_label"],
                    f"{conf_pct:.0f}% Confidence",
                    delta_color="off",
                    help="Direction forecasted by the classification model (UP or DOWN) with confidence percentage.",
                )
            with f_col2:
                hurdle_dollar_min = forecast_result.get("hurdle_dollar_min", current_price * (MIN_HURDLE_PCT / 100.0))
                hurdle_delta = (
                    f"Clears ${hurdle_dollar_min:,.0f} Fee Cost"
                    if forecast_result["cleared_hurdle"]
                    else f"Below ${hurdle_dollar_min:,.0f} Fee Cost (Hold)"
                )
                st.metric(
                    "Predicted 1h Price Change",
                    format_signed_usd(forecast_result['pred_dollar_move']),
                    hurdle_delta,
                    delta_color="normal" if forecast_result["cleared_hurdle"] else "off",
                    help=f"Expected dollar move over the next hour. To overcome trading fees, expected move must clear ${hurdle_dollar_min:,.0f}; otherwise trades are skipped to protect capital.",
                )
            with f_col3:
                st.metric(
                    "Predicted Price in 1h",
                    f"${forecast_result['pred_target']:,.2f}",
                    f"Current: ${current_price:,.2f}",
                    delta_color="off",
                    help="Projected Bitcoin price 1 hour from now based on model return forecast.",
                )
            with f_col4:
                candle_range_usd = forecast_result.get("candle_range_usd", 0.0)
                swing_delta = "High Price Swing Alert" if forecast_result["spread_pct"] > 3.8 else "Normal Price Swing"
                st.metric(
                    "Last 1h Price Range (High - Low)",
                    f"${candle_range_usd:,.2f}",
                    swing_delta,
                    delta_color="off",
                    help="The high-low price range in Dollars over the last 1-hour candle. Values > 3.8% trigger risk vetoes.",
                )
            with f_col5:
                taker = forecast_result["taker"]
                taker_lead = "Buyers Leading" if taker > 0.52 else ("Sellers Leading" if taker < 0.48 else "Balanced")
                st.metric(
                    "Buyer vs. Seller Activity",
                    taker_lead,
                    f"{taker * 100:.0f}% Buyers / {(1.0 - taker) * 100:.0f}% Sellers",
                    delta_color="off",
                    help="Share of volume initiated by market buyers taking liquidity from the order book.",
                )

    # 3-Angle Model Consensus & Next-Hour Dollar Price Range + FastMCP Derivatives Pulse (Always visible on Tab 1)
    if not live_market_df.empty:
        from src.engine.model_consensus import (
            estimate_next_hour_price_range_usd,
            evaluate_multi_angle_consensus,
        )
        from src.mcp.mudra_mcp_server import get_live_btc_derivatives_pulse
        from src.model_loader import get_registered_models

        range_est = estimate_next_hour_price_range_usd(live_market_df, current_price=current_price)
        c_dir = forecast_result["direction_label"] if forecast_result else "UP"
        c_prob = forecast_result["prob_up"] if forecast_result else 50.0
        c_move = forecast_result["pred_dollar_move"] if forecast_result else 0.0

        # Collect directional votes dynamically across all registered models
        registered_models = get_registered_models()
        model_votes = []
        for v_name, m_spec in registered_models.items():
            try:
                c_feats = m_spec.get("features", [])
                if c_feats and all(col in live_market_df.columns for col in c_feats):
                    c_row = live_market_df[c_feats].iloc[-1:]
                    p_val = float(m_spec["classifier"].predict_proba(c_row)[0, 1]) * 100.0
                    d_val = "UP" if p_val >= 50.0 else "DOWN"
                    model_votes.append({"model_version": v_name, "direction": d_val, "prob_up": p_val})
            except Exception:
                pass

        consensus_info = evaluate_multi_angle_consensus(
            current_price=current_price,
            primary_direction=c_dir,
            primary_prob_up=c_prob,
            primary_pred_dollar_move=c_move,
            predicted_range_usd=range_est["predicted_range_usd"],
            model_votes=model_votes,
        )
        deriv_pulse = get_live_btc_derivatives_pulse()

        st.markdown("#### 3-Angle Model Consensus & Next-Hour Price Range ($)")
        ang_c1, ang_c2, ang_c3, ang_c4 = st.columns(4)
        with ang_c1:
            st.metric(
                "Angle 1: Direction Vote",
                consensus_info["agreement_badge"],
                f"Consensus: {consensus_info['consensus_direction']}",
                delta_color="off",
                help="Multi-model directional vote across active and shadow classifiers.",
            )
        with ang_c2:
            range_ok_text = (
                f"Clears ${consensus_info['hurdle_dollar_min']:,.0f} Fee Hurdle"
                if consensus_info["range_clears_fees"]
                else "Tight Chop Range"
            )
            st.metric(
                "Angle 2: Predicted 1h Range (High - Low)",
                f"${range_est['predicted_range_usd']:,.2f}",
                range_ok_text,
                delta_color="normal" if consensus_info["range_clears_fees"] else "off",
                help="Forecasted next-hour High-Low dollar swing using volatility clustering (45% 1h + 35% 6h EMA + 20% 24h mean).",
            )
        with ang_c3:
            st.metric(
                "Predicted 1h High / Low Band",
                f"${range_est['predicted_high_1h']:,.0f} / ${range_est['predicted_low_1h']:,.0f}",
                f"±${range_est['predicted_range_usd'] / 2.0:,.0f} around spot",
                delta_color="off",
                help="Expected upper and lower price boundaries for the next 1-hour candle.",
            )
        with ang_c4:
            funding_pct = float(deriv_pulse.get("funding_rate_8h_pct", 0.01))
            mark_p = float(deriv_pulse.get("mark_price_usd", 0.0))
            mark_label = f"${mark_p:,.2f}" if mark_p > 0 else deriv_pulse.get("derivatives_regime", "Neutral")
            st.metric(
                "BTC Futures Pulse (FastMCP)",
                mark_label,
                f"8h Funding: {funding_pct:+.4f}% ({deriv_pulse.get('status', 'LIVE')})",
                delta_color="off",
                help="Live Binance USDS-M BTCUSDT Perpetual Futures Mark Price and 8-hour funding rate via Mudra FastMCP.",
            )


def format_risk_levels(
    action: str,
    verdict: str,
    current_price: float,
    council_state: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str]:
    """
    Returns plain-English Stop-Loss and Take-Profit strings.
    When trade is not approved or action is HOLD, displays 'None (No active trade opened — cash preserved)'.
    When BUY is approved, displays stop loss (-1.5%) and target take-profit (+2.5%).
    When SELL is approved, displays stop loss (+1.5%) and target take-profit (-2.5%).
    """
    c_state = council_state or {}
    act = str(action).upper().strip()
    vrd = str(verdict).upper().strip()
    if vrd != "APPROVED" or act == "HOLD":
        no_trade_msg = "None (No active trade opened — cash preserved)"
        return no_trade_msg, no_trade_msg

    if act == "BUY":
        sl = float(c_state.get("stop_loss_price") or (current_price * 0.985))
        tp = float(c_state.get("take_profit_price") or (current_price * 1.025))
        return f"${sl:,.2f} (-1.5%)", f"${tp:,.2f} (+2.5%)"
    elif act == "SELL":
        sl = float(c_state.get("stop_loss_price") or (current_price * 1.015))
        tp = float(c_state.get("take_profit_price") or (current_price * 0.975))
        return f"${sl:,.2f} (+1.5%)", f"${tp:,.2f} (-2.5%)"

    no_trade_msg = "None (No active trade opened — cash preserved)"
    return no_trade_msg, no_trade_msg


# -------------------------------------------------------------------------------------
# TAB 2: MODEL EVIDENCE
# -------------------------------------------------------------------------------------
with tab2:
    st.subheader("Model Evidence & Risk Review")
    st.caption("Review the live council decision, delayed ground-truth accuracy, and model explanations.")
    if current_price is None or is_stale_fallback:
        st.warning("Council unavailable: a fresh confirmed market candle is required before any decision can be calculated.")
        st.stop()
    if not db_state["db_ok"]:
        st.warning("Council unavailable: the persisted database state is offline, so portfolio and audit evidence cannot be verified.")
        st.stop()

    # Only deliberate on observed market data; synthetic scenarios are not evidence.
    run_deliberation = st.button(
        "Check current live market",
        use_container_width=True,
        disabled=forecast_result is None or is_stale_fallback,
        help="Runs the council only when the latest confirmed market feed is fresh and model inference succeeded.",
    )

    if run_deliberation:
        with st.spinner("Convening Strategist and Risk Critic nodes..."):
            m_data, t_data, f_data = build_council_scenario_payload(
                scenario_selected="live",
                current_price=current_price,
                live_forecast=forecast_result,
                live_df=live_market_df,
            )

            st.session_state["council_state"] = run_trading_graph(
                symbol="BTCUSDT",
                current_price=current_price,
                market_data=m_data,
                engineered_features=t_data,
                ml_forecast=f_data,
            )

    # Use the live session result or the latest persisted audit from the database.
    council_state = st.session_state.get("council_state")
    if not council_state and db_state.get("latest_audit"):
        audit = db_state["latest_audit"]
        council_state = {
            "final_decision": audit.get("final_decision", "HOLD"),
            "risk_verdict": audit.get("risk_verdict", "APPROVED"),
            "strategist_action": audit.get("strategist_action", "HOLD"),
            "strategist_conviction": audit.get("strategist_conviction", 5),
            "kelly_fraction": audit.get("kelly_fraction", 0.0),
            "veto_reasons": audit.get("veto_reasons", []),
            "thesis": (
                audit.get("execution_notes")
                or audit.get("thesis")
                or audit.get("strategist_thesis")
                or audit.get("executive_summary")
                or "Click a button above to run."
            ),
        }
    elif not council_state:
        council_state = {
            "final_decision": "HOLD",
            "risk_verdict": "APPROVED",
            "strategist_action": "HOLD",
            "strategist_conviction": 5,
            "kelly_fraction": 0.0,
            "veto_reasons": [],
            "thesis": "Click a button above to run.",
        }

    # Deliberation Outcome Banner
    act = council_state.get("final_decision", "HOLD")
    vrd = council_state.get("risk_verdict", "APPROVED")
    conv = council_state.get("strategist_conviction", 5)
    kelly_pct = float(council_state.get("kelly_fraction", 0.0)) * 100.0
    vetoes = council_state.get("veto_reasons", [])

    if act == "HOLD" or vrd != "APPROVED":
        effective_kelly_pct = 0.0
        trade_dollar_amount = 0.0
    else:
        effective_kelly_pct = kelly_pct
        trade_dollar_amount = 10000.0 * (effective_kelly_pct / 100.0)

    if vrd == "APPROVED" and act == "HOLD":
        st.info("Decision: HOLD (No Trade) — Predicted dollar move is smaller than trading fees. Keeping cash safe.")
    elif vrd == "APPROVED" and act in ("BUY", "SELL"):
        st.success(f"Decision: {act} Approved — Trade Amount: ~${trade_dollar_amount:,.0f} ({effective_kelly_pct:.0f}% of cash)")
    else:
        st.error(f"Decision: HOLD (Trade Blocked by Risk Check) — {', '.join(vetoes) if vetoes else 'Risk limit reached'}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        with st.expander("Step 1: Strategy Proposal", expanded=True):
            prop_action = council_state.get("strategist_action", act)
            st.markdown(f"**Proposed Action:** `{prop_action}` (Confidence: {conv}/10)")
            thesis_text = (
                council_state.get("thesis")
                or council_state.get("strategist_thesis")
                or council_state.get("execution_notes")
                or council_state.get("executive_summary")
                or "Click a button above to run."
            )
            st.markdown(f"**Reason:**\n\n{thesis_text}")
    with col_s2:
        with st.expander("Step 2: Risk Check & Trade Limits", expanded=True):
            risk_status = "Approved" if vrd == "APPROVED" else "Blocked"
            st.markdown(f"**Risk Check:** `{risk_status}`")
            if act == "HOLD" or vrd != "APPROVED":
                trade_amt_str = "$0 (No Trade — Holding Cash)"
            else:
                trade_amt_str = f"~${trade_dollar_amount:,.0f} ({effective_kelly_pct:.0f}% of cash)"
            st.markdown(f"**Amount to Trade:** `{trade_amt_str}`")
            sl_str, tp_str = format_risk_levels(action=act, verdict=vrd, current_price=current_price, council_state=council_state)
            st.markdown(f"**Stop-Loss:** `{sl_str}`")
            st.markdown(f"**Take-Profit:** `{tp_str}`")
            if vetoes and vrd != "APPROVED":
                st.markdown("**Risk Veto Reasons:**")
                for v in vetoes:
                    st.markdown(f"- `{v}`")

    with st.expander("Spot Hold-Until-Profit Check & 3x Isolated Futures Simulation", expanded=True):
        from src.engine.paper_portfolio import evaluate_spot_hold_until_profit_decision, evaluate_futures_long_short_plan
        from src.database.connection import get_db

        cs_ml = council_state.get("ml_forecast") if isinstance(council_state, dict) else None
        cs_ef = council_state.get("engineered_features") if isinstance(council_state, dict) else None

        if isinstance(cs_ml, dict) and "expected_return_pct" in cs_ml:
            fc_ret = float(cs_ml.get("expected_return_pct", 0.0))
            raw_cs_dir = str(cs_ml.get("direction", "")).upper()
            fc_dir = raw_cs_dir if raw_cs_dir in ("UP", "DOWN") else ("UP" if fc_ret >= 0.0 else "DOWN")
            fc_hurdle = bool(cs_ml.get("clears_hurdle", abs(fc_ret) >= MIN_HURDLE_PCT))
        elif forecast_result:
            fc_dir = forecast_result.get("direction_label", "UP")
            fc_ret = float(forecast_result.get("pred_ret_pct", 0.0))
            fc_hurdle = bool(forecast_result.get("cleared_hurdle", False))
        else:
            fc_dir = "UP" if act != "SELL" else "DOWN"
            fc_ret = 0.25 if act == "BUY" else (-0.25 if act == "SELL" else 0.0)
            fc_hurdle = act in ("BUY", "SELL")

        if isinstance(cs_ef, dict) and "volatility_range_pct" in cs_ef:
            fc_spread = float(cs_ef.get("volatility_range_pct", 1.0))
        elif forecast_result:
            fc_spread = float(forecast_result.get("spread_pct", 1.0))
        else:
            fc_spread = 1.0

        f_plan = evaluate_futures_long_short_plan(
            current_price=current_price,
            predicted_direction=fc_dir,
            expected_return_pct=fc_ret,
            clears_hurdle=fc_hurdle,
            volatility_range_pct=fc_spread,
            risk_approved=(vrd == "APPROVED"),
            conviction_score=conv,
            leverage=3.0,
        )

        # Count historical Spot BUY and SELL trades from the database portfolio ledger
        port_df_tab2 = db_state.get("port_df", pd.DataFrame())
        port_kpis_tab2 = db_state.get("port_kpis", {})
        live_cash_bal = float(port_kpis_tab2.get("cash_balance", 10000.0) or 10000.0)
        live_btc_bal = float(port_kpis_tab2.get("btc_balance", 0.0) or 0.0)
        buy_count = 0
        sell_count = 0
        if not port_df_tab2.empty and "action" in port_df_tab2.columns:
            actions_upper = port_df_tab2["action"].astype(str).str.upper().str.strip()
            buy_count = int((actions_upper == "BUY").sum())
            sell_count = int((actions_upper == "SELL").sum())

        f_col1, f_col2 = st.columns(2)
        with f_col1:
            st.markdown("#### 1x Spot Trading Status (Plain-English Tracker)")
            spot_eval = None
            try:
                with get_db() as s:
                    spot_eval = evaluate_spot_hold_until_profit_decision(
                        session=s,
                        current_price=current_price,
                        predicted_direction=fc_dir,
                        proposed_action=act,
                        clears_hurdle=fc_hurdle,
                    )
            except Exception:
                spot_eval = None

            if not spot_eval:
                if act == "BUY" and vrd == "APPROVED" and fc_hurdle:
                    spot_eval = {
                        "final_spot_action": "BUY",
                        "position_state": "ENTERING_LONG_SPOT",
                        "btc_held": live_btc_bal,
                        "avg_entry_with_fees": round(current_price * (1.0 + (TOTAL_FRICTION_PER_SIDE_PCT / 100.0)), 2),
                        "net_exit_price": round(current_price * (1.0 - (TOTAL_FRICTION_PER_SIDE_PCT / 100.0)), 2),
                        "estimated_net_pnl_usd": 0.0,
                        "reason": (
                            f"Buying 1x Spot BTC at ${current_price:,.2f} because the model predicts UP and clears the trading fee hurdle."
                        ),
                    }
                else:
                    spot_eval = {
                        "final_spot_action": "HOLD",
                        "position_state": "FLAT_HOLDING_CASH",
                        "btc_held": live_btc_bal,
                        "avg_entry_with_fees": 0.0,
                        "net_exit_price": round(current_price * (1.0 - (TOTAL_FRICTION_PER_SIDE_PCT / 100.0)), 2),
                        "estimated_net_pnl_usd": 0.0,
                        "reason": "Waiting for a strong UP signal that clears the round-trip fee cost.",
                    }

            hurdle_usd = current_price * (MIN_HURDLE_PCT / 100.0)
            pred_move_usd = current_price * (fc_ret / 100.0)
            btc_held_now = float(spot_eval.get("btc_held", live_btc_bal) or 0.0)

            # Clear Answer to "Did Any Spot Trade Happen?"
            st.markdown(
                f"**Completed Spot Trades So Far:** `{buy_count} Buys` | `{sell_count} Sells` "
                f"*(Account: `${live_cash_bal:,.2f} Cash`, `{btc_held_now:.4f} BTC`)*"
            )

            if btc_held_now <= 1e-8:
                # STATE 1: HOLDING 100% CASH (NO BITCOIN OWNED YET)
                if spot_eval["final_spot_action"] == "BUY":
                    st.success(f"Spot Action Now: BUY BITCOIN (~${trade_dollar_amount:,.0f})")
                else:
                    st.info("Spot Status: WAITING IN CASH (No Bitcoin Owned Yet — $0 Fees Paid)")

                chk1 = "[PASS] Predicting UP" if fc_dir == "UP" else "[WAIT] Predicting DOWN (Spot only buys on UP)"
                chk2 = (
                    f"[PASS] Move (${pred_move_usd:+,.2f}) > ${hurdle_usd:,.0f} Fee"
                    if fc_hurdle
                    else f"[WAIT] Move (${pred_move_usd:+,.2f}) is smaller than ${hurdle_usd:,.0f} Fee"
                )
                chk3 = "[PASS] Normal Swing (<3.8%)" if fc_spread <= 3.8 else f"[BLOCKED] Big Swing ({fc_spread:.2f}% > 3.8%)"

                st.markdown("**3 Checks Required to Trigger a Spot BUY:**")
                st.markdown(
                    f"1. **Direction Check:** `{chk1}`\n"
                    f"2. **Fee Hurdle Check (`${hurdle_usd:,.0f}` min move):** `{chk2}`\n"
                    f"3. **Price Swing Check:** `{chk3}`"
                )
                st.caption(
                    "What happens AFTER we buy BTC? Once we own BTC, the bot holds it at $0 fee while predictions stay UP, "
                    "and only sells on a DOWN prediction after the price is high enough to lock in a net dollar profit after fees "
                    "(with a -3.5% emergency stop-loss if the market crashes)."
                )
            else:
                # STATE 2: HOLDING BITCOIN (IN AN ACTIVE SPOT TRADE)
                be_price = float(spot_eval.get("avg_entry_with_fees") or current_price)
                net_exit = float(spot_eval.get("net_exit_price") or current_price)
                net_pnl = float(spot_eval.get("estimated_net_pnl_usd") or 0.0)
                emerg_stop = round(be_price * (1.0 - 0.035), 2)

                st.success(
                    f"Spot Status: IN AN ACTIVE TRADE — Holding {btc_held_now:.4f} BTC | "
                    f"Action This Hour: {spot_eval['final_spot_action']}"
                )
                st.markdown(
                    f"- **Your Break-Even Price (incl. fees):** `${be_price:,.2f}`\n"
                    f"- **If Sold Right Now (after fees):** `${net_exit:,.2f}` (**Net Profit/Loss: `${net_pnl:+,.2f}`**)\n"
                    f"- **Emergency Crash Stop (-3.5%):** `${emerg_stop:,.2f}`"
                )
                st.caption(spot_eval["reason"])

        with f_col2:
            st.markdown("#### 3x Isolated Margin Futures Simulation")
            st.markdown(f"**Futures Position:** `{f_plan['futures_Position']}`")
            st.markdown(f"**Isolated Margin:** `${f_plan['margin_allocated_usd']:,.0f}` (Notional: `${f_plan['notional_exposure_usd']:,.0f}`)")
            if f_plan["liquidation_price"]:
                st.markdown(f"**Liquidation Price:** `${f_plan['liquidation_price']:,.2f}`")
                st.markdown(f"**Stop-Loss:** `${f_plan['stop_loss_price']:,.2f}` | **Take-Profit:** `${f_plan['take_profit_price']:,.2f}`")
            st.caption(f_plan["summary"])


TAB3_MODEL_SELECTOR_OPTIONS: List[str] = ["All Models", ACTIVE_MODEL_VERSION]


def filter_reconciliation_records(df: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    """
    Filters reconciliation records for the selected model version.
    If selected_model is 'All Models' or None, returns all records.
    Preserves unknown attribution records under 'All Models'.
    """
    if df.empty or "model_version" not in df.columns:
        return df.copy()
    if selected_model and selected_model != "All Models":
        return df[df["model_version"] == selected_model].copy()
    return df.copy()


def prepare_reconciliation_table_display(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares reconciliation audit records for UI table rendering.
    Maps null/missing model_version to 'UNKNOWN' and preserves DataFrame integrity.
    """
    if df.empty:
        return df.copy()
    disp_df = df.copy()
    if "model_version" in disp_df.columns:
        disp_df["Model Version"] = disp_df["model_version"].fillna("UNKNOWN")
    else:
        disp_df["Model Version"] = "UNKNOWN"
    return disp_df


def compute_fallback_reconciliation_kpis(df: pd.DataFrame, model_version: Optional[str] = None) -> Dict[str, Any]:
    """
    Computes rolling performance metrics directly from loaded DataFrame records.
    Used as a fallback when database KPI query fails, ensuring filtered client records
    are used rather than leaking unfiltered database-wide KPIs.
    """
    if df.empty:
        return {
            "total_evaluated": 0,
            "total_reconciled": 0,
            "directional_accuracy_pct": None,
            "win_rate_pct": None,
            "mae_usd": None,
            "mape_pct": None,
            "information_coefficient": None,
            "information_coefficient_ic": None,
            "black_swans_filtered": 0,
            "black_swan_count": 0,
            "accurate_count": 0,
            "tag_breakdown": {},
            "model_version": model_version,
            "is_fallback": True,
        }

    total = len(df)
    if "direction_correct" in df.columns:
        correct = int(df["direction_correct"].sum())
    elif {"direction_predicted", "direction_realized"}.issubset(df.columns):
        correct = int((df["direction_predicted"] == df["direction_realized"]).sum())
    else:
        correct = 0

    win_rate = round((correct / total) * 100.0, 2) if total > 0 else 0.0
    mae = round(float(df["absolute_error_usd"].mean()), 2) if "absolute_error_usd" in df.columns else 0.0
    mape = round(float(df["percentage_error"].mean() * 100.0), 4) if "percentage_error" in df.columns else 0.0

    ic = 0.0
    if len(df) > 1 and {"predicted_return_pct", "actual_return_pct"}.issubset(df.columns):
        if float(df["actual_return_pct"].std()) > 0 and float(df["predicted_return_pct"].std()) > 0:
            corr = df["predicted_return_pct"].corr(df["actual_return_pct"])
            ic = round(float(corr), 3) if not np.isnan(corr) else 0.0

    black_swans = int(df["is_black_swan"].sum()) if "is_black_swan" in df.columns else 0
    accurate = int((df["error_attribution_tag"] == "ACCURATE").sum()) if "error_attribution_tag" in df.columns else 0
    tag_counts = df["error_attribution_tag"].value_counts().to_dict() if "error_attribution_tag" in df.columns else {}

    return {
        "total_evaluated": total,
        "total_reconciled": total,
        "directional_accuracy_pct": win_rate,
        "win_rate_pct": win_rate,
        "mae_usd": mae,
        "mape_pct": mape,
        "information_coefficient": ic,
        "information_coefficient_ic": ic,
        "black_swans_filtered": black_swans,
        "black_swan_count": black_swans,
        "accurate_count": accurate,
        "tag_breakdown": tag_counts,
        "model_version": model_version,
        "is_fallback": True,
    }


def resolve_tab3_reconciliation_kpis(
    selected_model: str,
    base_kpis: Dict[str, Any],
    filtered_df: pd.DataFrame,
    kpi_fetcher: Optional[Callable[[Optional[int], Optional[str]], Dict[str, Any]]] = None,
    lookback_hours: Optional[int] = 168,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Resolves the KPI metrics dictionary for Tab 3 based on selected model and lookback window.
    If 'All Models', returns (base_kpis, None) unless custom window or base_kpis failed.
    If a specific model is selected:
      - Attempts to fetch model-specific KPIs via kpi_fetcher.
      - Explicitly distinguishes query success from query failure via query_failed/error flags
        or thrown exceptions, never assuming a dictionary indicates success.
      - If the query genuinely succeeded with zero records, returns the valid zero-valued metrics.
      - If the query failed, computes fallback KPIs from filtered_df if records exist,
        or returns a defensive unavailable KPI structure with an error message.
    Under NO circumstance does this return unfiltered base_kpis when a specific model is selected.

    Returns:
        (kpi_dict, warning_message_or_none)
    """
    if selected_model == "All Models" or not selected_model:
        # If custom lookback window requested (not default 168), fetch window KPIs from DB
        if kpi_fetcher is not None and lookback_hours != 168:
            try:
                res = kpi_fetcher(lookback_hours, None)
                if isinstance(res, dict) and not res.get("query_failed") and not res.get("error"):
                    return res, None
            except Exception:
                pass

        if base_kpis and (base_kpis.get("query_failed") or base_kpis.get("error")):
            if not filtered_df.empty:
                fallback_kpis = compute_fallback_reconciliation_kpis(filtered_df, model_version="All Models")
                warning_msg = (
                    f"Database KPI query failed. "
                    f"Displaying fallback KPIs computed from {len(filtered_df)} loaded client records."
                )
                return fallback_kpis, warning_msg
            unavailable_kpis = {
                "total_evaluated": 0,
                "total_reconciled": 0,
                "directional_accuracy_pct": None,
                "win_rate_pct": None,
                "mae_usd": None,
                "mape_pct": None,
                "information_coefficient": None,
                "information_coefficient_ic": None,
                "black_swans_filtered": 0,
                "black_swan_count": 0,
                "accurate_count": 0,
                "tag_breakdown": {},
                "model_version": None,
                "kpi_unavailable": True,
            }
            return unavailable_kpis, "Reconciliation KPIs are currently unavailable due to a database error."
        return base_kpis, None

    # Specific model selected: query filtered KPIs
    query_succeeded = False
    fetched_kpis = None
    if kpi_fetcher is not None:
        try:
            res = kpi_fetcher(lookback_hours, selected_model)
            if isinstance(res, dict) and not res.get("query_failed") and not res.get("error"):
                fetched_kpis = res
                query_succeeded = True
        except Exception:
            query_succeeded = False

    # Valid successful query result (including genuinely empty 0-record results)
    if query_succeeded and fetched_kpis is not None:
        return fetched_kpis, None

    # Query failed or was unavailable: fallback to filtered DataFrame
    if not filtered_df.empty:
        fallback_kpis = compute_fallback_reconciliation_kpis(filtered_df, model_version=selected_model)
        warning_msg = (
            f"Database KPI query failed for model '{selected_model}'. "
            f"Displaying fallback KPIs computed from {len(filtered_df)} loaded client records."
        )
        return fallback_kpis, warning_msg

    # No records and query failed: unavailable state
    unavailable_kpis = {
        "total_evaluated": 0,
        "total_reconciled": 0,
        "directional_accuracy_pct": None,
        "win_rate_pct": None,
        "mae_usd": None,
        "mape_pct": None,
        "information_coefficient": None,
        "information_coefficient_ic": None,
        "black_swans_filtered": 0,
        "black_swan_count": 0,
        "accurate_count": 0,
        "tag_breakdown": {},
        "model_version": selected_model,
        "kpi_unavailable": True,
    }
    warning_msg = f"Model-specific KPIs for '{selected_model}' are currently unavailable."
    return unavailable_kpis, warning_msg


# -------------------------------------------------------------------------------------
# LIVE TRACK RECORD & DELAYED RECONCILIATION (NEON DB SSOT)
# -------------------------------------------------------------------------------------
with tab2:
    st.subheader("Past Prediction Accuracy (Checked 1 Hour Later)")
    st.caption("Realized model accuracy evaluated 1 hour later against actual Bitcoin closing price.")

    rec_df = db_state.get("rec_df", pd.DataFrame())
    base_kpis = db_state.get("rec_kpis", {})

    # Dynamically discover all evaluated models from database records and disk manifests
    from src.model_loader import get_registered_models
    discovered_model_keys = list(get_registered_models().keys())
    db_models = (
        [str(m) for m in rec_df["model_version"].dropna().unique() if m and str(m).strip()]
        if not rec_df.empty and "model_version" in rec_df.columns
        else []
    )
    dropdown_models = ["All Models"] + sorted(list(set([ACTIVE_MODEL_VERSION] + db_models + discovered_model_keys)))

    t3_col1, t3_col2 = st.columns([1, 1])
    with t3_col1:
        selected_model = st.selectbox(
            "Filter Delayed Track Record by Model",
            options=dropdown_models,
            key="tab3_model_selector",
        )
    with t3_col2:
        lookback_label = st.radio(
            "Evaluation Time Window",
            options=["All Time (Total History)", "Last 7 Days (168 Hours)", "Last 24 Hours (24 Hours)"],
            horizontal=True,
            key="tab3_lookback_selector",
        )

    selected_lookback = None if "All Time" in lookback_label else (24 if "24 Hours" in lookback_label else 168)

    # Filter reconciliation records and resolve KPIs safely
    filtered_rec_df = filter_reconciliation_records(rec_df, selected_model)
    rec_kpis, kpi_warning = resolve_tab3_reconciliation_kpis(
        selected_model=selected_model,
        base_kpis=base_kpis,
        filtered_df=filtered_rec_df,
        kpi_fetcher=get_reconciliation_kpis if db_state.get("db_ok", False) else None,
        lookback_hours=selected_lookback,
    )
    if kpi_warning:
        st.warning(f"⚠️ {kpi_warning}")

    tot_val = rec_kpis.get("total_reconciled") if rec_kpis.get("total_reconciled") is not None else rec_kpis.get("total_evaluated", 0)
    lifetime_total = rec_kpis.get("total_lifetime_evaluated", tot_val)
    st.info(f"Database Connected: Showing {tot_val} evaluated predictions in selected window ({lifetime_total} total evaluations recorded across all time in Neon DB).")

    # Rolling Metrics Ribbon
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        tot_val = rec_kpis.get("total_reconciled") if rec_kpis.get("total_reconciled") is not None else rec_kpis.get("total_evaluated", 0)
        st.metric(
            "Hours Checked",
            f"{tot_val}",
            help="Total number of historical 1-hour candles where an advance prediction was delayed and evaluated against realized close.",
        )
    with m_col2:
        val = rec_kpis.get("directional_accuracy_pct") if rec_kpis.get("directional_accuracy_pct") is not None else rec_kpis.get("win_rate_pct")
        st.metric(
            "Direction Accuracy (UP/DOWN)",
            f"{float(val):.1f}%" if val is not None else "N/A",
            delta_color="off",
            help="How often the model correctly predicted whether Bitcoin would close higher or lower 1 hour later.",
        )
    with m_col3:
        mae_usd = rec_kpis.get("mae_usd")
        st.metric(
            "Average Dollar Error",
            f"${float(mae_usd):,.2f}" if mae_usd is not None else "N/A",
            help="The average dollar difference between the forecasted price and the actual closing price.",
        )
    with m_col4:
        bs_count = rec_kpis.get("black_swans_filtered") if rec_kpis.get("black_swans_filtered") is not None else rec_kpis.get("black_swan_count", 0)
        st.metric(
            "Big Market Swings",
            f"{bs_count}",
            delta_color="off",
            help="Unusual market spikes (>3.8% hourly volatility) flagged by risk checks.",
        )

    with st.expander("Advanced Statistical Metrics (MAPE & Correlation)", expanded=False):
        adv_col1, adv_col2 = st.columns(2)
        with adv_col1:
            mape_val = rec_kpis.get("mape_pct")
            st.metric(
                "Mean Percentage Error (MAPE)",
                f"{float(mape_val):.2f}%" if mape_val is not None else "N/A",
                help="Mean absolute percentage error between the forecasted price and actual market price.",
            )
        with adv_col2:
            ic_val = rec_kpis.get("information_coefficient_ic") if rec_kpis.get("information_coefficient_ic") is not None else rec_kpis.get("information_coefficient")
            st.metric(
                "Information Coeff (IC)",
                f"{float(ic_val):.3f}" if ic_val is not None else "N/A",
                help="Information coefficient correlation between predicted return and actual return.",
            )

    # Hit / Miss Scatter Chart
    if not filtered_rec_df.empty:
        st.markdown("### Realized Price vs. Predicted Price Accuracy Chart")
        st.plotly_chart(create_reconciliation_scatter(filtered_rec_df), use_container_width=True)

        st.markdown("### Historical Delayed Reconciliation Audit Table")
        # Format table for display
        disp_df = prepare_reconciliation_table_display(filtered_rec_df)

        # 1. Plain-English Direction Columns (Strict binary UP / DOWN based on returns, healing historical 'FLAT' entries)
        def _get_dir_pred(row):
            ret = float(row.get("predicted_return_pct", 0.0) or 0.0)
            return "UP" if ret >= 0.0 else "DOWN"

        def _get_dir_real(row):
            ret = float(row.get("actual_return_pct", 0.0) or 0.0)
            return "UP" if ret >= 0.0 else "DOWN"

        disp_df["Predicted Direction"] = disp_df.apply(_get_dir_pred, axis=1)
        disp_df["Actual Direction"] = disp_df.apply(_get_dir_real, axis=1)
        disp_df["Direction Result"] = (disp_df["Predicted Direction"] == disp_df["Actual Direction"]).apply(
            lambda x: "[MATCH]" if x else "[MISMATCH]"
        )

        # 2. Combined Price Columns in Dollars
        def _fmt_pred_price(row):
            p = float(row.get("predicted_price", 0.0) or 0.0)
            return f"${p:,.2f}"

        def _fmt_actual_price(row):
            p = float(row.get("actual_price_at_t_plus_1", 0.0) or 0.0)
            return f"${p:,.2f}"

        disp_df["Predicted Price (Expected Move)"] = disp_df.apply(_fmt_pred_price, axis=1)
        disp_df["Actual Market Close (Realized Move)"] = disp_df.apply(_fmt_actual_price, axis=1)
        disp_df["Price Difference ($)"] = disp_df["absolute_error_usd"].apply(lambda x: f"${float(x or 0.0):,.2f}")
        disp_df["Time (UTC)"] = disp_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")

        # 3. Plain-English AI Diagnosis Sentence
        def _generate_diagnosis(row):
            pred_dir = str(row.get("Predicted Direction", "")).upper()
            act_dir = str(row.get("Actual Direction", "")).upper()
            is_match = (pred_dir == act_dir) and bool(pred_dir)
            diff = float(row.get("absolute_error_usd", 0.0) or 0.0)
            pred_ret = float(row.get("predicted_return_pct", 0.0) or 0.0)
            act_ret = float(row.get("actual_return_pct", 0.0) or 0.0)
            tag = str(row.get("error_attribution_tag", ""))

            if tag == "BLACK_SWAN_OUTLIER" or abs(act_ret) >= 3.8:
                return f"Market Flash Anomaly: Extreme liquidation spike ({act_ret:+.2f}% move)."

            if is_match:
                if diff <= 150.0 or tag == "ACCURATE":
                    return f"Target Hit: Accurate directional call within ${diff:,.2f} variance."
                elif act_dir == "DOWN":
                    if abs(act_ret) > abs(pred_ret):
                        return f"Stronger Drop: Market fell further than forecasted (-${diff:,.2f})."
                    else:
                        return f"Smaller Drop: Downward move was smaller than forecasted (+${diff:,.2f})."
                else:  # UP
                    if abs(act_ret) > abs(pred_ret):
                        return f"Stronger Rally: Market moved upward faster than anticipated (+${diff:,.2f})."
                    else:
                        return f"Weaker Move: Upward breakout was smaller than forecasted (-${diff:,.2f})."
            else:
                if diff <= 150.0 or tag == "SIDEWAYS_DRIFT":
                    return f"Target Drift: Sideways range chop drifted by ${diff:,.2f}."
                else:
                    return f"Direction Miss: Market reversed by {act_ret:+.2f}% (${diff:,.2f} gap)."

        disp_df["AI Diagnosis"] = disp_df.apply(_generate_diagnosis, axis=1)

        display_cols = [
            "Time (UTC)",
            "Model Version",
            "Predicted Direction",
            "Actual Direction",
            "Direction Result",
            "Predicted Price (Expected Move)",
            "Actual Market Close (Realized Move)",
            "Price Difference ($)",
            "AI Diagnosis",
        ]

        st.dataframe(
            disp_df[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(f"[INFO] No delayed reconciliation records found for selection '{selected_model}'. Delayed reconciliation runs automatically every hour as candles close.")

    # Read-Only Forecast Error Investigation Explorer
    st.markdown("---")
    with st.expander("Inspect Specific Prediction Error Details (Forecast Error Investigations & Root-Cause Audit Log)", expanded=False):
        st.caption(
            "Read-only diagnostic audit log exploring prediction anomalies, regime shifts, and causal hypotheses. "
            "Strictly distinguishes suspected root causes (unverified hypotheses) from confirmed ground-truth causes."
        )
        inv_col1, inv_col2 = st.columns([3, 2])
        with inv_col1:
            search_kw = st.text_input(
                "Search News Headlines / Macro Catalysts",
                value="",
                placeholder="e.g. CPI, ETF, liquidation, Fed",
                key="investigation_headline_search",
            )
        with inv_col2:
            status_opt = st.selectbox(
                "Filter by Verification Status",
                options=["ALL", "UNVERIFIED_HYPOTHESIS", "CONFIRMED_CAUSE", "REFUTED", "APPROVED_FOR_TRAINING"],
                key="investigation_status_filter",
            )

        try:
            from src.database.connection import get_db
            from src.database.investigations import get_prediction_investigations
            filter_status = None if status_opt == "ALL" else status_opt
            with get_db() as inv_session:
                inv_records = get_prediction_investigations(
                    inv_session,
                    headline_search=search_kw.strip() if search_kw.strip() else None,
                    status=filter_status,
                    limit=30,
                )
            if inv_records:
                inv_display = []
                for inv in inv_records:
                    st_badge = {
                        "UNVERIFIED_HYPOTHESIS": "🟡 Unverified Hypothesis",
                        "CONFIRMED_CAUSE": "🟢 Confirmed Cause",
                        "REFUTED": "🔴 Refuted Hypothesis",
                        "APPROVED_FOR_TRAINING": "🔵 Approved for Training",
                    }.get(inv.get("verification_status"), inv.get("verification_status"))

                    inv_display.append({
                        "ID": inv.get("id"),
                        "Status": st_badge,
                        "Suspected Cause": inv.get("suspected_cause_category"),
                        "Confidence": f"{float(inv.get('diagnostic_confidence', 0.0)):.1%}",
                        "Headline": inv.get("headline") or "N/A",
                        "Model Version": inv.get("model_version"),
                        "Ground Truth Label": inv.get("verified_ground_truth_label") or "Pending Verification",
                        "Verified By": inv.get("verified_by") or "—",
                        "Created (UTC)": pd.to_datetime(inv.get("created_at")).strftime("%Y-%m-%d %H:%M") if inv.get("created_at") else "—",
                    })
                st.dataframe(pd.DataFrame(inv_display), use_container_width=True, hide_index=True)
            else:
                st.info("No recorded prediction investigations match your query. Investigations are logged when significant prediction residuals or anomalies are analyzed.")
        except Exception:
            st.info("Diagnostic investigations database is currently operating in local test mode. Delayed reconciliation logs will populate when live anomalies occur.")


# -------------------------------------------------------------------------------------
# TAB 3: PAPER PORTFOLIO
# -------------------------------------------------------------------------------------
with tab3:
    st.subheader("$10,000 Paper Trading Account")
    st.caption(
        f"Simulated trading starting with $10,000 virtual cash (includes trading fees and price slippage). "
        f"Simulated paper-trading performance demonstrates execution accounting and fee-hurdle discipline, "
        f"but does not guarantee future live trading results or standalone statistical model edge."
    )

    port_df = db_state["port_df"]
    port_kpis = db_state["port_kpis"]

    total_equity = port_kpis.get("total_equity_usd")
    cash_balance = port_kpis.get("cash_balance")
    btc_balance = port_kpis.get("btc_balance")
    benchmark_btc_equity = port_kpis.get("benchmark_btc_equity")
    alpha_usd = port_kpis.get("alpha_usd")

    # Clear top-level Spot Trade Counter Banner in Tab 4
    t4_buys = 0
    t4_sells = 0
    t4_holds = 0
    if not port_df.empty and "action" in port_df.columns:
        act_series = port_df["action"].astype(str).str.upper().str.strip()
        t4_buys = int((act_series == "BUY").sum())
        t4_sells = int((act_series == "SELL").sum())
        t4_holds = int((act_series == "HOLD").sum())

    hurdle_usd_t4 = (
        current_price * (MIN_HURDLE_PCT / 100.0)
        if current_price is not None
        else None
    )
    if btc_balance is None:
        st.warning("Paper portfolio metrics are unavailable because the database snapshot could not be loaded.")
    elif btc_balance > 1e-8:
        st.success(
            f"Spot Trade Tracker: ACTIVE POSITION OPEN ({btc_balance:.4f} BTC held) | "
            f"Completed Spot Trades: {t4_buys} Buys, {t4_sells} Sells ({t4_holds} hours held)"
        )
    else:
        hurdle_message = (
            f"Why no Buy yet? The predicted 1h move must clear the ~${hurdle_usd_t4:,.0f} "
            f"({MIN_HURDLE_PCT:.2f}%) fee hurdle."
            if hurdle_usd_t4 is not None
            else "The fee hurdle will be evaluated after a live market candle is available."
        )
        st.info(
            f"Spot Trade Tracker: 100% IN CASH — {t4_buys} Buys, {t4_sells} Sells, "
            f"{t4_holds} Hours Skipped | {hurdle_message}"
        )

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.metric(
            "Account Value",
            f"${float(total_equity):,.2f}" if total_equity is not None else "Unavailable",
            (
                f"{format_signed_usd(float(total_equity) - 10000.0)} total"
                if total_equity is not None
                else "Database required"
            ),
        )
    with p_col2:
        st.metric(
            "Available Cash",
            f"${float(cash_balance):,.2f}" if cash_balance is not None else "Unavailable",
        )
    with p_col3:
        st.metric(
            "Bitcoin Held",
            f"{float(btc_balance):.4f} BTC" if btc_balance is not None else "Unavailable",
        )
    with p_col4:
        st.metric(
            "If You Just Held BTC",
            (
                f"${float(benchmark_btc_equity):,.2f}"
                if benchmark_btc_equity is not None
                else "Unavailable"
            ),
        )
    with p_col5:
        st.metric(
            "Difference vs. Holding BTC",
            format_signed_usd(float(alpha_usd)) if alpha_usd is not None else "Unavailable",
        )

    # Strategy Equity Curve vs Buy & Hold Benchmark
    if not port_df.empty:
        st.markdown("### Live Forward Equity Curve (Strategy vs. Buy & Hold Benchmark)")
        st.plotly_chart(create_alpha_equity_curve(port_df), use_container_width=True)

        st.markdown("### Transaction Execution Ledger")
        t_disp = port_df.iloc[::-1].copy()

        # 1. Plain-English Action Badges
        def _fmt_action(val):
            a = str(val).upper().strip()
            if a == "BUY":
                return "[BUY]"
            elif a == "SELL":
                return "[SELL]"
            elif a == "HOLD":
                return "[HOLD]"
            elif "DEPOSIT" in a:
                return "[DEPOSIT]"
            return f"[{a}]"

        t_disp["Order Action"] = t_disp["action"].apply(_fmt_action)

        # 2. Execution Price
        def _fmt_exec_price(row):
            act = str(row.get("action", "")).upper()
            p = float(row.get("execution_price", 0.0) or 0.0)
            if act in ["BUY", "SELL"] and p > 0:
                return f"${p:,.2f}"
            return "-"

        t_disp["Execution Price"] = t_disp.apply(_fmt_exec_price, axis=1)

        # 3. Trade Size (Quantity in BTC and USD)
        def _fmt_trade_size(row):
            act = str(row.get("action", "")).upper()
            amt = float(row.get("btc_amount", 0.0) or 0.0)
            price = float(row.get("execution_price", 0.0) or 0.0)
            usd_val = abs(amt * price)
            if act == "BUY" and amt > 0:
                return f"+{amt:.4f} BTC (~${usd_val:,.0f})"
            elif act == "SELL" and amt < 0:
                return f"{amt:.4f} BTC (~${usd_val:,.0f})"
            elif act == "HOLD":
                return "0.0000 BTC"
            return "-"

        t_disp["Trade Size"] = t_disp.apply(_fmt_trade_size, axis=1)

        # 4. Fees & Friction
        def _fmt_friction(row):
            act = str(row.get("action", "")).upper()
            fee = float(row.get("fee_deducted_usd", 0.0) or 0.0)
            slip = float(row.get("slippage_usd", 0.0) or 0.0)
            tot = fee + slip
            if act in ["BUY", "SELL"] and tot > 0:
                return f"${tot:,.2f}"
            elif act == "HOLD":
                return "$0.00"
            return "$0.00"

        t_disp["Fees & Friction"] = t_disp.apply(_fmt_friction, axis=1)

        # 5. Portfolio Value & Return
        def _fmt_portfolio_val(row):
            eq = float(row.get("total_equity_usd", 10000.0) or 10000.0)
            return f"${eq:,.2f}"

        t_disp["Portfolio Value"] = t_disp.apply(_fmt_portfolio_val, axis=1)

        # 6. Time (UTC)
        t_disp["Time (UTC)"] = pd.to_datetime(t_disp["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")

        # 7. Plain-English Strategy Rationale
        t_disp["Strategy Rationale"] = t_disp.apply(format_portfolio_rationale, axis=1)

        display_cols = [
            "Time (UTC)",
            "Order Action",
            "Execution Price",
            "Trade Size",
            "Fees & Friction",
            "Portfolio Value",
            "Strategy Rationale",
        ]

        st.dataframe(
            t_disp[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("[INFO] Portfolio initialized with $10,000.00 USDT virtual bankroll. Transactions log each hour as trades execute.")


# -------------------------------------------------------------------------------------
# TREESHAP ATTRIBUTION & TECHNICAL DETAILS
# -------------------------------------------------------------------------------------
with tab2:
    st.subheader("Top Signals Behind This Hour's Prediction")
    st.caption("TreeSHAP feature attributions and walk-forward chronological evaluation boundaries.")

    if is_stale_fallback:
        st.warning(
            "⚠️ Live market feed unavailable (offline replay active). "
            "Live TreeSHAP factor attribution inference is paused to prevent presenting stale explanations as live."
        )
    else:
        shap_res = generate_live_shap_attribution(
            live_df=live_market_df,
            explainer_clf=explainer_clf,
            is_stale=is_stale_fallback,
            top_n=4,
        )
        if shap_res:
            base_val, top_pos, top_neg, fig_shap = shap_res
            sh_col1, sh_col2 = st.columns(2)
            with sh_col1:
                st.markdown("#### Signals Pushing Price UP")
                if top_pos:
                    for driver in top_pos:
                        feat = driver["feature"]
                        plain = get_feature_display_name(feat)
                        st.markdown(f"- **{plain}** (`{feat}`)")
                else:
                    st.write("No strong upward signals this hour.")
            with sh_col2:
                st.markdown("#### Signals Pushing Price DOWN")
                if top_neg:
                    for driver in top_neg:
                        feat = driver["feature"]
                        plain = get_feature_display_name(feat)
                        st.markdown(f"- **{plain}** (`{feat}`)")
                else:
                    st.write("No strong downward signals this hour.")

            if fig_shap is not None:
                try:
                    st.plotly_chart(fig_shap, use_container_width=True)
                except Exception:
                    try:
                        st.pyplot(fig_shap)
                    except Exception:
                        pass
        else:
            st.info("[INFO] TreeSHAP factor attributions ready on next candle close.")

    with st.expander("Technical Model Details & Training History", expanded=False):
        g_col1, g_col2 = st.columns(2)
        with g_col1:
            st.markdown("#### Active Production Model Pedigree")
            st.markdown(build_model_pedigree_markdown())

        with g_col2:
            st.markdown("#### Anti-Leakage Chronological Split Boundary")
            st.markdown(
                """
                ```text
                CLEAN DATASET (34,990 Total 1-Hour Candles):
                ├── Range: 2022-07-08 13:00 UTC to 2026-07-06 08:00 UTC
                └── Note: 1 irregular exchange-halt row (2023-03-24 12:00 UTC) removed

                TRAIN PERIOD (80% Chronological — 27,991 Hours):
                ├── Start: 2022-07-08 13:00:00 UTC
                └── End:   2025-09-17 17:00:00 UTC (No future market data visible to model)

                PURGE BUFFER (1-Hour Causality Guard — 1 Hour):
                └── Excluded: 2025-09-17 18:00:00 UTC (Prevents T+1h target leakage across boundary)

                OUT-OF-SAMPLE TEST PERIOD (20% Holdout — 6,998 Hours):
                ├── Start: 2025-09-17 19:00:00 UTC
                └── End:   2026-07-06 08:00:00 UTC (Pure out-of-sample forward evaluation)
                ```
                """
            )

        st.markdown("#### Controlled Model Benchmark & Rejection Rationale")
        st.markdown(
            """
            | Model Version | Regressor RMSE (USD) | Direction Hit Rate | Classifier Accuracy | Classifier Macro F1 | Governance Status |
            | :--- | :--- | :--- | :--- | :--- | :--- |
            | **Naive Persistence** | **$398.77** | *Baseline* | *Baseline* | *Baseline* | Reference Benchmark |
            | **`xgb_onnx_v1.0.0`** *(Active)* | $399.47 | **50.50%** | **53.26%** | **0.5297** | **ACTIVE PRODUCTION** |
            | **`xgb_v2.0.0`** *(New)* | **$398.54** | **51.20%** | **53.80%** | **0.5340** | **CANDIDATE (SHADOW EVALUATION)** |
            | **`p1_controlled_candidate_v1`** | $399.37 | 50.36% | 53.03% | 0.5280 | **REJECTED CANDIDATE** |

            **Governance Policy & Candidate Evaluation:**
            - **Candidate `xgb_v2.0.0` (Shadow Testing):** Trained on 36 stationary features, beating the active baseline across RMSE ($398.54 vs $399.47), directional hit rate (51.20%), and accuracy (53.80%). Currently logging shadow predictions in Neon DB for forward validation against live candles.
            - **Candidate `p1_controlled_candidate_v1` (Rejected):** Rejected due to lower directional hit rate (50.36% vs 50.50%) and lower accuracy (53.03% vs 53.26%) on holdout evaluation.
            - **Automatic Model Onboarding:** Any new model placed into `artifacts/models/` with a manifest is automatically discovered, verified, served in consensus voting, and tracked in Neon DB without manual code changes.
            """
        )

        st.markdown("#### MLflow Experiment Runs & Bayesian Optimization Trials")
        try:
            import mlflow
            from config import MLFLOW_TRACKING_URI, MLFLOW_EXPERIMENT_NAME
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            runs_df = mlflow.search_runs(experiment_names=[MLFLOW_EXPERIMENT_NAME])
            if not runs_df.empty:
                display_cols = [c for c in [
                    "tags.mlflow.runName", "tags.model_type", "metrics.test_accuracy",
                    "metrics.test_rmse_price", "params.learning_rate", "params.max_depth",
                    "params.n_estimators", "start_time"
                ] if c in runs_df.columns]
                renamed_cols = {
                    "tags.mlflow.runName": "Run Name",
                    "tags.model_type": "Model Type",
                    "metrics.test_accuracy": "Holdout Accuracy",
                    "metrics.test_rmse_price": "Test RMSE",
                    "params.learning_rate": "Learning Rate",
                    "params.max_depth": "Max Depth",
                    "params.n_estimators": "Trees",
                    "start_time": "Date / Time",
                }
                st.dataframe(runs_df[display_cols].rename(columns=renamed_cols).head(10), use_container_width=True)
            else:
                st.info("No MLflow runs recorded yet. Execute `python scripts/run_pipeline.py --tune` to run Optuna Bayesian tuning.")
        except Exception:
            st.caption("Offline MLflow tracking active via 'mlruns/'.")