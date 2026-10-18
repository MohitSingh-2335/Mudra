# app.py
"""
Mudra — Institutional Quantitative Cryptocurrency Intelligence Terminal.
Real-Time Multi-Agent Deliberation, Delayed Ground Truth Reconciliation,
and Autonomous $10,000 Forward Paper Trading Ledger on Neon Cloud PostgreSQL.
"""
# 1. PRE-LOAD ONNXRUNTIME FIRST to prevent Windows OpenMP (0xC0000005) thread collision
try:
    import onnxruntime  # noqa: F401
except ImportError:
    pass

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
    ACTIVE_MODEL_VERSION,
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

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Mudra — Institutional Quant Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-Contrast Quantitative Terminal Styling
st.markdown(
    """
    <style>
    .main { background-color: #0E1117; color: #E0E0E0; }
    .stMetric { background-color: #1A1F2C; border: 1px solid #2B313E; border-radius: 8px; padding: 12px; }
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
    spread_pct = ((recent_candle["high"] - recent_candle["low"]) / recent_candle["close"]) * 100.0
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
    }


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


def build_council_scenario_payload(scenario_selected: str, current_price: float):
    """
    Constructs market_data, engineered_features, and ml_forecast dictionaries for Tab 2
    Multi-Agent Council stress/simulation scenarios.
    Enforces strict internal mathematical consistency across all generated fields:
      - high = current_price * (1.0 + volatility_spread / 2.0)
      - low  = current_price * (1.0 - volatility_spread / 2.0)
      - volatility_range_pct = round(((high - low) / current_price) * 100.0, 3)
      - predicted_dollar_move = round(current_price * (expected_return_pct / 100.0), 2)
      - clears_hurdle = bool(abs(expected_return_pct) >= MIN_HURDLE_PCT)
    """
    current_price = float(current_price)
    if scenario_selected == "bullish":
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
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def fetch_cloud_database_state(current_price: float):
    """Pulls track record, portfolio equity, and reconciliation KPIs directly from Neon."""
    fallback_state = {
        "db_ok": False,
        "db_ver": "Connecting / Offline",
        "db_lat": 0.0,
        "ticks_df": pd.DataFrame(),
        "rec_df": pd.DataFrame(),
        "rec_kpis": {
            "total_evaluated": 0, "total_reconciled": 0,
            "win_rate_pct": 0.0, "directional_accuracy_pct": 0.0,
            "mae_usd": 0.0, "mape_pct": 0.0,
            "information_coefficient": 0.0, "information_coefficient_ic": 0.0,
            "black_swans_filtered": 0, "accurate_count": 0
        },
        "port_df": pd.DataFrame(),
        "port_kpis": {
            "cash_balance": 10000.0, "btc_balance": 0.0,
            "total_equity_usd": 10000.0, "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": 10000.0, "alpha_usd": 0.0,
            "alpha_pct": 0.0, "drawdown_pct": 0.0, "total_trades": 0
        },
        "latest_audit": None,
    }
    try:
        db_ok, db_ver, db_lat = test_connection()
        if not db_ok:
            return fallback_state

        ticks_df = get_latest_market_ticks(limit=100)
        rec_df = get_reconciliation_history(limit=50)
        rec_kpis = get_reconciliation_kpis(lookback_hours=168)
        port_df = get_portfolio_history(limit=100)
        port_kpis = get_portfolio_kpis(current_price=current_price)
        latest_audit = get_latest_council_audit()

        return {
            "db_ok": db_ok,
            "db_ver": db_ver,
            "db_lat": db_lat,
            "ticks_df": ticks_df,
            "rec_df": rec_df,
            "rec_kpis": rec_kpis,
            "port_df": port_df,
            "port_kpis": port_kpis,
            "latest_audit": latest_audit,
        }
    except Exception:
        return fallback_state


# =====================================================================================
# SIDEBAR CONTROLS & STATUS
# =====================================================================================
st.sidebar.title("Mudra Intelligence")
st.sidebar.caption("Institutional Quantitative Cryptocurrency Platform")

# On-Demand Hourly Quant Deliberation & Sync Trigger
if st.sidebar.button("Sync Latest Hourly Candle Now", use_container_width=True):
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

if st.sidebar.button("Refresh Terminal Feeds", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Infrastructure Health")

# Fetch live market data (strictly confirmed closed hourly candles)
live_market_df = fetch_live_market_candles()
current_price = float(live_market_df["close"].iloc[-1]) if not live_market_df.empty else 83000.0

# Assess feed staleness / offline replay status
is_stale_fallback, stale_age_hours = check_feed_staleness(live_market_df)

# Fetch Neon cloud DB state
db_state = fetch_cloud_database_state(current_price=current_price)

if db_state["db_ok"]:
    st.sidebar.success(f"**Neon Cloud:** Connected ({db_state['db_lat']:.0f}ms)")
else:
    st.sidebar.error("**Neon Cloud:** Disconnected (SQLite test fallback)")

if is_stale_fallback:
    st.sidebar.warning(f"**Market Feed:** Offline Replay ({stale_age_hours:.0f}h lag)")
elif not live_market_df.empty:
    feed_src = str(live_market_df.attrs.get("source", "Live")).title()
    st.sidebar.success(f"**Market Feed:** {feed_src} Confirmed Closed")
else:
    st.sidebar.error("**Market Feed:** Feed Offline")
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Quant Architecture:**
    - Single Source of Truth: Neon PostgreSQL
    - Inference: Tuned XGBoost Regressor
    - Multi-Agent: LangGraph 2-Node Council
    - Forward Ledger: $10,000 Portfolio
    """
)


# =====================================================================================
# TOP EXECUTIVE KPI BANNER & 1-SECOND WEBSOCKET RIBBON
# =====================================================================================
st.title("Mudra Quantitative Cryptocurrency Terminal")

# Client-Side 1-Second Binance WebSocket Ticker (0% Server Load)
components.html(
    """
    <div style="background-color: #0E1117; border: 1px solid #1E2329; border-radius: 8px; padding: 10px 16px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; color: #FFFFFF; display: flex; justify-content: space-between; align-items: center; box-sizing: border-box;">
        <div style="display: flex; align-items: center; gap: 12px;">
            <span style="font-size: 1.05rem; font-weight: 700; color: #F0B90B;">BTC/USDT</span>
            <span id="price" style="font-size: 1.35rem; font-weight: 800; font-family: monospace; transition: color 0.15s ease;">$---.--</span>
            <span id="change" style="font-size: 0.88rem; font-weight: 600; padding: 2px 6px; border-radius: 4px;">---%</span>
        </div>
        <div style="display: flex; gap: 16px; font-size: 0.82rem; color: #848E9C; align-items: center;">
            <div>24h High: <span id="high" style="color: #EAECEF; font-weight: 600; font-family: monospace;">$---.--</span></div>
            <div>24h Low: <span id="low" style="color: #EAECEF; font-weight: 600; font-family: monospace;">$---.--</span></div>
            <div>24h Vol: <span id="vol" style="color: #EAECEF; font-weight: 600; font-family: monospace;">--- BTC</span></div>
            <div style="display: flex; align-items: center; gap: 6px; margin-left: 8px;">
                <span style="display: inline-block; width: 8px; height: 8px; border-radius: 50%; background-color: #0ECB81; box-shadow: 0 0 6px #0ECB81;"></span>
                <span style="color: #0ECB81; font-weight: 600;">Binance WebSocket 1s</span>
            </div>
        </div>
    </div>
    <script>
        const ws = new WebSocket("wss://stream.binance.com:9443/ws/btcusdt@ticker");
        let lastPrice = 0;
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            const price = parseFloat(data.c);
            const change = parseFloat(data.P);
            const high = parseFloat(data.h);
            const low = parseFloat(data.l);
            const vol = parseFloat(data.v);

            const priceEl = document.getElementById("price");
            const changeEl = document.getElementById("change");
            const highEl = document.getElementById("high");
            const lowEl = document.getElementById("low");
            const volEl = document.getElementById("vol");

            if (priceEl) {
                priceEl.innerText = "$" + price.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                if (lastPrice !== 0) {
                    priceEl.style.color = price >= lastPrice ? "#0ECB81" : "#F6465D";
                }
                lastPrice = price;
            }
            if (changeEl) {
                const isPos = change >= 0;
                changeEl.innerText = (isPos ? "+" : "") + change.toFixed(2) + "%";
                changeEl.style.color = isPos ? "#0ECB81" : "#F6465D";
                changeEl.style.backgroundColor = isPos ? "rgba(14, 203, 129, 0.15)" : "rgba(246, 70, 93, 0.15)";
            }
            if (highEl) highEl.innerText = "$" + high.toLocaleString(undefined, {minimumFractionDigits: 2});
            if (lowEl) lowEl.innerText = "$" + low.toLocaleString(undefined, {minimumFractionDigits: 2});
            if (volEl) volEl.innerText = vol.toLocaleString(undefined, {maximumFractionDigits: 0}) + " BTC";
        };
    </script>
    """,
    height=60,
)

st.caption(f"Real-Time BTC/USDT Multi-Agent Deliberation, Delayed Ground Truth Reconciliation & Autonomous Forward Ledger • Last Sync: {datetime.now(timezone.utc).strftime('%H:%M:%S UTC')}")


# Top Metrics Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

# Calculate 24h price change
price_24h_ago = float(live_market_df["close"].iloc[-24]) if len(live_market_df) >= 24 else current_price
delta_24h_pct = ((current_price - price_24h_ago) / price_24h_ago) * 100.0

with kpi1:
    spot_label = "BTC Spot Price [OFFLINE REPLAY]" if is_stale_fallback else "BTC Spot Price"
    st.metric(
        spot_label,
        f"${current_price:,.2f}",
        f"{delta_24h_pct:+.2f}% (24h)",
        help="Real-time BTC/USDT spot price streamed from Binance candles, with 24-hour percentage price change.",
    )

with kpi2:
    audit = db_state["latest_audit"]
    action_badge = audit.get("final_decision", "HOLD") if audit else "HOLD"
    verdict = audit.get("risk_verdict", "APPROVED") if audit else "APPROVED"
    st.metric(
        "Multi-Agent Verdict",
        f"{action_badge}",
        f"Critic: {verdict}",
        help="Autonomous decision from the 2-Node Multi-Agent Council (Lead Strategist proposal audited by Reflexion Risk Critic).",
    )

with kpi3:
    rec_kpis = db_state["rec_kpis"]
    win_rate = float(
        rec_kpis.get("directional_accuracy_pct")
        if rec_kpis.get("directional_accuracy_pct") is not None
        else rec_kpis.get("win_rate_pct", 0.0)
    )
    total_eval = int(
        rec_kpis.get("total_reconciled")
        if rec_kpis.get("total_reconciled") is not None
        else rec_kpis.get("total_evaluated", 0)
    )
    st.metric(
        "Recent Direction Accuracy (7d)",
        f"{win_rate:.1f}%",
        f"{total_eval} evaluated hours",
        help="Percentage of hourly forecasts in the last 7 days (168 hours) where the model's predicted direction (UP or DOWN) matched the realized market close.",
    )

with kpi4:
    port = db_state["port_kpis"]
    equity = float(port.get("total_equity_usd", 10000.0))
    ret_pct = float(port.get("cumulative_return_pct", 0.0))
    st.metric(
        "Portfolio Equity",
        f"${equity:,.2f}",
        f"{ret_pct:+.2f}% vs $10k",
        help="Total marked-to-market value of the simulated forward paper-trading account (Cash + Bitcoin Inventory) from the $10,000 initial bankroll.",
    )

with kpi5:
    alpha = float(port.get("alpha_usd", 0.0))
    alpha_pct = float(port.get("alpha_pct", 0.0))
    st.metric(
        "Excess Alpha",
        f"${alpha:+,.2f}",
        f"{alpha_pct:+.2f}% over B&H",
        help="Dollar and percentage outperformance of the Mudra strategy relative to a passive Bitcoin Buy & Hold benchmark over the paper-trading period.",
    )

st.markdown("---")


# =====================================================================================
# 5 INSTITUTIONAL TABS
# =====================================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Candlestick Terminal",
    "Multi-Agent Council (LangGraph)",
    "Live Track Record (Delayed Truth)",
    "$10,000 Paper Portfolio Ledger",
    "MLOps Governance & TreeSHAP",
])


# -------------------------------------------------------------------------------------
# TAB 1: LIVE CANDLESTICK TERMINAL
# -------------------------------------------------------------------------------------
with tab1:
    st.subheader("BTC/USDT Real-Time Streaming Candlestick Terminal")
    st.caption("Live market price action, technical momentum ribbons (EMA-20 / EMA-50), and machine-learning next-hour price forecasts.")

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

    with st.expander("Python Microstructural Technical Analysis (20/50 EMA Ribbon & Order Flow)"):
        if is_stale_fallback:
            st.info(f"ℹ️ Technical chart displaying historical local data replay ({stale_age_hours:.0f}h lag).")
        candlestick_fig = create_tradingview_candlestick(live_market_df)
        st.plotly_chart(candlestick_fig, use_container_width=True)

    # Next-Hour Forecast Panel
    st.markdown("### Next-Hour Model Forecast Breakdown")
    st.caption("Single-row forward inference produced by the active XGBoost model (`xgb_onnx_v1.0.0`) evaluated at the latest confirmed candle close.")
    if is_stale_fallback:
        st.warning(
            f"⚠️ Live market feed unavailable (offline replay active, {stale_age_hours:.0f}h lag). "
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
                st.metric(
                    "Predicted Direction",
                    forecast_result["direction_badge"],
                    f"Up Prob: {forecast_result['prob_up']:.1f}%",
                    help="The model's forecasted market direction for the next 60 minutes based on machine learning classification probabilities.",
                )
            with f_col2:
                st.metric(
                    "Expected 1h Move",
                    f"{forecast_result['pred_ret_pct']:+.3f}%",
                    f"Cleared {MIN_HURDLE_PCT:.2f}% Hurdle [PASS]" if forecast_result["cleared_hurdle"] else "Below Fee Hurdle [SKIP]",
                    help=f"Forecasted percentage price return over the next hour. To justify execution fees and slippage, the absolute expected move must clear the {MIN_HURDLE_PCT:.2f}% hurdle threshold.",
                )
            with f_col3:
                st.metric(
                    "Target Close Price",
                    f"${forecast_result['pred_target']:,.2f}",
                    f"${forecast_result['pred_dollar_move']:+,.2f}",
                    help="The absolute dollar price target predicted for the next hourly candle close, along with the expected dollar change from current price.",
                )
            with f_col4:
                taker = forecast_result["taker"]
                st.metric(
                    "Buyer Aggression (Taker Ratio)",
                    f"{taker:.4f}",
                    "Order Flow: " + ("Buyer Pressure (+)" if taker > 0.52 else ("Seller Pressure (-)" if taker < 0.48 else "Balanced (0)")),
                    help="Share of hourly volume triggered by market buyers (> 0.52 indicates buyer dominance; < 0.48 indicates seller dominance).",
                )
            with f_col5:
                spread_pct = forecast_result["spread_pct"]
                st.metric(
                    "Hourly Volatility Spread",
                    f"{spread_pct:.2f}%",
                    "Regime: " + ("Extreme Flush [ALERT]" if spread_pct > 3.8 else "Controlled [NORMAL]"),
                    help="The high-low price range as a percentage of close price over the last completed hourly candle (> 3.8% triggers high-volatility circuit breaker warnings).",
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
# TAB 2: MULTI-AGENT COUNCIL (LANGGRAPH)
# -------------------------------------------------------------------------------------
with tab2:
    st.subheader("Autonomous Multi-Agent Trading Council (LangGraph)")
    st.caption("Cyclical 2-node StateGraph: Lead Quantitative Strategist synthesizes thesis, and Reflexion Risk Critic enforces volatility & Kelly limits.")

    st.markdown("#### Quantitative Stress-Testing & Simulation Scenarios")
    st.caption("Click any scenario below to convene the AI agents live on demand:")

    # Deliberation Trigger & Stress Testing Scenarios
    c_btn1, c_btn2, c_btn3, c_btn4 = st.columns(4)
    run_deliberation = False
    scenario_selected = "live"

    if c_btn1.button("Deliberate Live Market", use_container_width=True):
        scenario_selected = "live"
        run_deliberation = True
    if c_btn2.button("Stress: Bullish Breakout", use_container_width=True):
        scenario_selected = "bullish"
        run_deliberation = True
    if c_btn3.button("Stress: Volatility Flush Veto", use_container_width=True):
        scenario_selected = "vol_spike"
        run_deliberation = True
    if c_btn4.button("Stress: Sub-Hurdle Chop Veto", use_container_width=True):
        scenario_selected = "chop"
        run_deliberation = True

    if run_deliberation:
        with st.spinner("Convening Strategist and Risk Critic nodes..."):
            m_data, t_data, f_data = build_council_scenario_payload(
                scenario_selected=scenario_selected,
                current_price=current_price,
            )

            st.session_state["council_state"] = run_trading_graph(
                symbol="BTCUSDT",
                current_price=current_price,
                market_data=m_data,
                engineered_features=t_data,
                ml_forecast=f_data,
            )

    # Use simulated/live state if triggered, otherwise fallback to latest audit from database
    council_state = st.session_state.get("council_state")
    if not council_state and db_state.get("latest_audit"):
        audit = db_state["latest_audit"]
        council_state = {
            "final_decision": audit.get("final_decision", "HOLD"),
            "risk_verdict": audit.get("risk_verdict", "APPROVED"),
            "strategist_action": audit.get("strategist_action", "HOLD"),
            "strategist_conviction": audit.get("strategist_conviction", 5),
            "kelly_fraction": audit.get("kelly_fraction", 0.10),
            "veto_reasons": audit.get("veto_reasons", []),
            "thesis": (
                audit.get("execution_notes")
                or audit.get("thesis")
                or audit.get("strategist_thesis")
                or audit.get("executive_summary")
                or "Awaiting live deliberation or click scenario button above."
            ),
        }
    elif not council_state:
        council_state = {
            "final_decision": "HOLD",
            "risk_verdict": "APPROVED",
            "strategist_action": "HOLD",
            "strategist_conviction": 5,
            "kelly_fraction": 0.10,
            "veto_reasons": [],
            "thesis": "Awaiting live deliberation. Click any scenario button above to convene the Multi-Agent Council.",
        }

    # Deliberation Outcome Banner
    act = council_state.get("final_decision", "HOLD")
    vrd = council_state.get("risk_verdict", "APPROVED")
    conv = council_state.get("strategist_conviction", 5)
    kelly_pct = float(council_state.get("kelly_fraction", 0.0)) * 100.0
    vetoes = council_state.get("veto_reasons", [])

    if vrd == "APPROVED":
        st.success(f"[APPROVED] COUNCIL VERDICT: {act} AUTHORIZED | Recommended Kelly Capital Allocation: {kelly_pct:.1f}%")
    else:
        st.error(f"[VETOED] ORDER VETOED BY RISK CRITIC: {', '.join(vetoes) if vetoes else 'Risk thresholds breached'}")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        with st.expander("Node 1: Lead Quantitative Strategist Thesis", expanded=True):
            st.markdown(f"**Proposed Action:** `{council_state.get('strategist_action', 'HOLD')}` (Conviction: {conv}/10)")
            thesis_text = (
                council_state.get("thesis")
                or council_state.get("strategist_thesis")
                or council_state.get("executive_summary")
                or "Thesis unavailable"
            )
            st.markdown(f"**Executive Thesis:**\n\n{thesis_text}")
            if council_state.get("circuit_breaker_active"):
                st.caption("*Running in Deterministic Circuit Breaker Mode.*")
    with col_s2:
        with st.expander("Node 2: Reflexion Risk Critic Veto Audit", expanded=True):
            st.markdown(f"**Critic Verdict:** `{vrd}`")
            st.markdown(f"**Allocated Sizing Fraction:** `{kelly_pct:.1f}%` of available cash")
            sl_str, tp_str = format_risk_levels(action=act, verdict=vrd, current_price=current_price, council_state=council_state)
            st.markdown(f"**Stop-Loss:** `{sl_str}`")
            st.markdown(f"**Target Take-Profit:** `{tp_str}`")
            if vetoes:
                st.markdown("**Breached Invariants:**")
                for v in vetoes:
                    st.markdown(f"- [VETO] `{v}`")


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
    kpi_fetcher: Optional[Callable[[int, Optional[str]], Dict[str, Any]]] = None,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """
    Resolves the KPI metrics dictionary for Tab 3 based on selected model.
    If 'All Models', returns (base_kpis, None) unless base_kpis itself failed.
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
            res = kpi_fetcher(168, selected_model)
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
# TAB 3: LIVE TRACK RECORD & DELAYED RECONCILIATION (NEON DB SSOT)
# -------------------------------------------------------------------------------------
with tab3:
    st.subheader("Live Prediction Track Record & Delayed Ground Truth Ledger")
    st.caption("Delayed ground truth reconciliation evaluates prediction at T against confirmed candle close at T+1. All records stored permanently in Neon PostgreSQL.")

    # Read-only Model Version Selector
    selected_model = st.selectbox(
        "Filter Delayed Track Record by Model",
        options=TAB3_MODEL_SELECTOR_OPTIONS,
        key="tab3_model_selector",
    )

    rec_df = db_state.get("rec_df", pd.DataFrame())
    base_kpis = db_state.get("rec_kpis", {})

    # Filter reconciliation records and resolve KPIs safely
    filtered_rec_df = filter_reconciliation_records(rec_df, selected_model)
    rec_kpis, kpi_warning = resolve_tab3_reconciliation_kpis(
        selected_model=selected_model,
        base_kpis=base_kpis,
        filtered_df=filtered_rec_df,
        kpi_fetcher=get_reconciliation_kpis if db_state.get("db_ok", False) else None,
    )
    if kpi_warning:
        st.warning(f"⚠️ {kpi_warning}")

    # Rolling Metrics Ribbon
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        tot_val = rec_kpis.get("total_reconciled") if rec_kpis.get("total_reconciled") is not None else rec_kpis.get("total_evaluated", 0)
        st.metric(
            "Total Evaluated Hours",
            f"{tot_val}",
            help="Total number of historical 1-hour candles where an advance prediction was delayed and evaluated against realized close.",
        )
    with m_col2:
        val = rec_kpis.get("directional_accuracy_pct") if rec_kpis.get("directional_accuracy_pct") is not None else rec_kpis.get("win_rate_pct")
        st.metric(
            "Direction Accuracy (Up/Down)",
            f"{float(val):.1f}%" if val is not None else "N/A",
            help="How often the model correctly predicted whether Bitcoin would close higher or lower 1 hour later.",
        )
    with m_col3:
        val = rec_kpis.get("mae_usd")
        st.metric(
            "Avg Dollar Error (MAE)",
            f"${float(val):,.2f}" if val is not None else "N/A",
            help="The average dollar difference between the forecasted price and the actual closing price.",
        )
    with m_col4:
        val = rec_kpis.get("mape_pct")
        st.metric(
            "Mean Percentage Error (MAPE)",
            f"{float(val):.2f}%" if val is not None else "N/A",
            help="Mean absolute percentage error between the forecasted price and actual market price.",
        )
    with m_col5:
        val = rec_kpis.get("information_coefficient_ic") if rec_kpis.get("information_coefficient_ic") is not None else rec_kpis.get("information_coefficient")
        st.metric(
            "Forecast Correlation (IC)",
            f"{float(val):.3f}" if val is not None else "N/A",
            help="Measures how well the size of predicted moves tracks actual moves (-1.0 to +1.0; above +0.02 shows useful signal).",
        )

    # Hit / Miss Scatter Chart
    if not filtered_rec_df.empty:
        st.markdown("### Realized Price vs. Predicted Price Accuracy Chart")
        st.plotly_chart(create_reconciliation_scatter(filtered_rec_df), use_container_width=True)

        st.markdown("### Historical Delayed Reconciliation Audit Table")
        with st.expander("📘 How to Read This Delayed Prediction Report Card (Plain-English Guide)", expanded=False):
            st.markdown(
                """
                * **Predicted vs. Actual Direction:** Compares the AI's forecasted direction (`UP` vs `DOWN`) against where the market actually closed (no deadbands, every micro-move is categorized).
                * **Direction Result (`[MATCH]` vs `[MISMATCH]`):** Confirms whether the realized market trend matched the forecast.
                * **Predicted Price (Expected Move):** The dollar price target and percentage return forecasted 60 minutes in advance.
                * **Actual Market Close (Realized Move):** The real closing price and percentage move confirmed by Binance.
                * **Price Difference ($):** The dollar gap between the AI's forecast and the final close.
                * **AI Diagnosis:** A plain-English summary explaining whether the hour was a tight target hit, sideways drift, momentum expansion, or a directional reversal.
                """
            )

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

        # 2. Combined Price and Percentage Move Columns
        def _fmt_pred_price(row):
            p = float(row.get("predicted_price", 0.0) or 0.0)
            r = float(row.get("predicted_return_pct", 0.0) or 0.0)
            return f"${p:,.2f} ({r:+.2f}%)"

        def _fmt_actual_price(row):
            p = float(row.get("actual_price_at_t_plus_1", 0.0) or 0.0)
            r = float(row.get("actual_return_pct", 0.0) or 0.0)
            return f"${p:,.2f} ({r:+.2f}%)"

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

    # Read-Only Forecast Error Investigation Explorer (Stage 5 Audit Surface)
    st.markdown("---")
    with st.expander("🔍 Forecast Error Investigations & Root-Cause Audit Log", expanded=False):
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
# TAB 4: $10,000 PAPER TRADING PORTFOLIO LEDGER (NEON DB SSOT)
# -------------------------------------------------------------------------------------
with tab4:
    st.subheader("Forward $10,000 Paper Trading Portfolio & Alpha Attribution")
    st.caption(
        f"Forward paper-trading simulation ledger on a $10,000 initial bankroll, automatically deducting "
        f"{TAKER_FEE_PCT:.2f}% taker fee and {SLIPPAGE_PCT:.2f}% slippage ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side, "
        f"{ROUND_TRIP_FRICTION_PCT:.2f}% round-trip). Simulated paper-trading performance demonstrates execution accounting "
        f"and fee-hurdle discipline, but does not guarantee future live trading results or standalone statistical model edge."
    )

    port_df = db_state["port_df"]
    port_kpis = db_state["port_kpis"]

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.metric(
            "Total Portfolio Equity",
            f"${float(port_kpis.get('total_equity_usd', 10000.0)):,.2f}",
            f"{float(port_kpis.get('cumulative_return_pct', 0.0)):+.2f}%",
            help="Combined cash and marked-to-market Bitcoin inventory value in the simulated $10,000 forward paper-trading account.",
        )
    with p_col2:
        st.metric(
            "Cash Balance",
            f"${float(port_kpis.get('cash_balance', 10000.0)):,.2f}",
            help="Available USDT cash balance held in reserve when cash-preservation rules or risk vetos prevent market orders.",
        )
    with p_col3:
        st.metric(
            "BTC Inventory",
            f"{float(port_kpis.get('btc_balance', 0.0)):.6f} BTC",
            help="Quantity of Bitcoin currently held in the forward simulation portfolio.",
        )
    with p_col4:
        st.metric(
            "Buy & Hold Benchmark",
            f"${float(port_kpis.get('benchmark_btc_equity', 10000.0)):,.2f}",
            help="What the initial $10,000 would be worth if 100% had been invested in Bitcoin at portfolio inception without active management.",
        )
    with p_col5:
        st.metric(
            "Strategy Alpha",
            f"${float(port_kpis.get('alpha_usd', 0.0)):+,.2f}",
            f"{float(port_kpis.get('alpha_pct', 0.0)):+.2f}% vs B&H",
            help="Excess dollar and percentage return generated by the active strategy over the passive Buy & Hold benchmark.",
        )

    # Strategy Equity Curve vs Buy & Hold Benchmark
    if not port_df.empty:
        st.markdown("### Live Forward Equity Curve (Strategy vs. Buy & Hold Benchmark)")
        st.plotly_chart(create_alpha_equity_curve(port_df), use_container_width=True)

        st.markdown("### Transaction Execution Ledger")
        with st.expander("📘 How to Read This Transaction Execution Ledger (Plain-English Guide)", expanded=False):
            st.markdown(
                f"""
                * **Order Action (`[BUY]`, `[SELL]`, `[HOLD]`, `[DEPOSIT]`):** The decision executed by the multi-agent council after evaluating market volatility and the fee hurdle.
                * **Execution Price:** The actual Binance market close price where the order was filled (or `-` when holding).
                * **Trade Size:** The quantity of Bitcoin bought or sold, along with its approximate dollar value.
                * **Fees & Friction:** The modeled taker fee ({TAKER_FEE_PCT:.2f}%) and slippage ({SLIPPAGE_PCT:.2f}%) per execution ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). When holding, this shows `$0.00 (Fees Saved)`.
                * **Portfolio Value:** Total account equity (Cash + Bitcoin inventory marked to market) and cumulative percentage return from the $10,000 start.
                * **Strategy Rationale:** A plain-English explanation of why the trade was executed or why cash was preserved.
                """
            )

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
                return f"${tot:,.2f} ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}%)"
            elif act == "HOLD":
                return "$0.00 (Fees Saved)"
            return "$0.00"

        t_disp["Fees & Friction"] = t_disp.apply(_fmt_friction, axis=1)

        # 5. Portfolio Value & Return
        def _fmt_portfolio_val(row):
            eq = float(row.get("total_equity_usd", 10000.0) or 10000.0)
            ret = float(row.get("cumulative_return_pct", 0.0) or 0.0)
            return f"${eq:,.2f} ({ret:+.2f}%)"

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
# TAB 5: MLOPS MODEL GOVERNANCE & TREESHAP ATTRIBUTION
# -------------------------------------------------------------------------------------
with tab5:
    st.subheader("Production MLOps Model Governance & TreeSHAP Factor Attribution")
    st.caption("Evidence boundary auditing, feature attributions, and Optuna Bayesian optimization lineage.")

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
        | **`p1_controlled_candidate_v1`** | $399.37 | 50.36% | 53.03% | 0.5280 | **REJECTED CANDIDATE** |

        **Governance Rejection Rationale:**
        *Why was candidate `p1_controlled_candidate_v1` rejected?* While the candidate regressor price RMSE was marginally lower ($399.37 vs $399.47), both models underperformed naive persistence ($398.77). More importantly, the candidate had a lower directional hit rate (50.36% vs 50.50%), lower classification accuracy (53.03% vs 53.26%), and lower macro F1. In accordance with Mudra's model governance policy, candidates that fail to beat the active baseline across key trading metrics are permanently rejected and blocked from production deployment.
        """
    )

    st.markdown("---")
    st.markdown("#### Live TreeSHAP Shapley Factor Attribution")
    st.caption("Exact mathematical feature contributions pushing the probability UP (Bullish) or DOWN (Bearish).")
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
                st.markdown("**Bullish Contributors (Pushed Up):**")
                for driver in top_pos:
                    st.markdown(f"- **`{driver['feature']}`**: SHAP value `+{driver['shap']:.4f}`")
            with sh_col2:
                st.markdown("**Bearish Contributors (Pushed Down):**")
                for driver in top_neg:
                    st.markdown(f"- **`{driver['feature']}`**: SHAP value `{driver['shap']:.4f}`")
        else:
            st.info("[INFO] TreeSHAP factor attributions ready on next candle close.")
