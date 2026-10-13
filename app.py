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
import sys
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Guard stdout reconfiguration against Windows CRT pipe corruption
if hasattr(sys.stdout, "reconfigure") and sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8")

from typing import Tuple, Dict, Any, Optional
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timezone
import requests
import plotly.graph_objects as go

from config import (
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    FEATURED_BTC_DATA_PATH,
    TAKER_FEE_PCT,
    SLIPPAGE_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
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
    """Loads production XGBoost regressor and TreeSHAP explainer."""
    try:
        reg = joblib.load(REGRESSOR_MODEL_PATH)
        clf = joblib.load(CLASSIFIER_MODEL_PATH)
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
    cleared_hurdle = abs(pred_ret_pct) >= 0.05

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
    st.metric(spot_label, f"${current_price:,.2f}", f"{delta_24h_pct:+.2f}% (24h)")

with kpi2:
    audit = db_state["latest_audit"]
    action_badge = audit.get("final_decision", "HOLD") if audit else "HOLD"
    verdict = audit.get("risk_verdict", "APPROVED") if audit else "APPROVED"
    st.metric("Multi-Agent Verdict", f"{action_badge}", f"Critic: {verdict}")

with kpi3:
    rec_kpis = db_state["rec_kpis"]
    win_rate = float(rec_kpis.get("directional_accuracy_pct") or rec_kpis.get("win_rate_pct", 0.0))
    total_eval = int(rec_kpis.get("total_reconciled") or rec_kpis.get("total_evaluated", 0))
    st.metric("All-Time Win Rate", f"{win_rate:.1f}%", f"{total_eval} evaluated hours")

with kpi4:
    port = db_state["port_kpis"]
    equity = float(port.get("total_equity_usd", 10000.0))
    ret_pct = float(port.get("cumulative_return_pct", 0.0))
    st.metric("Portfolio Equity", f"${equity:,.2f}", f"{ret_pct:+.2f}% vs $10k")

with kpi5:
    alpha = float(port.get("alpha_usd", 0.0))
    alpha_pct = float(port.get("alpha_pct", 0.0))
    st.metric("Excess Alpha", f"${alpha:+,.2f}", f"{alpha_pct:+.2f}% over B&H")

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
    st.caption("Official TradingView live Binance feed • Millisecond WebSocket streaming, real-time volume & interactive timeframes.")

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
                st.metric("Predicted Direction", forecast_result["direction_badge"], f"Up Prob: {forecast_result['prob_up']:.1f}%")
            with f_col2:
                st.metric(
                    "Expected 1h Move",
                    f"{forecast_result['pred_ret_pct']:+.3f}%",
                    "Cleared 0.05% Hurdle [PASS]" if forecast_result["cleared_hurdle"] else "Below Fee Hurdle [SKIP]",
                )
            with f_col3:
                st.metric("Target Close Price", f"${forecast_result['pred_target']:,.2f}", f"${forecast_result['pred_dollar_move']:+,.2f}")
            with f_col4:
                taker = forecast_result["taker"]
                st.metric(
                    "Taker Buy Ratio",
                    f"{taker:.4f}",
                    "Order Flow: " + ("Buyer Pressure (+)" if taker > 0.52 else ("Seller Pressure (-)" if taker < 0.48 else "Balanced (0)")),
                )
            with f_col5:
                spread_pct = forecast_result["spread_pct"]
                st.metric(
                    "Hourly Volatility Spread",
                    f"{spread_pct:.2f}%",
                    "Regime: " + ("Extreme Flush [ALERT]" if spread_pct > 3.8 else "Controlled [NORMAL]"),
                )


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
            if scenario_selected == "bullish":
                m_data = {"close": current_price, "high": current_price * 1.01, "low": current_price * 0.99, "volume": 5000.0, "taker_buy_ratio": 0.65}
                t_data = {"volatility_spread": 0.015, "order_flow_bias": "STRONG_BUYER_AGGRESSION", "rsi_estimate": 62.0}
                f_data = {"expected_return_pct": 0.42, "predicted_dollar_move": 350.0, "clears_hurdle": True, "confidence": 0.88, "direction": "UP", "direction_badge": "BULLISH [UP]"}
            elif scenario_selected == "vol_spike":
                m_data = {"close": current_price, "high": current_price * 1.05, "low": current_price * 0.95, "volume": 12000.0, "taker_buy_ratio": 0.35}
                t_data = {"volatility_spread": 0.052, "order_flow_bias": "EXTREME_LIQUIDATION", "rsi_estimate": 28.0}
                f_data = {"expected_return_pct": 0.20, "predicted_dollar_move": 160.0, "clears_hurdle": True, "confidence": 0.60, "direction": "UP", "direction_badge": "BULLISH [UP]"}
            elif scenario_selected == "chop":
                m_data = {"close": current_price, "high": current_price * 1.002, "low": current_price * 0.998, "volume": 400.0, "taker_buy_ratio": 0.49}
                t_data = {"volatility_spread": 0.004, "order_flow_bias": "FLAT_CHOP", "rsi_estimate": 50.0}
                f_data = {"expected_return_pct": 0.02, "predicted_dollar_move": 15.0, "clears_hurdle": False, "confidence": 0.45, "direction": "SIDEWAYS", "direction_badge": "SIDEWAYS [CHOP]"}
            else:
                m_data = {"close": current_price, "high": current_price * 1.005, "low": current_price * 0.995, "volume": 1200.0, "taker_buy_ratio": 0.51}
                t_data = {"volatility_spread": 0.018, "order_flow_bias": "NEUTRAL", "rsi_estimate": 51.0}
                f_data = {"expected_return_pct": 0.08, "predicted_dollar_move": 65.0, "clears_hurdle": True, "confidence": 0.65, "direction": "UP", "direction_badge": "BULLISH [UP]"}

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
            "thesis": audit.get("execution_notes", "Awaiting live deliberation or click scenario button above."),
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
            st.markdown(f"**Executive Thesis:**\n\n{council_state.get('thesis', 'Thesis unavailable')}")
            if council_state.get("circuit_breaker_active"):
                st.caption("*Running in Deterministic Circuit Breaker Mode.*")
    with col_s2:
        with st.expander("Node 2: Reflexion Risk Critic Veto Audit", expanded=True):
            st.markdown(f"**Critic Verdict:** `{vrd}`")
            st.markdown(f"**Allocated Sizing Fraction:** `{kelly_pct:.1f}%` of available cash")
            st.markdown(f"**Stop-Loss:** `${current_price * 0.985:,.2f}` (-1.5%)")
            st.markdown(f"**Target Take-Profit:** `${current_price * 1.025:,.2f}` (+2.5%)")
            if vetoes:
                st.markdown("**Breached Invariants:**")
                for v in vetoes:
                    st.markdown(f"- [VETO] `{v}`")


# -------------------------------------------------------------------------------------
# TAB 3: LIVE TRACK RECORD & DELAYED RECONCILIATION (NEON DB SSOT)
# -------------------------------------------------------------------------------------
with tab3:
    st.subheader("Live Prediction Track Record & Delayed Ground Truth Ledger")
    st.caption("Delayed ground truth reconciliation evaluates prediction at T against confirmed candle close at T+1. All records stored permanently in Neon PostgreSQL.")

    rec_df = db_state["rec_df"]
    rec_kpis = db_state["rec_kpis"]

    # Rolling Metrics Ribbon
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.metric("Total Evaluated Hours", f"{rec_kpis.get('total_reconciled') or rec_kpis.get('total_evaluated', 0)}")
    with m_col2:
        st.metric("Directional Win Rate", f"{float(rec_kpis.get('directional_accuracy_pct') or rec_kpis.get('win_rate_pct', 0.0)):.1f}%")
    with m_col3:
        st.metric("Mean Absolute Error (MAE)", f"${float(rec_kpis.get('mae_usd', 0.0)):,.2f}")
    with m_col4:
        st.metric("Mean Error (MAPE)", f"{float(rec_kpis.get('mape_pct', 0.0)):.2f}%")
    with m_col5:
        st.metric("Information Coeff (IC)", f"{float(rec_kpis.get('information_coefficient_ic') or rec_kpis.get('information_coefficient', 0.0)):.3f}")

    # Hit / Miss Scatter Chart
    if not rec_df.empty:
        st.markdown("### Realized Price vs. Predicted Price Accuracy Chart")
        st.plotly_chart(create_reconciliation_scatter(rec_df), use_container_width=True)

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
        disp_df = rec_df.copy()

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
            is_match = row["Predicted Direction"] == row["Actual Direction"]
            diff = float(row.get("absolute_error_usd", 0.0) or 0.0)
            pred_ret = float(row.get("predicted_return_pct", 0.0) or 0.0)
            act_ret = float(row.get("actual_return_pct", 0.0) or 0.0)
            tag = str(row.get("error_attribution_tag", ""))

            if tag == "BLACK_SWAN_OUTLIER" or abs(act_ret) >= 3.8:
                return f"Market Flash Anomaly: Extreme liquidation spike ({act_ret:+.2f}% move)."

            if is_match:
                if diff <= 150.0 or tag == "ACCURATE":
                    return f"Target Hit: Accurate directional call within ${diff:,.2f} variance."
                elif abs(act_ret) > abs(pred_ret):
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
        st.info("[INFO] Delayed reconciliation runs automatically every hour as candles close. First evaluated predictions will appear on the next hourly candle close.")


# -------------------------------------------------------------------------------------
# TAB 4: $10,000 PAPER TRADING PORTFOLIO LEDGER (NEON DB SSOT)
# -------------------------------------------------------------------------------------
with tab4:
    st.subheader("Forward $10,000 Paper Trading Portfolio & Alpha Attribution")
    st.caption(f"Live forward execution accounting for {TAKER_FEE_PCT:.2f}% taker fee and {SLIPPAGE_PCT:.2f}% slippage ({TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side). Proves excess Alpha over Bitcoin Buy & Hold benchmark.")

    port_df = db_state["port_df"]
    port_kpis = db_state["port_kpis"]

    p_col1, p_col2, p_col3, p_col4, p_col5 = st.columns(5)
    with p_col1:
        st.metric("Total Portfolio Equity", f"${float(port_kpis.get('total_equity_usd', 10000.0)):,.2f}", f"{float(port_kpis.get('cumulative_return_pct', 0.0)):+.2f}%")
    with p_col2:
        st.metric("Cash Balance", f"${float(port_kpis.get('cash_balance', 10000.0)):,.2f}")
    with p_col3:
        st.metric("BTC Inventory", f"{float(port_kpis.get('btc_balance', 0.0)):.6f} BTC")
    with p_col4:
        st.metric("Buy & Hold Benchmark", f"${float(port_kpis.get('benchmark_btc_equity', 10000.0)):,.2f}")
    with p_col5:
        st.metric("Strategy Alpha", f"${float(port_kpis.get('alpha_usd', 0.0)):+,.2f}", f"{float(port_kpis.get('alpha_pct', 0.0)):+.2f}% vs B&H")

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
        def _fmt_rationale(row):
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
                if "hurdle" in notes.lower() or "friction" in notes.lower() or "0.14" in notes:
                    return "Capital Preserved: Expected move is below 0.14% round-trip fee hurdle."
                elif "veto" in notes.lower():
                    return "Risk Veto: Trade prevented by Risk Critic to protect capital."
                return "Capital Preserved: Sideways chop; preserving cash until high-conviction setup."
            elif "DEPOSIT" in act:
                return "Portfolio Initialized: Starting bankroll of $10,000.00 USDT."
            return notes if notes else "Execution logged."

        t_disp["Strategy Rationale"] = t_disp.apply(_fmt_rationale, axis=1)

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
        st.markdown(
            """
            | Parameter | Active Value | Rationale |
            | :--- | :--- | :--- |
            | **Model Architecture** | `XGBRegressor (CPU)` | Sub-millisecond single-row latency |
            | **Model Checkpoint** | `xgb_onnx_v1.0.0` | Optuna 50 Bayesian trial champion |
            | **Cross-Validation RMSE**| `$396.22` | Chronological walk-forward split |
            | **Directional Accuracy**| `53.17%` | Out-of-sample test set |
            | **Round-Trip Friction** | `0.30%` | Tested over 7,000 continuous hours |
            | **Net Strategy Alpha** | `+33.55 pp` | Outperformed Buy & Hold |
            """
        )

    with g_col2:
        st.markdown("#### Anti-Leakage Chronological Split Boundary")
        st.markdown(
            """
            ```
            TRAIN PERIOD (80% Chronological):
            ├── 2023-01-01 00:00:00 UTC
            └── 2024-03-31 23:00:00 UTC (No future data visible)
            
            TEST PERIOD (20% Out-Of-Sample):
            ├── 2024-04-01 00:00:00 UTC
            └── 2024-10-31 23:00:00 UTC (Pure out-of-sample forward test)
            ```
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
