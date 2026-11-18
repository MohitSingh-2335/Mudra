import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure") and sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from config import (
    ACTIVE_MODEL_VERSION,
    APP_SUBTITLE,
    APP_TITLE,
    CLASSIFIER_MODEL_PATH,
    CLASSIFIER_MODEL_SHA256,
    FEATURE_DISPLAY_NAMES,
    MIN_HURDLE_PCT,
    REGRESSOR_MODEL_PATH,
    REGRESSOR_MODEL_SHA256,
    ROUND_TRIP_FRICTION_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.agents.tools import fetch_historical_candles
from src.agents.trading_graph import run_trading_graph
from src.database.connection import get_db, test_connection
from src.database.investigations import get_prediction_investigations
from src.database.queries import (
    get_latest_council_audit,
    get_latest_market_ticks,
    get_portfolio_history,
    get_portfolio_kpis,
    get_reconciliation_history,
    get_reconciliation_kpis,
)
from src.engine.sync import sync_hourly_quant_cycle
from src.explainability import get_tree_explainer
from src.model_loader import load_verified_model
from src.ui.charts import (
    create_alpha_equity_curve,
    create_reconciliation_scatter,
    create_tradingview_candlestick,
)

st.set_page_config(page_title=APP_TITLE, layout="wide", initial_sidebar_state="expanded")
st.markdown("<style>.main { background-color: #0E1117; color: #E0E0E0; }</style>", unsafe_allow_html=True)


# Load verified production XGBoost models
@st.cache_resource
def load_models():
    try:
        reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
        clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
        reg.set_params(device="cpu")
        clf.set_params(device="cpu")
        return reg, clf, get_tree_explainer(clf)
    except Exception as exc:
        st.error(f"Failed to load ML artifacts: {exc}")
        return None, None, None


reg_model, clf_model, explainer_clf = load_models()


# Helper to format signed USD values
def format_signed_usd(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "Unavailable"
    sign = "+" if val > 0 else ("-" if val < 0 else "")
    return f"{sign}${abs(val):,.{decimals}f}"


# Check if the market candle feed is stale
def check_feed_staleness(df: pd.DataFrame, max_lag_hours: float = 3.0) -> Tuple[bool, float]:
    if df is None or df.empty:
        return True, 0.0
    latest_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    lag = max(0.0, (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600.0)
    return (lag > max_lag_hours or df.attrs.get("source") == "local_fallback", lag)


# Generate next-hour return and direction predictions from live candle
def generate_live_forecast(live_df: pd.DataFrame, reg_model: Any, clf_model: Any, is_stale: bool = False) -> Optional[Dict[str, Any]]:
    if is_stale or live_df is None or live_df.empty or reg_model is None or len(live_df) < 35:
        return None

    from src.feature_engineering import (
        extract_features_for_inference,
        predict_proba_with_validation,
        predict_with_validation,
        validate_inference_features,
    )

    try:
        sample_input = extract_features_for_inference(live_df)
        recent = live_df.iloc[-1]
        curr_price = float(recent["close"])
        candle_range = float(recent["high"] - recent["low"])
        spread = (candle_range / curr_price) * 100.0 if curr_price > 0 else 0.0
        taker = float(recent.get("taker_buy_ratio", 0.5))

        raw_pred_ret = float(predict_with_validation(reg_model, sample_input, model_type="regressor")[0])
        pred_ret_pct = raw_pred_ret * 100.0
        pred_move = curr_price * raw_pred_ret
        cleared = abs(pred_ret_pct) >= MIN_HURDLE_PCT

        prob_up, prob_down = 50.0, 50.0
        direction = "UP" if pred_ret_pct >= 0 else "DOWN"
        if clf_model:
            clf_df = validate_inference_features(sample_input, model_type="classifier")
            code = int(predict_with_validation(clf_model, clf_df, model_type="classifier")[0])
            direction = "UP" if code == 1 else "DOWN"
            if hasattr(clf_model, "predict_proba"):
                probs = predict_proba_with_validation(clf_model, clf_df)[0]
                prob_down, prob_up = float(probs[0]) * 100.0, float(probs[1]) * 100.0

        direction_badge = "SIDEWAYS (CHOP)" if not cleared else ("BULLISH (UP)" if direction == "UP" else "BEARISH (DOWN)")

        return {
            "raw_pred_ret": raw_pred_ret,
            "pred_ret_pct": pred_ret_pct,
            "pred_dollar_move": pred_move,
            "pred_target": curr_price + pred_move,
            "cleared_hurdle": cleared,
            "direction_label": direction,
            "direction_badge": direction_badge,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "taker": taker,
            "spread_pct": spread,
            "candle_range_usd": candle_range,
            "hurdle_dollar_min": curr_price * (MIN_HURDLE_PCT / 100.0),
            "current_price": curr_price,
        }
    except Exception:
        return None


# Build agent council scenario dictionary
def build_council_scenario_payload(scenario_selected: str, current_price: float, live_forecast: Optional[Dict[str, Any]] = None, live_df: Optional[pd.DataFrame] = None):
    current_price = float(current_price)
    if scenario_selected == "live" and live_forecast:
        expected_ret = float(live_forecast.get("pred_ret_pct", 0.08))
        c_dir = live_forecast.get("direction_label", "UP")
        confidence = float(live_forecast.get("prob_up", 50.0)) / 100.0 if c_dir == "UP" else float(live_forecast.get("prob_down", 50.0)) / 100.0
        vol_spread = max(0.002, float(live_forecast.get("spread_pct", 1.8)) / 100.0)
        taker = float(live_forecast.get("taker", 0.51))
        order_bias = "BUYER_AGGRESSION" if taker > 0.52 else ("SELLER_AGGRESSION" if taker < 0.48 else "NEUTRAL")
        volume = float(live_df["volume"].iloc[-1]) if live_df is not None and not live_df.empty and "volume" in live_df.columns else 1200.0
        rsi = float(live_df["rsi"].iloc[-1]) if live_df is not None and not live_df.empty and "rsi" in live_df.columns else 50.0
    elif scenario_selected == "bullish":
        vol_spread, volume, taker = 0.015, 5000.0, 0.65
        order_bias, rsi, expected_ret, confidence = "STRONG_BUYER_AGGRESSION", 62.0, 0.42, 0.88
    elif scenario_selected == "vol_spike":
        vol_spread, volume, taker = 0.052, 12000.0, 0.35
        order_bias, rsi, expected_ret, confidence = "EXTREME_LIQUIDATION", 28.0, -0.20, 0.60
    elif scenario_selected == "chop":
        vol_spread, volume, taker = 0.004, 400.0, 0.49
        order_bias, rsi, expected_ret, confidence = "FLAT_CHOP", 50.0, 0.02, 0.45
    else:
        vol_spread, volume, taker = 0.018, 1200.0, 0.51
        order_bias, rsi, expected_ret, confidence = "NEUTRAL", 51.0, 0.08, 0.65

    clears = bool(abs(expected_ret) >= MIN_HURDLE_PCT)
    direction = "SIDEWAYS" if not clears else ("UP" if expected_ret >= 0 else "DOWN")
    m_data = {"close": current_price, "high": current_price * (1 + vol_spread / 2), "low": current_price * (1 - vol_spread / 2), "volume": volume, "taker_buy_ratio": taker}
    t_data = {"volatility_spread": vol_spread, "volatility_range_pct": round(vol_spread * 100, 3), "order_flow_bias": order_bias, "rsi_estimate": rsi}
    f_data = {"expected_return_pct": expected_ret, "predicted_dollar_move": round(current_price * (expected_ret / 100.0), 2), "clears_hurdle": clears, "confidence": confidence, "direction": direction}
    return m_data, t_data, f_data


# Human-readable rationale for paper portfolio trades
def format_portfolio_rationale(row) -> str:
    act = str(row.get("action", "")).upper()
    notes = str(row.get("notes", "") or "")
    if act == "BUY":
        amt = float(row.get("btc_amount", 0.0) or 0.0)
        p = float(row.get("execution_price", 0.0) or 0.0)
        return f"Position Opened: Allocated 10% Kelly fraction (~${amt * p:,.0f}) on approved bullish signal."
    if act == "SELL":
        pnl = float(row.get("realized_pnl_usd", 0.0) or 0.0)
        return f"Position Closed: Sold inventory at market (Realized PnL: ${pnl:+,.2f})."
    if act == "HOLD":
        notes_lower = notes.lower()
        if any(w in notes_lower for w in ["sub-hurdle", "below hurdle", "fee drag", "fails to clear"]):
            return f"Capital Preserved: Expected move is below {MIN_HURDLE_PCT:.2f}% round-trip fee hurdle."
        if "veto" in notes_lower:
            return "Risk Veto: Trade prevented by Risk Critic to protect capital."
        return "Capital Preserved: Sideways chop; preserving cash until high-conviction setup."
    if "DEPOSIT" in act:
        return "Portfolio Initialized: Starting bankroll of $10,000.00 USDT."
    return notes or "Execution logged."


# Markdown pedigree table for model transparency
def build_model_pedigree_markdown() -> str:
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


# Fetch recent market candles
@st.cache_data(ttl=60)
def fetch_live_market_candles(symbol: str = "BTCUSDT") -> pd.DataFrame:
    try:
        return fetch_historical_candles(symbol=symbol, limit=100)
    except Exception as exc:
        empty = pd.DataFrame()
        empty.attrs["error"] = str(exc)
        return empty


# Query cloud database state and metrics
@st.cache_data(ttl=60)
def fetch_cloud_database_state(current_price: Optional[float]):
    fallback = {"db_ok": False, "db_ver": "Offline", "db_lat": 0.0, "db_error": "Database unavailable", "ticks_df": pd.DataFrame(), "rec_df": pd.DataFrame(), "rec_kpis": {}, "port_df": pd.DataFrame(), "port_kpis": {}, "latest_audit": None}
    try:
        db_ok, db_ver, db_lat = test_connection()
        if not db_ok:
            return fallback
        return {
            "db_ok": True, "db_ver": db_ver, "db_lat": db_lat, "db_error": None,
            "ticks_df": get_latest_market_ticks(limit=500),
            "rec_df": get_reconciliation_history(limit=500),
            "rec_kpis": get_reconciliation_kpis(lookback_hours=168),
            "port_df": get_portfolio_history(limit=500),
            "port_kpis": get_portfolio_kpis(current_price=current_price) if current_price is not None else {},
            "latest_audit": get_latest_council_audit(),
        }
    except Exception as exc:
        fallback["db_error"] = str(exc)
        return fallback


# Format stop-loss and take-profit levels
def format_risk_levels(action: str, verdict: str, current_price: float, council_state: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    c_state = council_state or {}
    act = str(action).upper().strip()
    vrd = str(verdict).upper().strip()
    if vrd != "APPROVED" or act == "HOLD" or not act:
        no_trade = "None (No active trade opened — cash preserved)"
        return no_trade, no_trade

    if act == "BUY":
        sl = float(c_state.get("stop_loss_price") or (current_price * 0.985))
        tp = float(c_state.get("take_profit_price") or (current_price * 1.025))
        return f"${sl:,.2f} (-1.5%)", f"${tp:,.2f} (+2.5%)"
    if act == "SELL":
        sl = float(c_state.get("stop_loss_price") or (current_price * 1.015))
        tp = float(c_state.get("take_profit_price") or (current_price * 0.975))
        return f"${sl:,.2f} (+1.5%)", f"${tp:,.2f} (-2.5%)"

    no_trade = "None (No active trade opened — cash preserved)"
    return no_trade, no_trade


# Filter reconciliation records by model version
def filter_reconciliation_records(df: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    if df.empty or "model_version" not in df.columns or selected_model in ("All Models", ""):
        return df.copy()
    return df[df["model_version"] == selected_model].copy()


# Prepare display dataframe for reconciliation table
def prepare_reconciliation_table_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    disp = df.copy()
    disp["Model Version"] = disp["model_version"].fillna("UNKNOWN") if "model_version" in disp.columns else "UNKNOWN"
    return disp


# Fallback reconciliation KPI calculations
def compute_fallback_reconciliation_kpis(df: pd.DataFrame, model_version: Optional[str] = None) -> Dict[str, Any]:
    if df.empty:
        return {"total_evaluated": 0, "total_reconciled": 0, "directional_accuracy_pct": None, "win_rate_pct": None, "mae_usd": None, "model_version": model_version}
    total = len(df)
    correct = int(df["direction_correct"].sum()) if "direction_correct" in df.columns else 0
    win_rate = round((correct / total) * 100.0, 2) if total > 0 else 0.0
    mae = round(float(df["absolute_error_usd"].mean()), 2) if "absolute_error_usd" in df.columns else 0.0
    return {"total_evaluated": total, "total_reconciled": total, "directional_accuracy_pct": win_rate, "win_rate_pct": win_rate, "mae_usd": mae, "model_version": model_version}


# Resolve reconciliation KPIs for UI
def resolve_tab3_reconciliation_kpis(selected_model: str, base_kpis: Dict[str, Any], filtered_df: pd.DataFrame, kpi_fetcher: Optional[Callable] = None, lookback_hours: Optional[int] = 168):
    if selected_model in ("All Models", ""):
        if kpi_fetcher and lookback_hours != 168:
            try:
                res = kpi_fetcher(lookback_hours, None)
                if isinstance(res, dict) and not res.get("query_failed"):
                    return res, None
            except Exception:
                pass
        return base_kpis, None
    if kpi_fetcher:
        try:
            res = kpi_fetcher(lookback_hours, selected_model)
            if isinstance(res, dict) and not res.get("query_failed"):
                return res, None
        except Exception:
            pass
    if not filtered_df.empty:
        return compute_fallback_reconciliation_kpis(filtered_df, selected_model), f"Displaying fallback KPIs computed from {len(filtered_df)} records."
    return {"total_evaluated": 0, "total_reconciled": 0, "directional_accuracy_pct": None}, f"KPIs for '{selected_model}' currently unavailable."


# --- Sidebar ---
st.sidebar.title("Mudra BTC Terminal")
st.sidebar.caption("Hourly Bitcoin Price Prediction & Paper Trading")

if st.sidebar.button("Run Hourly Prediction Now", use_container_width=True):
    with st.spinner("Deliberating..."):
        try:
            synced, msg, _ = sync_hourly_quant_cycle(force=True)
            if synced:
                st.sidebar.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.info(msg)
        except Exception as exc:
            st.sidebar.error(f"Sync error: {exc}")

if st.sidebar.button("Refresh Data", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("System Status")

live_market_df = fetch_live_market_candles()
current_price = float(live_market_df["close"].iloc[-1]) if not live_market_df.empty else None
is_stale_fallback, stale_age_hours = check_feed_staleness(live_market_df)
db_state = fetch_cloud_database_state(current_price=current_price)

if db_state["db_ok"]:
    st.sidebar.success(f"Database: Connected ({db_state['db_lat']:.0f}ms)")
else:
    st.sidebar.error("Database: Offline")

if is_stale_fallback:
    st.sidebar.warning(f"Market Data: Stale replay ({stale_age_hours:.0f}h lag)" if not live_market_df.empty else "Market Data: Unavailable")
elif not live_market_df.empty:
    st.sidebar.success("Market Data: Live (1h Confirmed Candles)")
else:
    st.sidebar.error("Market Data: Offline")

st.title(APP_TITLE)
st.caption(APP_SUBTITLE)

# KPI Header Metrics
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
p_24h = float(live_market_df["close"].iloc[-24]) if len(live_market_df) >= 24 else None
d_24h = current_price - p_24h if current_price is not None and p_24h is not None else None

with kpi1:
    st.metric("Bitcoin Price", f"${current_price:,.2f}" if current_price and not is_stale_fallback else "Unavailable", f"{format_signed_usd(d_24h)} (24h)" if d_24h and not is_stale_fallback else "Fresh candle required")
with kpi2:
    audit = db_state["latest_audit"]
    st.metric("AI Trade Decision", audit.get("final_decision", "Unavailable") if audit else "Unavailable", "Risk Check: Passed" if audit and audit.get("risk_verdict") == "APPROVED" else "Database audit required", delta_color="off")
with kpi3:
    rk = db_state["rec_kpis"]
    win_rate = rk.get("directional_accuracy_pct") or rk.get("win_rate_pct")
    tot_eval = rk.get("total_reconciled") if rk.get("total_reconciled") is not None else rk.get("total_evaluated")
    st.metric("Direction Accuracy (7d)", f"{float(win_rate):.1f}%" if win_rate is not None else "Unavailable", f"{int(tot_eval)}h rolling" if tot_eval is not None else "Database required", delta_color="off")
with kpi4:
    eq = db_state["port_kpis"].get("total_equity_usd")
    st.metric("Paper Account Value", f"${float(eq):,.2f}" if eq is not None else "Unavailable", f"{format_signed_usd(float(eq) - 10000.0)} vs $10k start" if eq is not None else "Database required")
with kpi5:
    alpha = db_state["port_kpis"].get("alpha_usd")
    st.metric("vs. Holding Bitcoin", format_signed_usd(float(alpha)) if alpha is not None else "Unavailable", "Ahead of Buy & Hold" if alpha and float(alpha) >= 0 else "Behind Buy & Hold" if alpha else "Database required")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Overview", "Model Evidence", "Paper Portfolio"])
forecast_result = None

# --- TAB 1: OVERVIEW ---
with tab1:
    st.subheader("Live Bitcoin (BTC/USDT) 1-Hour Chart")
    components.html(
        """
        <div class="tradingview-widget-container" style="height:540px;width:100%;margin:0;padding:0;">
          <div id="tradingview_mudra" style="height:100%;width:100%;"></div>
          <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
          <script type="text/javascript">
          new TradingView.widget({
            "autosize": true, "symbol": "BINANCE:BTCUSDT", "interval": "60", "timezone": "Etc/UTC",
            "theme": "dark", "style": "1", "locale": "en", "toolbar_bg": "#0E1117", "enable_publishing": false,
            "container_id": "tradingview_mudra", "backgroundColor": "#0E1117", "gridColor": "rgba(43, 49, 62, 0.4)"
          });
          </script>
        </div>
        """,
        height=550,
    )

    with st.expander("Python Technical Chart (EMA 20/50 & Volume)", expanded=False):
        st.plotly_chart(create_tradingview_candlestick(live_market_df), use_container_width=True)

    st.markdown("### Next-Hour Model Prediction")
    if is_stale_fallback:
        st.warning("Live market feed unavailable (offline replay active). Model forecast inference paused.")
    else:
        forecast_result = generate_live_forecast(live_df=live_market_df, reg_model=reg_model, clf_model=clf_model, is_stale=is_stale_fallback)
        if forecast_result:
            f1, f2, f3, f4, f5 = st.columns(5)
            conf = forecast_result["prob_up"] if forecast_result["direction_label"] == "UP" else forecast_result["prob_down"]
            with f1:
                st.metric("Predicted Direction (1h)", forecast_result["direction_label"], f"{conf:.0f}% Confidence", delta_color="off")
            with f2:
                st.metric("Predicted 1h Price Change", format_signed_usd(forecast_result['pred_dollar_move']), f"Clears ${forecast_result['hurdle_dollar_min']:,.0f} Fee Cost" if forecast_result["cleared_hurdle"] else f"Below ${forecast_result['hurdle_dollar_min']:,.0f} Fee Cost (Hold)", delta_color="normal" if forecast_result["cleared_hurdle"] else "off")
            with f3:
                st.metric("Predicted Price in 1h", f"${forecast_result['pred_target']:,.2f}", f"Current: ${current_price:,.2f}", delta_color="off")
            with f4:
                st.metric("Last 1h Price Range", f"${forecast_result['candle_range_usd']:,.2f}", "High Price Swing Alert" if forecast_result["spread_pct"] > 3.8 else "Normal Price Swing", delta_color="off")
            with f5:
                tk = forecast_result["taker"]
                st.metric("Buyer vs. Seller Activity", "Buyers Leading" if tk > 0.52 else ("Sellers Leading" if tk < 0.48 else "Balanced"), f"{tk * 100:.0f}% Buyers / {(1.0 - tk) * 100:.0f}% Sellers", delta_color="off")

# --- TAB 2: MODEL EVIDENCE ---
with tab2:
    st.subheader("Model Evidence & Risk Review")
    st.caption("Review the live council decision, delayed ground-truth accuracy, and model explanations.")
    if current_price is None or is_stale_fallback or not db_state["db_ok"]:
        st.warning("Council unavailable: fresh confirmed market candle and database connection required.")
        st.stop()

    if st.button("Check current live market", use_container_width=True, disabled=forecast_result is None):
        with st.spinner("Deliberating..."):
            m_data, t_data, f_data = build_council_scenario_payload("live", current_price, forecast_result, live_market_df)
            st.session_state["council_state"] = run_trading_graph("BTCUSDT", current_price, m_data, t_data, f_data)

    council_state = st.session_state.get("council_state") or (db_state.get("latest_audit") or {"final_decision": "HOLD", "risk_verdict": "APPROVED", "strategist_action": "HOLD", "strategist_conviction": 5, "thesis": "Click button above to run."})
    act = council_state.get("final_decision", "HOLD")
    vrd = council_state.get("risk_verdict", "APPROVED")
    conv = council_state.get("strategist_conviction", 5)

    if vrd == "APPROVED" and act == "HOLD":
        st.info("Decision: HOLD (No Trade) — Predicted dollar move is smaller than trading fees. Keeping cash safe.")
    elif vrd == "APPROVED" and act in ("BUY", "SELL"):
        st.success(f"Decision: {act} Approved — Executing trade.")
    else:
        st.error("Decision: HOLD (Trade Blocked by Risk Check)")

    c_s1, c_s2 = st.columns(2)
    with c_s1:
        st.markdown(f"**Proposed Action:** `{council_state.get('strategist_action', act)}` (Confidence: {conv}/10)")
        st.markdown(f"**Thesis:** {council_state.get('thesis', council_state.get('execution_notes', 'None'))}")
    with c_s2:
        sl_str, tp_str = format_risk_levels(action=act, verdict=vrd, current_price=current_price, council_state=council_state)
        st.markdown(f"**Risk Verdict:** `{vrd}` | **Stop Loss:** `{sl_str}` | **Take Profit:** `{tp_str}`")

    # Reconciliation section
    st.markdown("---")
    st.subheader("Past Prediction Accuracy (Checked 1 Hour Later)")
    rec_df = db_state.get("rec_df", pd.DataFrame())
    base_kpis = db_state.get("rec_kpis", {})

    t3_col1, t3_col2 = st.columns([1, 1])
    with t3_col1:
        selected_model = st.selectbox("Filter Delayed Track Record by Model", options=["All Models", ACTIVE_MODEL_VERSION], key="tab3_model_selector")
    with t3_col2:
        lookback_label = st.radio("Evaluation Time Window", options=["All Time (Total History)", "Last 7 Days (168 Hours)", "Last 24 Hours (24 Hours)"], horizontal=True, key="tab3_lookback_selector")

    selected_lookback = None if "All Time" in lookback_label else (24 if "24 Hours" in lookback_label else 168)
    filtered_rec_df = filter_reconciliation_records(rec_df, selected_model)
    rec_kpis, _ = resolve_tab3_reconciliation_kpis(selected_model, base_kpis, filtered_rec_df, get_reconciliation_kpis if db_state.get("db_ok") else None, selected_lookback)

    if not filtered_rec_df.empty:
        st.plotly_chart(create_reconciliation_scatter(filtered_rec_df), use_container_width=True)
        disp_df = prepare_reconciliation_table_display(filtered_rec_df)
        disp_df["Predicted Direction"] = disp_df["predicted_return_pct"].apply(lambda r: "UP" if float(r or 0) >= 0 else "DOWN")
        disp_df["Actual Direction"] = disp_df["actual_return_pct"].apply(lambda r: "UP" if float(r or 0) >= 0 else "DOWN")
        disp_df["Direction Result"] = (disp_df["Predicted Direction"] == disp_df["Actual Direction"]).apply(lambda x: "[MATCH]" if x else "[MISMATCH]")
        disp_df["Predicted Price (Expected Move)"] = disp_df["predicted_price"].apply(lambda p: f"${float(p or 0):,.2f}")
        disp_df["Actual Market Close (Realized Move)"] = disp_df["actual_price_at_t_plus_1"].apply(lambda p: f"${float(p or 0):,.2f}")
        disp_df["Price Difference ($)"] = disp_df["absolute_error_usd"].apply(lambda x: f"${float(x or 0):,.2f}")
        disp_df["Time (UTC)"] = pd.to_datetime(disp_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")

        def _diag(row):
            pred_dir, act_dir = str(row.get("Predicted Direction", "")).upper(), str(row.get("Actual Direction", "")).upper()
            diff, act_ret = float(row.get("absolute_error_usd", 0) or 0), float(row.get("actual_return_pct", 0) or 0)
            tag = str(row.get("error_attribution_tag", ""))
            if tag == "BLACK_SWAN_OUTLIER" or abs(act_ret) >= 3.8:
                return f"Market Flash Anomaly: Extreme liquidation spike ({act_ret:+.2f}% move)."
            if pred_dir == act_dir:
                return f"Target Hit: Accurate directional call within ${diff:,.2f} variance."
            return f"Direction Miss: Market reversed by {act_ret:+.2f}% (${diff:,.2f} gap)."

        disp_df["AI Diagnosis"] = disp_df.apply(_diag, axis=1)
        st.dataframe(disp_df[["Time (UTC)", "Model Version", "Predicted Direction", "Actual Direction", "Direction Result", "Predicted Price (Expected Move)", "Actual Market Close (Realized Move)", "Price Difference ($)", "AI Diagnosis"]], use_container_width=True, hide_index=True)

    with st.expander("Inspect Specific Prediction Error Details (Forecast Error Investigations & Root-Cause Audit Log)", expanded=False):
        inv_col1, inv_col2 = st.columns([3, 2])
        with inv_col1:
            search_kw = st.text_input("Search News Headlines / Macro Catalysts", value="", key="investigation_headline_search")
        with inv_col2:
            status_opt = st.selectbox("Filter by Verification Status", options=["ALL", "UNVERIFIED_HYPOTHESIS", "CONFIRMED_CAUSE", "REFUTED", "APPROVED_FOR_TRAINING"], key="investigation_status_filter")
        try:
            f_status = None if status_opt == "ALL" else status_opt
            with get_db() as inv_session:
                inv_records = get_prediction_investigations(inv_session, headline_search=search_kw.strip() or None, status=f_status, limit=30)
            if inv_records:
                st.dataframe(pd.DataFrame(inv_records), use_container_width=True, hide_index=True)
            else:
                st.info("No recorded prediction investigations match your query.")
        except Exception:
            st.info("Diagnostic investigations database operating in local mode.")

    with st.expander("Technical Model Details & Training History", expanded=False):
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("#### Active Production Model Pedigree")
            st.markdown(build_model_pedigree_markdown())
        with g2:
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
            """
        )

# --- TAB 3: PAPER PORTFOLIO ---
with tab3:
    st.subheader("$10,000 Paper Trading Account")
    st.caption("Simulated paper-trading performance demonstrates execution accounting and fee-hurdle discipline.")
    port_df = db_state["port_df"]
    port_kpis = db_state["port_kpis"]
    total_equity = port_kpis.get("total_equity_usd")
    cash_balance = port_kpis.get("cash_balance")
    btc_balance = port_kpis.get("btc_balance")
    benchmark_btc = port_kpis.get("benchmark_btc_equity")
    alpha_usd = port_kpis.get("alpha_usd")

    p1, p2, p3, p4, p5 = st.columns(5)
    with p1:
        st.metric("Account Value", f"${float(total_equity):,.2f}" if total_equity is not None else "Unavailable", f"{format_signed_usd(float(total_equity) - 10000.0)} total" if total_equity is not None else "Database required")
    with p2:
        st.metric("Available Cash", f"${float(cash_balance):,.2f}" if cash_balance is not None else "Unavailable")
    with p3:
        st.metric("Bitcoin Held", f"{float(btc_balance):.4f} BTC" if btc_balance is not None else "Unavailable")
    with p4:
        st.metric("If You Just Held BTC", f"${float(benchmark_btc):,.2f}" if benchmark_btc is not None else "Unavailable")
    with p5:
        st.metric("Difference vs. Holding BTC", format_signed_usd(float(alpha_usd)) if alpha_usd is not None else "Unavailable")

    if not port_df.empty:
        st.plotly_chart(create_alpha_equity_curve(port_df), use_container_width=True)
        t_disp = port_df.iloc[::-1].copy()
        t_disp["Order Action"] = t_disp["action"].apply(lambda a: f"[{str(a).upper().strip()}]")
        t_disp["Execution Price"] = t_disp.apply(lambda r: f"${float(r.get('execution_price', 0.0)):,.2f}" if str(r.get("action")).upper() in ["BUY", "SELL"] and float(r.get("execution_price", 0.0)) > 0 else "-", axis=1)
        t_disp["Trade Size"] = t_disp.apply(lambda r: f"{float(r.get('btc_amount', 0.0)):+.4f} BTC" if str(r.get("action")).upper() in ["BUY", "SELL"] else "-", axis=1)
        t_disp["Fees & Friction"] = t_disp.apply(lambda r: f"${float(r.get('fee_deducted_usd', 0.0)) + float(r.get('slippage_usd', 0.0)):,.2f}", axis=1)
        t_disp["Portfolio Value"] = t_disp["total_equity_usd"].apply(lambda eq: f"${float(eq or 10000.0):,.2f}")
        t_disp["Time (UTC)"] = pd.to_datetime(t_disp["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")
        t_disp["Strategy Rationale"] = t_disp.apply(format_portfolio_rationale, axis=1)
        st.dataframe(t_disp[["Time (UTC)", "Order Action", "Execution Price", "Trade Size", "Fees & Friction", "Portfolio Value", "Strategy Rationale"]], use_container_width=True, hide_index=True)
    else:
        st.info("Portfolio initialized with $10,000.00 USDT virtual bankroll.")
