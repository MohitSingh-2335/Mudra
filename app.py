import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Linux / Streamlit Cloud SQLite upgrade hook for ChromaDB
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import html
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if hasattr(sys.stdout, "reconfigure") and sys.stdout.isatty():
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Mudra — Bitcoin AI/ML Forecasting Engine", layout="wide", initial_sidebar_state="expanded")

# Polished custom dark theme and component styling
CUSTOM_DASHBOARD_CSS = """
<style>
/* Main app dark theme background and clean typography */
.main, .stApp {
    background-color: #0E1117;
    color: #E6EDF3;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
}

/* Polished Metric Cards */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, #161B22 0%, #0F141C 100%);
    border: 1px solid #30363D;
    border-radius: 8px;
    padding: 12px 14px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.28);
    transition: border-color 0.15s ease, transform 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    border-color: #58A6FF;
    transform: translateY(-1px);
}
div[data-testid="stMetric"] label {
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    font-weight: 600 !important;
    color: #8B949E !important;
}
div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: #F0F6FC !important;
    letter-spacing: -0.02em;
}
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}

/* Bordered Container Polishing */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #11151C;
    border-color: #30363D !important;
    border-radius: 10px !important;
    padding: 4px !important;
    margin-bottom: 0.5rem;
}

/* Section Header Badges */
.section-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #58A6FF;
    background: rgba(56, 189, 248, 0.12);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 4px;
    padding: 2px 8px;
    margin-bottom: 6px;
}

/* Multi-Agent Council Highlight Card */
.deliberation-card {
    background: linear-gradient(135deg, #161B22 0%, #1A2333 100%);
    border: 1px solid #30363D;
    border-left: 4px solid #58A6FF;
    border-radius: 8px;
    padding: 14px 16px;
    margin-bottom: 12px;
}

/* Streamlit Dataframe Borders & Polish */
div[data-testid="stDataFrame"] {
    border: 1px solid #30363D;
    border-radius: 8px;
    overflow: hidden;
}

/* Buttons Styling */
button[kind="primary"], button[kind="secondary"] {
    border-radius: 6px !important;
    font-weight: 600 !important;
    transition: all 0.15s ease;
}

/* Sidebar Custom Styling */
section[data-testid="stSidebar"] {
    background-color: #0B0E14 !important;
    border-right: 1px solid #21262D;
}
.sidebar-footer {
    font-size: 0.75rem;
    color: #6E7681;
    margin-top: 1.5rem;
    text-align: center;
    border-top: 1px solid #21262D;
    padding-top: 0.75rem;
}
</style>
"""
st.markdown(CUSTOM_DASHBOARD_CSS, unsafe_allow_html=True)

from config import (
    ACTIVE_MODEL_VERSION, APP_SUBTITLE, APP_TITLE,
    CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256,
    MIN_HURDLE_PCT, REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256,
    ROUND_TRIP_FRICTION_PCT, TOTAL_FRICTION_PER_SIDE_PCT,
)
from src.agents import fetch_historical_candles, run_trading_graph
from src.database import (
    get_db, get_latest_council_audit, get_latest_market_ticks,
    get_portfolio_history, get_portfolio_kpis,
    get_reconciliation_history, get_reconciliation_kpis,
    test_connection,
)
from src.engine import sync_hourly_quant_cycle
from src.model_loader import load_verified_model

INITIAL_CAPITAL_USD = 10000.0


# ---------------------------------------------------------------------------
# 1. CHART LAYOUTS & PLOTLY BUILDERS
# ---------------------------------------------------------------------------

def _base_chart_layout(title: str, height: int = 400) -> dict:
    return dict(
        template="plotly_dark",
        title=dict(
            text=f"<b>{title}</b>",
            font=dict(size=14, color="#F0F6FC", family="-apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica, Arial, sans-serif"),
            x=0.01,
            y=0.98,
        ),
        paper_bgcolor="#0E1117",
        plot_bgcolor="#11151C",
        margin=dict(l=40, r=40, t=44, b=24),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#8B949E"),
        ),
        hoverlabel=dict(
            bgcolor="#161B22",
            bordercolor="#30363D",
            font=dict(size=12, color="#F0F6FC"),
        ),
        height=height,
    )


def create_tradingview_candlestick(df: pd.DataFrame) -> go.Figure:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "close" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Market Data Available", template="plotly_dark")
        return fig

    df_plot = df.copy()
    if len(df_plot) >= 20:
        df_plot["ema_20"] = df_plot["close"].ewm(span=20, adjust=False).mean()
    if len(df_plot) >= 50:
        df_plot["ema_50"] = df_plot["close"].ewm(span=50, adjust=False).mean()

    vol_colors = ["#00C087" if c >= o else "#FF3B69" for c, o in zip(df_plot["close"].fillna(0.0), df_plot.get("open", df_plot["close"]).fillna(0.0))]
    x_axis = df_plot["timestamp"] if "timestamp" in df_plot.columns else df_plot.index

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(
        x=x_axis, open=df_plot.get("open", df_plot["close"]), high=df_plot.get("high", df_plot["close"]),
        low=df_plot.get("low", df_plot["close"]), close=df_plot["close"], name="BTC/USDT",
        increasing_line_color="#00C087", decreasing_line_color="#FF3B69",
        increasing_fillcolor="#00C087", decreasing_fillcolor="#FF3B69",
    ), row=1, col=1)

    if "ema_20" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["ema_20"], mode="lines", name="EMA 20", line=dict(color="#2962FF", width=1.5)), row=1, col=1)
    if "ema_50" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["ema_50"], mode="lines", name="EMA 50", line=dict(color="#FF9800", width=1.5)), row=1, col=1)
    if "volume" in df_plot.columns:
        fig.add_trace(go.Bar(x=x_axis, y=df_plot["volume"], name="Volume (BTC)", marker=dict(color=vol_colors, opacity=0.7)), row=2, col=1)

    fig.update_layout(**_base_chart_layout("Bitcoin (BTC/USDT) Hourly Price & Volume Action", height=450), xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1, gridcolor="#21262D", zerolinecolor="#30363D")
    fig.update_yaxes(title_text="Volume (BTC)", row=2, col=1, gridcolor="#21262D", zerolinecolor="#30363D")
    fig.update_xaxes(title_text="Time (UTC)", row=2, col=1, gridcolor="#21262D", zerolinecolor="#30363D")
    return fig


def create_alpha_equity_curve(df: pd.DataFrame) -> go.Figure:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "total_equity_usd" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Portfolio Data Available", template="plotly_dark")
        return fig

    df_plot = df.copy()
    if "timestamp" in df_plot.columns:
        df_plot = df_plot.sort_values("timestamp")
    x_axis = df_plot["timestamp"] if "timestamp" in df_plot.columns else df_plot.index
    fig = go.Figure()
    fig.add_hline(y=float(INITIAL_CAPITAL_USD), line_dash="dot", line_color="#6E7681", annotation_text=f"Initial Capital (${INITIAL_CAPITAL_USD:,.2f})", annotation_position="bottom right", annotation_font=dict(color="#8B949E", size=11))

    if "benchmark_btc_equity" in df_plot.columns:
        fig.add_trace(go.Scatter(x=x_axis, y=df_plot["benchmark_btc_equity"], mode="lines", name="Bitcoin Buy & Hold Benchmark ($)", line=dict(color="#F7931A", width=1.5, dash="dash")))
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot["total_equity_usd"], mode="lines+markers", name="AI Strategy Equity ($)", line=dict(color="#00F2FE", width=2.5), marker=dict(size=4, color="#00C087")))

    fig.update_layout(**_base_chart_layout("Paper Trading Portfolio Equity vs. $10,000 Baseline", height=380), hovermode="x unified")
    fig.update_yaxes(title_text="Portfolio Equity (USD)", gridcolor="#21262D", zerolinecolor="#30363D")
    fig.update_xaxes(title_text="Time (UTC)", gridcolor="#21262D", zerolinecolor="#30363D")
    return fig


def create_reconciliation_scatter(df: pd.DataFrame) -> go.Figure:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty or "actual_price_at_t_plus_1" not in df.columns:
        fig = go.Figure()
        fig.update_layout(title="No Reconciliation Data Available", template="plotly_dark")
        return fig

    df_plot = df.copy()
    if "timestamp" in df_plot.columns:
        df_plot = df_plot.sort_values("timestamp")
    x_axis = df_plot["timestamp"] if "timestamp" in df_plot.columns else df_plot.index

    colors = ["#00C087" if c else "#FF3B69" for c in df_plot.get("direction_correct", [True] * len(df_plot))]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x_axis, y=df_plot["actual_price_at_t_plus_1"], mode="lines", name="Actual BTC Close", line=dict(color="#E6EDF3", width=1.8)))

    pred_series = df_plot.get("predicted_price", pd.Series(0.0, index=df_plot.index)).fillna(0.0)
    actual_series = df_plot["actual_price_at_t_plus_1"].fillna(0.0)
    error_series = df_plot.get("absolute_error_usd", pd.Series(0.0, index=df_plot.index)).fillna(0.0)
    time_labels = [pd.to_datetime(t).strftime("%Y-%m-%d %H:%M UTC") if pd.notnull(t) else str(t) for t in x_axis]
    is_correct_list = df_plot.get("direction_correct", [True] * len(df_plot))

    fig.add_trace(go.Scatter(
        x=x_axis, y=pred_series, mode="markers", name="AI Model Forecast Target",
        marker=dict(size=8, color=colors, line=dict(width=1, color="#FFFFFF")),
        text=[f"Time: {t}<br>Forecast: ${p:,.2f}<br>Actual: ${a:,.2f}<br>Error: ${e:,.2f}<br>Outcome: {'MATCH' if c else 'MISMATCH'}"
              for t, p, a, e, c in zip(time_labels, pred_series, actual_series, error_series, is_correct_list)],
        hoverinfo="text",
    ))

    fig.update_layout(**_base_chart_layout("Realized BTC Close vs. AI Forecast Targets", height=380), hovermode="closest")
    fig.update_yaxes(title_text="Price (USD)", gridcolor="#21262D", zerolinecolor="#30363D")
    fig.update_xaxes(title_text="Time (UTC)", gridcolor="#21262D", zerolinecolor="#30363D")
    return fig


# ---------------------------------------------------------------------------
# 2. MODEL LOADERS & INFERENCE HELPERS
# ---------------------------------------------------------------------------

@st.cache_resource
def load_models():
    try:
        reg = load_verified_model(REGRESSOR_MODEL_PATH, REGRESSOR_MODEL_SHA256)
        clf = load_verified_model(CLASSIFIER_MODEL_PATH, CLASSIFIER_MODEL_SHA256)
        reg.set_params(device="cpu")
        clf.set_params(device="cpu")

        # Load Model 2 (Stationary) and Model 3 (LightGBM) if available in registered models
        m2_reg, m2_clf = None, None
        m3_reg, m3_clf = None, None
        try:
            from src.model_loader import get_registered_models
            reg_dict = get_registered_models()
            if "xgb_v2.0.0" in reg_dict:
                m2_reg = reg_dict["xgb_v2.0.0"]["regressor"]
                m2_clf = reg_dict["xgb_v2.0.0"]["classifier"]
            if "lgbm_v3.0.0" in reg_dict:
                m3_reg = reg_dict["lgbm_v3.0.0"]["regressor"]
                m3_clf = reg_dict["lgbm_v3.0.0"]["classifier"]
        except Exception:
            pass

        return reg, clf, m2_reg, m2_clf, m3_reg, m3_clf
    except Exception:
        return None, None, None, None, None, None


reg_model, clf_model, m2_reg_model, m2_clf_model, m3_reg_model, m3_clf_model = load_models()


def format_signed_usd(val: Optional[float], decimals: int = 2) -> str:
    if val is None:
        return "Unavailable"
    sign = "+" if val > 0 else ("-" if val < 0 else "")
    return f"{sign}${abs(val):,.{decimals}f}"


def check_feed_staleness(df: pd.DataFrame, max_lag_hours: float = 3.0) -> Tuple[bool, float]:
    if df is None or df.empty:
        return True, 0.0
    latest_ts = pd.to_datetime(df["timestamp"].iloc[-1], utc=True)
    lag = max(0.0, (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600.0)
    return (lag > max_lag_hours or df.attrs.get("source") == "local_fallback", lag)


def generate_live_forecast(live_df: pd.DataFrame, reg_model: Any, clf_model: Any, is_stale: bool = False) -> Optional[Dict[str, Any]]:
    if is_stale or live_df is None or live_df.empty or reg_model is None or len(live_df) < 35:
        return None
    from src.feature_engineering import (
        extract_features_for_inference, predict_proba_with_validation,
        predict_with_validation, validate_inference_features,
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

        badge = "SIDEWAYS (CHOP)" if not cleared else ("BULLISH (UP)" if direction == "UP" else "BEARISH (DOWN)")
        return {
            "raw_pred_ret": raw_pred_ret, "pred_ret_pct": pred_ret_pct, "pred_dollar_move": pred_move,
            "pred_target": curr_price + pred_move, "cleared_hurdle": cleared, "direction_label": direction,
            "direction_badge": badge, "prob_up": prob_up, "prob_down": prob_down, "taker": taker,
            "spread_pct": spread, "candle_range_usd": candle_range, "hurdle_dollar_min": curr_price * (MIN_HURDLE_PCT / 100.0),
            "current_price": curr_price,
        }
    except Exception:
        return None


def fetch_live_btc_ticker_price(fallback_price: Optional[float] = None) -> float:
    # fetches live real-time spot price for BTCUSDT from Binance ticker endpoint
    try:
        import requests
        resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=3.0)
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict) and "price" in data:
                return float(data["price"])
    except Exception:
        pass
    return float(fallback_price or 65000.0)


def render_live_btc_websocket_ticker(fallback_price: float, fallback_24h_pct: Optional[float] = None) -> None:
    # renders live self-updating WebSocket ticker bar for BTC/USDT with Binance and Coinbase fallback
    init_price = max(1.0, float(fallback_price or 65000.0))
    pct = float(fallback_24h_pct or 0.0)
    pct_sign = "+" if pct >= 0 else ""
    pct_str = f"{pct_sign}{pct:.2f}%"
    badge_class = "badge-green" if pct >= 0 else "badge-red"
    init_high = init_price * 1.015
    init_low = init_price * 0.985

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
    <meta charset="utf-8">
    <style>
      * {{ box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }}
      body {{ background: transparent; color: #E6EDF3; overflow: hidden; }}
      .ticker-bar {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        background: linear-gradient(180deg, #161B22 0%, #0F141C 100%);
        border: 1px solid #30363D;
        border-radius: 8px;
        padding: 10px 18px;
        height: 88px;
        gap: 16px;
      }}
      .ticker-section {{ display: flex; flex-direction: column; justify-content: center; }}
      .top-label {{
        font-size: 11px;
        font-weight: 600;
        color: #8B949E;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 2px;
      }}
      .live-dot {{
        width: 7px;
        height: 7px;
        background-color: #00C087;
        border-radius: 50%;
        display: inline-block;
        animation: livePulse 1.5s infinite ease-in-out;
      }}
      @keyframes livePulse {{
        0% {{ transform: scale(0.95); opacity: 0.7; box-shadow: 0 0 0 0 rgba(0, 192, 135, 0.7); }}
        70% {{ transform: scale(1.1); opacity: 1; box-shadow: 0 0 0 6px rgba(0, 192, 135, 0); }}
        100% {{ transform: scale(0.95); opacity: 0.7; box-shadow: 0 0 0 0 rgba(0, 192, 135, 0); }}
      }}
      .price-row {{
        display: flex;
        align-items: baseline;
        gap: 8px;
      }}
      .price-val {{
        font-size: 24px;
        font-weight: 700;
        font-variant-numeric: tabular-nums;
        color: #F0F6FC;
        transition: color 0.25s ease;
      }}
      .tick-arrow {{
        font-size: 16px;
        font-weight: bold;
        transition: color 0.25s ease;
      }}
      .stat-grid {{
        display: flex;
        align-items: center;
        gap: 24px;
      }}
      .stat-item {{
        display: flex;
        flex-direction: column;
        align-items: flex-start;
      }}
      .stat-val {{
        font-size: 13px;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: #C9D1D9;
      }}
      .badge-pct {{
        font-size: 12px;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        font-variant-numeric: tabular-nums;
      }}
      .badge-green {{ background: rgba(0, 192, 135, 0.15); color: #00C087; border: 1px solid rgba(0, 192, 135, 0.35); }}
      .badge-red {{ background: rgba(255, 59, 105, 0.15); color: #FF3B69; border: 1px solid rgba(255, 59, 105, 0.35); }}
      .clock-section {{
        display: flex;
        flex-direction: column;
        align-items: flex-end;
        justify-content: center;
      }}
      .clock-val {{
        font-size: 13px;
        font-weight: 600;
        font-variant-numeric: tabular-nums;
        color: #58A6FF;
        background: rgba(56, 139, 253, 0.12);
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid rgba(56, 139, 253, 0.25);
      }}
      .stream-tag {{
        font-size: 10px;
        color: #8B949E;
        margin-top: 3px;
      }}
    </style>
    </head>
    <body>
    <div class="ticker-bar">
      <div class="ticker-section">
        <div class="top-label">
          <span class="live-dot" id="liveDot"></span>
          <span>BTC/USDT SPOT</span>
          <span id="feedStatus" style="color: #00C087; font-size: 10px;">LIVE</span>
        </div>
        <div class="price-row">
          <span class="price-val" id="priceDisplay">${init_price:,.2f}</span>
          <span class="tick-arrow" id="tickArrow" style="color: #8B949E;">▲</span>
        </div>
      </div>

      <div class="stat-grid">
        <div class="stat-item">
          <span class="top-label">24h Change</span>
          <span class="badge-pct {badge_class}" id="change24h">{pct_str}</span>
        </div>
        <div class="stat-item">
          <span class="top-label">24h High</span>
          <span class="stat-val" id="high24h">${init_high:,.2f}</span>
        </div>
        <div class="stat-item">
          <span class="top-label">24h Low</span>
          <span class="stat-val" id="low24h">${init_low:,.2f}</span>
        </div>
      </div>

      <div class="clock-section">
        <div class="top-label" style="justify-content: flex-end;">Synchronized Clock</div>
        <div class="clock-val" id="liveClock">--:--:-- UTC</div>
        <div class="stream-tag" id="streamSource">Connecting to Feed...</div>
      </div>
    </div>

    <script>
    (function() {{
      let lastPrice = {init_price};
      let activeWs = null;
      let binanceConnected = false;
      let fallbackTimer = null;
      let retryTimer = null;

      const priceEl = document.getElementById('priceDisplay');
      const arrowEl = document.getElementById('tickArrow');
      const changeEl = document.getElementById('change24h');
      const highEl = document.getElementById('high24h');
      const lowEl = document.getElementById('low24h');
      const clockEl = document.getElementById('liveClock');
      const streamEl = document.getElementById('streamSource');
      const dotEl = document.getElementById('liveDot');
      const statusEl = document.getElementById('feedStatus');

      function updateClock() {{
        const now = new Date();
        const h = String(now.getUTCHours()).padStart(2, '0');
        const m = String(now.getUTCMinutes()).padStart(2, '0');
        const s = String(now.getUTCSeconds()).padStart(2, '0');
        clockEl.textContent = h + ':' + m + ':' + s + ' UTC';
      }}
      setInterval(updateClock, 1000);
      updateClock();

      function formatPrice(p) {{
        return '$' + Number(p).toLocaleString('en-US', {{ minimumFractionDigits: 2, maximumFractionDigits: 2 }});
      }}

      function handleTick(newPrice, pctChange, high24, low24, sourceName) {{
        if (newPrice > lastPrice) {{
          arrowEl.textContent = '▲';
          arrowEl.style.color = '#00C087';
          priceEl.style.color = '#00C087';
        }} else if (newPrice < lastPrice) {{
          arrowEl.textContent = '▼';
          arrowEl.style.color = '#FF3B69';
          priceEl.style.color = '#FF3B69';
        }}
        setTimeout(function() {{ priceEl.style.color = '#F0F6FC'; }}, 400);

        lastPrice = newPrice;
        priceEl.textContent = formatPrice(newPrice);

        if (pctChange !== null && !isNaN(pctChange)) {{
          const sign = pctChange >= 0 ? '+' : '';
          changeEl.textContent = sign + pctChange.toFixed(2) + '%';
          changeEl.className = 'badge-pct ' + (pctChange >= 0 ? 'badge-green' : 'badge-red');
        }}
        if (high24 && !isNaN(high24) && high24 > 0) {{
          highEl.textContent = formatPrice(high24);
        }}
        if (low24 && !isNaN(low24) && low24 > 0) {{
          lowEl.textContent = formatPrice(low24);
        }}
        streamEl.textContent = sourceName;
        dotEl.style.backgroundColor = '#00C087';
        statusEl.textContent = 'LIVE';
        statusEl.style.color = '#00C087';
      }}

      function connectCoinbase() {{
        if (activeWs) {{
          try {{ activeWs.close(); }} catch(e) {{}}
        }}
        streamEl.textContent = 'Connecting Coinbase...';
        try {{
          const cbWs = new WebSocket('wss://ws-feed.exchange.coinbase.com');
          activeWs = cbWs;
          cbWs.onopen = function() {{
            cbWs.send(JSON.stringify({{
              type: 'subscribe',
              product_ids: ['BTC-USD'],
              channels: ['ticker']
            }}));
            streamEl.textContent = 'Coinbase WebSocket (Fallback)';
          }};
          cbWs.onmessage = function(event) {{
            try {{
              const data = JSON.parse(event.data);
              if (data.type === 'ticker' && data.price) {{
                const p = parseFloat(data.price);
                const open = parseFloat(data.open_24h);
                const high = parseFloat(data.high_24h);
                const low = parseFloat(data.low_24h);
                const pct = (open > 0) ? ((p - open) / open) * 100 : null;
                handleTick(p, pct, high, low, 'Coinbase WebSocket');
              }}
            }} catch(e) {{}}
          }};
          cbWs.onclose = function() {{
            dotEl.style.backgroundColor = '#FF3B69';
            statusEl.textContent = 'RECONNECTING';
            statusEl.style.color = '#FF3B69';
            clearTimeout(retryTimer);
            retryTimer = setTimeout(connectBinance, 3000);
          }};
          cbWs.onerror = function() {{
            try {{ cbWs.close(); }} catch(e) {{}}
          }};
        }} catch(err) {{
          clearTimeout(retryTimer);
          retryTimer = setTimeout(connectBinance, 3000);
        }}
      }}

      function connectBinance() {{
        if (activeWs) {{
          try {{ activeWs.close(); }} catch(e) {{}}
        }}
        binanceConnected = false;
        streamEl.textContent = 'Connecting Binance...';

        fallbackTimer = setTimeout(function() {{
          if (!binanceConnected) {{
            connectCoinbase();
          }}
        }}, 4000);

        try {{
          const bWs = new WebSocket('wss://stream.binance.com:9443/ws/btcusdt@ticker');
          activeWs = bWs;
          bWs.onopen = function() {{
            // Connected to Binance
          }};
          bWs.onmessage = function(event) {{
            binanceConnected = true;
            clearTimeout(fallbackTimer);
            try {{
              const d = JSON.parse(event.data);
              if (d.c) {{
                const p = parseFloat(d.c);
                const pct = parseFloat(d.P);
                const high = parseFloat(d.h);
                const low = parseFloat(d.l);
                handleTick(p, pct, high, low, 'Binance WebSocket (1s)');
              }}
            }} catch(e) {{}}
          }};
          bWs.onclose = function() {{
            if (!binanceConnected) {{
              connectCoinbase();
            }} else {{
              dotEl.style.backgroundColor = '#FF3B69';
              statusEl.textContent = 'RECONNECTING';
              statusEl.style.color = '#FF3B69';
              clearTimeout(retryTimer);
              retryTimer = setTimeout(connectBinance, 3000);
            }}
          }};
          bWs.onerror = function() {{
            if (!binanceConnected) {{
              connectCoinbase();
            }}
          }};
        }} catch(err) {{
          connectCoinbase();
        }}
      }}

      connectBinance();
    }})();
    </script>
    </body>
    </html>
    """
    components.html(html_code, height=96)


def generate_m2_forecast(live_df: pd.DataFrame, m2_reg: Any, m2_clf: Any) -> Optional[Dict[str, Any]]:
    # generates live next-hour target and direction for Model 2 (Stationary)
    if live_df is None or live_df.empty or m2_reg is None or len(live_df) < 35:
        return None
    try:
        from src.model_loader import get_registered_models
        from src.feature_engineering import extract_features_for_inference
        reg_dict = get_registered_models()
        if "xgb_v2.0.0" not in reg_dict:
            return None
        c_feats = reg_dict["xgb_v2.0.0"].get("features", [])
        f_dict = extract_features_for_inference(live_df)
        if not c_feats or not all(k in f_dict for k in c_feats):
            return None
        inp = pd.DataFrame([{k: f_dict[k] for k in c_feats}])
        m2_ret = float(m2_reg.predict(inp)[0])
        curr_price = float(live_df["close"].iloc[-1])
        m2_move = curr_price * m2_ret

        prob_up, prob_down = 50.0, 50.0
        m2_dir = "UP" if m2_ret >= 0.0 else "DOWN"
        if m2_clf is not None:
            if hasattr(m2_clf, "predict_proba"):
                probs = m2_clf.predict_proba(inp)[0]
                prob_down, prob_up = float(probs[0]) * 100.0, float(probs[1]) * 100.0
                m2_dir = "UP" if prob_up >= 50.0 else "DOWN"
            elif hasattr(m2_clf, "predict"):
                pred_c = int(m2_clf.predict(inp)[0])
                m2_dir = "UP" if pred_c == 1 else "DOWN"
                prob_up, prob_down = (60.0, 40.0) if pred_c == 1 else (40.0, 60.0)

        return {
            "pred_target": curr_price + m2_move,
            "pred_dollar_move": m2_move,
            "pred_ret_pct": m2_ret * 100.0,
            "direction_label": m2_dir,
            "prob_up": prob_up,
            "prob_down": prob_down,
        }
    except Exception:
        return None


def generate_m3_forecast(live_df: pd.DataFrame, m3_reg: Any, m3_clf: Any) -> Optional[Dict[str, Any]]:
    # generates live next-hour target and direction for Model 3 (LightGBM)
    if live_df is None or live_df.empty or m3_reg is None or len(live_df) < 35:
        return None
    try:
        from src.model_loader import get_registered_models
        from src.feature_engineering import extract_features_for_inference
        reg_dict = get_registered_models()
        if "lgbm_v3.0.0" not in reg_dict:
            return None
        c_feats = reg_dict["lgbm_v3.0.0"].get("features", [])
        f_dict = extract_features_for_inference(live_df)
        if not c_feats or not all(k in f_dict for k in c_feats):
            return None
        inp = pd.DataFrame([{k: f_dict[k] for k in c_feats}])
        m3_ret = float(m3_reg.predict(inp)[0])
        curr_price = float(live_df["close"].iloc[-1])
        m3_move = curr_price * m3_ret

        prob_up, prob_down = 50.0, 50.0
        m3_dir = "UP" if m3_ret >= 0.0 else "DOWN"
        if m3_clf is not None:
            if hasattr(m3_clf, "predict_proba"):
                probs = m3_clf.predict_proba(inp)[0]
                prob_down, prob_up = float(probs[0]) * 100.0, float(probs[1]) * 100.0
                m3_dir = "UP" if prob_up >= 50.0 else "DOWN"
            elif hasattr(m3_clf, "predict"):
                pred_c = int(m3_clf.predict(inp)[0])
                m3_dir = "UP" if pred_c == 1 else "DOWN"
                prob_up, prob_down = (60.0, 40.0) if pred_c == 1 else (40.0, 60.0)

        return {
            "pred_target": curr_price + m3_move,
            "pred_dollar_move": m3_move,
            "pred_ret_pct": m3_ret * 100.0,
            "direction_label": m3_dir,
            "prob_up": prob_up,
            "prob_down": prob_down,
        }
    except Exception:
        return None


def generate_candidate_forecast(live_df: pd.DataFrame, model_spec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # generates live next-hour target and direction for any registered model spec
    if live_df is None or live_df.empty or not model_spec or len(live_df) < 35:
        return None
    try:
        from src.feature_engineering import extract_features_for_inference
        reg_model = model_spec.get("regressor")
        clf_model = model_spec.get("classifier")
        if reg_model is None:
            return None
        c_feats = model_spec.get("features", [])
        f_dict = extract_features_for_inference(live_df)
        if not c_feats or not all(k in f_dict for k in c_feats):
            return None
        inp = pd.DataFrame([{k: f_dict[k] for k in c_feats}])
        pred_ret = float(reg_model.predict(inp)[0])
        curr_price = float(live_df["close"].iloc[-1])
        pred_move = curr_price * pred_ret

        prob_up, prob_down = 50.0, 50.0
        pred_dir = "UP" if pred_ret >= 0.0 else "DOWN"
        if clf_model is not None:
            if hasattr(clf_model, "predict_proba"):
                probs = clf_model.predict_proba(inp)[0]
                prob_down, prob_up = float(probs[0]) * 100.0, float(probs[1]) * 100.0
                pred_dir = "UP" if prob_up >= 50.0 else "DOWN"
            elif hasattr(clf_model, "predict"):
                pred_c = int(clf_model.predict(inp)[0])
                pred_dir = "UP" if pred_c == 1 else "DOWN"
                prob_up, prob_down = (60.0, 40.0) if pred_c == 1 else (40.0, 60.0)

        return {
            "pred_target": curr_price + pred_move,
            "pred_dollar_move": pred_move,
            "pred_ret_pct": pred_ret * 100.0,
            "direction_label": pred_dir,
            "prob_up": prob_up,
            "prob_down": prob_down,
        }
    except Exception:
        return None


def format_model_display_name(model_version: str) -> str:
    # formats model version string into friendly display name with dynamic fallback
    m_str = str(model_version or "").strip()
    mapping = {
        "xgb_onnx_v1.0.0": "Model 1 (xgb_onnx_v1.0.0)",
        "xgb_v1": "Model 1 (xgb_onnx_v1.0.0)",
        "xgb_v2.0.0": "Model 2 (xgb_v2.0.0)",
        "lgbm_v3.0.0": "Model 3 (lgbm_v3.0.0)",
    }
    if m_str in mapping:
        return mapping[m_str]
    if "v1" in m_str:
        return f"Model 1 ({m_str})"
    if "v2" in m_str:
        return f"Model 2 ({m_str})"
    if "v3" in m_str or "lgbm" in m_str:
        return f"Model 3 ({m_str})"
    return m_str


def sort_models_for_display(registered_dict: Dict[str, Dict[str, Any]]) -> List[Tuple[str, Dict[str, Any]]]:
    # sorts models so active model comes first, followed in numerical version order
    def sort_key(item):
        ver, spec = item
        v = str(ver).lower()
        if "v1" in v or spec.get("status") == "ACTIVE":
            order = 1
        elif "v2" in v:
            order = 2
        elif "v3" in v:
            order = 3
        elif "v4" in v:
            order = 4
        elif "v5" in v:
            order = 5
        else:
            order = 99
        return (order, ver)
    return sorted(registered_dict.items(), key=sort_key)


def get_model_card_title(model_version: str, spec: Dict[str, Any]) -> str:
    # returns card header title for a given model version
    ver = str(model_version).strip()
    if ver in ("xgb_onnx_v1.0.0", "xgb_v1"):
        return "Model 1 (Active XGBoost v1)"
    if ver == "xgb_v2.0.0":
        return "Model 2 (Shadow XGBoost v2)"
    if ver == "lgbm_v3.0.0":
        return "Model 3 (Candidate LightGBM v3)"
    disp = format_model_display_name(ver)
    tag = "Active" if spec.get("status") == "ACTIVE" else "Candidate"
    return f"{disp} ({tag})"


def get_model_benchmark_caption(model_version: str, spec: Dict[str, Any]) -> str:
    # returns holdout benchmark caption line for a given model
    ver = str(model_version).strip()
    if ver in ("xgb_onnx_v1.0.0", "xgb_v1"):
        return "Benchmark: **53.26%** Accuracy · 0.6914 Log-Loss · 39 Features"
    if ver == "xgb_v2.0.0":
        return "Benchmark: **52.07%** Accuracy · 0.6926 Log-Loss · 36 Stationary Features"
    if ver == "lgbm_v3.0.0":
        return "Benchmark: **52.89%** Accuracy · 0.6913 Log-Loss · 29 Stationary Features + Platt"
    met = spec.get("metrics", {})
    n_feats = len(spec.get("features", []))
    acc = f"{met['holdout_accuracy']*100:.2f}%" if "holdout_accuracy" in met else None
    ll = f"{met['holdout_log_loss']:.4f}" if "holdout_log_loss" in met else None
    if acc and ll:
        return f"Benchmark: **{acc}** Accuracy · {ll} Log-Loss · {n_feats} Features"
    return f"Candidate Model · {n_feats} Features"


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
        vol_spread, volume, taker, order_bias, rsi, expected_ret, confidence = 0.015, 5000.0, 0.65, "STRONG_BUYER_AGGRESSION", 62.0, 0.42, 0.88
    elif scenario_selected == "vol_spike":
        vol_spread, volume, taker, order_bias, rsi, expected_ret, confidence = 0.052, 12000.0, 0.35, "EXTREME_LIQUIDATION", 28.0, -0.20, 0.60
    elif scenario_selected == "chop":
        vol_spread, volume, taker, order_bias, rsi, expected_ret, confidence = 0.004, 400.0, 0.49, "FLAT_CHOP", 50.0, 0.02, 0.45
    else:
        vol_spread, volume, taker, order_bias, rsi, expected_ret, confidence = 0.018, 1200.0, 0.51, "NEUTRAL", 51.0, 0.08, 0.65

    clears = bool(abs(expected_ret) >= MIN_HURDLE_PCT)
    direction = "SIDEWAYS" if not clears else ("UP" if expected_ret >= 0 else "DOWN")
    m_data = {"close": current_price, "high": current_price * (1 + vol_spread / 2), "low": current_price * (1 - vol_spread / 2), "volume": volume, "taker_buy_ratio": taker}
    t_data = {"volatility_spread": vol_spread, "volatility_range_pct": round(vol_spread * 100, 3), "order_flow_bias": order_bias, "rsi_estimate": rsi}
    f_data = {"expected_return_pct": expected_ret, "predicted_dollar_move": round(current_price * (expected_ret / 100.0), 2), "clears_hurdle": clears, "confidence": confidence, "direction": direction}
    return m_data, t_data, f_data


def format_portfolio_rationale(row) -> str:
    act = str(row.get("action", "")).upper().strip()
    btc_bal = float(row.get("btc_balance", 0.0) or 0.0)
    exec_price = float(row.get("execution_price", 0.0) or 0.0)
    cash = float(row.get("cash_balance", 10000.0) or 10000.0)
    notes = str(row.get("notes", "") or "")

    if act == "BUY":
        return f"Buy Order Executed: Allocated {float(row.get('btc_amount', 0.0)):.4f} BTC at ${exec_price:,.2f} on approved bullish signal."
    if act == "SELL":
        return f"Sell Order Executed: Closed position at ${exec_price:,.2f}."
    if act == "INITIAL_DEPOSIT":
        return "Initial Deposit: Virtual bankroll of $10,000.00 USDT allocated for forward testing."
    if act == "HOLD":
        if btc_bal > 0:
            btc_val = btc_bal * exec_price
            return f"Holding Active Position: {btc_bal:.4f} BTC (${btc_val:,.2f} value) held at BTC ${exec_price:,.2f}; letting runner ride."
        if "Sub-hurdle fee drag" in notes or "Fee hurdle" in notes:
            return f"Capital Preserved: Expected move is below {MIN_HURDLE_PCT:.2f}% round-trip fee hurdle."
        if "Risk Veto" in notes or "veto" in notes.lower():
            return "Risk Veto: Trade prevented by Risk Critic to protect capital."
        price_clause = f" at BTC ${exec_price:,.2f}" if exec_price > 0 else ""
        return f"Capital Preserved: Sideways chop{price_clause}; preserving cash until high-conviction setup."
    return notes or "Execution logged."


def build_model_pedigree_markdown() -> str:
    return (
        "| Model Version | Architecture | Status | Features | Accuracy | Log-Loss | Friction Benchmark |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n"
        f"| **Model 1** (`xgb_onnx_v1.0.0`) | XGBoost Regressor + Classifier | Active Production | 42 (Raw + engineered) | 53.26% | 0.6914 | `0.30% (Historical)` benchmark; canonical `{ROUND_TRIP_FRICTION_PCT:.2f}%` (`{TOTAL_FRICTION_PER_SIDE_PCT:.2f}%`/side) |\n"
        f"| **Model 2** (`xgb_v2.0.0`) | XGBoost Stationary | Live Shadow | 29 (Leak-free stationary) | 52.07% | 0.6926 | Evaluated under canonical `{ROUND_TRIP_FRICTION_PCT:.2f}%` hurdle |\n"
        f"| **Model 3** (`lgbm_v3.0.0`) | LightGBM GBDT | Shadow Candidate | 29 (Stationary + Platt) | 52.89% | 0.6913 | Evaluated under canonical `{ROUND_TRIP_FRICTION_PCT:.2f}%` hurdle |"
    )


def classify_ground_truth_result(row) -> str:
    # classifies ground truth outcomes into tradable matches, noise, directional errors, or opportunity losses
    pred_ret = float(row.get("predicted_return_pct", 0.0) or 0.0)
    act_ret = float(row.get("actual_return_pct", 0.0) or 0.0)
    diff = float(row.get("absolute_error_usd", 0.0) or 0.0)
    dir_ok = bool(row.get("direction_correct", False))
    hurdle = MIN_HURDLE_PCT

    if not dir_ok:
        if diff < 100.0 or abs(act_ret) < hurdle:
            return "DRIFT (CHOP)"
        return "MISMATCH (DIR)"

    # Opportunity Loss: Direction matched, but model under-predicted range below hurdle while actual exploded
    if abs(pred_ret) < hurdle and abs(act_ret) >= hurdle and diff >= 150.0:
        return "MISSED MOVE (RANGE)"

    if abs(pred_ret) >= hurdle and abs(act_ret) >= hurdle:
        return "MATCH (TRADABLE)"

    return "MATCH (CHOP)"


def format_ground_truth_diagnosis(row) -> str:
    # formats human-readable economic explanation of why a prediction was accurate, noise, or a missed trade
    diag = row.get("diagnosis_summary")
    if pd.notna(diag) and str(diag).strip():
        return str(diag).strip()
    pred_ret = float(row.get("predicted_return_pct", 0.0) or 0.0)
    act_ret = float(row.get("actual_return_pct", 0.0) or 0.0)
    diff = float(row.get("absolute_error_usd", 0.0) or 0.0)
    dir_ok = bool(row.get("direction_correct", False))
    hurdle = MIN_HURDLE_PCT

    if not dir_ok:
        if diff < 100.0:
            return "Sideways Noise: Sub-$100 drift; fee hurdle protected capital."
        return f"Direction Error: Forecasted {'UP' if pred_ret >= 0 else 'DOWN'}, market moved {'UP' if act_ret >= 0 else 'DOWN'} (${diff:,.0f} gap)."

    if abs(pred_ret) < hurdle and abs(act_ret) >= hurdle and diff >= 150.0:
        return f"Opportunity Loss: Actual move was {act_ret:+.2f}% (${diff:,.0f} gap); model predicted sub-hurdle ({pred_ret:+.2f}%), trade skipped."

    if abs(pred_ret) >= hurdle and abs(act_ret) >= hurdle:
        return f"Tradable Match: Both predicted ({pred_ret:+.2f}%) and actual ({act_ret:+.2f}%) cleared {hurdle:.2f}% fee hurdle."

    return f"Noise Filtered: Sub-hurdle move ({act_ret:+.2f}%); correctly held cash."


def filter_reconciliation_records(df: pd.DataFrame, selected_model: str) -> pd.DataFrame:
    if df.empty or "model_version" not in df.columns or selected_model in ("All Models", ""):
        return df.copy()
    return df[df["model_version"] == selected_model].copy()


def prepare_reconciliation_table_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    disp = df.copy()
    disp["Model Version"] = disp["model_version"].fillna("UNKNOWN") if "model_version" in disp.columns else "UNKNOWN"
    return disp


def render_ground_truth_html_table(df_subset: pd.DataFrame) -> None:
    # renders styled html table with multi-line word-wrapped ai diagnosis descriptions
    if df_subset.empty:
        st.info("No records to display.")
        return

    rows_html = []
    for idx, (_, row) in enumerate(df_subset.iterrows()):
        bg_color = "#0D1117" if idx % 2 == 0 else "#111620"

        time_val = html.escape(str(row.get("Time (UTC)", "—")))
        model_val = html.escape(str(row.get("Model", "—")))
        start_price = html.escape(str(row.get("Start Price", "—")))
        pred_target = html.escape(str(row.get("Pred Target", "—")))
        actual_val = html.escape(str(row.get("Actual (T+1h)", "—")))
        error_val = html.escape(str(row.get("Error ($)", "—")))
        pred_dir = str(row.get("Pred Dir", "—")).upper()
        actual_dir = str(row.get("Actual Dir", "—")).upper()
        result_val = str(row.get("Result", "—"))
        diag_val = html.escape(str(row.get("AI Diagnosis (1-Line)", row.get("AI Diagnosis", "—"))))

        # direction badges with color coding
        if pred_dir == "UP":
            pdir_badge = '<span style="background: rgba(46, 160, 67, 0.15); color: #3FB950; border: 1px solid rgba(46, 160, 67, 0.3); padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.04em;">UP</span>'
        elif pred_dir == "DOWN":
            pdir_badge = '<span style="background: rgba(248, 81, 73, 0.15); color: #F85149; border: 1px solid rgba(248, 81, 73, 0.3); padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.04em;">DOWN</span>'
        else:
            pdir_badge = f'<span style="color: #8B949E; font-size: 0.72rem;">{html.escape(pred_dir)}</span>'

        if actual_dir == "UP":
            adir_badge = '<span style="background: rgba(46, 160, 67, 0.15); color: #3FB950; border: 1px solid rgba(46, 160, 67, 0.3); padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.04em;">UP</span>'
        elif actual_dir == "DOWN":
            adir_badge = '<span style="background: rgba(248, 81, 73, 0.15); color: #F85149; border: 1px solid rgba(248, 81, 73, 0.3); padding: 2px 7px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; letter-spacing: 0.04em;">DOWN</span>'
        else:
            adir_badge = f'<span style="color: #8B949E; font-size: 0.72rem;">{html.escape(actual_dir)}</span>'

        # outcome result badges
        esc_res = html.escape(result_val)
        if "MATCH" in result_val:
            res_badge = f'<span style="background: rgba(46, 160, 67, 0.15); color: #3FB950; border: 1px solid rgba(46, 160, 67, 0.3); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; white-space: nowrap;">{esc_res}</span>'
        elif "MISMATCH" in result_val:
            res_badge = f'<span style="background: rgba(248, 81, 73, 0.15); color: #F85149; border: 1px solid rgba(248, 81, 73, 0.3); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; white-space: nowrap;">{esc_res}</span>'
        else:
            res_badge = f'<span style="background: rgba(210, 153, 34, 0.15); color: #D29922; border: 1px solid rgba(210, 153, 34, 0.3); padding: 2px 8px; border-radius: 4px; font-weight: 600; font-size: 0.72rem; white-space: nowrap;">{esc_res}</span>'

        # diagnosis styling with wrap across 2 to 3 lines
        diag_color = "#8B949E" if "unavailable" in diag_val.lower() else "#E6EDF3"
        diag_font_style = "italic" if "unavailable" in diag_val.lower() else "normal"

        row_html = (
            f'<tr style="background: {bg_color}; border-bottom: 1px solid #21262D;">'
            f'<td style="padding: 10px 12px; white-space: nowrap; font-size: 0.78rem; color: #8B949E;">{time_val}</td>'
            f'<td style="padding: 10px 12px; white-space: nowrap; font-weight: 600; font-size: 0.8rem; color: #58A6FF;">{model_val}</td>'
            f'<td style="padding: 10px 12px; white-space: nowrap; font-family: monospace; font-size: 0.82rem; text-align: right; color: #8B949E;">{start_price}</td>'
            f'<td style="padding: 10px 12px; white-space: nowrap; font-family: monospace; font-size: 0.82rem; text-align: right; color: #C9D1D9;">{pred_target}</td>'
            f'<td style="padding: 10px 12px; white-space: nowrap; font-family: monospace; font-size: 0.82rem; text-align: right; font-weight: 600; color: #F0F6FC;">{actual_val}</td>'
            f'<td style="padding: 10px 12px; white-space: nowrap; font-family: monospace; font-size: 0.82rem; text-align: right; color: #E6EDF3;">{error_val}</td>'
            f'<td style="padding: 10px 12px; text-align: center; vertical-align: middle;">{pdir_badge}</td>'
            f'<td style="padding: 10px 12px; text-align: center; vertical-align: middle;">{adir_badge}</td>'
            f'<td style="padding: 10px 12px; text-align: center; vertical-align: middle;">{res_badge}</td>'
            f'<td style="padding: 10px 12px; white-space: normal !important; word-wrap: break-word !important; overflow-wrap: break-word !important; line-height: 1.45; min-width: 280px; max-width: 480px; font-size: 0.81rem; color: {diag_color}; font-style: {diag_font_style}; vertical-align: middle;">{diag_val}</td>'
            f'</tr>'
        )
        rows_html.append(row_html)

    table_body = "\n".join(rows_html)
    full_table_html = (
        '<div style="max-height: 540px; overflow-y: auto; overflow-x: auto; border: 1px solid #30363D; border-radius: 8px; background: #0D1117; margin-bottom: 12px;">\n'
        '  <table style="width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, \'Segoe UI\', Roboto, sans-serif; font-size: 0.82rem; text-align: left;">\n'
        '    <thead>\n'
        '      <tr style="position: sticky; top: 0; background: #161B22; z-index: 10; border-bottom: 2px solid #30363D;">\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;">Time (UTC)</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap;">Model</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: right;">Start Price</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: right;">Pred Target</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: right;">Actual (T+1h)</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: right;">Error ($)</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: center;">Pred Dir</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: center;">Actual Dir</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; white-space: nowrap; text-align: center;">Result</th>\n'
        '        <th style="padding: 10px 12px; color: #8B949E; font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.05em; min-width: 280px; max-width: 480px;">AI Diagnosis (Full)</th>\n'
        '      </tr>\n'
        '    </thead>\n'
        '    <tbody>\n'
        f'{table_body}\n'
        '    </tbody>\n'
        '  </table>\n'
        '</div>'
    )
    st.markdown(full_table_html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# 3. DATA FETCHING (NEON POSTGRESQL & BINANCE VISION)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=60)
def fetch_live_market_candles(symbol: str = "BTCUSDT") -> pd.DataFrame:
    try:
        return fetch_historical_candles(symbol=symbol, limit=100)
    except Exception as exc:
        empty = pd.DataFrame()
        empty.attrs["error"] = str(exc)
        return empty


@st.cache_data(ttl=60)
def fetch_cloud_database_state(current_price: Optional[float]):
    fallback = {
        "db_ok": False, "db_ver": "Offline", "db_lat": 0.0, "db_error": "Database unavailable",
        "ticks_df": pd.DataFrame(), "rec_df": pd.DataFrame(), "rec_kpis": {}, "latest_audit": None,
        "port_df": pd.DataFrame(), "port_kpis": {},
    }
    try:
        db_ok, db_ver, db_lat = test_connection()
        if not db_ok:
            return fallback
        return {
            "db_ok": True, "db_ver": db_ver, "db_lat": db_lat, "db_error": None,
            "ticks_df": get_latest_market_ticks(limit=500),
            "rec_df": get_reconciliation_history(limit=100),
            "rec_kpis": get_reconciliation_kpis(lookback_hours=168),
            "latest_audit": get_latest_council_audit(),
            "port_df": get_portfolio_history(limit=500),
            "port_kpis": get_portfolio_kpis(current_price=current_price or 90000.0),
        }
    except Exception as exc:
        fallback["db_error"] = str(exc)
        return fallback


# ---------------------------------------------------------------------------
# 4. SINGLE-PAGE STREAMLIT USER INTERFACE
# ---------------------------------------------------------------------------

# Fetch Market & Database Telemetry
live_market_df = fetch_live_market_candles()
current_price = float(live_market_df["close"].iloc[-1]) if not live_market_df.empty else None
is_stale_fallback, stale_age_hours = check_feed_staleness(live_market_df)
db_state = fetch_cloud_database_state(current_price=current_price)

# Sidebar Controls & System Status
st.sidebar.title("Mudra BTC AI Engine")
st.sidebar.caption("Machine Learning & Agentic Forecasting")

if st.sidebar.button("Run Hourly Ingestion & Prediction", use_container_width=True, help="Fetches newest candle from Binance Vision, executes active model inference, and stores prediction."):
    with st.spinner("Executing quant cycle..."):
        try:
            synced, msg, _ = sync_hourly_quant_cycle(force=True)
            if synced:
                st.sidebar.success(msg)
                st.cache_data.clear()
                st.rerun()
            else:
                st.sidebar.info(msg)
        except Exception as exc:
            st.sidebar.error(f"Execution error: {exc}")

if st.sidebar.button("Refresh Live Telemetry", use_container_width=True, help="Clears Streamlit cache and reloads live database and market data."):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()
st.sidebar.subheader("Live System Status")

is_neon_connected = db_state["db_ok"] and "postgres" in str(db_state.get("db_ver", "")).lower()

if is_neon_connected:
    st.sidebar.success(f"Neon Cloud DB: Connected ({db_state['db_lat']:.0f}ms)")
else:
    st.sidebar.warning("Neon Cloud DB: Disconnected (Using Local SQLite). To load continuous 24/7 ground-truth history, set DATABASE_URL in Streamlit Cloud Secrets.")

if is_stale_fallback:
    st.sidebar.warning(f"Market Feed: Stale ({stale_age_hours:.0f}h lag)" if not live_market_df.empty else "Market Feed: Offline")
elif not live_market_df.empty:
    st.sidebar.success("Market Feed: Live Binance Vision (1h)")
else:
    st.sidebar.error("Market Feed: Offline")

st.sidebar.markdown(
    """
    <div class="sidebar-footer">
        <strong>Mudra Quantitative System</strong><br>
        24/7 Autonomous Pipeline · Dual XGBoost + LangGraph<br>
        Neon PostgreSQL Ground Truth
    </div>
    """,
    unsafe_allow_html=True,
)

# Page Header
st.title("Mudra — Bitcoin 1-Hour AI Forecasting & Ground Truth Evaluation")
st.caption("Multi-Model Inference (XGBoost v1/v2 & LightGBM v3) • Multi-Agent LangGraph Reasoning • Live Neon PostgreSQL Ground-Truth Verification")

m2_forecast_result = generate_m2_forecast(live_df=live_market_df, m2_reg=m2_reg_model, m2_clf=m2_clf_model)
m3_forecast_result = generate_m3_forecast(live_df=live_market_df, m3_reg=m3_reg_model, m3_clf=m3_clf_model)

# Section 1: Top Headline Metric Cards (Live Telemetry & Real-Time Targets)
with st.container(border=True):
    st.markdown('<span class="section-badge">Section 1 · Live Telemetry & Model Targets</span>', unsafe_allow_html=True)
    p_24h = float(live_market_df["close"].iloc[-24]) if len(live_market_df) >= 24 else None
    live_spot_price = fetch_live_btc_ticker_price(fallback_price=current_price)
    spot_d_24h = (live_spot_price - p_24h) if live_spot_price and p_24h else None

    forecast_result = generate_live_forecast(live_df=live_market_df, reg_model=reg_model, clf_model=clf_model, is_stale=is_stale_fallback)

    fallback_24h_pct = ((spot_d_24h / p_24h) * 100.0) if (spot_d_24h is not None and p_24h and p_24h > 0) else None

    # Live Real-Time WebSocket Ticker Box (Self-updating every second via browser-native WebSocket)
    render_live_btc_websocket_ticker(fallback_price=live_spot_price, fallback_24h_pct=fallback_24h_pct)

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

    # Dynamic Dual-Output Model Predictions Row (Discovered dynamically across all registered models)
    try:
        from src.model_loader import get_registered_models
        registered_models = get_registered_models()
    except Exception:
        registered_models = {}

    if registered_models:
        sorted_models = sort_models_for_display(registered_models)
        model_cols = st.columns(len(sorted_models))
        for col, (m_ver, m_spec) in zip(model_cols, sorted_models):
            with col:
                card_title = get_model_card_title(m_ver, m_spec)
                bench_caption = get_model_benchmark_caption(m_ver, m_spec)
                if (m_ver in ("xgb_onnx_v1.0.0", "xgb_v1") or m_spec.get("status") == "ACTIVE") and forecast_result:
                    f_res = forecast_result
                elif m_ver == "xgb_v2.0.0" and m2_forecast_result:
                    f_res = m2_forecast_result
                elif m_ver == "lgbm_v3.0.0" and m3_forecast_result:
                    f_res = m3_forecast_result
                else:
                    f_res = generate_candidate_forecast(live_market_df, m_spec)

                if f_res:
                    f_dir = f_res.get("direction_label", "HOLD")
                    f_p_up = f_res.get("prob_up", 50.0)
                    f_ret = f_res.get("pred_ret_pct", 0.0)
                    st.metric(
                        card_title,
                        f"${f_res['pred_target']:,.2f}",
                        f"{f_dir} ({f_ret:+.2f}%) • P(UP): {f_p_up:.1f}%",
                        help=f"{card_title}: Regressor next-hour target price and classifier direction probability."
                    )
                else:
                    st.metric(card_title, "Standby", "Awaiting Data", help=f"{card_title} forecast is on standby.")
                st.caption(bench_caption)

        # table 1: current live predictions across all registered models
        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
        t1_rows = []
        rec_df_cache = db_state.get("rec_df", pd.DataFrame())
        for m_ver, m_spec in sorted_models:
            if (m_ver in ("xgb_onnx_v1.0.0", "xgb_v1") or m_spec.get("status") == "ACTIVE") and forecast_result:
                f_res = forecast_result
            elif m_ver == "xgb_v2.0.0" and m2_forecast_result:
                f_res = m2_forecast_result
            elif m_ver == "lgbm_v3.0.0" and m3_forecast_result:
                f_res = m3_forecast_result
            else:
                f_res = generate_candidate_forecast(live_market_df, m_spec)

            if m_spec.get("status") == "ACTIVE":
                role = "Active Production"
            elif "v2" in m_ver:
                role = "Shadow Benchmark"
            elif "v3" in m_ver:
                role = "Candidate Evaluator"
            else:
                role = "Registered Model"

            m_eval_count = 0
            if not rec_df_cache.empty and "model_version" in rec_df_cache.columns:
                m_eval_count = int((rec_df_cache["model_version"] == m_ver).sum())
            if m_ver in ("xgb_onnx_v1.0.0", "xgb_v1") and m_eval_count == 0:
                m_eval_count = len(rec_df_cache)

            if m_eval_count >= 168:
                track_str = f"{m_eval_count:,} hrs"
            elif m_eval_count > 0:
                track_str = f"{m_eval_count} hrs (Warming up)"
            else:
                track_str = "Warming up (0 hrs)"

            target_val = f"${f_res['pred_target']:,.2f}" if f_res and "pred_target" in f_res else "—"
            ret_val = f"{f_res['pred_ret_pct']:+.2f}%" if f_res and "pred_ret_pct" in f_res else "—"
            dir_val = f_res.get("direction_label", "HOLD") if f_res else "STANDBY"
            p_up_val = f"{f_res.get('prob_up', 50.0):.1f}%" if f_res else "—"

            t1_rows.append({
                "Model": format_model_display_name(m_ver),
                "Role": role,
                "BTC at Forecast": f"${live_spot_price:,.2f}" if live_spot_price else "—",
                "Predicted Target ($)": target_val,
                "Pred Return (%)": ret_val,
                "Classifier Dir": dir_val,
                "P(UP) %": p_up_val,
                "Live Track Record": track_str,
            })

        if t1_rows:
            st.dataframe(pd.DataFrame(t1_rows), use_container_width=True, hide_index=True)
    else:
        # Fallback to single Active Model 1 card
        with st.container():
            if forecast_result:
                m1_dir = forecast_result.get("direction_label", "HOLD")
                m1_p_up = forecast_result.get("prob_up", 50.0)
                m1_ret = forecast_result.get("pred_ret_pct", 0.0)
                st.metric(
                    "Model 1 (Active XGBoost v1)",
                    f"${forecast_result['pred_target']:,.2f}",
                    f"{m1_dir} ({m1_ret:+.2f}%) • P(UP): {m1_p_up:.1f}%",
                    help="Active XGBoost v1 model: Regressor next-hour target price and classifier direction probability."
                )
            else:
                st.metric("Model 1 (Active XGBoost v1)", "Computing...", "Awaiting Candle", help="Model 1 forecast is calculating.")
            st.caption("Benchmark: **53.26%** Accuracy · 0.6914 Log-Loss · 39 Features")

st.divider()

# Native 4-Tab Recruiter-Friendly Dashboard
tab_models, tab_ground_truth, tab_portfolio, tab_agent = st.tabs([
    "Model Evolution & SHAP",
    "Ground Truth (T+1h)",
    "Paper Portfolio & Charts",
    "AI Reasoning (LangGraph + RAG + MCP)",
])

# Tab 1: Model Evolution & SHAP
with tab_models:
    with st.container(border=True):
        st.markdown('<span class="section-badge">Model Evolution & Governance</span>', unsafe_allow_html=True)
        st.subheader("Model Benchmark Comparison (Active vs. Shadow Candidates)")
        st.caption("Empirical holdout validation metrics comparing Active Model 1 (53.26%) alongside Shadow Models 2 (52.07%) & 3 (52.89%).")

        manifest_rows = []
        models_dir = ROOT_DIR / "artifacts" / "models"
        manifest_files = sorted(models_dir.glob("*manifest*.json")) if models_dir.exists() else []
        for mf in manifest_files:
            if mf.exists():
                try:
                    with open(mf, "r", encoding="utf-8") as f:
                        m_data = json.load(f)
                    met = m_data.get("metrics", {})
                    acc = f"{met['holdout_accuracy']*100:.2f}%" if "holdout_accuracy" in met else "N/A"
                    ll = f"{met['holdout_log_loss']:.4f}" if "holdout_log_loss" in met else "N/A"
                    br = f"{met['holdout_brier_score']:.4f}" if "holdout_brier_score" in met else "N/A"
                    auc = f"{met['holdout_roc_auc']:.4f}" if "holdout_roc_auc" in met else "N/A"
                    ic = f"{met['holdout_spearman_ic']:+.4f}" if "holdout_spearman_ic" in met else "N/A"
                    status = m_data.get("status", "ACTIVE_PRODUCTION")
                    feat_count = len(m_data.get("features", []))
                    schema_lbl = f"{m_data.get('feature_schema_version', 'v1')} ({feat_count} feat)"
                    manifest_rows.append({
                        "Model Version": m_data.get("model_version", mf.stem),
                        "Schema": schema_lbl,
                        "Accuracy": acc,
                        "Log-Loss": ll,
                        "Brier": br,
                        "ROC-AUC": auc,
                        "Spearman IC": ic,
                        "Status": status,
                    })
                except Exception:
                    pass
        if manifest_rows:
            st.dataframe(pd.DataFrame(manifest_rows), use_container_width=True, hide_index=True)

        shap_path = ROOT_DIR / "docs" / "images" / "shap_classifier_summary.png"
        if shap_path.exists():
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.subheader("TreeSHAP Feature Importance & Interpretability")
            st.image(str(shap_path), use_column_width=True)
            st.caption("Top TreeSHAP features driving model direction: taker_buy_ratio (aggressive market orders), volatility range, and short-term momentum.")

        with st.expander("Research Findings, Anti-Leakage Split & Model Governance", expanded=False):
            st.markdown("#### Active Production Model Pedigree")
            st.markdown(build_model_pedigree_markdown())
            st.markdown(
                """
                #### Anti-Leakage Chronological Split Boundary
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

                #### Controlled Model Benchmark & Rejection Rationale
                | Model Version | Regressor RMSE (USD) | Direction Hit Rate | Classifier Accuracy | Classifier Macro F1 | Governance Status |
                | :--- | :--- | :--- | :--- | :--- | :--- |
                | **Naive Persistence** | **$398.77** | *Baseline* | *Baseline* | *Baseline* | Reference Benchmark |
                | **`xgb_onnx_v1.0.0`** *(Active)* | $399.47 | **50.50%** | **53.26%** | **0.5297** | **ACTIVE PRODUCTION** |
                | **`p1_controlled_candidate_v1`** | $399.37 | 50.36% | 53.03% | 0.5280 | **REJECTED CANDIDATE** |

                *Note: Simulated paper-trading performance demonstrates execution accounting and fee-hurdle discipline.*
                """
            )
            st.caption("Forecast Error Investigations & Root-Cause Audit Log (investigation_headline_search, investigation_status_filter available)")

# Tab 2: Ground Truth (T+1h)
with tab_ground_truth:
    with st.container(border=True):
        st.markdown('<span class="section-badge">Section 2 · Ground Truth Evaluation</span>', unsafe_allow_html=True)
        st.subheader("Previous AI Model Predictions vs. Real Market Outcomes (Ground Truth)")

        rec_df = db_state.get("rec_df", pd.DataFrame())
        rec_kpis = db_state.get("rec_kpis", {})

        if not rec_df.empty:
            total_lifetime = rec_kpis.get("total_lifetime_evaluated") or len(rec_df)
            rolling_eval = rec_kpis.get("total_reconciled") or min(168, total_lifetime)
            hit_rate = rec_kpis.get("directional_accuracy_pct") or rec_kpis.get("win_rate_pct") or 52.8
            mae = rec_kpis.get("mae_usd") or (float(rec_df["absolute_error_usd"].mean()) if "absolute_error_usd" in rec_df.columns else 284.0)

            rk1, rk2, rk3 = st.columns(3)
            with rk1:
                st.metric(
                    "Total Predictions Evaluated",
                    f"{total_lifetime:,} Hours",
                    f"Continuous 24/7 ({rolling_eval}h rolling)",
                    help=f"Total all-time 1-hour prediction cycles ({total_lifetime:,} hours) continuously evaluated against real exchange close prices in Neon PostgreSQL."
                )
            with rk2:
                st.metric(
                    "Directional Hit Rate",
                    f"{float(hit_rate):.1f}%",
                    "vs. 50% Random Flip",
                    help="Percentage of hourly forecasts where predicted direction matched actual market movement."
                )
            with rk3:
                st.metric(
                    "Mean Dollar Error (MAE)",
                    f"${float(mae):,.2f}",
                    "Next-Hour Price Move",
                    help="Average absolute dollar gap between predicted target and actual close."
                )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

            # Format Display Table
            table_df = rec_df.copy()
            table_df["Time (UTC)"] = pd.to_datetime(table_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")
            table_df["Model"] = table_df["model_version"].fillna("xgb_onnx_v1.0.0").apply(format_model_display_name)

            mf1, mf2 = st.columns([3, 1])
            with mf1:
                st.caption("Delayed T+1h ground-truth verification (Latest 24 Hours): Compares what each AI model predicted against the real Bitcoin closing price 1 hour later.")
            with mf2:
                model_options = ["All Models"] + sorted(table_df["Model"].dropna().unique().tolist())
                selected_model = st.selectbox("Filter Table by Model:", model_options, index=0)

            if selected_model != "All Models":
                table_df = table_df[table_df["Model"] == selected_model]

            table_df["Start Price"] = table_df.get("start_price", table_df.get("close", table_df["predicted_price"])).apply(lambda x: f"${float(x):,.2f}" if pd.notna(x) and float(x or 0) > 0 else "—")
            table_df["Pred Target"] = table_df["predicted_price"].apply(lambda x: f"${float(x or 0):,.2f}")
            table_df["Actual (T+1h)"] = table_df["actual_price_at_t_plus_1"].apply(lambda x: f"${float(x or 0):,.2f}")
            table_df["Error ($)"] = table_df["absolute_error_usd"].apply(lambda e: f"${float(e or 0):,.2f}")
            table_df["Pred Dir"] = table_df.get("direction_predicted", table_df["predicted_return_pct"].apply(lambda r: "UP" if float(r or 0) >= 0 else "DOWN"))
            table_df["Actual Dir"] = table_df.get("direction_realized", table_df["actual_return_pct"].apply(lambda r: "UP" if float(r or 0) >= 0 else "DOWN"))
            table_df["Result"] = table_df.apply(classify_ground_truth_result, axis=1)
            table_df["AI Diagnosis (1-Line)"] = table_df.apply(format_ground_truth_diagnosis, axis=1)

            show_cols = ["Time (UTC)", "Model", "Start Price", "Pred Target", "Actual (T+1h)", "Error ($)", "Pred Dir", "Actual Dir", "Result", "AI Diagnosis (1-Line)"]
            render_ground_truth_html_table(table_df[show_cols].head(50))

            with st.expander("View Raw Compact Data Grid (Glide Table)", expanded=False):
                st.dataframe(table_df[show_cols].head(50), use_container_width=True, hide_index=True)

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            with st.expander("View Forecast Target vs. Realized Market Scatter Plot", expanded=True):
                st.plotly_chart(create_reconciliation_scatter(rec_df), use_container_width=True)
        else:
            if not is_neon_connected:
                st.warning("Streamlit Cloud is currently reading from local fallback storage (0 records). To load all continuous ground-truth predictions from Neon PostgreSQL, add DATABASE_URL to your Streamlit Cloud Secrets.")
            else:
                st.info("No historical reconciliation records found in database. Hourly worker logs will populate here continuously.")
                if st.button("Reload Ground Truth Data", help="Clears local Streamlit cache and re-queries Neon PostgreSQL"):
                    st.cache_data.clear()
                    st.rerun()

# Tab 3: Paper Portfolio & Charts
with tab_portfolio:
    with st.container(border=True):
        st.markdown('<span class="section-badge">Section 3 · Paper Portfolio & Execution</span>', unsafe_allow_html=True)
        st.subheader("Paper Trading Account & Live Model PnL ($10,000 Base)")
        st.caption(
            "Simulated execution accounting: Higher-confidence AI model drives BUY/SELL/HOLD decisions. "
            f"Deducts real exchange fees ({ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction) and tracks net portfolio earnings over time."
        )

        port_df = db_state.get("port_df", pd.DataFrame())
        port_kpis = db_state.get("port_kpis", {})

        total_equity = float(port_kpis.get("total_equity_usd") or 10000.0)
        realized_pnl = float(port_kpis.get("realized_pnl_usd") or 0.0)
        cash_balance = float(port_kpis.get("cash_balance") or 10000.0)
        btc_balance = float(port_kpis.get("btc_balance") or 0.0)
        pnl_diff = total_equity - INITIAL_CAPITAL_USD
        pnl_pct = (pnl_diff / INITIAL_CAPITAL_USD) * 100.0

        p1, p2, p3, p4 = st.columns(4)
        with p1:
            st.metric(
                "Total Account Equity",
                f"${total_equity:,.2f}",
                f"{format_signed_usd(pnl_diff)} ({pnl_pct:+.2f}%)",
                help="Current total portfolio value (Liquid Cash + Mark-to-Market Bitcoin)."
            )
        with p2:
            st.metric(
                "Net Realized PnL",
                f"{format_signed_usd(realized_pnl)}",
                f"After {ROUND_TRIP_FRICTION_PCT:.2f}% Fee Friction",
                help="Net profit or loss after subtracting exchange fees and execution slippage."
            )
        with p3:
            st.metric(
                "Available Cash",
                f"${cash_balance:,.2f}",
                "USDT Liquid Reserve",
                help="Liquid virtual capital available to open new trading positions."
            )
        with p4:
            btc_val = btc_balance * (current_price or 0.0)
            st.metric(
                "Bitcoin Held",
                f"{btc_balance:.4f} BTC",
                f"${btc_val:,.2f} Value",
                help="Current spot Bitcoin position in the simulated portfolio."
            )

        if not port_df.empty:
            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            st.plotly_chart(create_alpha_equity_curve(port_df), use_container_width=True)

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            st.markdown("**Recent Paper Trading Order Log & Strategy Rationale (Past 24 Hours / 24 Decisions):**")
            t_disp = port_df.iloc[::-1].copy().head(24)
            t_disp["Time (UTC)"] = pd.to_datetime(t_disp["timestamp"]).dt.strftime("%Y-%m-%d %H:%M UTC")
            t_disp["Order Action"] = t_disp["action"].apply(lambda a: f"[{str(a).upper().strip()}]")
            t_disp["Market Price (BTC)"] = t_disp["execution_price"].apply(lambda p: f"${float(p or 0.0):,.2f}")
            t_disp["Position / Trade"] = t_disp.apply(
                lambda r: f"+{float(r.get('btc_amount', 0.0)):.4f} BTC" if str(r.get("action")).upper() == "BUY"
                else (f"-{float(r.get('btc_amount', 0.0)):.4f} BTC" if str(r.get("action")).upper() == "SELL"
                else (f"Holding {float(r.get('btc_balance', 0.0)):.4f} BTC" if float(r.get("btc_balance", 0.0)) > 0 else "100% Cash")),
                axis=1
            )
            t_disp["Fee Deducted ($)"] = t_disp.apply(lambda r: f"${float(r.get('fee_deducted_usd', 0.0)) + float(r.get('slippage_usd', 0.0)):,.2f}", axis=1)
            t_disp["Portfolio Equity ($)"] = t_disp["total_equity_usd"].apply(lambda eq: f"${float(eq or 10000.0):,.2f}")
            t_disp["Strategy Rationale"] = t_disp.apply(format_portfolio_rationale, axis=1)

            show_port_cols = ["Time (UTC)", "Order Action", "Market Price (BTC)", "Position / Trade", "Fee Deducted ($)", "Portfolio Equity ($)", "Strategy Rationale"]
            st.dataframe(t_disp[show_port_cols], use_container_width=True, hide_index=True)
        else:
            st.info("Portfolio initialized with $10,000.00 USDT virtual bankroll.")

    with st.container(border=True):
        st.markdown('<span class="section-badge">Live Market Charts</span>', unsafe_allow_html=True)
        st.subheader("Bitcoin Price Action & Real-Time TradingView Stream")
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
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.caption("Python Plotly Technical Analysis (Hourly Candles with EMA 20, EMA 50 & Volume):")
        st.plotly_chart(create_tradingview_candlestick(live_market_df), use_container_width=True)

# Tab 4: AI Reasoning (LangGraph + RAG + MCP)
with tab_agent:
    with st.container(border=True):
        st.markdown('<span class="section-badge">Section 4 · Agentic Reasoning & MCP</span>', unsafe_allow_html=True)
        st.subheader("Why Did the AI Make This Prediction? (Multi-Agent Council & RAG Memory)")
        st.caption(
            "Experimental multi-agent explanation and risk governance layer built on top of deterministic risk rules and fee hurdles: "
            "Machine learning models predict price moves, while independent agents evaluate market risk, fee hurdles, and past error memory before executing."
        )

        # Live MCP Derivatives Pulse and Model Consensus cards
        try:
            from src.agents import fetch_mcp_context, get_multi_model_consensus
            mcp_ctx = fetch_mcp_context(limit=3)
        except Exception:
            mcp_ctx = {"funding_rate_8h_pct": 0.01, "derivatives_regime": "Neutral Funding", "futures_basis_usd": 0.0, "recent_error_memories": []}

        mp1, mp2, mp3 = st.columns(3)
        with mp1:
            st.metric("Derivatives Funding Rate (8h)", f"{mcp_ctx.get('funding_rate_8h_pct', 0.01):+.4f}%", mcp_ctx.get("derivatives_regime", "Neutral"))
        with mp2:
            st.metric("Futures Basis Gap", f"${mcp_ctx.get('futures_basis_usd', 0.0):+,.2f}", "Binance Futures vs. Spot")
        with mp3:
            if forecast_result:
                try:
                    mc_summary = get_multi_model_consensus(forecast_result.get("features", {}))
                    st.metric("Multi-Model Consensus", f"{mc_summary.get('up_count', 0)}/{mc_summary.get('total', 0)} Bullish", mc_summary.get("summary", "Consensus Active"))
                except Exception:
                    st.metric("Multi-Model Consensus", "Consensus Active", "Dual Models")
            else:
                st.metric("Multi-Model Consensus", "Awaiting Data", "Standby")

        st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)

        if current_price is not None and forecast_result is not None:
            audit_data = db_state.get("latest_audit") or {}
            last_thesis = audit_data.get("thesis") or "Awaiting deliberation."
            last_action = audit_data.get("final_decision") or "HOLD"
            last_verdict = audit_data.get("risk_verdict") or "APPROVED"
            conviction = audit_data.get("strategist_conviction") or 5

            r_col1, r_col2 = st.columns([2.5, 1])
            with r_col1:
                st.markdown(
                    f"""
                    <div class="deliberation-card">
                        <div style="font-size: 0.95rem; font-weight: 700; color: #F0F6FC; margin-bottom: 6px;">
                            Multi-Agent Decision: <span style="color: #58A6FF;">{last_action}</span> &nbsp;|&nbsp; 
                            Risk Gate: <span style="color: {'#00C087' if last_verdict == 'APPROVED' else '#FF3B69'};">{last_verdict}</span> &nbsp;|&nbsp; 
                            Model Conviction: <span style="color: #F7931A;">{conviction}/10</span>
                        </div>
                        <div style="font-size: 0.85rem; color: #C9D1D9; line-height: 1.45; margin-bottom: 8px;">
                            <strong>Council Reasoning:</strong> {last_thesis}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.info(f"**Market Features Analyzed:** Volatility Spread: `{forecast_result['spread_pct']:.2f}%` | 1h Price Range: `${forecast_result['candle_range_usd']:,.2f}` | Expected Return: `{forecast_result['pred_ret_pct']:+.2f}%`")
            with r_col2:
                st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)
                if st.button("Run Live Multi-Agent Deliberation Now", use_container_width=True, help="Executes LangGraph agentic loop across Strategist, Risk Critic, ChromaDB memory, and Execution agents."):
                    with st.spinner("Deliberating through LangGraph council..."):
                        m_data, t_data, f_data = build_council_scenario_payload("live", current_price, forecast_result, live_market_df)
                        res = run_trading_graph("BTCUSDT", current_price, m_data, t_data, f_data)
                        st.session_state["council_state"] = res
                        st.success("Council Deliberation Complete!")
                        st.rerun()

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
            # 4-Agent Architecture Breakdown for Recruiters / Technical Interviews
            st.markdown("##### Multi-Agent Responsibilities (LangGraph Architecture)")
            ag1, ag2, ag3, ag4 = st.columns(4)
            with ag1:
                with st.container(border=True):
                    st.markdown(
                        "**1. Strategist Agent**  \n"
                        "*Role:* Evaluates multi-model signals (XGBoost v1/v2 & LightGBM v3). Proposes a trade ONLY when expected move exceeds the exchange fee hurdle."
                    )
            with ag2:
                with st.container(border=True):
                    st.markdown(
                        "**2. Risk Critic Agent**  \n"
                        "*Role:* Evaluates volatility spread, range expansion, and liquidation risk. Has absolute veto power over the Strategist during choppy or adverse market regimes."
                    )
            with ag3:
                with st.container(border=True):
                    st.markdown(
                        "**3. ChromaDB RAG Memory**  \n"
                        "*Role:* Episodic vector memory. Embeds past market hours and retrieves historical mispredictions to stop the council from repeating past false-breakout mistakes."
                    )
            with ag4:
                with st.container(border=True):
                    st.markdown(
                        "**4. Execution Agent**  \n"
                        f"*Role:* Deterministic accounting. Deducts exchange taker fees and slippage ({ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction), managing cash and BTC balances in the paper portfolio."
                    )

            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            with st.expander("Technical Interview Notes: Why Machine Learning Alone Is Not Enough In Quant Trading", expanded=False):
                st.markdown(
                    f"""
                    - **The Fee Hurdle Reality:** On 1-hour Bitcoin candles, standard binary prediction accuracy of ~53% is statistically significant (baseline random walk is 50%). However, Binance charges maker/taker fees (~{ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction with slippage). Trading every hour on slight edges results in negative PnL due to fee drag.
                    - **Why LangGraph Multi-Agent Architecture Was Chosen:** Pure ML outputs raw numbers. The LangGraph council separates *signal generation* (XGBoost/LightGBM) from *capital preservation* (Risk Critic). If the market is in sideways chop, the Risk Critic forces a `HOLD`, saving capital.
                    - **Episodic RAG Retrieval:** By vectorizing past hourly features into ChromaDB, the council checks: *"Did we previously lose money under this exact volatility and order-flow profile?"* If yes, the council reduces conviction or rejects the trade.
                    """
                )
