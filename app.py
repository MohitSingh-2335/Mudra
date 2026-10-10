# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import ta
import plotly.graph_objects as go
import requests
from binance.client import Client
from src.feature_engineering import create_features
from src.agents.fear_greed_agent import merge_fear_greed
from src.agents.onchain_agent import merge_onchain
from config import (
    REGRESSOR_FEATURES,
    CLASSIFIER_FEATURES,
    REGRESSOR_MODEL_PATH,
    CLASSIFIER_MODEL_PATH,
    FEATURED_BTC_DATA_PATH
)
import matplotlib.pyplot as plt
from src.explainability import get_tree_explainer, explain_single_prediction
from src.agents.market_analyst_agent import generate_market_commentary


st.set_page_config(page_title="BTC Predictor Suite", layout="wide")

# --- Load Models and Static Data ---
@st.cache_resource
def load_models_and_data():
    """Load models and the pre-featured static data file."""
    try:
        xgbr_model = joblib.load(REGRESSOR_MODEL_PATH)
        xgbc_model = joblib.load(CLASSIFIER_MODEL_PATH)
        xgbr_model.set_params(device="cpu")
        xgbc_model.set_params(device="cpu")
        explainer_clf = get_tree_explainer(xgbc_model)
        sim_df = pd.read_csv(FEATURED_BTC_DATA_PATH, parse_dates=['timestamp'])
        return xgbr_model, xgbc_model, explainer_clf, sim_df
    except FileNotFoundError as e:
        st.error(f"🚨 A required file is missing: {e}. Please ensure all model and data files are present.")
        return None, None, None, None

xgbr_model, xgbc_model, explainer_clf, sim_df = load_models_and_data()

def render_analyst_commentary(report):
    """Renders the AI Chief Quantitative Strategist commentary card."""
    st.markdown("---")
    st.subheader("🤖 AI Chief Quantitative Strategist Commentary")

    is_bull = "BULLISH" in report.trade_thesis
    badge_color = "🟢" if is_bull else ("🔴" if "BEARISH" in report.trade_thesis else "⚪")

    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        st.markdown(f"### {badge_color} Thesis: `{report.trade_thesis}`")
    with col_t2:
        st.metric("Model Conviction", f"{report.conviction_score} / 10")

    st.info(f"**Executive Summary:**\n\n{report.executive_summary}")

    with st.expander("🔬 View In-Depth Quant & Macro Synthesis", expanded=True):
        st.markdown("**📊 Quantitative Driver Analysis (TreeSHAP Interpretation):**")
        st.write(report.quantitative_driver_analysis)

        st.markdown("**🌐 Macro & Domain Regime Context (ChromaDB RAG):**")
        st.write(report.macro_market_context)

        st.markdown("**⚠️ Invalidation Scenarios & Key Risks:**")
        for risk in report.key_risks:
            st.markdown(f"- {risk}")

# --- Initialize Session State for Simulation Page ---
if 'current_index' not in st.session_state:
    st.session_state.current_index = 25
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = {}

# --- App Mode Selection ---
st.sidebar.title("BTC Predictor Suite 🤖")
app_mode = st.sidebar.radio(
    "Choose a Prediction Mode",
    ["Simulation from File", "Live Market Prediction", "Manual Prediction"]
)

# =====================================================================================
# --- LIVE PREDICTION PAGE ---
# =====================================================================================
if app_mode == "Live Market Prediction":
    st.title("🔴 Live Market Prediction")

    # --- Data Fetching and Processing ---
    @st.cache_data(ttl=60)
    def get_live_data():
        # Provider 1: Try Binance API first
        try:
            if hasattr(st, "secrets") and "binance" in st.secrets:
                api_key = st.secrets["binance"]["api_key"]
                api_secret = st.secrets["binance"]["api_secret"]
                client = Client(api_key, api_secret)
                klines = client.get_klines(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=100)
                df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                df = merge_fear_greed(df)
                df = merge_onchain(df)
                featured_df = create_features(df.copy())
                return featured_df, "Binance API"
        except Exception:
            pass

        # Provider 2: US Cloud-Friendly Fallback (Coinbase Institutional Public Feed)
        try:
            url = "https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity=3600"
            r = requests.get(url, headers={"User-Agent": "Mudra-Quant/1.0"}, timeout=10)
            if r.status_code == 200:
                raw = r.json()[:100]
                df = pd.DataFrame(raw, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df = df.sort_values('timestamp').reset_index(drop=True)
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col])
                df = merge_fear_greed(df)
                df = merge_onchain(df)
                featured_df = create_features(df.copy())
                return featured_df, "Coinbase Institutional Public Feed"
        except Exception as e:
            return None, f"Feed Error: {e}"

        return None, "All providers unavailable"

    if st.button("Refresh Live Data"):
        st.cache_data.clear()

    try:
        live_df, feed_source = get_live_data()
        if live_df is None or live_df.empty:
            st.error(f"Failed to fetch market data: {feed_source}. Please refresh in a moment.")
            st.stop()

        if "Coinbase" in feed_source:
            st.info("ℹ️ Live BTC/USD market feed streaming via Coinbase Institutional API (US cloud provider fallback).")

        last_updated_time = live_df['timestamp'].iloc[-1]
        st.markdown(f"**Last Updated:** `{last_updated_time}` | **Feed Provider:** `{feed_source}`")

        st.header("Recent Market Data")
        fig = go.Figure(data=go.Scatter(x=live_df['timestamp'], y=live_df['close'], mode='lines', name='Close Price'))
        fig.update_layout(title="BTC/USDT - Live 1-Hour Chart", xaxis_title="Time", yaxis_title="Price (USDT)")
        st.plotly_chart(fig, use_container_width=True)

        reg_features = REGRESSOR_FEATURES
        clf_features = CLASSIFIER_FEATURES

        # Defensive check: ensure all required model features are present and non-null
        live_df = live_df.bfill().ffill().fillna(0)
        for col in reg_features:
            if col not in live_df.columns:
                live_df[col] = 0.0
        for col in clf_features:
            if col not in live_df.columns:
                live_df[col] = 0.0

        st.header("Prediction for the Current Hour")
        prediction_input = live_df.iloc[-2]
        
        input_reg = pd.DataFrame([prediction_input[reg_features]], columns=reg_features)
        input_clf = pd.DataFrame([prediction_input[clf_features]], columns=clf_features)
        
        pred_return = xgbr_model.predict(input_reg)[0]
        pred_price = prediction_input['close'] * (1 + pred_return)
        pred_move_code = xgbc_model.predict(input_clf)[0]
        pred_move_text = "Upward 📈" if pred_move_code == 1 else "Downward 📉"

        col1, col2 = st.columns(2)
        col1.metric("Predicted Price for this Hour", f"${pred_price:,.2f}")
        col2.metric("Predicted Movement for this Hour", pred_move_text)
        st.info("This prediction is based on the data from the previous completed hour.")

        with st.expander("🔍 AI Prediction Explanation (SHAP Attribution)"):
            base_val, top_pos, top_neg, fig = explain_single_prediction(explainer_clf, input_clf, top_n=4)
            exp_col1, exp_col2 = st.columns(2)
            with exp_col1:
                st.markdown("**🟢 Bullish Drivers (Pushed UP):**")
                for item in top_pos:
                    st.markdown(f"- **{item['feature']}**: `+{item['shap']:.4f}` (value: `{item['value']:.2f}`)")
            with exp_col2:
                st.markdown("**🔴 Bearish Drivers (Pushed DOWN):**")
                for item in top_neg:
                    st.markdown(f"- **{item['feature']}**: `{item['shap']:.4f}` (value: `{item['value']:.2f}`)")
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("---")
        if st.button("🧠 Synthesize AI Strategist Commentary", key="btn_live_agent"):
            with st.spinner("Synthesizing quantitative ML predictions, TreeSHAP drivers, and ChromaDB knowledge..."):
                report = generate_market_commentary(
                    current_price=float(prediction_input['close']),
                    pred_price=float(pred_price),
                    pred_return=float(pred_return),
                    pred_move_text=pred_move_text,
                    top_pos_drivers=top_pos,
                    top_neg_drivers=top_neg
                )
                st.session_state['live_analyst_report'] = report

        if 'live_analyst_report' in st.session_state:
            render_analyst_commentary(st.session_state['live_analyst_report'])

    except Exception as e:
        st.error(f"An error occurred while fetching or processing live data: {e}")

# =====================================================================================
# --- SIMULATION FROM FILE PAGE ---
# =====================================================================================
elif app_mode == "Simulation from File":
    st.title("⏳ Trading Simulation (from CSV File)")

    st.markdown(f"**Current Time:** `{sim_df.loc[st.session_state.current_index, 'timestamp']}`")
    st.header("Recent Market Data")
    history_df = sim_df.iloc[st.session_state.current_index-24 : st.session_state.current_index+1]
    fig = go.Figure(data=go.Scatter(x=history_df['timestamp'], y=history_df['close'], mode='lines+markers', name='Close Price'))
    fig.update_layout(title="BTC/USDT - Last 24 Hours", xaxis_title="Time", yaxis_title="Price (USDT)")
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.previous_prediction:
        st.header("Last Hour's Prediction vs. Actual")
        prev_pred = st.session_state.previous_prediction
        actual_data = sim_df.loc[st.session_state.current_index]
        actual_movement = "Upward 📈" if actual_data['close'] > actual_data['open'] else "Downward 📉"
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Predicted Price", f"${prev_pred['price']:.2f}")
            st.metric("Predicted Movement", prev_pred['movement'])
        with col2:
            st.metric("Actual Price", f"${actual_data['close']:.2f}")
            st.metric("Actual Movement", actual_movement)
        if prev_pred['movement'] == actual_movement:
            st.success("✅ Prediction was CORRECT")
        else:
            st.error("❌ Prediction was INCORRECT")
    else:
        st.info("Advancing to the next hour will show the first prediction verification.")
        

    st.header("Prediction for the Next Hour")
    current_data = sim_df.loc[st.session_state.current_index]
    reg_features = REGRESSOR_FEATURES
    clf_features = CLASSIFIER_FEATURES
    
    input_reg = pd.DataFrame([current_data[reg_features]], columns=reg_features)
    input_clf = pd.DataFrame([current_data[clf_features]], columns=clf_features)
    pred_return = xgbr_model.predict(input_reg)[0]
    pred_price = current_data['close'] * (1 + pred_return)
    pred_move_code = xgbc_model.predict(input_clf)[0]
    pred_move_text = "Upward 📈" if pred_move_code == 1 else "Downward 📉"
    
    col1, col2 = st.columns(2)
    col1.metric("Predicted Next Price", f"${pred_price:.2f}")
    col2.metric("Predicted Next Movement", pred_move_text)
    st.session_state.previous_prediction = {'price': pred_price, 'movement': pred_move_text}
    with st.expander("🔍 AI Prediction Explanation (SHAP Attribution)"):
        base_val, top_pos, top_neg, fig = explain_single_prediction(explainer_clf, input_clf, top_n=4)
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("**🟢 Bullish Drivers (Pushed UP):**")
            for item in top_pos:
                st.markdown(f"- **{item['feature']}**: `+{item['shap']:.4f}` (value: `{item['value']:.2f}`)")
        with exp_col2:
            st.markdown("**🔴 Bearish Drivers (Pushed DOWN):**")
            for item in top_neg:
                st.markdown(f"- **{item['feature']}**: `{item['shap']:.4f}` (value: `{item['value']:.2f}`)")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("---")
    sim_key = f"sim_report_{st.session_state.current_index}"
    if st.button("🧠 Synthesize AI Strategist Commentary", key="btn_sim_agent"):
        with st.spinner("Synthesizing quantitative ML predictions, TreeSHAP drivers, and ChromaDB knowledge..."):
            report = generate_market_commentary(
                current_price=float(current_data['close']),
                pred_price=float(pred_price),
                pred_return=float(pred_return),
                pred_move_text=pred_move_text,
                top_pos_drivers=top_pos,
                top_neg_drivers=top_neg
            )
            st.session_state[sim_key] = report

    if sim_key in st.session_state:
        render_analyst_commentary(st.session_state[sim_key])

    if st.button("Advance to Next Hour ->"):
        st.session_state.current_index += 1
        st.rerun()

# =====================================================================================
# --- MANUAL PREDICTION PAGE ---
# =====================================================================================
elif app_mode == "Manual Prediction":
    st.title("✍️ Manual Prediction")
    st.markdown("Enter market data to get a one-off prediction.")
    
    st.sidebar.header("Input Features")
    open_price = st.sidebar.number_input("Open Price", value=68000.0, step=100.0)
    high_price = st.sidebar.number_input("High Price", value=68500.0, step=100.0)
    low_price = st.sidebar.number_input("Low Price", value=67500.0, step=100.0)
    close_price = st.sidebar.number_input("Close Price", value=68200.0, step=100.0)
    volume = st.sidebar.number_input("Volume", value=1500.0, step=100.0)
    
    if st.sidebar.button("🔮 Predict"):
        try:
            timestamp = datetime.now()
            price_change = close_price - open_price
            volatility = (high_price - low_price) / open_price
            high_low_ratio = high_price / low_price
            close_open_diff = close_price - open_price
            hour, dayofweek, day = timestamp.hour, timestamp.dayofweek, timestamp.day
            hour_sin, hour_cos = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
            day_sin, day_cos = np.sin(2*np.pi*dayofweek/7), np.cos(2*np.pi*dayofweek/7)
            close_series = pd.Series([close_price] * 35)
            rsi = ta.momentum.RSIIndicator(close=close_series, window=14).rsi().iloc[-1]
            macd = ta.trend.MACD(close=close_series).macd().iloc[-1]
            bb = ta.volatility.BollingerBands(close=close_series)
            bb_high = bb.bollinger_hband().iloc[-1]
            bb_low = bb.bollinger_lband().iloc[-1]
            ema_10 = ta.trend.EMAIndicator(close=close_series, window=10).ema_indicator().iloc[-1]
            ema_30 = ta.trend.EMAIndicator(close=close_series, window=30).ema_indicator().iloc[-1]
            reg_features = REGRESSOR_FEATURES
            input_reg = pd.DataFrame([{
                    'volume': volume,
                    'Price Change': price_change,
                    'Rolling_Std_Close': 0,
                    'vol_1h': high_price - low_price,
                    'vol_mean_6h': 0, 'vol_std_6h': 0, 'vol_max_6h': 0, 'vol_min_6h': 0,
                    'hour': hour, 'dayofweek': dayofweek, 'day': day,
                    'rsi': rsi, 'high_low_ratio': high_low_ratio,
                    'hour_sin': hour_sin, 'hour_cos': hour_cos,
                    'day_sin': day_sin, 'day_cos': day_cos,
                    'close_lag_1': close_price,
                    'taker_buy_ratio': 0, 'taker_buy_ratio_mean_6h': 0, 'trades_mean_6h': 0,
                    'fng_value': 50, 'fng_mean_3d': 50,
                    'onchain_num_tx_change': 0, 'onchain_hash_rate_change': 0,
                    'onchain_miners_revenue_change': 0
                }])[REGRESSOR_FEATURES]
            clf_features = CLASSIFIER_FEATURES
            input_clf = pd.DataFrame([{
                    'volume': volume, 'Price Change': price_change,
                    'Volatility': volatility, 'Rolling_Mean_Close': close_price,
                    'Rolling_Std_Close': 0, 'vol_mean_6h': 0, 'vol_std_6h': 0,
                    'vol_max_6h': 0, 'vol_min_6h': 0, 'return_mean_6h': 0, 'return_std_6h': 0,
                    'hour': hour, 'dayofweek': dayofweek, 'day': day,
                    'rsi': rsi, 'macd': macd, 'bb_high': bb_high, 'bb_low': bb_low,
                    'ema_10': ema_10, 'ema_30': ema_30,
                    'high_low_ratio': high_low_ratio, 'close_open_diff': close_open_diff,
                    'close_lag_1': close_price, 'volume_lag_1': volume,
                    'rolling_max_6h': high_price, 'rolling_min_6h': low_price,
                    'price_volatility_interaction': close_price * volatility,
                    'hour_sin': hour_sin, 'hour_cos': hour_cos,
                    'day_sin': day_sin, 'day_cos': day_cos,
                    'taker_buy_ratio': 0, 'taker_buy_ratio_mean_6h': 0, 'trades_mean_6h': 0,
                    'fng_value': 50, 'fng_mean_3d': 50,
                    'onchain_num_tx_change': 0, 'onchain_hash_rate_change': 0,
                    'onchain_miners_revenue_change': 0
                }])[CLASSIFIER_FEATURES]

            pred_return = xgbr_model.predict(input_reg)[0]
            pred_price = close_price * (1 + pred_return)
            pred_move_code = xgbc_model.predict(input_clf)[0]
            pred_move_text = "Upward 📈" if pred_move_code == 1 else "Downward 📉"
            
            st.header("Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Close Price", f"${pred_price:,.2f}")
            col2.metric("Predicted Movement", pred_move_text)
            st.success("Prediction generated successfully!")
            with st.expander("🔍 AI Prediction Explanation (SHAP Attribution)"):
                base_val, top_pos, top_neg, fig = explain_single_prediction(explainer_clf, input_clf, top_n=4)
                exp_col1, exp_col2 = st.columns(2)
                with exp_col1:
                    st.markdown("**🟢 Bullish Drivers (Pushed UP):**")
                    for item in top_pos:
                        st.markdown(f"- **{item['feature']}**: `+{item['shap']:.4f}` (value: `{item['value']:.2f}`)")
                with exp_col2:
                    st.markdown("**🔴 Bearish Drivers (Pushed DOWN):**")
                    for item in top_neg:
                        st.markdown(f"- **{item['feature']}**: `{item['shap']:.4f}` (value: `{item['value']:.2f}`)")
                st.pyplot(fig)
                plt.close(fig)

            st.markdown("---")
            if st.button("🧠 Synthesize AI Strategist Commentary", key="btn_manual_agent"):
                with st.spinner("Synthesizing quantitative ML predictions, TreeSHAP drivers, and ChromaDB knowledge..."):
                    report = generate_market_commentary(
                        current_price=float(close_price),
                        pred_price=float(pred_price),
                        pred_return=float(pred_return),
                        pred_move_text=pred_move_text,
                        top_pos_drivers=top_pos,
                        top_neg_drivers=top_neg
                    )
                    st.session_state['manual_analyst_report'] = report

            if 'manual_analyst_report' in st.session_state:
                render_analyst_commentary(st.session_state['manual_analyst_report'])

        except Exception as e:
            st.error(f"❌ An error occurred: {e}")