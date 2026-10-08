# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import ta
import plotly.graph_objects as go
from binance.client import Client
from src.feature_engineering import create_features
from config import XGB_FEATURES, SVC_FEATURES

st.set_page_config(page_title="BTC Predictor Suite", layout="wide")

# --- Load Models and Static Data ---
@st.cache_resource
def load_models_and_data():
    """Load models and the pre-featured static data file."""
    try:
        xgb_model = joblib.load('artifacts/models/best_xgb_model.pkl')
        svc_model = joblib.load('artifacts/models/best_svc_model.pkl')
        scaler = joblib.load('artifacts/models/scaler.pkl')
        # Load the data for the simulation page
        sim_data = pd.read_csv('data/processed/featured_btc_data.csv', parse_dates=['timestamp'])
        return xgb_model, svc_model, scaler, sim_data
    except FileNotFoundError as e:
        st.error(f"🚨 A required file is missing: {e}. Please ensure all model and data files are present.")
        return None, None, None, None

xgb_pipeline, svc_model, scaler, sim_df = load_models_and_data()

# --- Initialize Session State for Simulation Page ---
if 'current_index' not in st.session_state:
    st.session_state.current_index = 25
if 'previous_prediction' not in st.session_state:
    st.session_state.previous_prediction = {}

# --- App Mode Selection ---
st.sidebar.title("BTC Predictor Suite 🤖")
app_mode = st.sidebar.radio(
    "Choose a Prediction Mode",
    ["Live Prediction (Binance)", "Simulation from File", "Manual Prediction"]
)

# =====================================================================================
# --- LIVE PREDICTION (BINANCE) PAGE ---
# =====================================================================================
if app_mode == "Live Prediction (Binance)":
    st.title("🔴 Live Prediction (from Binance API)")

    # --- Binance API Connection ---
    try:
        api_key = st.secrets["binance"]["api_key"]
        api_secret = st.secrets["binance"]["api_secret"]
        client = Client(api_key, api_secret)
    except Exception as e:
        st.error(f"Failed to connect to Binance API. Check your .streamlit/secrets.toml file. Error: {e}")
        st.stop()

    # --- Data Fetching and Processing ---
    @st.cache_data(ttl=60)
    def get_live_data(symbol='BTCUSDT', interval=Client.KLINE_INTERVAL_1HOUR, limit=100):
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
        featured_df = create_features(df.copy())
        return featured_df

    if st.button("Refresh Live Data"):
        st.cache_data.clear()

    try:
        live_df = get_live_data()
        last_updated_time = live_df['timestamp'].iloc[-1]
        st.markdown(f"**Last Updated:** `{last_updated_time}`")

        st.header("Recent Market Data")
        fig = go.Figure(data=go.Scatter(x=live_df['timestamp'], y=live_df['close'], mode='lines', name='Close Price'))
        fig.update_layout(title="BTC/USDT - Live 1-Hour Chart", xaxis_title="Time", yaxis_title="Price (USDT)")
        st.plotly_chart(fig, use_container_width=True)

        st.header("Prediction for the Current Hour")
        prediction_input = live_df.iloc[-2]

        reg_features = XGB_FEATURES
        clf_features = SVC_FEATURES
        
        input_reg = pd.DataFrame([prediction_input[reg_features]], columns=reg_features)
        input_clf = pd.DataFrame([prediction_input[clf_features]], columns=clf_features)
        
        pred_price = xgb_pipeline.predict(input_reg)[0]
        scaled_input_svc = scaler.transform(input_clf)
        pred_move_code = svc_model.predict(scaled_input_svc)[0]
        pred_move_text = "Upward 📈" if pred_move_code == 1 else "Downward 📉"

        col1, col2 = st.columns(2)
        col1.metric("Predicted Price for this Hour", f"${pred_price:,.2f}")
        col2.metric("Predicted Movement for this Hour", pred_move_text)
        st.info("This prediction is based on the data from the previous completed hour.")

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
    reg_features = XGB_FEATURES
    clf_features = SVC_FEATURES
    
    input_reg = pd.DataFrame([current_data[reg_features]], columns=reg_features)
    input_clf = pd.DataFrame([current_data[clf_features]], columns=clf_features)
    pred_price = xgb_pipeline.predict(input_reg)[0]
    scaled_input_svc = scaler.transform(input_clf)
    pred_move_code = svc_model.predict(scaled_input_svc)[0]
    pred_move_text = "Upward 📈" if pred_move_code == 1 else "Downward 📉"
    
    col1, col2 = st.columns(2)
    col1.metric("Predicted Next Price", f"${pred_price:.2f}")
    col2.metric("Predicted Next Movement", pred_move_text)
    st.session_state.previous_prediction = {'price': pred_price, 'movement': pred_move_text}

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
            reg_features = XGB_FEATURES
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
                    'onchain_miners_revenue_change': 0,
                    'sentiment_score': 0, 'sentiment_mean_3d': 0
                }])[XGB_FEATURES]
            clf_features = SVC_FEATURES
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
                    'onchain_miners_revenue_change': 0,
                    'sentiment_score': 0, 'sentiment_mean_3d': 0
                }])[SVC_FEATURES]

            pred_price = xgb_pipeline.predict(input_reg)[0]
            scaled_input_svc = scaler.transform(input_clf)
            pred_move_code = svc_model.predict(scaled_input_svc)[0]
            pred_move_text = "Upward 📈" if pred_move_code == 1 else "Downward 📉"
            
            st.header("Prediction Results")
            col1, col2 = st.columns(2)
            col1.metric("Predicted Close Price", f"${pred_price:,.2f}")
            col2.metric("Predicted Movement", pred_move_text)
            st.success("Prediction generated successfully!")
        except Exception as e:
            st.error(f"❌ An error occurred: {e}")