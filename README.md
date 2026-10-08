# Mudra: BTC/USDT Trading Research Platform

> Mudra is an experimental Python platform for BTC/USDT market analysis, next-hour machine-learning prediction, and fee-aware trading-strategy evaluation.

---

## Documentation

See [MUDRA_PROJECT_ANALYSIS.md](MUDRA_PROJECT_ANALYSIS.md) for the current implementation, architecture, outputs, and known gaps.

Historical reports are in `docs/reports/`. Design and planning material is in `archive/planning/` and should be treated as future scope, not as implemented functionality.

---

## Main Components

1. **Data ingestion and preparation**: `scripts/ingest/` and `scripts/prepare_data.py` fetch and prepare hourly Binance data.
2. **Feature engineering**: `src/feature_engineering.py` combines technical, volume, Fear & Greed, on-chain, and sentiment features.
3. **Machine learning**: `src/model_training.py` trains XGBoost regression and SVC classification models.
4. **Research experiments**: `scripts/experiments/` and `experiments/backtesting/` compare models and validate strategies chronologically.
5. **Dashboard**: `app.py` provides live, file-based simulation, and manual prediction modes through Streamlit.

---

## 🛠️ Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch fresh market data
python scripts/ingest/fetch_fresh_data.py

# 3. Generate features
python scripts/prepare_data.py

# 4. Train models
python -m src.model_training

# 5. Launch the interactive dashboard
streamlit run app.py
```

---

## 🔒 Security & Environment
Binance credentials are loaded from environment variables or `.streamlit/secrets.toml`. These files are local-only and must never be committed. The dashboard is for research and prediction; it does not place exchange orders.
