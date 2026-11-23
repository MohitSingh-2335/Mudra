# Architectural Decision Records (ADRs)

Lightweight decision records documenting core technology choices, trade-offs, and live verification evidence in Mudra.

---

## ADR 01: Neon Serverless PostgreSQL (Unified Relational + JSONB Store)

- **Why we use it:** Provides strict ACID relational foreign keys linking market candles, multi-model predictions, and hourly post-mortems with schema validation, while native `JSONB` columns store arbitrary 29–39 engineered feature vectors and LLM post-mortem payloads without breaking relational integrity.
- **Why not a simpler alternative:** Local SQLite cannot persist across stateless cloud runners (GitHub Actions hourly workers wipe local disk upon job exit), while pure NoSQL databases (e.g. MongoDB) lack atomic relational joins required for exact $T+1\text{h}$ reconciliation.
- **Where live proof is visible:** Dashboard Tab 1 (*Inspect Latest Stored Feature Snapshot*) and live audit report `data/proof/neon_column_audit.json` (100% active non-null telemetry across 8 tables).
- **30-second interview answer:** *"We use Neon PostgreSQL as our single source of ground truth because GitHub Actions and Streamlit Cloud run in stateless containers. Combining structured SQL tables for time-series candles with JSONB for dynamic feature snapshots allows deterministic point-in-time reconciliation without managing separate database clusters."*

---

## ADR 02: FastMCP Server (`src/mcp_server.py`)

- **Why we use it:** Standardizes external market data ingestion (Deribit/Kraken perpetual funding rates, futures basis gap, and live CryptoCompare/CoinDesk news) into a decoupled Model Context Protocol tool interface with typed schemas and dual US-friendly fallbacks.
- **Why not a simpler alternative:** Hardcoding ad-hoc REST/scraping scripts directly inside the quantitative loop tightly couples market feeds to the engine; if an endpoint is blocked or changes schemas, the entire ML pipeline fails.
- **Where live proof is visible:** Dashboard Sidebar (*24h API Health Telemetry*), Tab 4 top metric cards (*Derivatives Funding Rate & Futures Basis*), and `data/proof/live_verification_proof.json` (`mcp_derivatives` status `LIVE`).
- **30-second interview answer:** *"We implemented a FastMCP server to isolate external crypto exchange and news APIs behind typed Model Context Protocol tool interfaces. It provides automatic fallback from Deribit to Kraken Futures and CoinDesk RSS, ensuring external feed outages never crash our autonomous inference pipeline."*

---

## ADR 03: Persistent Episodic RAG (Neon-Hydrated ChromaDB)

- **Why we use it:** Enables the multi-agent council to retrieve historical misprediction patterns and market regime rules before placing trades, stopping the system from repeating false-breakout mistakes under identical volatility or order-flow conditions.
- **Why not a simpler alternative:** Pure in-memory vector databases wipe their index on cloud container shutdown, whereas heavy managed cloud vector databases (Pinecone, Weaviate) exceed free-tier limits or require ongoing subscription costs.
- **Where live proof is visible:** Dashboard Tab 4 (*Live Episodic RAG Memory Proof: 72+ vectors indexed, 62+ hydrated from Neon*) and `data/proof/live_verification_proof.json` (`rag_hydration` proof block).
- **30-second interview answer:** *"We designed a hybrid episodic vector memory where ChromaDB automatically hydrates its embeddings from reconciled Neon records on startup. This gives our LangGraph council long-term memory of past prediction errors across stateless cloud runs at zero infrastructure cost."*

---

## ADR 04: LangGraph + Grounded Gemini 3.1 Flash-Lite Post-Mortems

- **Why we use it:** Implements stateful multi-agent governance where a Risk Critic has deterministic veto power over the Strategist, and utilizes Gemini 3.1 Flash-Lite with structured `application/json` output and strict prompt templates to generate verifiable, feature-level error attribution.
- **Why not a simpler alternative:** Unstructured LLM prompts frequently hallucinate non-existent prices, exceed free-tier rate limits (RPD quotas), or emit unescaped control characters; hardcoded heuristic rules alone lack human-readable contextual synthesis for quantitative audits.
- **Where live proof is visible:** Dashboard Tab 2 (*AI Diagnosis 1-line column citing specific RSI and taker order drivers*) and `data/proof/live_verification_proof.json` (`gemini_postmortem` status `OK`, model `gemini-3.1-flash-lite`).
- **30-second interview answer:** *"We use LangGraph to separate aggressive signal generation from capital preservation with an independent Risk Critic agent. For post-mortems, we use Gemini 3.1 Flash-Lite with strict JSON schema enforcement to attribute errors directly to micro-structure features like taker buy ratios and funding rates rather than generic price commentary."*

---

## ADR 05: Classical GBDTs (`XGBoost v1/v2` + `LightGBM v3` + Platt Calibration)

- **Why we use it:** Tabular financial time series with high noise-to-signal ratios are empirically dominated by gradient-boosted decision trees over deep learning; combining XGBoost with LightGBM and Platt probability calibration provides well-calibrated directional confidence intervals and clear TreeSHAP feature attribution.
- **Why not a simpler alternative:** Deep learning architectures (LSTMs, Transformers) require massive parameter tuning, are prone to severe overfitting on non-stationary crypto regimes, and exhibit high inference latency without providing verifiable feature attribution.
- **Where live proof is visible:** Dashboard Tab 1 (*Model Benchmark Comparison: 53.26% Active vs 52.07% & 52.89% Shadow Candidates*) and TreeSHAP summary plots in `docs/images/`.
- **30-second interview answer:** *"We deployed classical GBDT ensembles—XGBoost and LightGBM—with Platt probability calibration because tree-based models consistently outperform neural networks on noisy 1-hour tabular crypto data. They train in seconds, avoid non-stationary regime memorization, and output exact TreeSHAP attribution for every trading decision."*
