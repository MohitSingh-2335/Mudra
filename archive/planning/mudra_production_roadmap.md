# ðŸš€ Mudra â€” Production-Ready & Resume-Ready Roadmap

> **Target:** B.Tech CSE Fresher | Roles: ML Engineer, Data Scientist, Data Analyst, Python Dev  
> **Timeline:** 2â€“3 weeks (Full-time grind)  
> **Current State:** Research prototype â†’ **Goal:** Production-grade, deployable, interview-ready AI/ML system

---

## ðŸ§­ Big Picture: What This Project Will Become

By the end of this roadmap, **Mudra** will be a fully deployable, explainable, RAG-powered Agentic AI crypto ML platform with:
- A **FastAPI backend** serving model predictions as REST endpoints
- A **beautiful Streamlit frontend** with SHAP explainability and live predictions
- **MLflow** for experiment tracking and model registry
- **Optuna** for hyperparameter tuning (automated)
- **SHAP** for explainability (why did the model decide this?)
- **RAG pipeline** (ChromaDB + embeddings) â€” retrieves relevant crypto news at prediction time
- **LangChain** â€” chains retrieval â†’ prompt formatting â†’ LLM call cleanly
- **Gemini Tool-Calling Agent** â€” a real agentic AI analyst that decides which tools to call and synthesizes a market verdict
- **Docker + GitHub Actions CI/CD** for production deployment
- **Deployed online** (HuggingFace Spaces + Render.com)
- **Clean GitHub** with proper README, badges, and documentation

> **IBM Course Skills Used:** RAG, ChromaDB, LangChain, Agentic AI (tool-calling), embeddings, prompt engineering â€” all genuinely applied, nothing forced.

---

## ðŸ“Š Current State Audit (Honest Assessment)

| Component | Status | Priority Fix |
|---|---|---|
| Data pipeline | âœ… Works but has bugs (duplicate columns) | High |
| Feature engineering | âœ… Good but inconsistent feature lists | High |
| Model training (XGBoost + SVC) | âœ… Works | Medium |
| Backtesting | âœ… Good logic | Low |
| SHAP explainability | âŒ Missing | High |
| Optuna tuning | âŒ Missing | High |
| MLflow tracking | âŒ Missing | High |
| FastAPI backend | âŒ Missing | High |
| LLM/GenAI integration | âŒ Missing | High |
| Docker | âŒ Missing | Medium |
| CI/CD (GitHub Actions) | âŒ Missing | Medium |
| Tests | âŒ Missing | Medium |
| Online deployment | âŒ Not deployed | High |
| Professional README | âŒ Current one is aspirational/fake | High |

---

## ðŸ—“ï¸ PHASE-BY-PHASE ROADMAP

---

### âš¡ PHASE 0 â€” Code Cleanup & Bug Fixes (Day 1-2)
> **Why first:** You can't build a house on a broken foundation. These bugs will cause errors during interviews.

#### Tasks:
- [ ] **Fix duplicate on-chain columns** in `scripts/prepare_data.py`
- [ ] **Align feature lists** between `src/feature_engineering.py` and `app.py` (they are mismatched â†’ causes crashes)
- [ ] **Fix `requirements.txt`** â€” add missing deps (`shap`, `mlflow`, `optuna`, `fastapi`, `uvicorn`, `python-dotenv`)
- [ ] **Add `.env` file** + `.env.example` for API keys (Binance, Gemini API)
- [ ] **Add a `config.py` single source of truth** â€” one place for all paths, feature lists, model names
- [ ] **Remove dead/duplicate code** in scripts

#### Interview Value:
- Shows you understand **code quality**, **debugging**, and **maintainability**
- Interviewers WILL look at your GitHub code

---

### ðŸ§ª PHASE 1 â€” MLflow Experiment Tracking (Day 3-4)
> **Why:** MLflow is one of the **most asked-about MLOps tools** in ML engineer interviews. It shows you understand the full ML lifecycle.

#### What to Build:
- Wrap `src/model_training.py` with **MLflow runs**
- Log: hyperparameters, metrics (MAE, RMSE, accuracy, Sharpe ratio), model artifacts
- Register best model in **MLflow Model Registry**
- Run MLflow UI locally: `mlflow ui`

#### Key Concepts You'll Learn:
- **Experiment tracking** â€” why it matters (reproducibility)
- **Model versioning** â€” staging vs. production models
- **Artifact logging** â€” saving plots, CSVs, models

#### Resume Bullet:
> *"Integrated MLflow for experiment tracking, logging 15+ metrics per run and managing model registry across XGBoost regression and SVC classification experiments"*

---

### ðŸ”¬ PHASE 2 â€” Optuna Hyperparameter Tuning (Day 4-5)
> **Why:** Optuna is the **industry standard** for AutoML/HPO. Shows you go beyond default params.

#### What to Build:
- Replace hardcoded XGBoost params with **Optuna study**
- Define search space: `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`
- Run 50-100 trials with **pruning** (stops bad trials early)
- Log best params to **MLflow** automatically
- Visualize optimization history (Optuna has built-in plots)

#### Key Concepts You'll Learn:
- **Bayesian optimization** vs grid search (smarter search)
- **Trial pruning** â€” efficiency
- **Hyperparameter importance** plots

#### Resume Bullet:
> *"Implemented Optuna-based Bayesian hyperparameter optimization with 100 trials and MedianPruner, reducing XGBoost MAE by X% over default configuration"*

---

### ðŸ§  PHASE 3 â€” SHAP Explainability (Day 5-6)
> **Why:** Explainability is **critical** in 2024-2026 AI hiring. It shows you understand WHY your model works, not just that it works.

#### What to Build:
- Add `shap` to `src/model_training.py`
- Generate **SHAP summary plots** (global feature importance)
- Generate **SHAP waterfall plots** (why did THIS prediction happen?)
- Save SHAP plots as images, display them in Streamlit
- Add a "Why did the model predict this?" section in the dashboard

#### Key Concepts You'll Learn:
- **SHAP values** â€” game-theory based feature attribution
- **Global vs. local explanations**
- **Feature importance** beyond just model `.feature_importances_`

#### Resume Bullet:
> *"Implemented SHAP-based model explainability, identifying top 5 features driving BTC direction predictions (RSI, Fear & Greed index, trading volume, MACD, lagged returns)"*

---

### ðŸ¤– PHASE 4 â€” RAG + LangChain + Gemini Agentic AI (Day 7-9)
> **Why:** This is the direct application of your IBM RAG & Agentic AI certificate. Three of the hottest skills in one phase. Done right, this alone makes your resume stand out from 95% of freshers.

#### What to Build â€” Part A: RAG Pipeline (ChromaDB)

- Install: `chromadb`, `langchain`, `langchain-google-genai`, `sentence-transformers`
- Create `src/rag/news_store.py`:
  - Fetch recent crypto news headlines from **CryptoPanic free API** (or GDELT which is already in the project)
  - **Embed** the headlines using a free sentence-transformer model (`all-MiniLM-L6-v2`)
  - Store embeddings in a **local ChromaDB** collection (no cloud, no cost)
- Create `src/rag/retriever.py`:
  - Given the current market situation (e.g., "BTC RSI oversold, Fear=22"), retrieve the **top 3 most relevant recent news articles**

#### What to Build â€” Part B: LangChain Pipeline

- Create `src/agents/langchain_pipeline.py`:
  - Use **LangChain** to chain:
    1. Retrieve relevant news (RAG step)
    2. Format a structured prompt with: ML prediction + SHAP top features + Fear & Greed + retrieved news
    3. Call **Gemini** with the assembled prompt
  - This is a proper **LangChain RAG chain** â€” exactly what you studied

#### What to Build â€” Part C: Gemini Tool-Calling Agent

- Create `src/agents/market_analyst_agent.py`:
  - Define **tools** that Gemini can call:
    - `get_fear_greed()` â†’ returns current Fear & Greed index
    - `get_ml_prediction()` â†’ runs the XGBoost/SVC model and returns prediction + confidence
    - `get_onchain_metrics()` â†’ returns hash rate, miner revenue changes
    - `search_recent_news(query)` â†’ queries the ChromaDB RAG store
  - Build a **Gemini function-calling agent** that:
    - Receives a user question like *"Should I be cautious about BTC right now?"*
    - **Decides on its own** which tools to call
    - Synthesizes all results into a final verdict
- Integrate into Streamlit: show the agent's reasoning + verdict on the dashboard

#### Key Concepts You're Demonstrating (from your IBM course):
- **RAG** â€” retrieval-augmented generation with ChromaDB
- **Embeddings** â€” converting text to vectors for semantic search
- **LangChain** â€” chaining retrieval + prompt + LLM
- **Agentic AI / Tool-Calling** â€” LLM that decides which tools to invoke
- **Prompt engineering** â€” structured prompts for financial analysis

#### Tools: All FREE
- `chromadb` â€” free, runs locally
- `langchain` + `langchain-google-genai` â€” free, open-source
- `sentence-transformers` â€” free, runs locally (no API calls for embeddings)
- Gemini API â€” free tier (1,500 requests/day)
- CryptoPanic API â€” free tier available

#### Resume Bullet:
> *"Built a RAG-powered Agentic AI market analyst using LangChain + ChromaDB + Google Gemini function-calling â€” agent autonomously retrieves relevant crypto news, queries ML predictions, and synthesizes a natural-language risk verdict. Directly applies IBM RAG & Agentic AI Professional Certificate skills."*

---

### âš¡ PHASE 5 â€” FastAPI Backend (Day 9-10)
> **Why:** Moving from Streamlit-only to FastAPI shows you understand **production ML deployment**, not just research notebooks.

#### What to Build:
- Create `api/` directory with:
  - `api/main.py` â€” FastAPI app
  - `api/routers/predict.py` â€” prediction endpoint
  - `api/routers/health.py` â€” health check endpoint
  - `api/schemas.py` â€” Pydantic request/response models
- Endpoints:
  - `GET /health` â€” returns `{"status": "ok", "model_version": "..."}`
  - `POST /predict` â€” accepts OHLCV + features, returns prediction + confidence + SHAP
  - `GET /model-info` â€” returns model metadata from MLflow
  - `GET /latest-prediction` â€” returns last live prediction

#### Key Concepts You'll Learn:
- **REST API design** â€” endpoints, HTTP methods, status codes
- **Pydantic validation** â€” type-safe request/response
- **Model serving** â€” loading a trained model and making it an API

#### Resume Bullet:
> *"Built FastAPI REST backend serving ML model predictions with Pydantic validation, SHAP explanations per request, and MLflow model registry integration"*

---

### ðŸ³ PHASE 6 â€” Docker + GitHub Actions CI/CD (Day 11-13)
> **Why:** Docker is **non-negotiable** in 2025 for any production ML role. CI/CD shows software engineering maturity.

#### What to Build:

**Docker:**
- `Dockerfile` for the FastAPI backend
- `Dockerfile.streamlit` for the Streamlit frontend  
- `docker-compose.yml` to run both together + MLflow server

**GitHub Actions CI/CD (`.github/workflows/`):**
- `ci.yml` â€” runs on every push:
  - Lint with `flake8` or `ruff`
  - Run unit tests with `pytest`
  - Build Docker image (validates it)
- `cd.yml` â€” runs on merge to main:
  - Deploy to HuggingFace Spaces (free)

#### Key Concepts You'll Learn:
- **Containerization** â€” making your app environment-independent
- **CI/CD pipeline** â€” automated testing and deployment
- **Docker Compose** â€” multi-service orchestration

#### Resume Bullet:
> *"Containerized full ML stack with Docker Compose (FastAPI + Streamlit + MLflow) and implemented GitHub Actions CI/CD pipeline for automated testing and deployment"*

---

### ðŸ§ª PHASE 7 â€” Unit Tests (Day 13-14)
> **Why:** Tests are what separate **professional developers from beginners**. Interviewers ask about this.

#### What to Build:
- `tests/` directory with:
  - `test_feature_engineering.py` â€” test feature output shapes and values
  - `test_api.py` â€” test FastAPI endpoints with `httpx`/`TestClient`
  - `test_backtesting.py` â€” test Sharpe ratio calculation, fee math
- Use `pytest` as the test runner
- Add **coverage badge** to README

#### Key Concepts You'll Learn:
- **Unit testing** for data science code
- **API testing** with FastAPI TestClient
- **Test-driven thinking**

---

### ðŸš€ PHASE 8 â€” Deployment (Day 14-15)
> **Why:** A LIVE, clickable link in your resume/portfolio is 10x more impressive than "I built this locally."

#### Options (Pick ONE):
1. **HuggingFace Spaces** (Recommended â€” Free, easy, impressive for ML)
   - Deploy Streamlit app directly
   - Add model files as HF datasets
2. **Streamlit Cloud** (Simplest â€” Free tier available)
3. **Render.com** (For FastAPI backend â€” Free tier)

#### What to Do:
- Deploy Streamlit app to HuggingFace Spaces (free)
- Deploy FastAPI to Render.com (free)
- Add **live demo link** to GitHub README

---

### âœ¨ PHASE 9 â€” Professional GitHub & Documentation (Day 15-16)
> **Why:** Your GitHub IS your resume for tech companies. Bad README = rejected before interview.

#### What to Build:
- **Rewrite README.md** with:
  - Project demo GIF/screenshot
  - Architecture diagram (Mermaid)
  - Live demo link
  - Honest tech stack (not aspirational)
  - How to run locally
  - GitHub Actions badge
  - Coverage badge
- Add `CONTRIBUTING.md`
- Add proper `LICENSE` (MIT)
- Pin dependencies properly in `requirements.txt`
- Add GitHub Topics: `machine-learning`, `crypto`, `fastapi`, `streamlit`, `mlops`, `xgboost`, `shap`

---

## ðŸŽ¯ INTERVIEW PREPARATION â€” Questions You WILL Be Asked

### MLflow Questions:
- *"How did you track experiments?"* â†’ "I used MLflow to log hyperparameters, metrics, and model artifacts for every training run, and registered the best model in MLflow Model Registry."
- *"What is a model registry?"* â†’ "It's a central store for versioned models with staging/production/archived states â€” it lets you promote a model from experiment to production safely."

### SHAP Questions:
- *"How do you explain your model's predictions?"* â†’ "I use SHAP values based on Shapley game theory â€” each feature gets a contribution score for a specific prediction, not just global importance."
- *"What were the most important features?"* â†’ "RSI, Fear & Greed Index, and lagged returns had the highest mean absolute SHAP values globally."

### Optuna Questions:
- *"How did you tune your model?"* â†’ "I used Optuna's Bayesian optimization with TPESampler and MedianPruner â€” it intelligently searches the hyperparameter space and stops bad trials early."
- *"Why Optuna over GridSearch?"* â†’ "Grid search is exhaustive and slow. Optuna uses past trial results to decide where to search next â€” it's much more efficient."

### RAG Questions (from your IBM course â€” be confident here):
- *"What is RAG and why did you use it?"* â†’ "RAG stands for Retrieval-Augmented Generation. Instead of relying on the LLM's static training knowledge, I retrieve live, relevant crypto news from ChromaDB and feed it to Gemini as context â€” so the analysis is grounded in real current events."
- *"What vector database did you use?"* â†’ "ChromaDB â€” it's open-source, runs locally, and stores sentence-transformer embeddings of news headlines for semantic search."
- *"How does the retrieval work?"* â†’ "I embed the current market situation description as a vector, then do a cosine similarity search against stored news embeddings to find the top-3 most contextually relevant articles."

### Agentic AI Questions (from your IBM course â€” be confident here):
- *"What is an AI agent?"* â†’ "An AI agent is an LLM that can decide which tools to call based on the task, execute them, and synthesize the results â€” unlike a simple LLM that just generates text."
- *"How did you implement the agent?"* â†’ "I used Gemini's function-calling API. I defined 4 tools â€” Fear & Greed, ML prediction, on-chain metrics, and RAG news search. The agent receives a question, autonomously decides which tools are relevant, calls them, and generates a final market verdict."
- *"What framework did you use?"* â†’ "LangChain for the RAG pipeline and Gemini's native function-calling for the agent. I chose not to use LangGraph because my workflow is linear â€” LangGraph adds complexity only needed for stateful, looping workflows."

### FastAPI Questions:
- *"How is your model deployed?"* â†’ "As a REST API using FastAPI with Pydantic validation, served via uvicorn. It exposes /predict, /health, and /model-info endpoints."

### Docker Questions:
- *"How do you ensure reproducibility?"* â†’ "I containerized the entire stack with Docker and docker-compose â€” FastAPI backend, Streamlit frontend, and MLflow server all run as separate containers."

---

## ðŸ“‹ COMPLETE TASK CHECKLIST

### Phase 0 â€” Fixes (Day 1-2)
- [ ] Fix duplicate on-chain merge bug in `scripts/prepare_data.py`
- [ ] Align feature lists in `feature_engineering.py` and `app.py`
- [ ] Update `requirements.txt` with all missing packages
- [ ] Create `.env.example` file
- [ ] Consolidate config into `config.py`

### Phase 1 â€” MLflow (Day 3-4)
- [ ] Install `mlflow`
- [ ] Wrap model training with `mlflow.start_run()`
- [ ] Log params, metrics, model artifact
- [ ] Register model in MLflow Model Registry
- [ ] Verify `mlflow ui` works

### Phase 2 â€” Optuna (Day 4-5)
- [ ] Install `optuna`
- [ ] Create Optuna study for XGBoost
- [ ] Define search space (5-6 params)
- [ ] Log best params to MLflow
- [ ] Generate optimization history plot

### Phase 3 â€” SHAP (Day 5-6)
- [ ] Install `shap`
- [ ] Generate SHAP summary plot (save as image)
- [ ] Generate SHAP waterfall plot for individual predictions
- [ ] Add SHAP section to Streamlit app

### Phase 4 â€” RAG + LangChain + Agentic AI (Day 7-9)
**Part A: RAG Pipeline**
- [ ] Install `chromadb`, `langchain`, `langchain-google-genai`, `sentence-transformers`
- [ ] Create `src/rag/news_store.py` â€” fetch + embed + store crypto news in ChromaDB
- [ ] Create `src/rag/retriever.py` â€” semantic search over stored news
- [ ] Verify ChromaDB retrieval works locally

**Part B: LangChain Pipeline**
- [ ] Create `src/agents/langchain_pipeline.py`
- [ ] Build RAG chain: retrieve news â†’ format prompt â†’ call Gemini
- [ ] Test chain end-to-end with real data

**Part C: Gemini Tool-Calling Agent**
- [ ] Create `src/agents/market_analyst_agent.py`
- [ ] Define 4 tools: `get_fear_greed`, `get_ml_prediction`, `get_onchain_metrics`, `search_recent_news`
- [ ] Build Gemini function-calling agent
- [ ] Integrate agent verdict into Streamlit dashboard
- [ ] Test: agent should call tools autonomously and return a synthesized verdict

### Phase 5 â€” FastAPI (Day 9-10)
- [ ] Create `api/` directory structure
- [ ] Build `/health`, `/predict`, `/model-info` endpoints
- [ ] Add Pydantic schemas
- [ ] Test with `uvicorn api.main:app --reload`
- [ ] Test with Swagger UI at `/docs`

### Phase 6 â€” Docker + CI/CD (Day 11-13)
- [ ] Write `Dockerfile` for API
- [ ] Write `Dockerfile.streamlit` for frontend
- [ ] Write `docker-compose.yml`
- [ ] Write `.github/workflows/ci.yml`
- [ ] Write `.github/workflows/cd.yml`
- [ ] Verify GitHub Actions runs green âœ…

### Phase 7 â€” Tests (Day 13-14)
- [ ] Create `tests/` directory
- [ ] Write `test_feature_engineering.py`
- [ ] Write `test_api.py`
- [ ] Write `test_backtesting.py`
- [ ] Run `pytest --cov` and get >60% coverage

### Phase 8 â€” Deployment (Day 14-15)
- [ ] Deploy Streamlit to HuggingFace Spaces
- [ ] Deploy FastAPI to Render.com
- [ ] Add live demo URLs to README

### Phase 9 â€” Documentation (Day 15-16)
- [ ] Rewrite README.md professionally
- [ ] Add architecture diagram
- [ ] Add demo GIF
- [ ] Add all badges (CI, coverage, deployed)
- [ ] Add GitHub Topics

---

## ðŸ› ï¸ Final Tech Stack (Resume-Ready)

```
Data:           Binance API, CCXT, Pandas, NumPy
Features:       TA-Lib, custom indicators, Fear & Greed, on-chain data
ML Models:      XGBoost, SVC (Scikit-learn)
Tuning:         Optuna (Bayesian optimization)
Tracking:       MLflow (experiment tracking + model registry)
Explainability: SHAP (feature attribution)
RAG:            ChromaDB (vector store) + sentence-transformers (embeddings)
Orchestration:  LangChain (RAG chain: retrieve â†’ prompt â†’ LLM)
Agentic AI:     Gemini Function-Calling Agent + 4 custom tools
GenAI:          Google Gemini API (free tier)
Backend API:    FastAPI + Uvicorn + Pydantic
Frontend:       Streamlit + Plotly
Testing:        pytest + pytest-cov
DevOps:         Docker, Docker Compose, GitHub Actions CI/CD
Deployment:     HuggingFace Spaces (UI) + Render.com (API)
Certification:  IBM RAG & Agentic AI Professional Certificate (Coursera)
```

---

## ðŸ’¡ What To Say in Interviews

> *"Mudra is an end-to-end production ML system for crypto price prediction. I built a 40+ feature pipeline with technical, on-chain, and sentiment data. I trained XGBoost and SVC models tuned with Optuna and tracked all experiments with MLflow. I implemented SHAP for prediction explainability. For GenAI, I built a RAG pipeline using LangChain and ChromaDB that retrieves relevant crypto news as context, and a Gemini function-calling agent that autonomously uses tools â€” ML predictions, Fear & Greed index, on-chain metrics, and RAG news search â€” to generate a grounded natural-language market verdict. The system is served via FastAPI, containerized with Docker, deployed via GitHub Actions CI/CD, and live on HuggingFace Spaces."*

---

> [!IMPORTANT]
> **Start with Phase 0.** Do not skip straight to the fancy stuff. A project with clean, working code + 3 advanced features is MUCH better than a broken project with 8 half-implemented features.

> [!TIP]
> After completing each phase, **commit to GitHub immediately** with a meaningful commit message. Interviewers check your commit history!
