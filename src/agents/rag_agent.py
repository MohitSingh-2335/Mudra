# src/agents/rag_agent.py

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Silence ChromaDB internal telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

from typing import List, Optional
import google.generativeai as genai
import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config import CHROMADB_DIR, CHROMA_COLLECTION_NAME


def _get_gemini_api_key() -> Optional[str]:
    """Retrieves the Gemini API key across .env, st.secrets, and secrets.toml."""
    # 1. Environment variables
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key != "your_gemini_api_key_here":
        return key.strip()

    # 2. Active Streamlit session
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for section in ["Google", "google", "gemini", "Gemini"]:
                if section in st.secrets and "api_key" in st.secrets[section]:
                    return str(st.secrets[section]["api_key"]).strip()
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in st.secrets:
                return str(st.secrets["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    # 3. Standalone secrets.toml fallback
    try:
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            import tomllib
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
            for section in ["Google", "google", "gemini", "Gemini"]:
                if section in data and "api_key" in data[section]:
                    return str(data[section]["api_key"]).strip()
            if "GEMINI_API_KEY" in data:
                return str(data["GEMINI_API_KEY"]).strip()
            if "GOOGLE_API_KEY" in data:
                return str(data["GOOGLE_API_KEY"]).strip()
    except Exception:
        pass

    return None


class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Lightweight, pure Python embedding function using Google AI Studio's
    gemini-embedding-001 API (768-dimensional). Zero local PyTorch/ONNX dependencies.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: str = "models/gemini-embedding-001"):
        self.api_key = api_key or _get_gemini_api_key()
        self.model_name = model_name
        if self.api_key:
            genai.configure(api_key=self.api_key)

    def __call__(self, input: Documents) -> Embeddings:
        if not self.api_key:
            # Fallback if no API key is configured
            return [[0.0] * 768 for _ in input]

        try:
            res = genai.embed_content(
                model=self.model_name,
                content=input,
                output_dimensionality=768,
                task_type="retrieval_document"
            )
            return res["embedding"]
        except Exception as e:
            print(f"⚠️ Gemini Embedding API warning ({e}), using safe fallback.")
            return [[0.0] * 768 for _ in input]


# Foundational domain knowledge to pre-seed the vector store
FOUNDATIONAL_KNOWLEDGE = [
    {
        "id": "order_flow_taker_ratio",
        "doc": "High Taker Buy Ratio indicates aggressive market buyers paying the spread to take liquidity, signaling institutional buying pressure and bullish short-term momentum. Conversely, low taker buy ratios indicate seller-dominated order flow.",
        "category": "microstructure"
    },
    {
        "id": "momentum_mean_return_6h",
        "doc": "A positive 6-hour mean return combined with expanding volatility typically confirms an intraday trend continuation regime in Bitcoin markets, while negative mean returns indicate distribution.",
        "category": "trend_momentum"
    },
    {
        "id": "macro_fed_liquidity",
        "doc": "Federal Reserve interest rate cuts and quantitative easing expand global M2 liquidity, historically acting as strong macro tailwinds for Bitcoin. Hawkish rate hike cycles suppress valuations and increase correlation to US equities.",
        "category": "macro"
    },
    {
        "id": "spot_etf_flows",
        "doc": "Sustained net inflows into Spot Bitcoin ETFs absorb daily miner issuance and reduce liquid exchange reserves, creating structural upward price pressure and establishing strong support floors.",
        "category": "institutional"
    },
    {
        "id": "fear_greed_contrarian",
        "doc": "Extreme Fear (index below 25) historically correlates with local market bottoms and capitulation events, offering favorable risk-to-reward long entries. Extreme Greed (above 75) indicates retail euphoria and heightened risk of leverage liquidation flushes.",
        "category": "sentiment"
    },
    {
        "id": "onchain_miner_security",
        "doc": "Significant increases in on-chain transaction volume coupled with rising miner hash rate indicate strong network security and active capital transfer, often preceding major volatility expansions.",
        "category": "onchain"
    },
    {
        "id": "rsi_overbought_pullback",
        "doc": "An hourly RSI exceeding 70 indicates short-term overbought conditions with elevated probability of a temporary mean-reversion pullback, especially if volume is declining on the push higher.",
        "category": "technical"
    },
    {
        "id": "halving_supply_shock",
        "doc": "Bitcoin halving events reduce daily block subsidy rewards by 50%, restricting structural supply. Over 6-18 month horizons, halvings have historically catalyzed cyclical bull markets.",
        "category": "macro_supply"
    }
]


class MarketKnowledgeRAG:
    """Manages persistent ChromaDB vector storage and semantic retrieval for Mudra."""

    def __init__(self, db_path=CHROMADB_DIR, collection_name=CHROMA_COLLECTION_NAME):
        self.client = chromadb.PersistentClient(
            path=db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_fn = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn
        )
        self._seed_if_empty()

    def _seed_if_empty(self):
        """Pre-seeds the knowledge base if currently empty."""
        if self.collection.count() == 0:
            print("🌱 Seeding ChromaDB with foundational market domain knowledge via Gemini text-embedding-004...")
            ids = [item["id"] for item in FOUNDATIONAL_KNOWLEDGE]
            documents = [item["doc"] for item in FOUNDATIONAL_KNOWLEDGE]
            metadatas = [{"category": item["category"]} for item in FOUNDATIONAL_KNOWLEDGE]
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Pre-seeded {len(ids)} knowledge documents into ChromaDB.")

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None):
        """Adds or updates a document in the vector store."""
        metadata = metadata or {}
        self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])

    def retrieve_market_context(self, query: str, n_results: int = 3) -> List[str]:
        """
        Performs semantic similarity search over ChromaDB using Gemini embeddings.
        Returns a list of relevant text excerpts.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        documents = results.get("documents", [[]])[0]
        return documents


_rag_instance = None

def get_rag_agent() -> MarketKnowledgeRAG:
    """Singleton getter for MarketKnowledgeRAG."""
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MarketKnowledgeRAG()
    return _rag_instance


if __name__ == '__main__':
    rag = get_rag_agent()
    test_query = "bullish breakout with high taker volume and positive momentum"
    print(f"\n🔎 Testing Semantic RAG Retrieval for: '{test_query}'")
    context = rag.retrieve_market_context(test_query, n_results=2)
    for i, doc in enumerate(context, 1):
        print(f"   [{i}] {doc}")
