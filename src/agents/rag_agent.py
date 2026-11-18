# src/agents/rag_agent.py
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)

import chromadb
from chromadb.config import Settings
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from config import CHROMADB_DIR, CHROMA_COLLECTION_NAME

__all__ = [
    "MarketKnowledgeRAG", "get_rag_agent", "get_rag_collection",
    "initialize_knowledge_base", "retrieve_rag_context",
    "index_reconciled_outcome_memory", "query_similar_error_memories",
]


def _get_gemini_api_key() -> Optional[str]:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    return key.strip() if key and key.strip() != "your_gemini_api_key_here" else None


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or _get_gemini_api_key()
        if self.api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=self.api_key)
            except Exception:
                pass

    def __call__(self, input: Documents) -> Embeddings:
        if not self.api_key:
            return [[0.0] * 768 for _ in input]
        try:
            import google.generativeai as genai
            res = genai.embed_content(model="models/text-embedding-004", content=input, output_dimensionality=768)
            return res["embedding"]
        except Exception:
            return [[0.0] * 768 for _ in input]


FOUNDATIONAL_KNOWLEDGE = [
    {"id": "taker_ratio", "doc": "High Taker Buy Ratio signals aggressive buying and bullish momentum; low ratios indicate seller dominance.", "category": "microstructure"},
    {"id": "momentum_return", "doc": "Positive 6-hour mean return with expanding volatility confirms trend continuation.", "category": "trend"},
    {"id": "macro_liquidity", "doc": "Fed rate cuts and QE expand liquidity, serving as historical macro tailwinds for Bitcoin.", "category": "macro"},
    {"id": "etf_flows", "doc": "Sustained net Spot Bitcoin ETF inflows absorb miner issuance, establishing support floors.", "category": "institutional"},
    {"id": "fear_greed", "doc": "Extreme Fear (<25) often marks local bottoms; Extreme Greed (>75) indicates euphoric liquidation risk.", "category": "sentiment"},
    {"id": "onchain_activity", "doc": "Surges in on-chain transaction volume and miner hashrate precede major volatility expansions.", "category": "onchain"},
    {"id": "rsi_reversion", "doc": "Hourly RSI >70 indicates overbought conditions with mean-reversion pullback risk.", "category": "technical"},
    {"id": "halving_shock", "doc": "Halvings cut block subsidies by 50%, historically catalyzing bull markets over 6-18 months.", "category": "supply"},
]


class MarketKnowledgeRAG:
    def __init__(self, db_path: str = CHROMADB_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False, is_persistent=True))
        self.embedding_fn = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)
        if self.collection.count() == 0:
            self.collection.add(
                ids=[item["id"] for item in FOUNDATIONAL_KNOWLEDGE],
                documents=[item["doc"] for item in FOUNDATIONAL_KNOWLEDGE],
                metadatas=[{"category": item["category"]} for item in FOUNDATIONAL_KNOWLEDGE],
            )

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata or {}])

    def retrieve_market_context(self, query: str, n_results: int = 3) -> List[str]:
        results = self.collection.query(query_texts=[query], n_results=n_results)
        docs = results.get("documents")
        return docs[0] if docs and len(docs) > 0 else []


_rag_instance: Optional[MarketKnowledgeRAG] = None


def get_rag_agent() -> MarketKnowledgeRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MarketKnowledgeRAG()
    return _rag_instance


def get_rag_collection():
    try:
        return get_rag_agent().collection
    except Exception:
        return None


def initialize_knowledge_base() -> MarketKnowledgeRAG:
    return get_rag_agent()


def retrieve_rag_context(query: str, n_results: int = 3) -> List[str]:
    return get_rag_agent().retrieve_market_context(query=query, n_results=n_results)


def index_reconciled_outcome_memory(
    tick_id: int, timestamp_iso: str, market_regime: str, direction_predicted: str,
    direction_realized: str, direction_correct: bool, predicted_dollar_move: float,
    actual_dollar_move: float, absolute_error_usd: float, error_attribution_tag: str,
    plain_english_notes: str,
) -> bool:
    col = get_rag_collection()
    if col is None:
        return False
    doc_id = f"reconciled_tick_{int(tick_id)}"
    doc_text = f"[{timestamp_iso}] Regime: {market_regime} | Pred: {direction_predicted} (${predicted_dollar_move:+,.2f}) | Realized: {direction_realized} (${actual_dollar_move:+,.2f}) | Correct: {direction_correct} | Tag: {error_attribution_tag} | Notes: {plain_english_notes}"
    meta = {
        "type": "episodic_outcome", "tick_id": int(tick_id), "regime": str(market_regime or "UNKNOWN"),
        "direction_predicted": str(direction_predicted or "UP"), "direction_realized": str(direction_realized or "UP"),
        "direction_correct": bool(direction_correct), "error_attribution_tag": str(error_attribution_tag or "ACCURATE"),
        "absolute_error_usd": float(absolute_error_usd or 0.0),
    }
    try:
        col.upsert(ids=[doc_id], documents=[doc_text], metadatas=[meta])
        return True
    except Exception:
        return False


def query_similar_error_memories(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    col = get_rag_collection()
    if col is None:
        return []
    try:
        results = col.query(query_texts=[query_text], n_results=n_results, where={"type": "episodic_outcome"})
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]
    except Exception:
        return []
