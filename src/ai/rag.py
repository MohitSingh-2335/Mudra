import logging, os, sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from config import CHROMADB_DIR, CHROMA_COLLECTION_NAME

# Linux / Streamlit SQLite upgrade hook & silence ChromaDB telemetry
try:
    if sys.platform != "win32":
        __import__("pysqlite3"); sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception: pass
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)
logger = logging.getLogger("mudra.ai.rag")

try:
    import chromadb
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
    from chromadb.config import Settings
    HAS_CHROMADB = True
except Exception:
    chromadb, HAS_CHROMADB = None, False
    Documents, Embeddings = Any, Any  # type: ignore
    class EmbeddingFunction: pass  # type: ignore
    class Settings:
        def __init__(self, *args, **kwargs): pass


class RAGMemoryItem(BaseModel):
    doc_id: str
    document: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    distance: Optional[float] = None


class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self, api_key: Optional[str] = None):
        self.api_key, self._local_fallback = api_key, None

    def _hash_vector(self, text: str, dim: int = 768) -> List[float]:
        rng = np.random.RandomState(sum(ord(c) for c in text) % 10000 + 1)
        vec = rng.randn(dim).astype(float)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
            return [self._hash_vector(t) for t in texts]
        try:
            from src.ai.llm import embed_texts
            return embed_texts(texts, api_key=self.api_key, output_dimensionality=768)
        except Exception as exc:
            logger.warning("Gemini embed_texts failed, falling back to local: %s", exc)

        if self._local_fallback is None:
            try:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                self._local_fallback = DefaultEmbeddingFunction()
            except Exception: self._local_fallback = False
        if self._local_fallback:
            try:
                embs = self._local_fallback(texts)
                return [list(vec) + list(vec) if len(vec) == 384 else list(vec) for vec in embs]
            except Exception: pass
        return [self._hash_vector(t) for t in texts]


class MarketKnowledgeRAG:
    FOUNDATIONAL_KNOWLEDGE = [
        {"id": "taker_ratio", "doc": "High Taker Buy Ratio signals aggressive buying; low ratios indicate seller dominance.", "category": "microstructure"},
        {"id": "momentum_return", "doc": "Positive 6-hour mean return with expanding volatility confirms trend continuation.", "category": "trend"},
        {"id": "macro_liquidity", "doc": "Fed rate cuts and QE expand liquidity, serving as historical macro tailwinds for Bitcoin.", "category": "macro"},
        {"id": "etf_flows", "doc": "Sustained net Spot Bitcoin ETF inflows absorb miner issuance, establishing support floors.", "category": "liquidity"},
        {"id": "fear_greed", "doc": "Extreme Fear (<25) often marks local bottoms; Extreme Greed (>75) indicates euphoric liquidation risk.", "category": "sentiment"},
        {"id": "onchain_activity", "doc": "Surges in on-chain transaction volume and miner hashrate precede major volatility expansions.", "category": "onchain"},
        {"id": "rsi_reversion", "doc": "Hourly RSI >70 indicates overbought conditions with mean-reversion pullback risk.", "category": "technical"},
        {"id": "halving_shock", "doc": "Halvings cut block subsidies by 50%, historically catalyzing bull markets over 6-18 months.", "category": "supply"},
    ]

    def __init__(self, db_path: str = CHROMADB_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        self.has_chroma, self.collection = False, None
        if HAS_CHROMADB and chromadb is not None:
            try:
                Path(db_path).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(path=db_path, settings=Settings(anonymized_telemetry=False, is_persistent=True))
                self.embedding_fn = GeminiEmbeddingFunction()
                self.collection = self.client.get_or_create_collection(name=collection_name, embedding_function=self.embedding_fn)
                if self.collection.count() == 0:
                    self.collection.upsert(
                        ids=[item["id"] for item in self.FOUNDATIONAL_KNOWLEDGE],
                        documents=[item["doc"] for item in self.FOUNDATIONAL_KNOWLEDGE],
                        metadatas=[{"category": item["category"]} for item in self.FOUNDATIONAL_KNOWLEDGE],
                    )
                self.has_chroma = True
            except Exception as exc: logger.warning("ChromaDB initialization failed: %s", exc)

    def add_document(self, doc_id: str, text: str, metadata: Optional[dict] = None) -> None:
        if self.has_chroma and self.collection is not None:
            self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata or {}])

    def retrieve_market_context(self, query: str, n_results: int = 3) -> List[str]:
        if self.has_chroma and self.collection is not None:
            try:
                res = self.collection.query(query_texts=[query], n_results=n_results)
                docs = res.get("documents", [[]])[0]
                if docs: return docs
            except Exception: pass
        q = query.lower()
        matches = [k["doc"] for k in self.FOUNDATIONAL_KNOWLEDGE if any(w in k["doc"].lower() for w in q.split() if len(w) > 3)]
        return matches[:n_results] if matches else [self.FOUNDATIONAL_KNOWLEDGE[0]["doc"]]

    def hydrate_from_neon(self, limit: int = 50) -> int:
        if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules or not self.has_chroma or self.collection is None: return 0
        try:
            from src.database import ModelPrediction, get_db
            from sqlalchemy import or_, select
            with get_db() as s:
                rows = s.execute(select(ModelPrediction).where(or_(ModelPrediction.diagnosis_summary.isnot(None), ModelPrediction.error_attribution_tag.isnot(None))).order_by(ModelPrediction.id.desc()).limit(max(1, min(int(limit), 200)))).scalars().all()
                if not rows: return 0
                ids = [f"neon_pred_{r.id}" for r in rows]
                docs = [f"Model: {r.model_version} | Pred: {r.direction_predicted} (${float(r.predicted_price_1h or 0.0):,.2f}) | Actual: {r.direction_realized or 'N/A'} (${float(r.actual_price_at_t_plus_1 or 0.0):,.2f}) | Correct: {bool(r.direction_correct)} | Tag: {r.error_attribution_tag or 'NONE'} | Notes: {r.diagnosis_summary or f'Error: ${float(r.absolute_error_usd or 0.0):,.2f}'}" for r in rows]
                metas = [{"type": "episodic_outcome", "tick_id": int(r.tick_id or 0), "model_version": str(r.model_version or "unknown"), "direction_predicted": str(r.direction_predicted or "UP"), "direction_realized": str(r.direction_realized or "UP"), "direction_correct": bool(r.direction_correct), "error_attribution_tag": str(r.error_attribution_tag or "ACCURATE"), "absolute_error_usd": float(r.absolute_error_usd or 0.0)} for r in rows]
                self.collection.upsert(ids=ids, documents=docs, metadatas=metas)
                return len(ids)
        except Exception as exc:
            logger.debug("Neon hydration skipped: %s", exc)
            return 0

    def index_outcome_memory(self, tick_id: int, timestamp_iso: str, market_regime: str, direction_predicted: str, direction_realized: str, direction_correct: bool, predicted_dollar_move: float, actual_dollar_move: float, absolute_error_usd: float, error_attribution_tag: str, plain_english_notes: str) -> bool:
        if not self.has_chroma or self.collection is None: return False
        doc_id = f"reconciled_tick_{int(tick_id)}"
        doc_text = f"[{timestamp_iso}] Regime: {market_regime} | Pred: {direction_predicted} (${predicted_dollar_move:+,.2f}) | Realized: {direction_realized} (${actual_dollar_move:+,.2f}) | Correct: {direction_correct} | Tag: {error_attribution_tag} | Notes: {plain_english_notes}"
        meta = {"type": "episodic_outcome", "tick_id": int(tick_id), "regime": str(market_regime or "UNKNOWN"), "direction_predicted": str(direction_predicted or "UP"), "direction_realized": str(direction_realized or "UP"), "direction_correct": bool(direction_correct), "error_attribution_tag": str(error_attribution_tag or "ACCURATE"), "absolute_error_usd": float(absolute_error_usd or 0.0)}
        try:
            self.collection.upsert(ids=[doc_id], documents=[doc_text], metadatas=[meta])
            return True
        except Exception as exc:
            logger.warning("Index outcome memory failed: %s", exc)
            return False

    def query_memories(self, query_text: str, n_results: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.has_chroma or self.collection is None: return []
        try:
            kwargs: Dict[str, Any] = {"query_texts": [query_text], "n_results": n_results}
            if where: kwargs["where"] = where
            res = self.collection.query(**kwargs)
            ids, docs = res.get("ids", [[]])[0], res.get("documents", [[]])[0]
            metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else [{}] * len(docs)
            dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)
            return [RAGMemoryItem(doc_id=str(i), document=str(d), metadata=m or {}, distance=float(dist) if dist is not None else None).model_dump() for i, d, m, dist in zip(ids, docs, metas, dists)]
        except Exception as exc:
            logger.warning("RAG memory query failed: %s", exc)
            return []


FOUNDATIONAL_KNOWLEDGE = MarketKnowledgeRAG.FOUNDATIONAL_KNOWLEDGE
_rag_instance: Optional[MarketKnowledgeRAG] = None

def get_rag_agent() -> MarketKnowledgeRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MarketKnowledgeRAG()
        if not os.getenv("PYTEST_CURRENT_TEST") and "pytest" not in sys.modules:
            try: _rag_instance.hydrate_from_neon(limit=50)
            except Exception as e: logger.debug("Auto-hydration skipped: %s", e)
    return _rag_instance

def get_rag_collection(): return getattr(get_rag_agent(), "collection", None)
def initialize_knowledge_base() -> MarketKnowledgeRAG: return get_rag_agent()
def retrieve_rag_context(query: str, n_results: int = 3) -> List[str]: return get_rag_agent().retrieve_market_context(query, n_results)
def index_reconciled_outcome_memory(tick_id: int, timestamp_iso: str, market_regime: str, direction_predicted: str, direction_realized: str, direction_correct: bool, predicted_dollar_move: float, actual_dollar_move: float, absolute_error_usd: float, error_attribution_tag: str, plain_english_notes: str) -> bool:
    return get_rag_agent().index_outcome_memory(tick_id, timestamp_iso, market_regime, direction_predicted, direction_realized, direction_correct, predicted_dollar_move, actual_dollar_move, absolute_error_usd, error_attribution_tag, plain_english_notes)
def query_similar_error_memories(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    return get_rag_agent().query_memories(query_text=query_text, n_results=n_results, where={"type": "episodic_outcome"})
def hydrate_rag_from_neon(limit: int = 50) -> int: return get_rag_agent().hydrate_from_neon(limit)
