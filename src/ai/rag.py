import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from config import (
    CHROMADB_DIR,
    CHROMA_COLLECTION_NAME,
    CHROMA_GEMINI_COOLDOWN_SEC,
    CHROMA_GEMINI_COLLECTION_NAME,
)

# Linux / Streamlit SQLite upgrade hook & silence ChromaDB telemetry
try:
    if sys.platform != "win32":
        __import__("pysqlite3")
        sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
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
    chromadb, HAS_CHROMADB = False, False
    Documents, Embeddings = Any, Any  # type: ignore

    class EmbeddingFunction:  # type: ignore
        pass

    class Settings:
        def __init__(self, *args, **kwargs):
            pass


def _is_test_blocked() -> bool:
    # Disable external embedding calls during pytest unless a test explicitly enables mock RAG tests.
    return bool(os.getenv("PYTEST_CURRENT_TEST")) or (
        "pytest" in sys.modules and not os.getenv("MUDRA_TEST_GEMINI_RAG")
    )


class GeminiEmbeddingFunction(EmbeddingFunction):
    """Generates 768-dim embeddings using Google gemini-embedding-001."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.last_embedding_mode = "gemini-embedding-001"

    def __call__(self, input: Documents) -> Embeddings:
        from src.ai.llm import embed_texts
        embs = embed_texts(list(input), api_key=self.api_key, output_dimensionality=768, caller="rag_embed")
        self.last_embedding_mode = "gemini-embedding-001"
        return embs


class LocalMiniLMEmbedding(EmbeddingFunction):
    """Embeds texts with local CPU ONNX MiniLM (384-dim) and deterministic hash fallback in pytest."""

    def __init__(self):
        self._local_fn: Any = None
        self.last_embedding_mode = "hash_test_fallback" if _is_test_blocked() else "local_minilm"

    def _hash_vector(self, text: str, dim: int = 384) -> List[float]:
        rng = np.random.RandomState(sum(ord(c) for c in text) % 10000 + 1)
        vec = rng.randn(dim).astype(float)
        norm = np.linalg.norm(vec)
        return (vec / norm).tolist() if norm > 0 else vec.tolist()

    def _get_local_minilm(self) -> Any:
        if self._local_fn is None:
            try:
                from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
                self._local_fn = DefaultEmbeddingFunction()
            except Exception as exc:
                logger.warning("DefaultEmbeddingFunction unavailable, falling back to hash vectors: %s", exc)
                self._local_fn = False
        return self._local_fn

    def __call__(self, input: Documents) -> Embeddings:
        texts = list(input)
        if _is_test_blocked():
            self.last_embedding_mode = "hash_test_fallback"
            return [self._hash_vector(t, dim=384) for t in texts]

        local_fn = self._get_local_minilm()
        if local_fn:
            try:
                embs = local_fn(texts)
                self.last_embedding_mode = "local_minilm"
                return [list(vec) for vec in embs]
            except Exception as exc:
                logger.warning("Local MiniLM embedding failed: %s", exc)

        self.last_embedding_mode = "hash_test_fallback"
        return [self._hash_vector(t, dim=384) for t in texts]


def _fetch_neon_rag_records(limit: int = 15) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    from sqlalchemy import or_, select
    from src.database import ModelPrediction, get_db

    with get_db() as s:
        rows = s.execute(
            select(ModelPrediction)
            .where(or_(ModelPrediction.diagnosis_summary.isnot(None), ModelPrediction.error_attribution_tag.isnot(None)))
            .order_by(ModelPrediction.id.desc())
            .limit(max(1, min(int(limit), 200)))
        ).scalars().all()
        if not rows:
            return [], [], []
        ids = [f"neon_pred_{r.id}" for r in rows]
        docs = [
            f"Model: {r.model_version} | Pred: {r.direction_predicted} (${float(r.predicted_price_1h or 0.0):,.2f}) "
            f"| Actual: {r.direction_realized or 'N/A'} (${float(r.actual_price_at_t_plus_1 or 0.0):,.2f}) "
            f"| Correct: {bool(r.direction_correct)} | Tag: {r.error_attribution_tag or 'NONE'} "
            f"| Notes: {r.diagnosis_summary or f'Error: ${float(r.absolute_error_usd or 0.0):,.2f}'}"
            for r in rows
        ]
        metas = [
            {
                "type": "episodic_outcome",
                "tick_id": int(r.tick_id or 0),
                "model_version": str(r.model_version or "unknown"),
                "direction_predicted": str(r.direction_predicted or "UP"),
                "direction_realized": str(r.direction_realized or "UP"),
                "direction_correct": bool(r.direction_correct),
                "error_attribution_tag": str(r.error_attribution_tag or "ACCURATE"),
                "absolute_error_usd": float(r.absolute_error_usd or 0.0),
            }
            for r in rows
        ]
        return ids, docs, metas


def _upsert_missing_docs(col: Any, ids: List[str], docs: List[str], metas: List[Dict[str, Any]]) -> int:
    # Embed only new IDs so persistent Chroma collections do not burn duplicate embedding calls.
    if not ids:
        return 0
    existing_res = col.get(ids=ids)
    existing = set(existing_res.get("ids", []) if existing_res else [])
    new_items = [(i, d, m) for i, d, m in zip(ids, docs, metas) if i not in existing]
    if not new_items:
        return 0
    col.upsert(
        ids=[x[0] for x in new_items],
        documents=[x[1] for x in new_items],
        metadatas=[x[2] for x in new_items],
    )
    return len(new_items)


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

    def __init__(
        self,
        db_path: str = CHROMADB_DIR,
        collection_name: str = CHROMA_COLLECTION_NAME,
        gemini_collection_name: str = CHROMA_GEMINI_COLLECTION_NAME,
        cooldown_sec: int = CHROMA_GEMINI_COOLDOWN_SEC,
    ):
        self.has_chroma = False
        self.client: Any = None
        self.collection: Any = None
        self.local_collection: Any = None
        self._gemini_collection: Any = None
        self._gemini_collection_name = gemini_collection_name
        self._cooldown_sec = max(1, int(cooldown_sec))
        self.gemini_cooldown_until = 0.0
        self._gemini_hydrated = False
        self._local_hydrated = False
        self.embedding_fn = LocalMiniLMEmbedding()
        self._active_mode = self.embedding_fn.last_embedding_mode

        if HAS_CHROMADB and chromadb is not None:
            try:
                Path(db_path).mkdir(parents=True, exist_ok=True)
                self.client = chromadb.PersistentClient(
                    path=db_path,
                    settings=Settings(anonymized_telemetry=False, is_persistent=True),
                )
                self.local_collection = self.client.get_or_create_collection(
                    name=collection_name,
                    embedding_function=self.embedding_fn,
                )
                self._seed_foundational_knowledge(self.local_collection)
                self.collection = self.local_collection
                self._active_mode = self.embedding_fn.last_embedding_mode
                self.has_chroma = True
            except Exception as exc:
                logger.warning("ChromaDB initialization failed: %s", exc)

    def _seed_foundational_knowledge(self, col: Any) -> None:
        if col is not None and col.count() == 0:
            col.upsert(
                ids=[item["id"] for item in self.FOUNDATIONAL_KNOWLEDGE],
                documents=[item["doc"] for item in self.FOUNDATIONAL_KNOWLEDGE],
                metadatas=[{"category": item["category"]} for item in self.FOUNDATIONAL_KNOWLEDGE],
            )

    def _can_try_gemini(self) -> bool:
        if not self.has_chroma or self.client is None or _is_test_blocked():
            return False
        if time.time() < self.gemini_cooldown_until:
            return False
        from src.ai.llm import _get_gemini_api_key
        return bool(_get_gemini_api_key())

    def _get_gemini_collection(self) -> Any:
        # Keep 768-dim Gemini embeddings isolated from 384-dim MiniLM vectors to avoid dimension mismatch.
        if self._gemini_collection is None:
            self._gemini_collection = self.client.get_or_create_collection(
                name=self._gemini_collection_name,
                embedding_function=GeminiEmbeddingFunction(),
            )
        self._seed_foundational_knowledge(self._gemini_collection)
        return self._gemini_collection

    def _enter_fallback(self, exc: Exception) -> None:
        logger.warning("Gemini RAG unavailable, switching to local_minilm for %ss: %s", self._cooldown_sec, exc)
        self.gemini_cooldown_until = time.time() + self._cooldown_sec
        self.collection = self.local_collection
        self._active_mode = self.embedding_fn.last_embedding_mode

    def _hydrate_target(self, target_col: Any, limit: int) -> int:
        if _is_test_blocked() or target_col is None:
            return 0
        ids, docs, metas = _fetch_neon_rag_records(limit=limit)
        return _upsert_missing_docs(target_col, ids, docs, metas)

    def hydrate_from_neon(self, limit: int = 50) -> int:
        if _is_test_blocked() or not self.has_chroma:
            return 0

        if self._can_try_gemini():
            try:
                g_col = self._get_gemini_collection()
                added = self._hydrate_target(g_col, limit)
                self._gemini_hydrated = True
                self.collection = g_col
                self._active_mode = "gemini-embedding-001"
                return added
            except Exception as exc:
                self._enter_fallback(exc)

        try:
            added = self._hydrate_target(self.local_collection, limit)
            self._local_hydrated = True
            self.collection = self.local_collection
            self._active_mode = self.embedding_fn.last_embedding_mode
            return added
        except Exception as exc:
            logger.debug("Neon hydration skipped: %s", exc)
            return 0

    def _prepare_active_collection(self) -> Any:
        if self._can_try_gemini():
            try:
                g_col = self._get_gemini_collection()
                if not self._gemini_hydrated:
                    self._hydrate_target(g_col, limit=15)
                    self._gemini_hydrated = True
                self.collection = g_col
                self._active_mode = "gemini-embedding-001"
                return g_col
            except Exception as exc:
                self._enter_fallback(exc)

        if not self._local_hydrated and not _is_test_blocked():
            try:
                self._hydrate_target(self.local_collection, limit=15)
                self._local_hydrated = True
            except Exception as exc:
                logger.debug("Local Neon hydration skipped: %s", exc)
        self.collection = self.local_collection
        self._active_mode = self.embedding_fn.last_embedding_mode
        return self.local_collection

    def retrieve_market_context(self, query: str, n_results: int = 3) -> List[str]:
        if self.has_chroma:
            if self._can_try_gemini():
                try:
                    g_col = self._prepare_active_collection()
                    docs = g_col.query(query_texts=[query], n_results=n_results).get("documents", [[]])[0]
                    if docs:
                        return docs
                except Exception as exc:
                    self._enter_fallback(exc)
            try:
                l_col = self.local_collection
                self.collection = l_col
                docs = l_col.query(query_texts=[query], n_results=n_results).get("documents", [[]])[0]
                self._active_mode = self.embedding_fn.last_embedding_mode
                if docs:
                    return docs
            except Exception:
                pass
        q = query.lower()
        matches = [k["doc"] for k in self.FOUNDATIONAL_KNOWLEDGE if any(w in k["doc"].lower() for w in q.split() if len(w) > 3)]
        return matches[:n_results] if matches else [self.FOUNDATIONAL_KNOWLEDGE[0]["doc"]]

    def index_outcome_memory(
        self, tick_id: int, timestamp_iso: str, market_regime: str,
        direction_predicted: str, direction_realized: str, direction_correct: bool,
        predicted_dollar_move: float, actual_dollar_move: float,
        absolute_error_usd: float, error_attribution_tag: str, plain_english_notes: str,
    ) -> bool:
        if not self.has_chroma or self.local_collection is None:
            return False
        doc_id = f"reconciled_tick_{int(tick_id)}"
        doc_text = (
            f"[{timestamp_iso}] Regime: {market_regime} | Pred: {direction_predicted} (${predicted_dollar_move:+,.2f}) "
            f"| Realized: {direction_realized} (${actual_dollar_move:+,.2f}) | Correct: {direction_correct} "
            f"| Tag: {error_attribution_tag} | Notes: {plain_english_notes}"
        )
        meta = {
            "type": "episodic_outcome", "tick_id": int(tick_id), "regime": str(market_regime or "UNKNOWN"),
            "direction_predicted": str(direction_predicted or "UP"), "direction_realized": str(direction_realized or "UP"),
            "direction_correct": bool(direction_correct), "error_attribution_tag": str(error_attribution_tag or "ACCURATE"),
            "absolute_error_usd": float(absolute_error_usd or 0.0),
        }
        if self._can_try_gemini():
            try:
                g_col = self._get_gemini_collection()
                g_col.upsert(ids=[doc_id], documents=[doc_text], metadatas=[meta])
                self.collection = g_col
                self._active_mode = "gemini-embedding-001"
                return True
            except Exception as exc:
                self._enter_fallback(exc)
        try:
            self.local_collection.upsert(ids=[doc_id], documents=[doc_text], metadatas=[meta])
            self.collection = self.local_collection
            self._active_mode = self.embedding_fn.last_embedding_mode
            return True
        except Exception as exc:
            logger.warning("Index outcome memory failed: %s", exc)
            return False

    @property
    def embedding_mode(self) -> str:
        return self._active_mode

    @property
    def cooldown_remaining_sec(self) -> int:
        return max(0, int(self.gemini_cooldown_until - time.time()))

    def query_memories(self, query_text: str, n_results: int = 3, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.has_chroma or self.local_collection is None:
            return []
        kwargs: Dict[str, Any] = {"query_texts": [query_text], "n_results": n_results}
        if where:
            kwargs["where"] = where

        res = None
        if self._can_try_gemini():
            try:
                g_col = self._prepare_active_collection()
                res = g_col.query(**kwargs)
            except Exception as exc:
                self._enter_fallback(exc)

        if res is None:
            try:
                if not self._local_hydrated and not _is_test_blocked():
                    self._hydrate_target(self.local_collection, limit=15)
                    self._local_hydrated = True
                self.collection = self.local_collection
                res = self.local_collection.query(**kwargs)
                self._active_mode = self.embedding_fn.last_embedding_mode
            except Exception as exc:
                logger.warning("RAG memory query failed: %s", exc)
                return []

        ids, docs = res.get("ids", [[]])[0], res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0] if res.get("metadatas") else [{}] * len(docs)
        dists = res.get("distances", [[]])[0] if res.get("distances") else [None] * len(docs)
        return [
            {"doc_id": str(i), "document": str(d), "metadata": m or {}, "distance": float(dist) if dist is not None else None}
            for i, d, m, dist in zip(ids, docs, metas, dists)
        ]


_rag_instance: Optional[MarketKnowledgeRAG] = None


def get_rag_agent() -> MarketKnowledgeRAG:
    global _rag_instance
    if _rag_instance is None:
        _rag_instance = MarketKnowledgeRAG()
        if not _is_test_blocked():
            try:
                _rag_instance.hydrate_from_neon(limit=15)
            except Exception as e:
                logger.debug("Auto-hydration skipped: %s", e)
    return _rag_instance


def get_rag_collection():
    return getattr(get_rag_agent(), "collection", None)


def index_reconciled_outcome_memory(
    tick_id: int, timestamp_iso: str, market_regime: str, direction_predicted: str,
    direction_realized: str, direction_correct: bool, predicted_dollar_move: float,
    actual_dollar_move: float, absolute_error_usd: float, error_attribution_tag: str,
    plain_english_notes: str,
) -> bool:
    return get_rag_agent().index_outcome_memory(
        tick_id, timestamp_iso, market_regime, direction_predicted, direction_realized,
        direction_correct, predicted_dollar_move, actual_dollar_move, absolute_error_usd,
        error_attribution_tag, plain_english_notes,
    )


def query_similar_error_memories(query_text: str, n_results: int = 3) -> List[Dict[str, Any]]:
    return get_rag_agent().query_memories(query_text=query_text, n_results=n_results, where={"type": "episodic_outcome"})


def get_rag_embedding_mode() -> str:
    return getattr(get_rag_agent(), "embedding_mode", "uninitialized")


def get_rag_cooldown_remaining_sec() -> int:
    return getattr(get_rag_agent(), "cooldown_remaining_sec", 0)
