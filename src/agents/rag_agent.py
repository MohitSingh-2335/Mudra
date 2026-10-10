# src/agents/rag_agent.py

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')

# Silence ChromaDB internal telemetry warnings
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import chromadb
from chromadb.config import Settings
from config import CHROMADB_DIR, CHROMA_COLLECTION_NAME

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
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self._seed_if_empty()

    def _seed_if_empty(self):
        """Pre-seeds the knowledge base if currently empty."""
        if self.collection.count() == 0:
            print("🌱 Seeding ChromaDB with foundational market domain knowledge...")
            ids = [item["id"] for item in FOUNDATIONAL_KNOWLEDGE]
            documents = [item["doc"] for item in FOUNDATIONAL_KNOWLEDGE]
            metadatas = [{"category": item["category"]} for item in FOUNDATIONAL_KNOWLEDGE]
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
            print(f"✅ Pre-seeded {len(ids)} knowledge documents into ChromaDB.")

    def add_document(self, doc_id, text, metadata=None):
        """Adds or updates a document in the vector store."""
        metadata = metadata or {}
        self.collection.upsert(ids=[doc_id], documents=[text], metadatas=[metadata])

    def retrieve_market_context(self, query, n_results=3):
        """
        Performs semantic similarity search over ChromaDB.
        Returns a list of relevant text excerpts.
        """
        results = self.collection.query(query_texts=[query], n_results=n_results)
        documents = results.get("documents", [[]])[0]
        return documents


_rag_instance = None

def get_rag_agent():
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
