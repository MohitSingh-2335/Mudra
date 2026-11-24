"""
Mudra Multi-Agent Intelligence Layer & Episodic RAG Memory.
Backward-compatible re-export shim delegating to modularized src.ai subsystems.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import onnxruntime
except ImportError:
    pass

from typing import Any, Dict, List

from src.ai.llm import (
    GEMINI_MODEL_CANDIDATES,
    HourlyPostmortemOutput,
    MarketAnalystReport,
    ModelDiagnosis,
    POSTMORTEM_PROMPT_TEMPLATE,
    _clean_json_response,
    _generate_fallback_report,
    _get_gemini_api_key,
    _read_raw_gemini_api_key,
    embed_texts,
    generate_market_commentary,
    generate_with_gemini_fallback,
    get_gemini_status,
)
import src.ai.llm as _ai_llm

from src.ai.mcp_client import (
    MCPContextPayload,
    fetch_mcp_context,
)

from src.ai.rag import (
    FOUNDATIONAL_KNOWLEDGE,
    GeminiEmbeddingFunction,
    MarketKnowledgeRAG,
    RAGMemoryItem,
    get_rag_agent,
    get_rag_collection,
    hydrate_rag_from_neon,
    index_reconciled_outcome_memory,
    initialize_knowledge_base,
    query_similar_error_memories,
    retrieve_rag_context,
    get_rag_embedding_mode,
)

from src.ai.workflow import (
    ALLOWED_TRADE_ACTIONS,
    AgentState,
    build_trading_graph,
    calculate_technical_signals,
    create_initial_agent_state,
    fetch_historical_candles,
    get_live_market_data,
    get_model_return_forecast,
    get_multi_model_consensus,
    query_macro_sentiment_rag,
    risk_critic_node,
    run_trading_graph,
    strategist_node,
    trading_graph_app,
)


def generate_hourly_postmortem_with_gemini(
    tick_features: Dict[str, Any],
    mcp_context: Dict[str, Any],
    model_outcomes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Delegates to src.ai.llm while preserving any monkeypatches on agents."""
    current_gen = globals().get("generate_with_gemini_fallback", _ai_llm.generate_with_gemini_fallback)
    current_key = globals().get("_get_gemini_api_key", _ai_llm._get_gemini_api_key)
    saved_gen = _ai_llm.generate_with_gemini_fallback
    saved_key = _ai_llm._get_gemini_api_key
    try:
        _ai_llm.generate_with_gemini_fallback = current_gen
        _ai_llm._get_gemini_api_key = current_key
        return _ai_llm.generate_hourly_postmortem_with_gemini(tick_features, mcp_context, model_outcomes)
    finally:
        _ai_llm.generate_with_gemini_fallback = saved_gen
        _ai_llm._get_gemini_api_key = saved_key


__all__ = [
    # Layer 1: LLM
    "GEMINI_MODEL_CANDIDATES",
    "HourlyPostmortemOutput",
    "MarketAnalystReport",
    "ModelDiagnosis",
    "POSTMORTEM_PROMPT_TEMPLATE",
    "_clean_json_response",
    "_generate_fallback_report",
    "_get_gemini_api_key",
    "_read_raw_gemini_api_key",
    "embed_texts",
    "generate_hourly_postmortem_with_gemini",
    "generate_market_commentary",
    "generate_with_gemini_fallback",
    "get_gemini_status",
    # Layer 2: MCP Client
    "MCPContextPayload",
    "fetch_mcp_context",
    # Layer 3: RAG
    "FOUNDATIONAL_KNOWLEDGE",
    "GeminiEmbeddingFunction",
    "MarketKnowledgeRAG",
    "RAGMemoryItem",
    "get_rag_agent",
    "get_rag_collection",
    "hydrate_rag_from_neon",
    "index_reconciled_outcome_memory",
    "initialize_knowledge_base",
    "query_similar_error_memories",
    "retrieve_rag_context",
    # Layer 4: Workflow
    "ALLOWED_TRADE_ACTIONS",
    "AgentState",
    "build_trading_graph",
    "calculate_technical_signals",
    "create_initial_agent_state",
    "fetch_historical_candles",
    "get_live_market_data",
    "get_model_return_forecast",
    "get_multi_model_consensus",
    "query_macro_sentiment_rag",
    "risk_critic_node",
    "run_trading_graph",
    "strategist_node",
    "trading_graph_app",
]
