"""Mudra LLM & Gemini Post-Mortem Subsystem."""

import json
import logging
import os
import re
import sys
import time
import tomllib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore
        return lambda func: func

load_dotenv(".env")
logger = logging.getLogger("mudra.ai.llm")

GEMINI_MODEL_CANDIDATES = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-3.8-flash",
)

POSTMORTEM_PROMPT_TEMPLATE = """### ROLE
You are Mudra AI's Quantitative ML Error-Analysis Specialist. Diagnose why tabular BTC/USDT models succeeded or failed on a completed 1-hour candle.

### INPUT EVIDENCE (Grounded JSON)
```json
{evidence_json}
```

### STRICT RULES
1. Grounding: Never invent price, indicator values, or news headlines not present in JSON.
2. No Outcome Restatement: Do NOT simply restate the dollar error or UP/DOWN outcome.
3. Feature Attribution: Cite AT LEAST 2 SPECIFIC features/drivers (RSI, taker_buy_ratio, funding_rate, etc.).
4. Root Cause Driver: 2-4 word snake_case/CAPS label (e.g. TAKER_SELL_CONFIRMATION, OVERBOUGHT_RSI_REVERSAL).
5. Honest Unknown Rule: If unexplained by features/news, set root_cause_driver to UNEXPLAINED_OR_MISSING_SIGNAL and state current features do not explain the move.

### OUTPUT JSON SCHEMA
Respond strictly with valid JSON:
{{
  "market_summary": "1-2 sentence factual recap of move and order flow",
  "per_model": {{
    "<model_version>": {{
      "root_cause_driver": "CONCISE_LABEL",
      "diagnosis_summary": "1 concise sentence citing at least 2 specific features explaining why model succeeded/failed"
    }}
  }}
}}"""


def _sync_langsmith_env() -> None:
    # Loads LangSmith tracing env vars from secrets.toml outside pytest runs
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return
    secrets_path = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "secrets.toml"
    if not secrets_path.is_file():
        return
    try:
        data = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
        blocks = [data] + [data[s] for s in ("langsmith", "langchain") if isinstance(data.get(s), dict)]
        for block in blocks:
            for key in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
                if not os.getenv(key) and key in block:
                    val = block[key]
                    os.environ[key] = str(val).lower() if isinstance(val, bool) else str(val).strip()
        if os.getenv("LANGSMITH_TRACING", "").lower() == "true" and not os.getenv("LANGCHAIN_TRACING_V2"):
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
    except Exception as exc:
        logger.warning("Failed to parse secrets.toml for LangSmith: %s", type(exc).__name__)


_sync_langsmith_env()


def _log_llm_telemetry(endpoint: str, latency_ms: float, status_code: int, error_msg: Optional[str] = None) -> None:
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        return
    try:
        from src.database import get_db, log_telemetry
        with get_db() as session:
            log_telemetry(session, "gemini_api", endpoint, latency_ms, status_code, error_msg)
    except Exception:
        pass


class MarketAnalystReport(BaseModel):
    trade_thesis: str = Field(description="STRONG BULLISH, MODERATE BULLISH, NEUTRAL, MODERATE BEARISH, STRONG BEARISH")
    conviction_score: int = Field(description="Conviction rating 1-10")
    executive_summary: str = Field(description="Strategic executive summary")
    quantitative_driver_analysis: str = Field(description="Breakdown of top SHAP features")
    macro_market_context: str = Field(description="Retrieved domain knowledge")
    key_risks: List[str] = Field(description="Risk factors that invalidate thesis")


class ModelDiagnosis(BaseModel):
    root_cause_driver: str
    diagnosis_summary: str


class HourlyPostmortemOutput(BaseModel):
    market_summary: str
    per_model: Dict[str, ModelDiagnosis]


def _get_gemini_api_key() -> Optional[str]:
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(key)
        if val and val.strip() and val.strip() != "your_gemini_api_key_here":
            return val.strip()
    secrets_path = Path(__file__).resolve().parent.parent.parent / ".streamlit" / "secrets.toml"
    if secrets_path.is_file():
        try:
            sec = tomllib.loads(secrets_path.read_text(encoding="utf-8"))
            for section in ("google", "Google", "gemini", "Gemini"):
                if isinstance(sec.get(section), dict) and sec[section].get("api_key"):
                    v = str(sec[section]["api_key"]).strip()
                    if v and v != "your_gemini_api_key_here":
                        return v
            for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if key in sec and str(sec[key]).strip() != "your_gemini_api_key_here":
                    return str(sec[key]).strip()
        except Exception:
            pass
    return None


def _clean_json_response(raw_text: str) -> str:
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text.strip())
    return match.group(1).strip() if match else raw_text.strip()


@traceable(name="gemini_fallback_generation")
def generate_with_gemini_fallback(
    api_key: str,
    prompt: str,
    system_instruction: Optional[str] = None,
    timeout_sec: float = 20.0,
    total_deadline_sec: float = 45.0,
    return_model: bool = False,
    response_mime_type: Optional[str] = "application/json",
    response_schema: Optional[Any] = None,
    caller: str = "general",
) -> Union[str, Tuple[str, str]]:
    """Generates response via Google GenAI SDK with model candidate fallback and 429 cooldown."""
    if not api_key:
        raise ValueError("Missing Gemini API key. Cannot generate response.")

    from google import genai
    from google.genai import types

    custom_model = os.getenv("GEMINI_MODEL_NAME")
    candidates = (custom_model,) + tuple(m for m in GEMINI_MODEL_CANDIDATES if m != custom_model) if custom_model else GEMINI_MODEL_CANDIDATES
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)))
    cfg = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )

    t0, last_exc = time.perf_counter(), None
    for model_name in candidates:
        if (time.perf_counter() - t0) >= total_deadline_sec:
            break
        start_cand = time.perf_counter()
        try:
            resp = client.models.generate_content(model=model_name, contents=prompt, config=cfg)
            lat_ms = (time.perf_counter() - start_cand) * 1000.0
            text_out = getattr(resp, "text", None)
            if text_out and str(text_out).strip():
                _log_llm_telemetry(f"{model_name}:{caller}", lat_ms, 200)
                return (str(text_out).strip(), model_name) if return_model else str(text_out).strip()
            _log_llm_telemetry(f"{model_name}:{caller}", lat_ms, 503, "EMPTY_RESPONSE")
        except AssertionError:
            raise
        except Exception as exc:
            last_exc = exc
            lat_ms = (time.perf_counter() - start_cand) * 1000.0
            err_msg = str(exc)
            status_code = 429 if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) else (404 if ("404" in err_msg or "NOT_FOUND" in err_msg) else 503)
            _log_llm_telemetry(f"{model_name}:{caller}", lat_ms, status_code, err_msg[:300])
            logger.warning("Gemini candidate %s failed: %s", model_name, exc)
            if status_code == 429:
                time.sleep(4.0)

    raise RuntimeError(f"All Gemini model candidates failed or timed out. Last error: {last_exc}")


def _build_mcp_snapshot(mcp_context: Any) -> Dict[str, Any]:
    raw_mcp = mcp_context if isinstance(mcp_context, dict) else {}
    raw_funding = raw_mcp.get("funding_rate_8h_pct")
    raw_basis = raw_mcp.get("futures_basis_usd")
    raw_headlines = [str(h) for h in (raw_mcp.get("news_headlines") or [])[:5]]
    return {
        "funding_rate_8h_pct": float(raw_funding) if raw_funding is not None else None,
        "futures_basis_usd": float(raw_basis) if raw_basis is not None else None,
        "derivatives_regime": str(raw_mcp.get("derivatives_regime") or "Unavailable"),
        "derivatives_source": str(raw_mcp.get("derivatives_source") or "none"),
        "news_headlines": raw_headlines,
        "news_headline_count": len(raw_headlines),
    }


def _unavailable_postmortem(
    error_msg: str, model_versions: List[str], outcomes: List[Dict[str, Any]],
    mcp_snapshot: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "gemini_status": "UNAVAILABLE",
        "model_used": None,
        "error_detail": error_msg,
        "market_summary": "Gemini unavailable (offline or missing key)",
        "per_model": {
            mv: {"root_cause_driver": "GEMINI_UNAVAILABLE", "diagnosis_summary": "Gemini API unavailable — manual review needed."}
            for mv in model_versions
        },
        "model_outcomes": outcomes,
        "mcp_snapshot": mcp_snapshot or {},
    }


@traceable(name="gemini_hourly_postmortem")
def generate_hourly_postmortem_with_gemini(
    tick_features: Dict[str, Any],
    mcp_context: Dict[str, Any],
    model_outcomes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generates grounded per-model post-mortem explanation using Gemini, failing honestly if offline."""
    model_versions = [str(m.get("model_version", "unknown")) for m in model_outcomes]
    mcp_snapshot = _build_mcp_snapshot(mcp_context)
    api_key = _get_gemini_api_key()
    if not api_key:
        return _unavailable_postmortem("Gemini API key missing or test mode active", model_versions, model_outcomes, mcp_snapshot)

    try:
        evidence_features = {k: v for k, v in tick_features.items() if isinstance(v, (int, float, str, bool))}
        evidence_payload = {"tick_features": evidence_features, "mcp_context": mcp_snapshot, "model_outcomes": model_outcomes}
        prompt = POSTMORTEM_PROMPT_TEMPLATE.format(evidence_json=json.dumps(evidence_payload, indent=2))

        raw_resp, model_used = generate_with_gemini_fallback(
            api_key, prompt, return_model=True, response_mime_type="application/json",
            timeout_sec=20.0, total_deadline_sec=45.0, caller="postmortem",
        )
        clean_text = _clean_json_response(raw_resp)
        try:
            parsed = HourlyPostmortemOutput.model_validate_json(clean_text)
            market_summary = parsed.market_summary
            raw_per_model = {k: v.model_dump() for k, v in parsed.per_model.items()}
        except Exception:
            raw_data = json.loads(clean_text, strict=False)
            market_summary = str(raw_data.get("market_summary", "Hourly reconciliation completed."))
            raw_per_model = raw_data.get("per_model", {})

        per_model = {}
        for mv in model_versions:
            m_data = raw_per_model.get(mv, {}) if isinstance(raw_per_model, dict) else {}
            per_model[mv] = {
                "root_cause_driver": str((m_data.get("root_cause_driver") if isinstance(m_data, dict) else None) or "TELEMETRY_DRIVER"),
                "diagnosis_summary": str((m_data.get("diagnosis_summary") if isinstance(m_data, dict) else None) or "Reconciliation completed."),
            }

        return {
            "gemini_status": "OK",
            "model_used": model_used,
            "error_detail": None,
            "market_summary": market_summary,
            "per_model": per_model,
            "model_outcomes": model_outcomes,
            "mcp_snapshot": mcp_snapshot,
        }
    except Exception as exc:
        logger.warning("Hourly postmortem generation failed: %s", exc)
        return _unavailable_postmortem(str(exc), model_versions, model_outcomes, mcp_snapshot)


def _generate_fallback_report(pred_price: float, pred_return: float, pred_move_text: str) -> MarketAnalystReport:
    return MarketAnalystReport(
        trade_thesis="NEUTRAL",
        conviction_score=1,
        executive_summary="Gemini API unavailable — deterministic ML forecast only.",
        quantitative_driver_analysis=f"Forecast: ${pred_price:,.2f} ({pred_return:+.2%}, {pred_move_text}). Deterministic models only.",
        macro_market_context="No LLM market commentary available (offline fallback).",
        key_risks=["LLM advisor offline.", "Relying strictly on mathematical fee hurdle and risk critic gates."],
    )


def generate_market_commentary(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]],
    rag_docs: Optional[List[str]] = None,
) -> MarketAnalystReport:
    """Generates structured market commentary via Gemini, falling back honestly if offline."""
    api_key = _get_gemini_api_key()
    if not api_key:
        return _generate_fallback_report(pred_price, pred_return, pred_move_text)

    try:
        pos_str = ", ".join(f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_pos_drivers) or "None"
        neg_str = ", ".join(f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_neg_drivers) or "None"
        rag_context_str = "\n".join(f"- {d}" for d in (rag_docs or []))
        prompt = (
            f"You are Mudra AI's Lead Quantitative Strategist.\n"
            f"Current BTC: ${current_price:,.2f}, Predicted: ${pred_price:,.2f} ({pred_return:+.4%}, {pred_move_text})\n"
            f"Bullish drivers: {pos_str}\nBearish drivers: {neg_str}\nRAG context: {rag_context_str}\n"
            f"Return strictly JSON matching MarketAnalystReport schema."
        )
        clean_text = _clean_json_response(
            str(generate_with_gemini_fallback(api_key, prompt, response_mime_type="application/json", timeout_sec=20.0, total_deadline_sec=45.0, caller="commentary"))
        )
        try:
            return MarketAnalystReport.model_validate_json(clean_text)
        except Exception:
            report_data = json.loads(clean_text, strict=False)
            valid_theses = {"STRONG BULLISH", "MODERATE BULLISH", "NEUTRAL", "MODERATE BEARISH", "STRONG BEARISH"}
            raw_thesis = str(report_data.get("trade_thesis", "NEUTRAL")).upper().strip()
            report_data["trade_thesis"] = raw_thesis if raw_thesis in valid_theses else "NEUTRAL"
            report_data["conviction_score"] = max(1, min(10, int(report_data.get("conviction_score", 5))))
            return MarketAnalystReport(**report_data)
    except Exception as exc:
        logger.warning("Market commentary generation failed: %s", exc)
        return _generate_fallback_report(pred_price, pred_return, pred_move_text)


def embed_texts(
    texts: List[str],
    api_key: Optional[str] = None,
    output_dimensionality: int = 768,
    caller: str = "rag_embed",
) -> List[List[float]]:
    """Generates semantic embeddings via Gemini gemini-embedding-001 and logs quota telemetry."""
    key = api_key or _get_gemini_api_key()
    if not key:
        raise RuntimeError("Missing Gemini API key for embeddings.")

    from google import genai
    from google.genai import types

    t0 = time.perf_counter()
    try:
        client = genai.Client(api_key=key)
        cfg = types.EmbedContentConfig(output_dimensionality=output_dimensionality)
        res = client.models.embed_content(model="gemini-embedding-001", contents=texts, config=cfg)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        embs = getattr(res, "embeddings", [getattr(res, "embedding", None)])
        _log_llm_telemetry(f"gemini-embedding-001:{caller}", lat_ms, 200)
        return [list(getattr(e, "values", e)) for e in embs if e is not None]
    except Exception as exc:
        lat_ms = (time.perf_counter() - t0) * 1000.0
        err_msg = str(exc)
        status_code = 429 if ("429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg) else 503
        _log_llm_telemetry(f"gemini-embedding-001:{caller}", lat_ms, status_code, err_msg[:300])
        logger.warning("Gemini embed_content failed (%s): %s", status_code, exc)
        raise RuntimeError(f"Gemini embed_content error ({status_code}): {exc}") from exc
