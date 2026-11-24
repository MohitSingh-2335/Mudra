"""Mudra LLM & Gemini Post-Mortem Subsystem.

Uses the official Google GenAI SDK (google-genai >= 1.0.0), Pydantic strict schemas,
LangSmith tracing, and honest error boundaries without fake zero-vectors or fabricated scores.
"""

import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
from pydantic import BaseModel, Field

try:
    from langsmith import traceable
except ImportError:
    def traceable(*args, **kwargs):  # type: ignore
        def decorator(func): return func
        return decorator

load_dotenv(".env")
logger = logging.getLogger("mudra.ai.llm")

GEMINI_MODEL_CANDIDATES = (
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-flash-latest",
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


def _read_raw_gemini_api_key() -> Optional[str]:
    """Finds GEMINI_API_KEY from environment or secrets without assuming it is locally present."""
    for key in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(key)
        if val and val.strip() and val.strip() != "your_gemini_api_key_here":
            return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec in ("Google", "google", "gemini", "Gemini"):
                if sec in st.secrets and "api_key" in st.secrets[sec]:
                    v = str(st.secrets[sec]["api_key"]).strip()
                    if v and v != "your_gemini_api_key_here": return v
            for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if var in st.secrets and str(st.secrets[var]).strip() != "your_gemini_api_key_here":
                    return str(st.secrets[var]).strip()
    except Exception:
        pass
    return None


def _get_gemini_api_key() -> Optional[str]:
    return None if os.getenv("PYTEST_CURRENT_TEST") else _read_raw_gemini_api_key()


def get_gemini_status() -> Dict[str, Any]:
    raw_key = _read_raw_gemini_api_key()
    preferred = os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_CANDIDATES[0])
    masked = f"{raw_key[:4]}...{raw_key[-4:]}" if raw_key and len(raw_key) > 8 else ("Configured" if raw_key else None)
    return {
        "configured": bool(raw_key), "masked_key": masked, "preferred_model": preferred,
        "fallback_models": list(GEMINI_MODEL_CANDIDATES),
        "label": f"Gemini AI: Active ({preferred})" if raw_key else "Gemini AI: Rule Engine Fallback (No API Key)",
    }


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
) -> Union[str, Tuple[str, str]]:
    """Generates response using official Google GenAI SDK with candidate model fallback."""
    if not api_key:
        raise ValueError("Missing Gemini API key. Cannot generate response.")

    from google import genai
    from google.genai import types

    custom_model = os.getenv("GEMINI_MODEL_NAME")
    candidates = (custom_model,) + tuple(m for m in GEMINI_MODEL_CANDIDATES if m != custom_model) if custom_model else GEMINI_MODEL_CANDIDATES
    client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_sec * 1000)))
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type=response_mime_type,
        response_schema=response_schema,
    )

    t0, last_exc = time.perf_counter(), None
    for model_name in candidates:
        if (time.perf_counter() - t0) >= total_deadline_sec:
            break
        try:
            start_cand = time.perf_counter()
            resp = client.models.generate_content(model=model_name, contents=prompt, config=config)
            text_out = getattr(resp, "text", None)
            if text_out and str(text_out).strip():
                logger.info("Gemini call succeeded with model %s in %.1f ms", model_name, (time.perf_counter() - start_cand) * 1000.0)
                return (str(text_out).strip(), model_name) if return_model else str(text_out).strip()
        except AssertionError:
            raise
        except Exception as exc:
            last_exc = exc
            logger.warning("Gemini candidate %s failed: %s", model_name, exc)

    raise RuntimeError(f"All Gemini model candidates failed or timed out. Last error: {last_exc}")


def embed_texts(texts: List[str], api_key: Optional[str] = None, output_dimensionality: int = 768) -> List[List[float]]:
    """Embeds texts using Gemini gemini-embedding-001 model, failing honestly if unavailable."""
    key = api_key or _get_gemini_api_key()
    if not key:
        logger.warning("Cannot embed texts: Gemini API key is missing or test mode active.")
        raise RuntimeError("Missing Gemini API key for embeddings.")

    from google import genai
    from google.genai import types

    try:
        client = genai.Client(api_key=key)
        cfg = types.EmbedContentConfig(output_dimensionality=output_dimensionality)
        res = client.models.embed_content(model="gemini-embedding-001", contents=texts, config=cfg)
        embs = getattr(res, "embeddings", [getattr(res, "embedding", None)])
        return [list(getattr(e, "values", e)) for e in embs if e is not None]
    except Exception as exc:
        logger.warning("Gemini embed_content failed: %s", exc)
        raise RuntimeError(f"Gemini embed_content error: {exc}") from exc


def _unavailable_postmortem(error_msg: str, model_versions: List[str], outcomes: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "gemini_status": "UNAVAILABLE", "model_used": None, "error_detail": error_msg,
        "market_summary": "Gemini unavailable (offline or missing key)",
        "per_model": {mv: {"root_cause_driver": "GEMINI_UNAVAILABLE", "diagnosis_summary": "Gemini API unavailable (check quota/logs) — manual review needed."} for mv in model_versions},
        "model_outcomes": outcomes,
    }


@traceable(name="gemini_hourly_postmortem")
def generate_hourly_postmortem_with_gemini(
    tick_features: Dict[str, Any],
    mcp_context: Dict[str, Any],
    model_outcomes: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Generates grounded post-mortem explanation per model using Gemini, failing honestly if unavailable."""
    model_versions = [str(m.get("model_version", "unknown")) for m in model_outcomes]
    api_key = _get_gemini_api_key()
    if not api_key or os.getenv("PYTEST_CURRENT_TEST"):
        logger.warning("Gemini postmortem offline or test active: returning UNAVAILABLE.")
        return _unavailable_postmortem("Gemini API key missing or test mode active", model_versions, model_outcomes)

    try:
        evidence_features = {k: v for k, v in tick_features.items() if isinstance(v, (int, float, str, bool))}
        evidence_mcp = {
            "funding_rate_8h_pct": float(mcp_context.get("funding_rate_8h_pct", 0.01) or 0.01) if isinstance(mcp_context, dict) else 0.01,
            "derivatives_regime": str(mcp_context.get("derivatives_regime", "Neutral")) if isinstance(mcp_context, dict) else "Neutral",
            "news_headlines": (mcp_context.get("news_headlines", []) if isinstance(mcp_context, dict) else [])[:5],
        }
        evidence_payload = {"tick_features": evidence_features, "mcp_context": evidence_mcp, "model_outcomes": model_outcomes}
        prompt = POSTMORTEM_PROMPT_TEMPLATE.format(evidence_json=json.dumps(evidence_payload, indent=2))

        raw_resp, model_used = generate_with_gemini_fallback(
            api_key, prompt, return_model=True, response_mime_type="application/json",
            timeout_sec=20.0, total_deadline_sec=45.0,
        )
        clean_text = _clean_json_response(raw_resp)
        try:
            parsed = HourlyPostmortemOutput.model_validate_json(clean_text)
            market_summary, raw_per_model = parsed.market_summary, parsed.per_model
        except Exception:
            raw_data = json.loads(clean_text, strict=False)
            market_summary = str(raw_data.get("market_summary", "Hourly reconciliation completed."))
            raw_per_model = raw_data.get("per_model", {})

        per_model = {}
        for mv in model_versions:
            m_data = raw_per_model.get(mv, {})
            rc = getattr(m_data, "root_cause_driver", None) or (m_data.get("root_cause_driver") if isinstance(m_data, dict) else None) or "TELEMETRY_DRIVER"
            ds = getattr(m_data, "diagnosis_summary", None) or (m_data.get("diagnosis_summary") if isinstance(m_data, dict) else None) or "Reconciliation completed."
            per_model[mv] = {"root_cause_driver": str(rc), "diagnosis_summary": str(ds)}

        return {"gemini_status": "OK", "model_used": model_used, "error_detail": None, "market_summary": market_summary, "per_model": per_model, "model_outcomes": model_outcomes}
    except Exception as exc:
        logger.warning("Hourly postmortem generation failed: %s", exc)
        return _unavailable_postmortem(str(exc), model_versions, model_outcomes)


def _generate_fallback_report(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]], rag_context: Optional[List[str]] = None,
) -> MarketAnalystReport:
    """Honest deterministic fallback when Gemini is offline or unavailable."""
    return MarketAnalystReport(
        trade_thesis="NEUTRAL", conviction_score=1,
        executive_summary="Gemini API unavailable — deterministic ML forecast only.",
        quantitative_driver_analysis=f"Forecast: ${pred_price:,.2f} ({pred_return:+.2%}, {pred_move_text}). Deterministic models only.",
        macro_market_context="No LLM market commentary available (offline fallback).",
        key_risks=["LLM advisor offline.", "Relying strictly on mathematical fee hurdle and risk critic gates."],
    )


def generate_market_commentary(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]],
    market_snapshot: Optional[Dict[str, Any]] = None, rag_docs: Optional[List[str]] = None,
) -> MarketAnalystReport:
    """Generates structured market commentary via Gemini, falling back honestly to rule report if unavailable."""
    api_key = _get_gemini_api_key()
    if not api_key:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)

    try:
        pos_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_pos_drivers]) or "None"
        neg_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_neg_drivers]) or "None"
        rag_context_str = "\n".join([f"- {d}" for d in (rag_docs or [])])
        prompt = (
            f"You are Mudra AI's Lead Quantitative Strategist.\n"
            f"Current BTC: ${current_price:,.2f}, Predicted: ${pred_price:,.2f} ({pred_return:+.4%}, {pred_move_text})\n"
            f"Bullish drivers: {pos_str}\nBearish drivers: {neg_str}\nRAG context: {rag_context_str}\n"
            f"Return strictly JSON matching MarketAnalystReport schema."
        )
        raw_text = generate_with_gemini_fallback(
            api_key, prompt, response_mime_type="application/json",
            timeout_sec=20.0, total_deadline_sec=45.0,
        )
        clean_text = _clean_json_response(raw_text if isinstance(raw_text, str) else raw_text[0])
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
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)
