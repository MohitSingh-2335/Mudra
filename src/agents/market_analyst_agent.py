import json
import logging
import os
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)
load_dotenv('.env')


class MarketAnalystReport(BaseModel):
    trade_thesis: str = Field(description="STRONG BULLISH, MODERATE BULLISH, NEUTRAL, MODERATE BEARISH, STRONG BEARISH")
    conviction_score: int = Field(description="Conviction rating 1-10")
    executive_summary: str = Field(description="Strategic executive summary")
    quantitative_driver_analysis: str = Field(description="Breakdown of top SHAP features")
    macro_market_context: str = Field(description="Retrieved domain knowledge")
    key_risks: List[str] = Field(description="Risk factors that invalidate thesis")


GEMINI_MODEL_CANDIDATES = ("gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash")


def _read_raw_gemini_api_key() -> Optional[str]:
    for key_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        val = os.getenv(key_name)
        if val and val.strip() and val.strip() != "your_gemini_api_key_here":
            return val.strip()
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for sec in ("Google", "google", "gemini", "Gemini"):
                if sec in st.secrets and "api_key" in st.secrets[sec]:
                    v = str(st.secrets[sec]["api_key"]).strip()
                    if v and v != "your_gemini_api_key_here":
                        return v
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


def generate_with_gemini_fallback(api_key: str, prompt: str, system_instruction: Optional[str] = None) -> str:
    # Generates LLM content with automatic tiered model fallback (2.5 -> 2.0 -> 1.5)
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    custom_model = os.getenv("GEMINI_MODEL_NAME")
    candidates = (custom_model,) + tuple(m for m in GEMINI_MODEL_CANDIDATES if m != custom_model) if custom_model else GEMINI_MODEL_CANDIDATES

    last_exc: Optional[Exception] = None
    for idx, model_name in enumerate(candidates):
        try:
            if system_instruction and idx == 0:
                try:
                    model = genai.GenerativeModel(model_name=model_name, system_instruction=system_instruction)
                except TypeError:
                    model = genai.GenerativeModel(model_name)
            else:
                model = genai.GenerativeModel(model_name)
            resp = model.generate_content(prompt)
            text_out = getattr(resp, "text", None)
            if text_out and str(text_out).strip():
                return str(text_out).strip()
        except AssertionError:
            raise
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_exc}")


def _clean_json_response(raw_text: str) -> str:
    text = raw_text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    return match.group(1).strip() if match else text


def _generate_fallback_report(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]], rag_context: List[str]
) -> MarketAnalystReport:
    is_bullish = "Upward" in pred_move_text or pred_return > 0
    thesis = "MODERATE BULLISH" if is_bullish else "MODERATE BEARISH"
    conviction = 7 if abs(pred_return) > 0.002 else 6
    pos_summary = ", ".join([f"{d['feature']} (+{d['shap']:.3f})" for d in top_pos_drivers[:2]]) or "steady order book flow"
    neg_summary = ", ".join([f"{d['feature']} ({d['shap']:.3f})" for d in top_neg_drivers[:2]]) or "minor overhead resistance"

    return MarketAnalystReport(
        trade_thesis=thesis, conviction_score=conviction,
        executive_summary=f"Mudra ensemble forecasts {thesis.lower()} toward ${pred_price:,.2f} ({pred_return:+.2%}).",
        quantitative_driver_analysis=f"Upside catalysts: {pos_summary}. Downward drag: {neg_summary}.",
        macro_market_context=f"Market principles: {rag_context[0] if rag_context else 'Order flow imbalance dominates intraday.'}",
        key_risks=["Volatility expansion causing stop-out.", "Rapid mean-reversion at local resistance."],
    )


def generate_market_commentary(
    current_price: float, pred_price: float, pred_return: float, pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]], top_neg_drivers: List[Dict[str, Any]],
    market_snapshot: Optional[Dict[str, Any]] = None
) -> MarketAnalystReport:
    # Synthesizes quantitative model forecast, SHAP drivers, and RAG into executive report
    from src.agents.rag_agent import get_rag_agent
    rag = get_rag_agent()
    query_terms = [pred_move_text] + ([top_pos_drivers[0].get("feature", "")] if top_pos_drivers else []) + ([top_neg_drivers[0].get("feature", "")] if top_neg_drivers else [])
    rag_docs = rag.retrieve_market_context(" ".join(query_terms).strip(), n_results=2)

    api_key = _get_gemini_api_key()
    if not api_key:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)

    try:
        pos_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_pos_drivers]) or "None"
        neg_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_neg_drivers]) or "None"
        rag_context_str = "\n".join([f"- {d}" for d in rag_docs])
        prompt = (
            f"You are Mudra AI's Lead Quantitative Strategist.\n"
            f"Current BTC: ${current_price:,.2f}, Predicted: ${pred_price:,.2f} ({pred_return:+.4%}, {pred_move_text})\n"
            f"Bullish drivers: {pos_str}\nBearish drivers: {neg_str}\nRAG context: {rag_context_str}\n"
            f"Return strictly JSON with keys: trade_thesis, conviction_score, executive_summary, quantitative_driver_analysis, macro_market_context, key_risks."
        )
        raw_text = generate_with_gemini_fallback(api_key, prompt)
        clean_text = raw_text.strip()
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean_text)
        report_data = json.loads(match.group(1).strip() if match else clean_text)

        valid_theses = {"STRONG BULLISH", "MODERATE BULLISH", "NEUTRAL", "MODERATE BEARISH", "STRONG BEARISH"}
        raw_thesis = str(report_data.get("trade_thesis", "NEUTRAL")).upper().strip()
        report_data["trade_thesis"] = raw_thesis if raw_thesis in valid_theses else "NEUTRAL"
        report_data["conviction_score"] = max(1, min(10, int(report_data.get("conviction_score", 5))))
        return MarketAnalystReport(**report_data)
    except Exception:
        return _generate_fallback_report(current_price, pred_price, pred_return, pred_move_text, top_pos_drivers, top_neg_drivers, rag_docs)
