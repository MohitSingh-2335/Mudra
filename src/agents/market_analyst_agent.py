# src/agents/market_analyst_agent.py
"""
Mudra Market Analyst Agent.

Synthesizes quantitative ML forecasts, local TreeSHAP feature attributions,
and ChromaDB macro domain knowledge into structured institutional market commentary.
Exposes core LLM gateway utilities for Node 1 (Strategist) and Node 2 (Risk Critic).
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

if hasattr(sys.stdout, "reconfigure") and sys.stdout.isatty():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Silence ChromaDB internal telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"
logging.getLogger("chromadb.telemetry").setLevel(logging.ERROR)
logging.getLogger("chromadb.telemetry.posthog").setLevel(logging.ERROR)

from pydantic import BaseModel, Field
from dotenv import load_dotenv

from src.agents.rag_agent import get_rag_agent

# Load environment variables if present
load_dotenv('.env')


class MarketAnalystReport(BaseModel):
    """Structured quantitative market commentary synthesized by Gemini AI + RAG."""
    trade_thesis: str = Field(description="One of: STRONG BULLISH, MODERATE BULLISH, NEUTRAL, MODERATE BEARISH, STRONG BEARISH")
    conviction_score: int = Field(description="Conviction rating from 1 to 10")
    executive_summary: str = Field(description="2-3 sentence strategic executive summary")
    quantitative_driver_analysis: str = Field(description="Breakdown of how the top SHAP features justify the numeric prediction")
    macro_market_context: str = Field(description="Synthesis of retrieved ChromaDB domain knowledge and market regime")
    key_risks: List[str] = Field(description="Specific risk factors that would invalidate the thesis")


GEMINI_MODEL_CANDIDATES = (
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
)


def _read_raw_gemini_api_key() -> Optional[str]:
    """
    Retrieves the Gemini API key with multi-tiered resolution:
    1. Environment variables: GEMINI_API_KEY or GOOGLE_API_KEY (.env)
    2. Active Streamlit runtime: st.secrets['Google']['api_key'] / st.secrets['gemini']['api_key']
    3. Standalone file fallback: direct reading of .streamlit/secrets.toml via Python 3.11's tomllib
    """
    try:
        load_dotenv(".env")
    except Exception:
        pass

    # 1. Environment variables
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if key and key.strip() and key.strip() != "your_gemini_api_key_here":
        return key.strip()

    # 2. Active Streamlit session
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for section in ["Google", "google", "gemini", "Gemini"]:
                if section in st.secrets and "api_key" in st.secrets[section]:
                    val = str(st.secrets[section]["api_key"]).strip()
                    if val and val != "your_gemini_api_key_here":
                        return val
            for sec_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if sec_name in st.secrets:
                    val = str(st.secrets[sec_name]).strip()
                    if val and val != "your_gemini_api_key_here":
                        return val
    except Exception:
        pass

    # 3. Direct .streamlit/secrets.toml check (for CLI / testing outside of Streamlit)
    try:
        secrets_path = os.path.join(".streamlit", "secrets.toml")
        if os.path.exists(secrets_path):
            import tomllib
            with open(secrets_path, "rb") as f:
                data = tomllib.load(f)
            for section in ["Google", "google", "gemini", "Gemini"]:
                if section in data and "api_key" in data[section]:
                    val = str(data[section]["api_key"]).strip()
                    if val and val != "your_gemini_api_key_here":
                        return val
            for sec_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                if sec_name in data:
                    val = str(data[sec_name]).strip()
                    if val and val != "your_gemini_api_key_here":
                        return val
    except Exception:
        pass

    return None


def _get_gemini_api_key() -> Optional[str]:
    """
    Safely retrieves GEMINI_API_KEY or GOOGLE_API_KEY from OS env, .env, or Streamlit secrets.
    During automated pytest execution (PYTEST_CURRENT_TEST), returns None unless explicitly
    monkeypatched by a test so deterministic unit tests never make live external API calls.
    """
    if os.getenv("PYTEST_CURRENT_TEST"):
        return None
    return _read_raw_gemini_api_key()


def get_gemini_status() -> Dict[str, Any]:
    """Returns diagnostic status of the Google Gemini API configuration for UI display."""
    raw_key = _read_raw_gemini_api_key()
    preferred_model = os.getenv("GEMINI_MODEL_NAME", GEMINI_MODEL_CANDIDATES[0])
    if raw_key:
        masked = f"{raw_key[:4]}...{raw_key[-4:]}" if len(raw_key) > 8 else "Configured"
        return {
            "configured": True,
            "masked_key": masked,
            "preferred_model": preferred_model,
            "fallback_models": list(GEMINI_MODEL_CANDIDATES),
            "label": f"Gemini AI: Active ({preferred_model})",
        }
    return {
        "configured": False,
        "masked_key": None,
        "preferred_model": preferred_model,
        "fallback_models": list(GEMINI_MODEL_CANDIDATES),
        "label": "Gemini AI: Rule Engine Fallback (No API Key)",
    }


def generate_with_gemini_fallback(
    api_key: str,
    prompt: str,
    system_instruction: Optional[str] = None,
) -> str:
    """
    Calls Google Gemini using google.generativeai with automatic model-version fallback
    (gemini-2.5-flash -> gemini-2.0-flash -> gemini-1.5-flash) so retired model endpoints
    (404 Not Found) seamlessly upgrade to active Gemini models while preserving unit test mocks.
    """
    import google.generativeai as genai
    genai.configure(api_key=api_key)

    custom_model = os.getenv("GEMINI_MODEL_NAME")
    candidates = (
        (custom_model,) + tuple(m for m in GEMINI_MODEL_CANDIDATES if m != custom_model)
        if custom_model
        else GEMINI_MODEL_CANDIDATES
    )

    last_exc: Optional[Exception] = None
    for idx, model_name in enumerate(candidates):
        try:
            if system_instruction and idx == 0:
                try:
                    model = genai.GenerativeModel(
                        model_name=model_name,
                        system_instruction=system_instruction,
                    )
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
            continue

    raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_exc}")


def _clean_json_response(raw_text: str) -> str:
    """Strips markdown code blocks from LLM output to extract pure JSON."""
    text = raw_text.strip()
    # Remove markdown code fences if present (e.g. ```json ... ```)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    return text


def _generate_fallback_report(
    current_price: float,
    pred_price: float,
    pred_return: float,
    pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]],
    top_neg_drivers: List[Dict[str, Any]],
    rag_context: List[str]
) -> MarketAnalystReport:
    """Deterministic fallback synthesis when the Gemini API is unreachable."""
    is_bullish = "Upward" in pred_move_text or pred_return > 0
    thesis = "MODERATE BULLISH" if is_bullish else "MODERATE BEARISH"
    conviction = 7 if abs(pred_return) > 0.002 else 6

    pos_summary = ", ".join([f"{d['feature']} (+{d['shap']:.3f})" for d in top_pos_drivers[:2]]) or "steady order book flow"
    neg_summary = ", ".join([f"{d['feature']} ({d['shap']:.3f})" for d in top_neg_drivers[:2]]) or "minor overhead resistance"

    exec_summary = (
        f"Mudra's quant ensemble forecasts a {thesis.lower()} continuation toward ${pred_price:,.2f} "
        f"({pred_return:+.2%}). Intraday momentum and order flow indicate short-term directional bias."
    )

    quant_analysis = (
        f"Primary upside catalysts identified by TreeSHAP: {pos_summary}. "
        f"Conversely, downward drag is observed from: {neg_summary}."
    )

    macro_context = (
        f"Retrieved Market Principles: {rag_context[0] if rag_context else 'Order flow imbalance and momentum dominate intraday Bitcoin regimes.'}"
    )

    risks = [
        "Sudden volatility expansion causing liquidation wicks against the prevailing trend.",
        "Rapid mean-reversion if aggressive taker flow exhausts at local resistance."
    ]

    return MarketAnalystReport(
        trade_thesis=thesis,
        conviction_score=conviction,
        executive_summary=exec_summary,
        quantitative_driver_analysis=quant_analysis,
        macro_market_context=macro_context,
        key_risks=risks
    )


def generate_market_commentary(
    current_price: float,
    pred_price: float,
    pred_return: float,
    pred_move_text: str,
    top_pos_drivers: List[Dict[str, Any]],
    top_neg_drivers: List[Dict[str, Any]],
    market_snapshot: Optional[Dict[str, Any]] = None
) -> MarketAnalystReport:
    """
    Synthesizes ML predictions, TreeSHAP drivers, and ChromaDB RAG knowledge into a structured report.
    """
    rag = get_rag_agent()
    market_snapshot = market_snapshot or {}

    # 1. Retrieve RAG domain context matching top drivers
    query_terms = [pred_move_text]
    if top_pos_drivers:
        query_terms.append(top_pos_drivers[0].get("feature", ""))
    if top_neg_drivers:
        query_terms.append(top_neg_drivers[0].get("feature", ""))
    rag_query = " ".join(query_terms).strip()

    rag_docs = rag.retrieve_market_context(rag_query, n_results=2)
    rag_context_str = "\n".join([f"- {d}" for d in rag_docs])

    # 2. Retrieve Gemini API key
    api_key = _get_gemini_api_key()
    if not api_key:
        print("[market_analyst_agent] No API key found, generating deterministic fallback report.")
        return _generate_fallback_report(
            current_price, pred_price, pred_return, pred_move_text,
            top_pos_drivers, top_neg_drivers, rag_docs
        )

    # 3. Call Gemini with Structured Prompt
    try:
        # Gemini model invocation handled via generate_with_gemini_fallback

        pos_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_pos_drivers]) or "None"
        neg_str = ", ".join([f"{d['feature']}: SHAP={d['shap']:+.4f}" for d in top_neg_drivers]) or "None"

        prompt = f"""
You are Mudra AI's Lead Quantitative Market Strategist for Bitcoin (BTC/USDT).
Analyze the following quantitative ML predictions, TreeSHAP feature attributions, and retrieved market domain knowledge.

=== QUANTITATIVE PREDICTION ===
* Current BTC Price: ${current_price:,.2f}
* Predicted Next Price: ${pred_price:,.2f}
* Predicted Hourly Return: {pred_return:+.4%}
* Directional Movement Forecast: {pred_move_text}

=== TREESHAP EXPLAINABILITY DRIVERS ===
* Bullish Drivers (pushed probability UP): {pos_str}
* Bearish Drivers (pushed probability DOWN): {neg_str}

=== RETRIEVED RAG DOMAIN KNOWLEDGE ===
{rag_context_str}

=== ADDITIONAL SNAPSHOT ===
{json.dumps(market_snapshot, indent=2)}

=== OUTPUT INSTRUCTIONS ===
Return a strictly valid JSON object conforming exactly to this schema:
{{
  "trade_thesis": "STRONG BULLISH" | "MODERATE BULLISH" | "NEUTRAL" | "MODERATE BEARISH" | "STRONG BEARISH",
  "conviction_score": <integer 1 to 10>,
  "executive_summary": "<2-3 sentence strategic executive summary>",
  "quantitative_driver_analysis": "<Detailed explanation of how the SHAP features justify the numeric prediction>",
  "macro_market_context": "<Synthesis incorporating the retrieved RAG domain knowledge and market regime>",
  "key_risks": ["<risk 1>", "<risk 2>", "<risk 3>"]
}}
Do not include any explanation outside the JSON object.
"""
        raw_text = generate_with_gemini_fallback(api_key, prompt)
        cleaned_json = _clean_json_response(raw_text)
        report_data = json.loads(cleaned_json)

        # Defensive clamping and normalization
        valid_theses = {"STRONG BULLISH", "MODERATE BULLISH", "NEUTRAL", "MODERATE BEARISH", "STRONG BEARISH"}
        raw_thesis = str(report_data.get("trade_thesis", "NEUTRAL")).upper().strip()
        report_data["trade_thesis"] = raw_thesis if raw_thesis in valid_theses else "NEUTRAL"
        raw_conv = report_data.get("conviction_score", 5)
        report_data["conviction_score"] = max(1, min(10, int(raw_conv)))

        return MarketAnalystReport(**report_data)

    except Exception as e:
        print(f"[market_analyst_agent] Gemini API call exception ({e}), falling back to deterministic synthesis.")
        return _generate_fallback_report(
            current_price, pred_price, pred_return, pred_move_text,
            top_pos_drivers, top_neg_drivers, rag_docs
        )


if __name__ == '__main__':
    print("[INFO] Testing Market Analyst Agent...")
    sample_pos = [{"feature": "taker_buy_ratio", "shap": 0.0932, "value": 0.65}, {"feature": "return_mean_6h", "shap": 0.0711, "value": 0.008}]
    sample_neg = [{"feature": "volume_lag_1", "shap": -0.0409, "value": 1200.0}]

    report = generate_market_commentary(
        current_price=62904.0,
        pred_price=63150.0,
        pred_return=0.0039,
        pred_move_text="Upward",
        top_pos_drivers=sample_pos,
        top_neg_drivers=sample_neg
    )

    print(f"\n[REPORT] Thesis: {report.trade_thesis} (Conviction: {report.conviction_score}/10)")
    print(f"[EXECUTIVE] Executive Summary:\n   {report.executive_summary}")
    print(f"\n[ANALYSIS] Quant Driver Analysis:\n   {report.quantitative_driver_analysis}")
    print(f"\n[MACRO] Macro Context:\n   {report.macro_market_context}")
    print("\n[RISKS] Key Risks:")
    for r in report.key_risks:
        print(f"   - {r}")

