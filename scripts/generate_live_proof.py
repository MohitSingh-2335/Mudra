import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
try:
    import onnxruntime
except ImportError:
    pass
os.environ.pop("PYTEST_CURRENT_TEST", None)

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure repository root is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Clear pytest flag to ensure live production execution
os.environ.pop("PYTEST_CURRENT_TEST", None)

from src.mcp_server import get_live_btc_derivatives_pulse, get_recent_btc_news_headlines
from src.agents import (
    fetch_mcp_context,
    generate_hourly_postmortem_with_gemini,
    hydrate_rag_from_neon,
    get_rag_collection,
    retrieve_rag_context,
    get_gemini_status,
)
from src.database import get_db, HourlyPostmortem, ModelPrediction, ApiTelemetry
from sqlalchemy import select, desc

proof = {
    "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    "environment": {
        "python_version": sys.version.split()[0],
        "active_gemini_status": get_gemini_status(),
    },
}

# 1. Real Live MCP Derivatives Pulse
print("[PROOF 1/5] Calling get_live_btc_derivatives_pulse()...", flush=True)
pulse = get_live_btc_derivatives_pulse()
proof["mcp_derivatives"] = pulse
print(f"  -> status: {pulse.get('status')}, source: {pulse.get('source')}, mark_price: ${pulse.get('mark_price_usd'):,.2f}", flush=True)

# 2. Real Live MCP News Headlines
print("[PROOF 2/5] Calling get_recent_btc_news_headlines(hours_back=6, max_items=5)...", flush=True)
news = get_recent_btc_news_headlines(hours_back=6, max_items=5)
proof["mcp_news"] = news
print(f"  -> status: {news.get('status')}, count: {news.get('count')}, first headline: {news.get('headlines', [''])[0][:60]}...", flush=True)

# 3. Real Live Gemini Postmortem Analysis
print("[PROOF 3/5] Calling generate_hourly_postmortem_with_gemini()...", flush=True)
mcp_ctx = fetch_mcp_context(limit=3)
postmortem = generate_hourly_postmortem_with_gemini(
    tick_features={"close": pulse.get("mark_price_usd", 60800.0), "volatility_range_pct": 1.45, "taker_buy_ratio": 0.54, "rsi_estimate": 62.0},
    mcp_context=mcp_ctx,
    model_outcomes=[
        {
            "model_version": "lgbm_v3.0.0",
            "predicted_price": round(pulse.get("mark_price_usd", 60800.0) * 1.0025, 2),
            "actual_price": pulse.get("mark_price_usd", 60800.0),
            "predicted_return_pct": 0.25,
            "actual_return_pct": -0.10,
            "direction_predicted": "UP",
            "direction_realized": "DOWN",
            "direction_correct": False,
            "error_usd": 150.0,
        },
        {
            "model_version": "xgb_v2.0.0",
            "predicted_price": round(pulse.get("mark_price_usd", 60800.0) * 0.9985, 2),
            "actual_price": pulse.get("mark_price_usd", 60800.0),
            "predicted_return_pct": -0.15,
            "actual_return_pct": -0.10,
            "direction_predicted": "DOWN",
            "direction_realized": "DOWN",
            "direction_correct": True,
            "error_usd": 30.0,
        },
    ],
)
proof["gemini_postmortem"] = postmortem
print(f"  -> gemini_status: {postmortem.get('gemini_status')}, model_used: {postmortem.get('model_used')}", flush=True)

# 4. Real Live ChromaDB RAG Hydration from Neon
print("[PROOF 4/5] Calling hydrate_rag_from_neon()...", flush=True)
hydrated_count = hydrate_rag_from_neon(limit=50)
col = get_rag_collection()
total_docs = col.count() if col else 0
sample_retrieved = retrieve_rag_context("Bitcoin derivatives funding and trend momentum", n_results=2)
proof["rag_hydration"] = {
    "hydrated_count": hydrated_count,
    "chromadb_total_docs": total_docs,
    "sample_retrieved_documents": sample_retrieved,
}
print(f"  -> hydrated_from_neon: {hydrated_count}, total_chroma_docs: {total_docs}", flush=True)

# 5. Real Live Neon Cloud Database Snapshot
print("[PROOF 5/5] Querying Neon PostgreSQL latest rows...", flush=True)
with get_db() as s:
    pms = s.execute(select(HourlyPostmortem).order_by(desc(HourlyPostmortem.id)).limit(3)).scalars().all()
    preds = s.execute(select(ModelPrediction).order_by(desc(ModelPrediction.id)).limit(6)).scalars().all()
    telems = s.execute(select(ApiTelemetry).order_by(desc(ApiTelemetry.id)).limit(4)).scalars().all()

    proof["neon_snapshot"] = {
        "latest_hourly_postmortems": [
            {
                "id": p.id,
                "tick_id": p.tick_id,
                "status": p.gemini_status,
                "news_count": len(p.news_headlines or []),
                "created_at": str(p.created_at),
            }
            for p in pms
        ],
        "latest_model_predictions": [
            {
                "id": x.id,
                "tick_id": x.tick_id,
                "model_version": x.model_version,
                "direction_predicted": x.direction_predicted,
                "direction_realized": x.direction_realized,
                "direction_correct": x.direction_correct,
                "diagnosis_summary": x.diagnosis_summary,
            }
            for x in preds
        ],
        "latest_api_telemetry": [
            {
                "id": t.id,
                "service_name": t.service_name,
                "endpoint": t.endpoint,
                "latency_ms": round(float(t.latency_ms), 1),
                "status_code": t.status_code,
                "error_message": t.error_message,
                "timestamp": str(t.timestamp),
            }
            for t in telems
        ],
    }
print(f"  -> snapshot rows: postmortems={len(pms)}, predictions={len(preds)}, telemetry={len(telems)}", flush=True)

out_file = ROOT_DIR / "data" / "proof" / "live_verification_proof.json"
out_file.parent.mkdir(parents=True, exist_ok=True)
out_file.write_text(json.dumps(proof, indent=2, default=str), encoding="utf-8")

print("\n" + "=" * 60)
print(f"SUCCESS: Raw live verification proof written to:\n{out_file.resolve()}")
print("=" * 60)
