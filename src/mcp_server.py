# src/mcp/mudra_mcp_server.py
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Mudra-Quant-Intelligence-MCP")
except ImportError:
    class _FallbackMCP:
        def __init__(self, name: str): self.name = name
        def tool(self): return lambda fn: fn
        def run(self): pass
    mcp = _FallbackMCP("Mudra-Quant-Intelligence-MCP")

from config import EXCHANGE_CONNECT_TIMEOUT, EXCHANGE_READ_TIMEOUT
BINANCE_FUTURES_PREMIUM_URL = "https://fapi.binance.com/fapi/v1/premiumIndex?symbol=BTCUSDT"


@mcp.tool()
def get_live_btc_derivatives_pulse() -> Dict[str, Any]:
    """Fetches real-time BTC/USDT perpetual futures Mark Price, Index Price, and Funding Rate."""
    try:
        resp = requests.get(BINANCE_FUTURES_PREMIUM_URL, timeout=(EXCHANGE_CONNECT_TIMEOUT, EXCHANGE_READ_TIMEOUT))
        if resp.status_code == 200:
            data = resp.json()
            mark, idx, funding = float(data.get("markPrice", 0.0)), float(data.get("indexPrice", 0.0)), float(data.get("lastFundingRate", 0.0)) * 100.0
            sentiment = "Crowded Longs (Positive Funding)" if funding > 0.015 else ("Crowded Shorts (Negative Funding)" if funding < -0.005 else "Neutral Funding")
            return {
                "status": "LIVE", "symbol": "BTCUSDT-PERP", "mark_price_usd": round(mark, 2),
                "spot_index_price_usd": round(idx, 2), "futures_basis_usd": round(mark - idx, 2),
                "funding_rate_8h_pct": round(funding, 4), "derivatives_regime": sentiment,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
    except Exception:
        pass
    return {
        "status": "OFFLINE_FALLBACK", "symbol": "BTCUSDT-PERP", "mark_price_usd": 0.0,
        "spot_index_price_usd": 0.0, "futures_basis_usd": 0.0, "funding_rate_8h_pct": 0.0100,
        "derivatives_regime": "Neutral Funding (Offline Fallback)", "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def get_live_btc_spot_and_range() -> Dict[str, Any]:
    """Returns the latest confirmed 1h BTC/USDT spot candle and predicted price range."""
    from src.database import get_latest_market_ticks
    from src.engine import estimate_next_hour_price_range_usd
    df = get_latest_market_ticks(limit=48)
    if df.empty:
        return {"status": "OFFLINE_FALLBACK", "current_price": 65000.0, "predicted_range_usd": 650.0}
    current_price = float(df["close"].iloc[-1])
    price_24h_ago = float(df["close"].iloc[-24]) if len(df) >= 24 else current_price
    range_info = estimate_next_hour_price_range_usd(df, current_price)
    return {
        "status": "LIVE", "current_price": round(current_price, 2), "change_24h_usd": round(current_price - price_24h_ago, 2),
        "predicted_range_usd": range_info["predicted_range_usd"], "predicted_high_1h": range_info["predicted_high_1h"],
        "predicted_low_1h": range_info["predicted_low_1h"],
    }


@mcp.tool()
def get_recent_prediction_error_memories(limit: int = 5) -> List[Dict[str, Any]]:
    """Returns the most recent reconciled prediction outcomes from database."""
    from src.database import get_reconciliation_history
    df = get_reconciliation_history(limit=max(1, min(int(limit), 20)))
    if df.empty:
        return []
    return [{
        "timestamp": str(r.get("timestamp", "")), "direction_predicted": r.get("direction_predicted"),
        "direction_realized": r.get("direction_realized"), "direction_correct": bool(r.get("direction_correct")),
        "predicted_price": round(float(r.get("predicted_price") or 0.0), 2),
        "actual_price": round(float(r.get("actual_price_at_t_plus_1") or 0.0), 2),
        "dollar_error_usd": round(float(r.get("absolute_error_usd") or 0.0), 2),
        "diagnosis_tag": r.get("error_attribution_tag"),
    } for r in df.to_dict("records")]


@mcp.tool()
def get_spot_and_futures_trade_plan(
    current_price: float, predicted_direction: str, expected_return_pct: float,
    conviction_score: int = 5, volatility_range_pct: float = 1.0,
    risk_approved: bool = True, clears_hurdle: Optional[bool] = None,
) -> Dict[str, Any]:
    # returns spot hold-until-net-profit decision and marks leveraged futures out of scope
    from config import MIN_HURDLE_PCT
    from src.engine import evaluate_spot_hold_until_profit_decision
    from src.database import get_db

    clears = bool(clears_hurdle) if clears_hurdle is not None else (abs(float(expected_return_pct)) >= float(MIN_HURDLE_PCT))
    try:
        with get_db() as session:
            spot_plan = evaluate_spot_hold_until_profit_decision(
                session=session, current_price=current_price, predicted_direction=predicted_direction,
                proposed_action="BUY" if predicted_direction == "UP" else "SELL", clears_hurdle=clears,
            )
    except Exception:
        spot_plan = {"final_spot_action": "HOLD", "position_state": "DATABASE_OFFLINE_FALLBACK", "reason": "Database connection offline; retaining cash safety floor."}

    futures_plan = {
        "status": "OUT_OF_SCOPE",
        "summary": "Mudra is a spot BTC/USDT ML execution pipeline; leveraged futures trading is out of scope.",
    }
    return {"spot_plan": spot_plan, "futures_plan": futures_plan, "spot_hold_until_profit_plan": spot_plan, "futures_3x_isolated_plan": futures_plan}


@mcp.tool()
def get_recent_btc_news_headlines(hours_back: int = 2, max_items: int = 5) -> Dict[str, Any]:
    # fetches recent public bitcoin rss news headlines from coindesk and cointelegraph
    import xml.etree.ElementTree as ET
    from datetime import timedelta
    from email.utils import parsedate_to_datetime

    feeds = [
        ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
        ("Cointelegraph", "https://cointelegraph.com/rss/tag/bitcoin"),
    ]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours_back)))
    headlines: List[Dict[str, Any]] = []
    errors: List[str] = []

    for source_name, url in feeds:
        try:
            resp = requests.get(url, timeout=(3.0, 5.0), headers={"User-Agent": "Mudra-Quant/1.0"})
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item"):
                    title_elem = item.find("title")
                    pub_elem = item.find("pubDate")
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                    if not title:
                        continue
                    pub_dt = None
                    if pub_elem is not None and pub_elem.text:
                        try:
                            pub_dt = parsedate_to_datetime(pub_elem.text.strip())
                            if not pub_dt.tzinfo:
                                pub_dt = pub_dt.replace(tzinfo=timezone.utc)
                            else:
                                pub_dt = pub_dt.astimezone(timezone.utc)
                        except Exception:
                            pub_dt = None
                    if pub_dt and pub_dt >= cutoff:
                        headlines.append({
                            "title": title,
                            "published_utc": pub_dt.isoformat(),
                            "source": source_name,
                        })
        except Exception as exc:
            errors.append(f"{source_name}: {exc}")

    headlines.sort(key=lambda x: str(x.get("published_utc", "")), reverse=True)
    headlines = headlines[:max_items]

    if not headlines and errors:
        return {
            "status": "UNAVAILABLE",
            "headlines": [],
            "error": "; ".join(errors),
        }

    return {
        "status": "LIVE" if headlines else "OK_NO_HEADLINES",
        "headlines": headlines,
    }


if __name__ == "__main__":
    mcp.run()
