# src/mcp_server.py
import logging
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import Any, Dict, List
import requests

# Ensure all logs go exclusively to stderr so stdout carries clean JSON-RPC
logging.basicConfig(stream=sys.stderr, level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("Mudra-Quant-Intelligence-MCP")
except ImportError:
    class _FallbackMCP:
        def __init__(self, name: str):
            self.name = name

        def tool(self):
            return lambda fn: fn

        def run(self):
            pass

    mcp = _FallbackMCP("Mudra-Quant-Intelligence-MCP")

from config import EXCHANGE_CONNECT_TIMEOUT, EXCHANGE_READ_TIMEOUT

DERIBIT_TICKER_URL = "https://www.deribit.com/api/v2/public/ticker?instrument_name=BTC-PERPETUAL"
KRAKEN_FUTURES_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"
CRYPTOCOMPARE_NEWS_URL = "https://min-api.cryptocompare.com/data/v2/news/?lang=EN&categories=BTC"
BROWSER_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"


def _format_derivatives_pulse(source: str, symbol: str, mark: float, idx: float, funding_raw: float) -> Dict[str, Any]:
    funding = funding_raw * 100.0
    sentiment = (
        "Crowded Longs (Positive Funding)" if funding > 0.015
        else ("Crowded Shorts (Negative Funding)" if funding < -0.005 else "Neutral Funding")
    )
    return {
        "status": "LIVE",
        "source": source,
        "symbol": symbol,
        "mark_price_usd": round(mark, 2),
        "spot_index_price_usd": round(idx, 2),
        "futures_basis_usd": round(mark - idx, 2),
        "funding_rate_8h_pct": round(funding, 6),
        "derivatives_regime": sentiment,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def get_live_btc_derivatives_pulse() -> Dict[str, Any]:
    """Fetches live BTC perpetual futures basis and 8h funding rate from Deribit or Kraken."""
    errors: List[str] = []

    try:
        resp = requests.get(DERIBIT_TICKER_URL, timeout=(EXCHANGE_CONNECT_TIMEOUT, EXCHANGE_READ_TIMEOUT), headers={"User-Agent": BROWSER_USER_AGENT})
        if resp.status_code == 200:
            data = resp.json().get("result", {})
            mark, idx, funding_raw = float(data.get("mark_price", 0.0)), float(data.get("index_price", 0.0)), float(data.get("funding_8h", 0.0))
            if mark > 0 and idx > 0:
                return _format_derivatives_pulse("deribit", "BTC-PERPETUAL", mark, idx, funding_raw)
        else:
            errors.append(f"deribit: HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"deribit: {exc}")

    try:
        resp = requests.get(KRAKEN_FUTURES_TICKERS_URL, timeout=(EXCHANGE_CONNECT_TIMEOUT, EXCHANGE_READ_TIMEOUT), headers={"User-Agent": BROWSER_USER_AGENT})
        if resp.status_code == 200:
            xbt = next((t for t in resp.json().get("tickers", []) if t.get("symbol") == "PF_XBTUSD"), None)
            if xbt:
                mark, idx, funding_raw = float(xbt.get("markPrice", 0.0)), float(xbt.get("indexPrice", 0.0)), float(xbt.get("fundingRate", 0.0))
                if mark > 0 and idx > 0:
                    return _format_derivatives_pulse("kraken_futures", "PF_XBTUSD", mark, idx, funding_raw)
            else:
                errors.append("kraken_futures: PF_XBTUSD ticker not found")
        else:
            errors.append(f"kraken_futures: HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"kraken_futures: {exc}")

    return {
        "status": "OFFLINE_FALLBACK",
        "source": "none",
        "symbol": "BTC-PERPETUAL",
        "mark_price_usd": None,
        "spot_index_price_usd": None,
        "futures_basis_usd": None,
        "funding_rate_8h_pct": None,
        "derivatives_regime": "Unavailable",
        "error": " ; ".join(errors) if errors else "DERIVATIVES_FEEDS_UNAVAILABLE",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@mcp.tool()
def get_recent_prediction_error_memories(limit: int = 5) -> List[Dict[str, Any]]:
    """Returns the most recent reconciled prediction outcomes from Neon."""
    from src.database import get_reconciliation_history
    df = get_reconciliation_history(limit=max(1, min(int(limit), 20)))
    if df.empty:
        return []
    return [
        {
            "timestamp": str(r.get("timestamp", "")),
            "direction_predicted": r.get("direction_predicted"),
            "direction_realized": r.get("direction_realized"),
            "direction_correct": bool(r.get("direction_correct")),
            "predicted_price": round(float(r.get("predicted_price") or 0.0), 2),
            "actual_price": round(float(r.get("actual_price_at_t_plus_1") or 0.0), 2),
            "dollar_error_usd": round(float(r.get("absolute_error_usd") or 0.0), 2),
            "diagnosis_tag": r.get("error_attribution_tag"),
        }
        for r in df.to_dict("records")
    ]


@mcp.tool()
def get_recent_btc_news_headlines(hours_back: int = 2, max_items: int = 5) -> Dict[str, Any]:
    """Fetches recent Bitcoin news headlines from CryptoCompare with CoinDesk/Cointelegraph RSS fallback."""
    import xml.etree.ElementTree as ET
    from datetime import timedelta
    from email.utils import parsedate_to_datetime

    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, int(hours_back)))
    cutoff_ts = cutoff.timestamp()
    headlines: List[Dict[str, Any]] = []
    errors: List[str] = []

    try:
        resp = requests.get(CRYPTOCOMPARE_NEWS_URL, timeout=(3.0, 5.0), headers={"User-Agent": BROWSER_USER_AGENT})
        if resp.status_code == 200:
            for item in resp.json().get("Data", []):
                title = str(item.get("title", "")).strip()
                pub_on = float(item.get("published_on", 0))
                if title and pub_on >= cutoff_ts:
                    headlines.append({
                        "title": title,
                        "published_utc": datetime.fromtimestamp(pub_on, tz=timezone.utc).isoformat(),
                        "source": str(item.get("source", "CryptoCompare")).strip(),
                    })
        else:
            errors.append(f"CryptoCompare: HTTP {resp.status_code}")
    except Exception as exc:
        errors.append(f"CryptoCompare: {exc}")

    if not headlines:
        feeds = [
            ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
            ("Cointelegraph", "https://cointelegraph.com/rss/tag/bitcoin"),
        ]
        for source_name, url in feeds:
            try:
                resp = requests.get(url, timeout=(3.0, 5.0), headers={"User-Agent": BROWSER_USER_AGENT})
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    for item in root.findall(".//item"):
                        title_elem, pub_elem = item.find("title"), item.find("pubDate")
                        title = title_elem.text.strip() if title_elem is not None and title_elem.text else ""
                        if not title or pub_elem is None or not pub_elem.text:
                            continue
                        try:
                            pub_dt = parsedate_to_datetime(pub_elem.text.strip())
                            pub_dt = pub_dt.replace(tzinfo=timezone.utc) if not pub_dt.tzinfo else pub_dt.astimezone(timezone.utc)
                        except Exception:
                            pub_dt = None
                        if pub_dt and pub_dt >= cutoff:
                            headlines.append({"title": title, "published_utc": pub_dt.isoformat(), "source": source_name})
                else:
                    errors.append(f"{source_name}: HTTP {resp.status_code}")
            except Exception as exc:
                errors.append(f"{source_name}: {exc}")

    headlines.sort(key=lambda x: str(x.get("published_utc", "")), reverse=True)
    headlines = headlines[:max_items]
    titles_only = [h["title"] for h in headlines]
    status = "LIVE" if headlines else ("OK_NO_HEADLINES" if not errors else "UNAVAILABLE")

    return {
        "status": status,
        "hours_back": int(hours_back),
        "count": len(titles_only),
        "headlines": titles_only,
        "articles": headlines,
        "error": " ; ".join(errors) if (status == "UNAVAILABLE" and errors) else None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    mcp.run()
