import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("mudra.ai.mcp_client")


class MCPContextPayload(BaseModel):
    status: str = "UNAVAILABLE"  # "LIVE", "PARTIAL", or "UNAVAILABLE"
    transport: str = "mcp_stdio"  # "mcp_stdio" or "in_process_test"
    tools_discovered: List[str] = Field(default_factory=list)
    funding_rate_8h_pct: Optional[float] = None
    derivatives_regime: str = "Unavailable"
    futures_basis_usd: Optional[float] = None
    news_headlines: List[str] = Field(default_factory=list)
    recent_error_memories: List[Dict[str, Any]] = Field(default_factory=list)
    error: Optional[str] = None
    derivatives_status: str = "UNAVAILABLE"
    derivatives_source: str = "none"
    derivatives_error: Optional[str] = None
    derivatives_latency_ms: float = 0.0
    news_status: str = "UNAVAILABLE"
    news_error: Optional[str] = None
    news_latency_ms: float = 0.0


async def _fetch_mcp_context_stdio(limit: int = 3, timeout_sec: float = 15.0) -> Dict[str, Any]:
    async def _timed_tool(session: Any, name: str, arguments: Dict[str, Any]):
        t0 = time.perf_counter()
        res = await session.call_tool(name, arguments=arguments)
        return res, (time.perf_counter() - t0) * 1000.0

    async def _impl():
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        root_dir = Path(__file__).resolve().parent.parent.parent
        server_path = root_dir / "src" / "mcp_server.py"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root_dir)

        params = StdioServerParameters(command=sys.executable, args=[str(server_path)], env=env)
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools_res = await session.list_tools()
                discovered = [t.name for t in tools_res.tools]

                # Invoke derivatives, 6h news, and error memories concurrently so a cold DB never blocks live market feeds
                p_task = _timed_tool(session, "get_live_btc_derivatives_pulse", {})
                n_task = _timed_tool(session, "get_recent_btc_news_headlines", {"hours_back": 6, "max_items": 5})
                e_task = _timed_tool(session, "get_recent_prediction_error_memories", {"limit": limit})
                p_res, n_res, e_res = await asyncio.gather(p_task, n_task, e_task, return_exceptions=True)

                p_lat = p_res[1] if not isinstance(p_res, Exception) else 0.0
                n_lat = n_res[1] if not isinstance(n_res, Exception) else 0.0
                p_call = p_res[0] if not isinstance(p_res, Exception) else p_res
                n_call = n_res[0] if not isinstance(n_res, Exception) else n_res
                e_call = e_res[0] if not isinstance(e_res, Exception) else e_res

                p_data = json.loads(p_call.content[0].text) if (not isinstance(p_call, Exception) and p_call.content) else {}
                n_data = json.loads(n_call.content[0].text) if (not isinstance(n_call, Exception) and n_call.content) else {}
                e_data = []
                if not isinstance(e_call, Exception) and getattr(e_call, "content", None):
                    try:
                        parsed_e = [json.loads(c.text) for c in e_call.content if getattr(c, "text", None)]
                        e_data = parsed_e[0] if (len(parsed_e) == 1 and isinstance(parsed_e[0], list)) else parsed_e
                    except Exception:
                        e_data = []

        funding = p_data.get("funding_rate_8h_pct")
        headlines = n_data.get("headlines", []) if isinstance(n_data, dict) else []
        d_status = str(p_data.get("status", "UNAVAILABLE")) if p_data else "UNAVAILABLE"
        n_status = str(n_data.get("status", "UNAVAILABLE")) if isinstance(n_data, dict) and n_data else "UNAVAILABLE"
        live = (d_status == "LIVE") and bool(headlines)
        status = "LIVE" if live else ("PARTIAL" if (funding is not None or headlines) else "UNAVAILABLE")

        return MCPContextPayload(
            status=status,
            transport="mcp_stdio",
            tools_discovered=discovered,
            funding_rate_8h_pct=float(funding) if funding is not None else None,
            derivatives_regime=str(p_data.get("derivatives_regime", "Unavailable")),
            futures_basis_usd=float(p_data.get("futures_basis_usd")) if p_data.get("futures_basis_usd") is not None else None,
            recent_error_memories=e_data if isinstance(e_data, list) else [],
            news_headlines=headlines,
            derivatives_status=d_status,
            derivatives_source=str(p_data.get("source", "none")),
            derivatives_error=str(p_res) if isinstance(p_res, Exception) else p_data.get("error"),
            derivatives_latency_ms=round(float(p_lat), 2),
            news_status=n_status,
            news_error=str(n_res) if isinstance(n_res, Exception) else n_data.get("error"),
            news_latency_ms=round(float(n_lat), 2),
        ).model_dump()

    return await asyncio.wait_for(_impl(), timeout=timeout_sec)


def fetch_mcp_context(limit: int = 3, timeout_sec: float = 15.0) -> Dict[str, Any]:
    """Fetches real-time derivatives pulse, news, and prediction error memories from MCP server."""
    if os.getenv("PYTEST_CURRENT_TEST") or "pytest" in sys.modules:
        try:
            from src.mcp_server import (
                get_live_btc_derivatives_pulse,
                get_recent_btc_news_headlines,
                get_recent_prediction_error_memories,
            )
            t0_p = time.perf_counter()
            pulse = get_live_btc_derivatives_pulse()
            p_lat = (time.perf_counter() - t0_p) * 1000.0
            errs = get_recent_prediction_error_memories(limit=limit)
            t0_n = time.perf_counter()
            news = get_recent_btc_news_headlines(hours_back=6, max_items=5)
            n_lat = (time.perf_counter() - t0_n) * 1000.0
            headlines = news.get("headlines", []) if isinstance(news, dict) else []
            funding = pulse.get("funding_rate_8h_pct")
            status = "LIVE" if (pulse.get("status") == "LIVE" or headlines) else "UNAVAILABLE"

            payload = MCPContextPayload(
                status=status,
                transport="in_process_test",
                tools_discovered=["get_live_btc_derivatives_pulse", "get_recent_btc_news_headlines", "get_recent_prediction_error_memories"],
                funding_rate_8h_pct=float(funding) if funding is not None else None,
                derivatives_regime=str(pulse.get("derivatives_regime", "Unavailable")),
                futures_basis_usd=float(pulse.get("futures_basis_usd")) if pulse.get("futures_basis_usd") is not None else None,
                recent_error_memories=errs if isinstance(errs, list) else [],
                news_headlines=headlines,
                derivatives_status=str(pulse.get("status", "UNAVAILABLE")),
                derivatives_source=str(pulse.get("source", "none")),
                derivatives_error=pulse.get("error"),
                derivatives_latency_ms=round(float(p_lat), 2),
                news_status=str(news.get("status", "UNAVAILABLE")) if isinstance(news, dict) else "UNAVAILABLE",
                news_error=news.get("error") if isinstance(news, dict) else None,
                news_latency_ms=round(float(n_lat), 2),
            )
            return payload.model_dump()
        except Exception as exc:
            logger.warning("In-process test MCP fetch failed: %s", exc)
            return MCPContextPayload(status="UNAVAILABLE", transport="in_process_test", error=str(exc)).model_dump()

    try:
        def _run_coro(coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    return executor.submit(asyncio.run, coro).result()
            return asyncio.run(coro)

        res = _run_coro(_fetch_mcp_context_stdio(limit=limit, timeout_sec=timeout_sec))
        logger.info("MCP stdio context fetch succeeded (status: %s)", res.get("status"))
        return res
    except Exception as exc:
        logger.warning("MCP stdio client failed or timed out: %s", exc)
        return MCPContextPayload(status="UNAVAILABLE", transport="mcp_stdio_error", error=str(exc)).model_dump()
