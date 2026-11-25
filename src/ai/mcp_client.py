import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("mudra.ai.mcp_client")


def _build_context_payload(
    transport: str,
    tools_discovered: Optional[List[str]] = None,
    pulse: Optional[Dict[str, Any]] = None,
    news: Optional[Dict[str, Any]] = None,
    errors: Optional[List[Dict[str, Any]]] = None,
    p_lat: float = 0.0,
    n_lat: float = 0.0,
    p_err: Optional[str] = None,
    n_err: Optional[str] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds a uniform derivatives, news, and episodic-error context dictionary."""
    p = pulse if isinstance(pulse, dict) else {}
    n = news if isinstance(news, dict) else {}
    funding = p.get("funding_rate_8h_pct")
    basis = p.get("futures_basis_usd")
    headlines = [str(h) for h in (n.get("headlines") or [])]
    d_status = str(p.get("status", "UNAVAILABLE")) if p else "UNAVAILABLE"
    n_status = str(n.get("status", "UNAVAILABLE")) if n else "UNAVAILABLE"

    if error:
        status = "UNAVAILABLE"
    elif d_status == "LIVE" and headlines:
        status = "LIVE"
    elif funding is not None or headlines:
        status = "PARTIAL"
    else:
        status = "UNAVAILABLE"

    return {
        "status": status,
        "transport": transport,
        "tools_discovered": tools_discovered or [],
        "funding_rate_8h_pct": float(funding) if funding is not None else None,
        "derivatives_regime": str(p.get("derivatives_regime", "Unavailable")),
        "futures_basis_usd": float(basis) if basis is not None else None,
        "news_headlines": headlines,
        "recent_error_memories": errors if isinstance(errors, list) else [],
        "error": error,
        "derivatives_status": d_status,
        "derivatives_source": str(p.get("source", "none")),
        "derivatives_error": p_err or p.get("error"),
        "derivatives_latency_ms": round(float(p_lat), 2),
        "news_status": n_status,
        "news_error": n_err or n.get("error"),
        "news_latency_ms": round(float(n_lat), 2),
    }


async def _call_tool_safe(session: Any, name: str, args: Dict[str, Any], timeout: float) -> Tuple[Any, float, Optional[str]]:
    """Executes one MCP stdio tool call with an isolated timeout and parses JSON output."""
    t0 = time.perf_counter()
    try:
        res = await asyncio.wait_for(session.call_tool(name, arguments=args), timeout=timeout)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        texts = [c.text for c in (getattr(res, "content", None) or []) if getattr(c, "text", None)]
        if not texts:
            return {}, lat_ms, None
        parsed = [json.loads(t) for t in texts]
        data = parsed[0] if len(parsed) == 1 else parsed
        return data, lat_ms, None
    except Exception as exc:
        return None, (time.perf_counter() - t0) * 1000.0, (str(exc) or type(exc).__name__)


async def _fetch_mcp_context_stdio(limit: int = 3, timeout_sec: float = 15.0) -> Dict[str, Any]:
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

                # Run fast derivatives and news HTTP tools concurrently before synchronous Neon query
                (p_data, p_lat, p_err), (n_data, n_lat, n_err) = await asyncio.gather(
                    _call_tool_safe(session, "get_live_btc_derivatives_pulse", {}, timeout=6.0),
                    _call_tool_safe(session, "get_recent_btc_news_headlines", {"hours_back": 6, "max_items": 5}, timeout=6.0),
                )
                e_data, _, _ = await _call_tool_safe(session, "get_recent_prediction_error_memories", {"limit": limit}, timeout=8.0)

        return _build_context_payload(
            transport="mcp_stdio",
            tools_discovered=discovered,
            pulse=p_data,
            news=n_data,
            errors=e_data if isinstance(e_data, list) else [],
            p_lat=p_lat,
            n_lat=n_lat,
            p_err=p_err,
            n_err=n_err,
        )

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
            return _build_context_payload(
                transport="in_process_test",
                tools_discovered=["get_live_btc_derivatives_pulse", "get_recent_btc_news_headlines", "get_recent_prediction_error_memories"],
                pulse=pulse, news=news, errors=errs, p_lat=p_lat, n_lat=n_lat,
            )
        except Exception as exc:
            logger.warning("In-process test MCP fetch failed: %s", exc)
            return _build_context_payload(transport="in_process_test", error=str(exc))

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
        err_str = str(exc) or type(exc).__name__
        logger.warning("MCP stdio client failed or timed out: %s", err_str)
        return _build_context_payload(transport="mcp_stdio_error", error=err_str)
