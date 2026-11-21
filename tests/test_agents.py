import pytest
from src.agents import (
    AgentState,
    build_trading_graph,
    calculate_technical_signals,
    get_model_return_forecast,
    risk_critic_node,
    run_trading_graph,
)


def test_trading_graph_compilation():
    app = build_trading_graph()
    assert app is not None
    assert hasattr(app, "invoke")


def test_technical_tools_execution():
    res = calculate_technical_signals.invoke({
        "close": 65000.0, "high": 65800.0, "low": 64700.0, "volume": 2500.0, "taker_buy_ratio": 0.62,
    })
    assert res["close_price"] == 65000.0
    assert res["high_low_spread_usd"] == 1100.0
    assert res["taker_buy_ratio"] == 0.62
    assert "Bullish Aggressive Buying" in res["order_flow_bias"]


def test_trading_graph_bullish_approval():
    state = run_trading_graph(
        symbol="BTCUSDT", current_price=65000.0,
        market_data={"close": 65000.0, "high": 65500.0, "low": 64800.0, "volume": 2000.0, "taker_buy_ratio": 0.65, "number_of_trades": 25000},
        engineered_features={"volatility_range_pct": 1.08, "taker_buy_ratio": 0.65, "rsi_estimate": 62.0},
        ml_forecast={"expected_return_pct": 0.25, "predicted_dollar_move": 162.5, "clears_hurdle": True},
    )
    assert state["trade_action"] == "BUY"
    assert state["risk_approved"] is True
    assert state["recommended_position_pct"] > 0.0
    assert state["stop_loss_price"] < 65000.0
    assert state["take_profit_price"] > 65000.0
    assert state["veto_reason"] is None
    assert "AUTHORIZED" in state["risk_critic_critique"] or "Approved" in state["risk_critic_critique"]


def test_risk_critic_sub_hurdle_veto():
    input_state: AgentState = {
        "symbol": "BTCUSDT", "current_price": 65000.0,
        "market_data": {"close": 65000.0, "high": 65300.0, "low": 64900.0, "volume": 1200.0, "taker_buy_ratio": 0.52},
        "engineered_features": {"volatility_range_pct": 0.62, "taker_buy_ratio": 0.52},
        "ml_forecast": {"expected_return_pct": 0.02, "predicted_dollar_move": 13.0, "clears_hurdle": False},
        "rag_context": "Neutral macro backdrop", "timestamp": "2026-10-11T12:00:00Z",
        "tools_called": ["get_live_market_data", "calculate_technical_signals"],
        "circuit_breaker_active": False, "trade_action": "BUY", "conviction_score": 7,
        "executive_summary": "Recommending long entry on micro momentum",
        "strategist_thesis": "Potential breakout despite low margin",
        "risk_approved": None, "veto_reason": None, "recommended_position_pct": 0.0,
        "stop_loss_price": None, "take_profit_price": None, "risk_critic_critique": "",
    }
    state = risk_critic_node(input_state)
    assert state["risk_approved"] is False
    assert state["recommended_position_pct"] == 0.0
    assert state["veto_reason"] is not None
    assert "Fee Drag" in state["veto_reason"]
    assert "VETOED" in state["risk_critic_critique"]


def test_trading_graph_volatility_spike_veto():
    state = run_trading_graph(
        symbol="BTCUSDT", current_price=65000.0,
        market_data={"close": 65000.0, "high": 68000.0, "low": 63500.0, "volume": 8000.0, "taker_buy_ratio": 0.60},
        engineered_features={"volatility_range_pct": 6.92, "taker_buy_ratio": 0.60},
        ml_forecast={"expected_return_pct": 0.15, "predicted_dollar_move": 97.5, "clears_hurdle": True},
    )
    assert state["risk_approved"] is False
    assert state["recommended_position_pct"] == 0.0
    assert "Volatility" in state["veto_reason"]
    assert "VETOED" in state["risk_critic_critique"]


def test_trading_graph_hold_preservation():
    state = run_trading_graph(
        symbol="BTCUSDT", current_price=65000.0,
        market_data={"close": 65000.0, "high": 65100.0, "low": 64950.0, "volume": 800.0, "taker_buy_ratio": 0.50},
        engineered_features={"volatility_range_pct": 0.23, "taker_buy_ratio": 0.50},
        ml_forecast={"expected_return_pct": 0.005, "predicted_dollar_move": 3.25, "clears_hurdle": False},
    )
    assert state["trade_action"] == "HOLD"
    assert state["risk_approved"] is True
    assert state["recommended_position_pct"] == 0.0
    assert "Capital preservation" in state["risk_critic_critique"]


def test_mcp_server_tools_and_context_fetcher():
    # verifies mcp server tool functions and agent context fetcher work cleanly
    import src.mcp_server as mcp_srv
    from src.agents import fetch_mcp_context

    assert hasattr(mcp_srv, "get_live_btc_derivatives_pulse")
    assert hasattr(mcp_srv, "get_live_btc_spot_and_range")
    assert hasattr(mcp_srv, "get_recent_prediction_error_memories")
    assert hasattr(mcp_srv, "get_spot_and_futures_trade_plan")

    pulse = mcp_srv.get_live_btc_derivatives_pulse()
    assert "funding_rate_8h_pct" in pulse
    assert "derivatives_regime" in pulse

    ctx = fetch_mcp_context(limit=2)
    assert isinstance(ctx, dict)
    assert "funding_rate_8h_pct" in ctx
    assert "derivatives_regime" in ctx
    assert isinstance(ctx["recent_error_memories"], list)
