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
    assert "news_headlines" in ctx


def test_mcp_news_headlines_and_agent_integration():
    # verifies news headlines tool executes cleanly and returns honest telemetry
    from src.mcp_server import get_recent_btc_news_headlines
    res = get_recent_btc_news_headlines(hours_back=2, max_items=3)
    assert isinstance(res, dict)
    assert "status" in res
    assert "headlines" in res
    assert isinstance(res["headlines"], list)


def test_gemini_postmortem_offline_handling():
    # verifies gemini postmortem generator returns honest unavailable status under test
    from src.agents import generate_hourly_postmortem_with_gemini
    tick_features = {"close": 65000.0, "volatility_range_pct": 1.2, "taker_buy_ratio": 0.52, "rsi_estimate": 55.0}
    mcp_context = {"news_headlines": ["Bitcoin holds above support amid steady ETF flows."]}
    model_outcomes = [
        {
            "model_version": "xgb_onnx_v1.0.0",
            "predicted_price": 65150.0,
            "actual_price": 65200.0,
            "predicted_return_pct": 0.23,
            "actual_return_pct": 0.31,
            "direction_predicted": "UP",
            "direction_realized": "UP",
            "direction_correct": True,
            "error_usd": 50.0,
        }
    ]
    res = generate_hourly_postmortem_with_gemini(tick_features, mcp_context, model_outcomes)
    assert isinstance(res, dict)
    assert res["gemini_status"] == "UNAVAILABLE"
    assert "xgb_onnx_v1.0.0" in res["per_model"]
    diag = res["per_model"]["xgb_onnx_v1.0.0"]["diagnosis_summary"]
    assert "unavailable" in diag.lower() or "manual review" in diag.lower()


def test_gemini_postmortem_raw_control_character_parsing(monkeypatch):
    # verifies postmortem parser tolerates unescaped control characters such as raw newlines
    import json
    from src import agents
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(agents, "_get_gemini_api_key", lambda: "mock_api_key")

    raw_json_with_control_chars = (
        '{\n'
        '  "market_summary": "First line summary.\nSecond line summary with unescaped newline.",\n'
        '  "per_model": {\n'
        '    "xgb_onnx_v1.0.0": {\n'
        '      "root_cause_driver": "VOLATILITY",\n'
        '      "diagnosis_summary": "Line one diagnosis.\nLine two diagnosis with raw newline."\n'
        '    }\n'
        '  }\n'
        '}'
    )

    # ensure standard strict json.loads fails on this string
    with pytest.raises(json.JSONDecodeError):
        json.loads(raw_json_with_control_chars)

    # mock fallback generator returning raw unescaped json
    monkeypatch.setattr(
        agents,
        "generate_with_gemini_fallback",
        lambda api_key, prompt, **kwargs: (raw_json_with_control_chars, "gemini-3.1-flash-lite") if kwargs.get("return_model") else raw_json_with_control_chars,
    )

    tick_features = {"close": 65000.0, "volatility_range_pct": 1.2, "taker_buy_ratio": 0.52}
    mcp_context = {"news_headlines": ["ETF inflows continue."]}
    model_outcomes = [
        {
            "model_version": "xgb_onnx_v1.0.0",
            "predicted_price": 65150.0,
            "actual_price": 65200.0,
            "predicted_return_pct": 0.23,
            "actual_return_pct": 0.31,
            "direction_predicted": "UP",
            "direction_realized": "UP",
            "direction_correct": True,
            "error_usd": 50.0,
        }
    ]

    res = agents.generate_hourly_postmortem_with_gemini(tick_features, mcp_context, model_outcomes)
    assert res["gemini_status"] == "OK"
    assert res["model_used"] == "gemini-3.1-flash-lite"
    assert res["market_summary"] == "First line summary.\nSecond line summary with unescaped newline."
    diag = res["per_model"]["xgb_onnx_v1.0.0"]["diagnosis_summary"]
    assert diag == "Line one diagnosis.\nLine two diagnosis with raw newline."


def test_get_top_shap_drivers_calibrated_classifier():
    # verifies get_top_shap_drivers extracts valid top-k shap attributions for calibrated classifier
    import numpy as np
    import pandas as pd
    from sklearn.calibration import CalibratedClassifierCV
    import xgboost as xgb
    from src.explainability import get_top_shap_drivers

    # create minimal synthetic training data for tree model
    np.random.seed(42)
    X = pd.DataFrame({
        "taker_buy_ratio": np.random.uniform(0.3, 0.7, 50),
        "rsi": np.random.uniform(30.0, 70.0, 50),
        "vol_log_z24": np.random.normal(0.0, 1.0, 50),
        "return_4h": np.random.normal(0.0, 1.0, 50),
    })
    y = (X["taker_buy_ratio"] > 0.5).astype(int)

    base_clf = xgb.XGBClassifier(n_estimators=5, max_depth=2, random_state=42, n_jobs=1)
    base_clf.fit(X, y)

    cal_clf = CalibratedClassifierCV(estimator=base_clf, method="sigmoid", cv="prefit")
    cal_clf.fit(X, y)

    test_row = pd.DataFrame([{
        "taker_buy_ratio": 0.65,
        "rsi": 62.0,
        "vol_log_z24": 1.2,
        "return_4h": 0.5,
    }])

    # 1. test unwrapped tree classifier
    drivers_base = get_top_shap_drivers(base_clf, test_row, top_k=3)
    assert isinstance(drivers_base, list)
    assert len(drivers_base) == 3
    for d in drivers_base:
        assert "feature" in d
        assert "value" in d
        assert "shap_impact" in d
        assert isinstance(d["shap_impact"], float)

    # 2. test CalibratedClassifierCV wrapper
    drivers_cal = get_top_shap_drivers(cal_clf, test_row, top_k=3)
    assert isinstance(drivers_cal, list)
    assert len(drivers_cal) == 3
    for d in drivers_cal:
        assert "feature" in d
        assert "value" in d
        assert "shap_impact" in d
        assert isinstance(d["shap_impact"], float)

    # 3. test graceful handling of empty or None inputs
    assert get_top_shap_drivers(None, test_row, top_k=3) == []
    assert get_top_shap_drivers(cal_clf, pd.DataFrame(), top_k=3) == []
    assert get_top_shap_drivers(cal_clf, None, top_k=3) == []


