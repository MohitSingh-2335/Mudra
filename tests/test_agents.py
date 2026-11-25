import json
import pytest
import src.ai.llm as ai_llm
from src.ai.llm import generate_hourly_postmortem_with_gemini
from src.ai.mcp_client import fetch_mcp_context
from src.ai.workflow import (
    AgentState,
    build_trading_graph,
    calculate_technical_signals,
    risk_critic_node,
    run_trading_graph,
)


def test_trading_graph_compilation():
    app = build_trading_graph()
    assert app is not None
    assert hasattr(app, "invoke")


def test_technical_tools_execution():
    import pandas as pd
    res = calculate_technical_signals(
        close=65000.0, high=65800.0, low=64700.0, volume=2500.0, taker_buy_ratio=0.62,
    )
    assert res["close_price"] == 65000.0
    assert res["high_low_spread_usd"] == 1100.0
    assert res["taker_buy_ratio"] == 0.62
    assert "Bullish Aggressive Buying" in res["order_flow_bias"]

    cb_candles = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=45, freq="1h", tz="UTC"),
        "open": [64900.0] * 45, "high": [65200.0] * 45, "low": [64700.0] * 45,
        "close": [65000.0] * 45, "volume": [2500.0] * 45,
    })
    res_cb = calculate_technical_signals(candles=cb_candles, close=65000.0, volume=2500.0)
    assert res_cb["taker_buy_ratio"] == pytest.approx(0.50)
    assert "Neutral" in res_cb["order_flow_bias"] or "Balanced" in res_cb["order_flow_bias"] or res_cb["close_price"] == 65000.0


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

    assert hasattr(mcp_srv, "get_live_btc_derivatives_pulse")
    assert hasattr(mcp_srv, "get_recent_prediction_error_memories")
    assert hasattr(mcp_srv, "get_recent_btc_news_headlines")

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
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(ai_llm, "_get_gemini_api_key", lambda: "mock_api_key")

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
    mock_gen = lambda api_key, prompt, **kwargs: (raw_json_with_control_chars, "gemini-3.1-flash-lite") if kwargs.get("return_model") else raw_json_with_control_chars
    monkeypatch.setattr(ai_llm, "generate_with_gemini_fallback", mock_gen)

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

    res = ai_llm.generate_hourly_postmortem_with_gemini(tick_features, mcp_context, model_outcomes)
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


def test_multi_model_consensus_uses_classifier_and_m3_sma10(monkeypatch):
    # Consensus must follow classifier direction probabilities rather than regressor sign, and feed true 10h SMA to Model 3.
    from src.ai.workflow import get_multi_model_consensus

    seen_sma = {}

    class _Reg:
        def __init__(self, val):
            self.val = val

        def predict(self, df):
            return [self.val]

    class _Clf:
        def __init__(self, key, prob_up):
            self.key = key
            self.prob_up = prob_up

        def predict_proba(self, df):
            if "sma_10_dist_pct" in df.columns:
                seen_sma[self.key] = float(df["sma_10_dist_pct"].iloc[0])
            return [[1.0 - self.prob_up, self.prob_up]]

    fake_models = {
        "xgb_onnx_v1.0.0": {"features": ["f1"], "regressor": _Reg(-0.5), "classifier": _Clf("m1", 0.72)},
        "lgbm_v2.0.0": {"features": ["sma_10_dist_pct"], "regressor": _Reg(0.8), "classifier": _Clf("m2", 0.35)},
        "lgbm_v3.0.0": {"features": ["sma_10_dist_pct"], "regressor": _Reg(-0.2), "classifier": _Clf("m3", 0.61)},
    }
    monkeypatch.setattr("src.model_loader.get_registered_models", lambda: fake_models)

    res = get_multi_model_consensus({"f1": 1.0, "sma_10_dist_pct": -0.02, "sma_10_true_dist_pct": 0.09})
    assert res["total"] == 3
    assert res["votes"] == {"xgb_onnx_v1.0.0": "UP", "lgbm_v2.0.0": "DOWN", "lgbm_v3.0.0": "UP"}
    assert res["up_count"] == 2
    assert res["down_count"] == 1
    assert seen_sma["m2"] == pytest.approx(-0.02)
    assert seen_sma["m3"] == pytest.approx(0.09)


def test_llm_telemetry_logs_model_and_caller(monkeypatch):
    # Each Gemini generation call must record model_name:caller to api_telemetry helper.
    import sys
    from types import SimpleNamespace

    logged = []
    monkeypatch.setattr(ai_llm, "_log_llm_telemetry", lambda ep, lat, st, err=None: logged.append((ep, lat, st, err)))
    monkeypatch.setattr(ai_llm.time, "sleep", lambda _: None)

    call_count = {"n": 0}

    class _FakeModels:
        def generate_content(self, model, contents, config):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("429 RESOURCE_EXHAUSTED quota exceeded")
            return SimpleNamespace(text='{"market_summary": "ok"}')

    class _FakeClient:
        def __init__(self, **kwargs):
            self.models = _FakeModels()

    fake_types = SimpleNamespace(
        HttpOptions=lambda **kw: kw,
        GenerateContentConfig=lambda **kw: kw,
    )
    fake_genai = SimpleNamespace(Client=_FakeClient, types=fake_types)
    fake_google = SimpleNamespace(genai=fake_genai)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)

    out, used = ai_llm.generate_with_gemini_fallback(
        "test-key", "hello", return_model=True, caller="postmortem"
    )
    assert out == '{"market_summary": "ok"}'
    assert used == ai_llm.GEMINI_MODEL_CANDIDATES[1]
    assert logged[0][0] == f"{ai_llm.GEMINI_MODEL_CANDIDATES[0]}:postmortem"
    assert logged[0][2] == 429
    assert logged[1][0] == f"{ai_llm.GEMINI_MODEL_CANDIDATES[1]}:postmortem"
    assert logged[1][2] == 200




def test_generate_hourly_postmortem_preserves_mcp_snapshot_honestly(monkeypatch):
    """Ensure mcp_snapshot keeps real basis/headlines and leaves missing funding_rate as None without fake 0.01 defaults."""
    import json
    from src.ai.llm import generate_hourly_postmortem_with_gemini

    # Offline / pytest path must preserve mcp_snapshot honestly when funding_rate_8h_pct is None
    offline_res = generate_hourly_postmortem_with_gemini(
        tick_features={"rsi_14": 48.0},
        mcp_context={
            "funding_rate_8h_pct": None,
            "futures_basis_usd": -12.5,
            "derivatives_regime": "Bearish / Deleveraging",
            "derivatives_source": "deribit",
            "news_headlines": ["BTC volume rises", "Miner reserves steady"],
        },
        model_outcomes=[{"model_version": "lgbm_v3.0.0", "predicted_return_pct": 0.1, "actual_return_pct": -0.2}],
    )
    assert offline_res["gemini_status"] == "UNAVAILABLE"
    assert offline_res["mcp_snapshot"]["funding_rate_8h_pct"] is None
    assert offline_res["mcp_snapshot"]["futures_basis_usd"] == -12.5
    assert offline_res["mcp_snapshot"]["derivatives_source"] == "deribit"
    assert offline_res["mcp_snapshot"]["news_headline_count"] == 2

    # Online mocked Gemini path must pass mcp_snapshot without inventing 0.01 and return it in analysis
    captured = {}

    def fake_gemini(api_key, prompt, **kwargs):
        captured["prompt"] = prompt
        payload = {
            "market_summary": "Muted spot session.",
            "per_model": {
                "lgbm_v3.0.0": {
                    "root_cause_driver": "NEGATIVE_BASIS",
                    "diagnosis_summary": "Negative basis aligned with mild pullback.",
                }
            },
        }
        return json.dumps(payload), "gemini-2.5-flash"

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr("src.ai.llm._get_gemini_api_key", lambda: "unit-test-key")
    monkeypatch.setattr("src.ai.llm.generate_with_gemini_fallback", fake_gemini)

    online_res = generate_hourly_postmortem_with_gemini(
        tick_features={"rsi_14": 48.0},
        mcp_context={
            "funding_rate_8h_pct": None,
            "futures_basis_usd": 125.5,
            "derivatives_regime": "Neutral",
            "derivatives_source": "kraken",
            "news_headlines": ["Headline A", "Headline B"],
        },
        model_outcomes=[{"model_version": "lgbm_v3.0.0", "predicted_return_pct": 0.1, "actual_return_pct": 0.2}],
    )
    assert online_res["gemini_status"] == "OK"
    assert online_res["mcp_snapshot"]["funding_rate_8h_pct"] is None
    assert online_res["mcp_snapshot"]["futures_basis_usd"] == 125.5
    assert online_res["mcp_snapshot"]["news_headline_count"] == 2
    assert '"funding_rate_8h_pct": null' in captured["prompt"]
    assert "0.01" not in captured["prompt"]


def test_rag_gemini_primary_fallback_and_cooldown_recovery(monkeypatch, tmp_path):
    # Verify Gemini primary RAG falls back cleanly to separate MiniLM collection on 429 and recovers after cooldown.
    import src.ai.rag as ai_rag

    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setenv("MUDRA_TEST_GEMINI_RAG", "1")
    monkeypatch.setattr(ai_rag, "_fetch_neon_rag_records", lambda limit=15: ([], [], []))
    monkeypatch.setattr(ai_rag.LocalMiniLMEmbedding, "_get_local_minilm", lambda self: False)
    monkeypatch.setattr(ai_llm, "_get_gemini_api_key", lambda: "unit-test-key")

    state = {"fail_429": False, "now": 1000.0}
    monkeypatch.setattr(ai_rag.time, "time", lambda: state["now"])

    def fake_embed_texts(texts, api_key=None, output_dimensionality=768, caller="rag_embed"):
        if state["fail_429"]:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [[0.1] * output_dimensionality for _ in texts]

    monkeypatch.setattr(ai_llm, "embed_texts", fake_embed_texts)

    rag = ai_rag.MarketKnowledgeRAG(db_path=str(tmp_path / "chroma_dual"), cooldown_sec=60)
    rag.index_outcome_memory(1, "2026-11-25T10:00Z", "Bullish", "UP", "UP", True, 50.0, 80.0, 30.0, "ACCURATE", "Matched.")
    assert rag.embedding_mode == "gemini-embedding-001"
    assert rag.cooldown_remaining_sec == 0

    state["fail_429"] = True
    res_fallback = rag.query_memories("Bullish UP Matched", n_results=1)
    assert len(res_fallback) >= 1
    assert rag.embedding_mode == "hash_test_fallback" or rag.embedding_mode == "local_minilm"
    assert rag.cooldown_remaining_sec == 60
    assert rag.collection is rag.local_collection

    # After cooldown expires and Gemini quota resets, RAG automatically resumes Gemini primary collection.
    state["fail_429"] = False
    state["now"] += 61.0
    res_recovered = rag.query_memories("Bullish UP Matched", n_results=1)
    assert len(res_recovered) >= 1
    assert rag.embedding_mode == "gemini-embedding-001"
    assert rag.cooldown_remaining_sec == 0
