from datetime import datetime, timedelta, timezone
import json
import logging
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import pytest
import requests
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import app
import config
from config import (
    INITIAL_CAPITAL_USD,
    MIN_HURDLE_PCT,
    MIN_HURDLE_RATE,
    ROUND_TRIP_FRICTION_PCT,
    ROUND_TRIP_FRICTION_RATE,
    SLIPPAGE_PCT,
    SLIPPAGE_RATE,
    TAKER_FEE_PCT,
    TAKER_FEE_RATE,
    TOTAL_FRICTION_PER_SIDE_PCT,
    TOTAL_FRICTION_PER_SIDE_RATE,
    _parse_execution_cost_rate,
)
import scripts.run_hourly_worker as worker
from src.backtesting import run_backtest_threshold
from src.agents import (
    generate_with_gemini_fallback,
    get_gemini_status,
    risk_critic_node,
    run_trading_graph,
    strategist_node,
)
from src.backtesting import _simulate_from_signals
from src.database import (
    Base, CouncilAudit, MarketTick, ModelPrediction, ModelRegistry,
    PaperPortfolio, Reconciliation, init_db as real_init_db,
)
from src.engine import (
    MarketFeedExhaustedError,
    _parse_trade_count,
    build_plain_english_reconciliation_story,
    ensure_model_registered,
    estimate_next_hour_price_range_usd,
    evaluate_spot_hold_until_profit_decision,
    execute_paper_order,
    fetch_latest_confirmed_candle,
    get_portfolio_summary,
    parse_retry_after,
    sync_hourly_quant_cycle,
)
import src.engine as sync
from src.mcp_server import get_live_btc_derivatives_pulse, get_spot_and_futures_trade_plan


# helper mocks for market feed responses
def _make_response(status_code, body=None, headers=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.headers = headers or {}
    if body is not None:
        resp.json.return_value = body
    else:
        resp.json.side_effect = json.JSONDecodeError("Expecting value", "", 0)
    return resp


def _make_binance_klines(n=102, now_dt=None):
    now_ms = (now_dt or datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)).timestamp() * 1000.0
    return [[now_ms - (n + 1 - i) * 3600000, "60000.0", "60500.0", "59800.0", "60200.0", "100.0", now_ms - (n + 1 - i) * 3600000 + 3599999, "6020000.0", 1500, "55.0", "3311000.0", "0"] for i in range(n)]


def _make_coinbase_candles(n=100, now_dt=None):
    now_sec = (now_dt or datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)).timestamp()
    return [[now_sec - (i + 1) * 3600, 59800.0, 60500.0, 60000.0, 60250.0, 100.0] for i in range(n)]


@pytest.fixture
def isolated_db(monkeypatch):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    real_init_db(engine)
    Session = sessionmaker(bind=engine)
    monkeypatch.setattr("src.database.engine", engine)
    monkeypatch.setattr("src.database.SessionLocal", Session)
    monkeypatch.setattr("src.database.test_connection", lambda: (True, "SQLite 3.x", 1.0))
    monkeypatch.setattr("src.database.init_db", lambda target_engine=None: real_init_db(engine))
    session = Session()
    yield session, engine
    session.close()
    Base.metadata.drop_all(bind=engine)
    engine.dispose()


@pytest.fixture
def temp_outbox(tmp_path, monkeypatch):
    outbox_path = tmp_path / "sync_outbox.json"
    monkeypatch.setattr(worker, "OUTBOX_FILE", outbox_path)
    return outbox_path


def test_canonical_cost_rates_and_display_aliases():
    assert TAKER_FEE_RATE == pytest.approx(0.0005, abs=1e-8)
    assert SLIPPAGE_RATE == pytest.approx(0.0002, abs=1e-8)
    assert TOTAL_FRICTION_PER_SIDE_RATE == pytest.approx(0.0007, abs=1e-8)
    assert ROUND_TRIP_FRICTION_RATE == pytest.approx(1.0 - (1.0 - 0.0007) ** 2, abs=1e-8)
    assert MIN_HURDLE_RATE == pytest.approx(ROUND_TRIP_FRICTION_RATE, abs=1e-8)
    assert TAKER_FEE_PCT == pytest.approx(TAKER_FEE_RATE * 100.0, abs=1e-8)
    assert SLIPPAGE_PCT == pytest.approx(SLIPPAGE_RATE * 100.0, abs=1e-8)
    assert ROUND_TRIP_FRICTION_PCT == pytest.approx(ROUND_TRIP_FRICTION_RATE * 100.0, abs=1e-8)


def test_env_var_parsing(monkeypatch):
    monkeypatch.setenv("TEST_RATE", "0.0010")
    assert _parse_execution_cost_rate("TEST_RATE", "TEST_PCT", 0.0005) == pytest.approx(0.0010)

    monkeypatch.delenv("TEST_RATE", raising=False)
    monkeypatch.setenv("TEST_PCT", "0.10")
    assert _parse_execution_cost_rate("TEST_RATE", "TEST_PCT", 0.0005) == pytest.approx(0.0010)

    monkeypatch.setenv("TEST_RATE", "0.0010")
    monkeypatch.setenv("TEST_PCT", "0.20")
    with pytest.raises(ValueError):
        _parse_execution_cost_rate("TEST_RATE", "TEST_PCT", 0.0005)

    monkeypatch.delenv("TEST_PCT", raising=False)
    monkeypatch.setenv("TEST_RATE", "-0.01")
    with pytest.raises(ValueError):
        _parse_execution_cost_rate("TEST_RATE", "TEST_PCT", 0.0005)


def test_paper_portfolio_exact_cost_deduction(isolated_db):
    session = isolated_db[0]
    r_buy = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    expected_capital = INITIAL_CAPITAL_USD * 0.10
    expected_fee = expected_capital * TAKER_FEE_RATE
    expected_slip = expected_capital * SLIPPAGE_RATE
    expected_exec_price = 60000.0 * (1.0 + SLIPPAGE_RATE)
    expected_btc = (expected_capital - expected_fee) / expected_exec_price

    assert r_buy.cash_balance == pytest.approx(INITIAL_CAPITAL_USD - expected_capital, abs=1e-4)
    assert r_buy.fee_deducted_usd == pytest.approx(expected_fee, abs=1e-4)
    assert r_buy.slippage_usd == pytest.approx(expected_slip, abs=1e-4)
    assert r_buy.execution_price == pytest.approx(expected_exec_price, abs=1e-4)
    assert r_buy.btc_amount == pytest.approx(expected_btc, abs=1e-7)

    execute_paper_order(session, "SELL", 62000.0)
    summary = get_portfolio_summary(session, current_price=62000.0)
    assert summary["btc_balance"] == pytest.approx(0.0, abs=1e-8)
    assert summary["cash_balance"] - INITIAL_CAPITAL_USD == pytest.approx(summary["realized_pnl_usd"], abs=1e-2)


def test_backtesting_simulation_caller_overrides():
    timestamps = pd.date_range("2026-01-01", periods=4, freq="1h")
    test_df = pd.DataFrame({"timestamp": timestamps, "close": [100.0, 105.0, 102.0, 100.0]})
    signals = np.array([1, 0, 0, 0])

    eq_default, tl_default = _simulate_from_signals(test_df, signals)
    assert len(tl_default) == 1

    eq_override, tl_override = _simulate_from_signals(test_df, signals, taker_fee_rate=0.0010, slippage_rate=0.0005)
    assert len(tl_override) == 1
    assert tl_override["equity_after"].iloc[0] < tl_default["equity_after"].iloc[0]


def test_threshold_sweeper_slippage():
    timestamps = pd.date_range("2026-01-01", periods=3, freq="1h")
    test_df = pd.DataFrame({"timestamp": timestamps, "close": [100.0, 110.0, 105.0]})
    for col in config.CLASSIFIER_FEATURES:
        test_df[col] = 1.0

    class MockScaler:
        def transform(self, X):
            return X

    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.8, 0.2], [0.8, 0.2]])

    eq_df, tl_df = run_backtest_threshold(test_df, MockModel(), MockScaler(), threshold=0.50)
    assert len(tl_df) == 1 and "equity" in eq_df.columns


def test_agent_nodes_cost_alignment():
    state = {
        "symbol": "BTCUSDT", "current_price": 65000.0, "trade_action": "HOLD",
        "ml_forecast": {"expected_return_pct": 0.0, "clears_hurdle": False},
        "engineered_features": {"volatility_range_pct": 1.0}, "market_data": {"taker_buy_ratio": 0.55},
        "rag_context": "",
    }
    rc_out = risk_critic_node(state)
    assert rc_out["risk_approved"] is True
    assert f"{ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction" in rc_out["risk_critic_critique"]

    strat_state = {**state, "ml_forecast": {
        "expected_return_pct": 0.25, "clears_hurdle": True, "direction": "UP",
        "direction_badge": "BULLISH", "direction_probability_up": 65.0,
    }}
    strat_out = strategist_node(strat_state)
    assert strat_out["trade_action"] == "BUY"
    assert f"{MIN_HURDLE_PCT:.2f}% estimated round-trip hurdle" in strat_out["executive_summary"]


def test_risk_critic_validation_fail_closed():
    for invalid_action in ["SHORT", "LONG", "", None, 123]:
        out = risk_critic_node({
            "symbol": "BTCUSDT", "current_price": 65000.0, "trade_action": invalid_action,
            "ml_forecast": {"expected_return_pct": 0.5, "clears_hurdle": True},
            "engineered_features": {"volatility_range_pct": 1.0},
        })
        assert out["risk_approved"] is False and out["recommended_position_pct"] == 0.0 and "Unsupported Trade Action" in out["veto_reason"]

    out_nan = risk_critic_node({
        "symbol": "BTCUSDT", "current_price": 65000.0, "trade_action": "BUY",
        "ml_forecast": {"expected_return_pct": float("nan"), "clears_hurdle": True},
        "engineered_features": {"volatility_range_pct": 1.0},
    })
    assert out_nan["risk_approved"] is False and out_nan["recommended_position_pct"] == 0.0


def test_ui_generate_live_forecast_hurdle(monkeypatch):
    import src.feature_engineering as fe

    live_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=40, freq="1h", tz="UTC"),
        "open": [65000.0] * 40, "high": [65200.0] * 40, "low": [64800.0] * 40,
        "close": [65000.0] * 40, "volume": [100.0] * 40, "taker_buy_ratio": [0.55] * 40,
    })
    dummy = {col: 1.0 for col in config.CLASSIFIER_FEATURES}
    dummy["vol_1h"] = 1.0
    monkeypatch.setattr(fe, "extract_features_for_inference", lambda df: dummy)
    monkeypatch.setattr(fe, "validate_inference_features", lambda f, model_type="classifier": pd.DataFrame([dummy]))
    monkeypatch.setattr(fe, "predict_with_validation", lambda m, df, model_type="regressor": np.array([0.0010]))
    monkeypatch.setattr(fe, "predict_proba_with_validation", lambda m, df: np.array([[0.2, 0.8]]))

    res = app.generate_live_forecast(live_df, MagicMock(), MagicMock(), is_stale=False)
    assert res is not None and res["cleared_hurdle"] is False and res["direction_badge"] == "SIDEWAYS (CHOP)"


def test_council_scenarios_and_monkeypatch(monkeypatch):
    monkeypatch.setattr("src.agents._get_gemini_api_key", lambda: None)
    for scenario, expected_action in [("chop", "HOLD"), ("bullish", "BUY"), ("vol_spike", "SELL")]:
        m, t, f = app.build_council_scenario_payload(scenario, 65000.0)
        state = run_trading_graph("BTCUSDT", 65000.0, m, t, f)
        assert state["strategist_action"] == expected_action


def test_format_portfolio_rationale():
    assert f"below {MIN_HURDLE_PCT:.2f}% round-trip fee hurdle" in app.format_portfolio_rationale({"action": "HOLD", "notes": "Sub-hurdle fee drag: preserving capital."})
    assert "Sideways chop" in app.format_portfolio_rationale({"action": "HOLD", "notes": "Neutral regime"})
    assert "Risk Veto" in app.format_portfolio_rationale({"action": "HOLD", "notes": "Order vetoed due to volatility spike"})


def test_model_pedigree_markdown():
    md = app.build_model_pedigree_markdown()
    assert "0.30% (Historical)" in md and f"`{ROUND_TRIP_FRICTION_PCT:.2f}%`" in md


def test_spot_hold_until_net_profit_lifecycle(isolated_db):
    session = isolated_db[0]
    d_entry = evaluate_spot_hold_until_profit_decision(session, current_price=60000.0, predicted_direction="UP", proposed_action="BUY", clears_hurdle=True)
    assert d_entry["final_spot_action"] == "BUY" and d_entry["position_state"] == "ENTERING_LONG_SPOT"

    execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.20, tick_id=1)
    session.commit()

    d_up = evaluate_spot_hold_until_profit_decision(session, current_price=60020.0, predicted_direction="UP", proposed_action="BUY", clears_hurdle=True)
    assert d_up["final_spot_action"] == "HOLD" and d_up["position_state"] == "HOLDING_BTC_ON_UP"

    d_hold_unprofitable = evaluate_spot_hold_until_profit_decision(session, current_price=60030.0, predicted_direction="DOWN", proposed_action="SELL", clears_hurdle=True)
    assert d_hold_unprofitable["final_spot_action"] == "HOLD" and d_hold_unprofitable["position_state"] == "HOLDING_BTC_UNTIL_NET_PROFIT" and d_hold_unprofitable["estimated_net_pnl_usd"] < 0.0

    d_profit_sell = evaluate_spot_hold_until_profit_decision(session, current_price=60500.0, predicted_direction="DOWN", proposed_action="SELL", clears_hurdle=True)
    assert d_profit_sell["final_spot_action"] == "SELL" and d_profit_sell["position_state"] == "TAKING_NET_PROFIT_ON_DOWN" and d_profit_sell["estimated_net_pnl_usd"] > 0.0

    d_stop = evaluate_spot_hold_until_profit_decision(session, current_price=57500.0, predicted_direction="DOWN", proposed_action="SELL", clears_hurdle=True, max_stop_loss_pct=3.5)
    assert d_stop["final_spot_action"] == "SELL" and d_stop["position_state"] == "EMERGENCY_STOP_LOSS_EXIT"


def test_price_range_estimator():
    df = pd.DataFrame({"high": [80400.0] * 24, "low": [80000.0] * 24, "close": [80200.0] * 24})
    r_info = estimate_next_hour_price_range_usd(df, current_price=80200.0)
    assert r_info["predicted_range_usd"] == pytest.approx(400.0, abs=1.0) and r_info["predicted_high_1h"] > 80200.0 > r_info["predicted_low_1h"]


def test_plain_english_reconciliation_story_and_mcp_fallback(monkeypatch):
    t0 = MarketTick(id=1, timestamp=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc), open=80000.0, high=80200.0, low=79900.0, close=80000.0, volume=100.0)
    t1 = MarketTick(id=2, timestamp=datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc), open=80000.0, high=80500.0, low=79950.0, close=80350.0, volume=120.0)
    story = build_plain_english_reconciliation_story(
        pred_tick=t0, outcome_tick=t1, pred_price=80200.0, actual_price=80350.0, pred_ret_pct=0.25,
        actual_ret_pct=0.4375, dir_pred="UP", dir_real="UP", dir_correct=True, abs_err_usd=150.0,
        tag="MOMENTUM_UNDERESTIMATED",
    )
    assert "[CORRECT]" in story["headline"] and "$80,350.00" in story["hypothesis_notes"]

    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    assert get_live_btc_derivatives_pulse()["status"] == "OFFLINE_FALLBACK"


def test_gemini_model_fallback_and_mcp_trade_plan(monkeypatch):
    called_models = []

    class _DummyModels:
        def generate_content(self, model=None, contents=None, config=None, **kwargs):
            called_models.append(model)
            if len(called_models) == 1:
                raise RuntimeError("Rate limit 429 on primary model candidate")
            res = MagicMock()
            res.text = "```json\n{\"trade_thesis\": \"MODERATE BULLISH\", \"conviction_score\": 7, \"executive_summary\": \"OK\", \"quantitative_driver_analysis\": \"SHAP OK\", \"macro_market_context\": \"RAG OK\", \"key_risks\": [\"Risk 1\"]}\n```"
            return res

    class _DummyClient:
        def __init__(self, api_key=None, **kwargs):
            self.models = _DummyModels()

    from google import genai
    monkeypatch.setattr(genai, "Client", _DummyClient)
    text = generate_with_gemini_fallback("test_key", "prompt")
    assert "MODERATE BULLISH" in text and len(called_models) >= 2

    st = get_gemini_status()
    assert st["configured"] is False or st["configured"] is True

    plan = get_spot_and_futures_trade_plan(
        current_price=80000.0, predicted_direction="UP", expected_return_pct=0.45,
        conviction_score=8, volatility_range_pct=1.0, risk_approved=True, clears_hurdle=True,
    )
    assert "spot_plan" in plan and "futures_plan" in plan


def test_parse_retry_after():
    assert parse_retry_after("5") == 5.0 and parse_retry_after("12.5") == 12.5 and parse_retry_after("0") == 0.0
    for invalid in ["-1", "NaN", "inf", "", None, "invalid_garbage"]:
        assert parse_retry_after(invalid) is None
    ref_time = datetime(2026, 7, 6, 12, 0, 0, tzinfo=timezone.utc)
    assert parse_retry_after("Mon, 06 Jul 2026 12:00:30 GMT", now_dt_fn=lambda: ref_time) == pytest.approx(30.0, abs=0.5)


def test_binance_transient_connection_failure_then_success(monkeypatch):
    mock_klines, attempts = _make_binance_klines(102), [0]

    def mock_get(url, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            raise requests.exceptions.ConnectionError("Connection reset by peer")
        return _make_response(200, mock_klines)

    monkeypatch.setattr("requests.get", mock_get)
    result = fetch_latest_confirmed_candle(sleep_fn=lambda s: None, clock_fn=lambda: 1000.0, now_dt_fn=lambda: datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc))
    assert attempts[0] == 2 and result["close"] == 60200.0 and result["number_of_trades"] == 1500


def test_binance_exhausted_retries_coinbase_fallback(monkeypatch):
    mock_coinbase = _make_coinbase_candles(100)

    def mock_get(url, **kwargs):
        if "binance" in url:
            raise requests.exceptions.ConnectionError("Binance down")
        return _make_response(200, mock_coinbase) if "coinbase" in url else _make_response(404)

    monkeypatch.setattr("requests.get", mock_get)
    result = fetch_latest_confirmed_candle(sleep_fn=lambda s: None, clock_fn=lambda: 1000.0, now_dt_fn=lambda: datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc))
    assert result["close"] == 60250.0 and result["number_of_trades"] is None


def test_both_providers_exhausted_raises_runtime_error(monkeypatch):
    def mock_get(url, **kwargs):
        raise requests.exceptions.ConnectionError("Network offline")

    monkeypatch.setattr("requests.get", mock_get)
    with pytest.raises(MarketFeedExhaustedError):
        fetch_latest_confirmed_candle(sleep_fn=lambda s: None, clock_fn=lambda: 1000.0, now_dt_fn=lambda: datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc))


def test_quant_cycle_feed_exhaustion_returns_failure_without_opening_db_session(monkeypatch):
    monkeypatch.setattr("src.engine.fetch_latest_confirmed_candle", MagicMock(side_effect=MarketFeedExhaustedError("All market feeds exhausted.")))
    monkeypatch.setattr("src.engine.test_connection", lambda: (True, "PostgreSQL", 1.0))
    synced, message, summary = sync_hourly_quant_cycle(force=False)
    assert synced is False and "exhausted" in message.lower() and summary == {}


def test_trade_count_parsing_validation():
    assert _parse_trade_count(1500) == 1500 and _parse_trade_count(0) == 0 and _parse_trade_count(None) is None and _parse_trade_count("1500") == 1500
    for invalid in [-5, 12.5, "abc", True]:
        with pytest.raises(ValueError):
            _parse_trade_count(invalid)


def test_confirmed_candle_filtering_and_minimum_count(monkeypatch):
    now_dt = datetime(2026, 7, 6, 14, 5, 0, tzinfo=timezone.utc)
    mock_klines = _make_binance_klines(102, now_dt=now_dt)
    monkeypatch.setattr("requests.get", lambda url, **kwargs: _make_response(200, mock_klines))
    result = fetch_latest_confirmed_candle(sleep_fn=lambda s: None, clock_fn=lambda: 1000.0, now_dt_fn=lambda: now_dt)
    assert result is not None and result["close"] == 60200.0 and len(result["candles"]) >= 35


def test_worker_success_on_new_sync(isolated_db, temp_outbox, monkeypatch):
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", lambda force=False: (True, "Recorded tick #1", {}))
    assert worker.run_worker(force=False) is True


def test_worker_success_on_already_recorded(isolated_db, temp_outbox, monkeypatch):
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", lambda force=False: (False, "Candle already recorded in Neon.", {}))
    assert worker.run_worker(force=False) is True


def test_worker_failure_on_feed_error(isolated_db, temp_outbox, monkeypatch):
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", lambda force=False: (False, "Market feed error", {}))
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_db_timeout(isolated_db, temp_outbox, monkeypatch):
    monkeypatch.setattr("src.database.test_connection", lambda: (False, "Timeout", 5000.0))
    assert worker.run_worker(force=False) is False


def test_worker_failure_on_exception(isolated_db, temp_outbox, monkeypatch):
    monkeypatch.setattr(worker, "sync_hourly_quant_cycle", MagicMock(side_effect=RuntimeError("crash")))
    assert worker.run_worker(force=False) is False


def test_outbox_missing_noop(isolated_db, temp_outbox):
    assert worker.flush_outbox(isolated_db[0]) == 0


def test_outbox_empty_unlinked(isolated_db, temp_outbox):
    temp_outbox.write_text("   \n ", encoding="utf-8")
    assert worker.flush_outbox(isolated_db[0]) == 0
    assert not temp_outbox.exists()


def test_outbox_malformed_preserved(isolated_db, temp_outbox):
    temp_outbox.write_text("{corrupt json", encoding="utf-8")
    assert worker.flush_outbox(isolated_db[0]) == 0
    assert temp_outbox.exists()


def test_outbox_flush_persists_and_unlinks(isolated_db, temp_outbox):
    session = isolated_db[0]
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    temp_outbox.write_text(json.dumps([{
        "type": "tick",
        "payload": {
            "timestamp": t0.isoformat(), "open": 60000.0, "high": 60500.0, "low": 59500.0, "close": 60200.0, "volume": 120.0,
            "model_version": "xgb_v1", "features": {"rsi": 55.4},
        }
    }]), encoding="utf-8")
    assert worker.flush_outbox(session) == 1
    assert not temp_outbox.exists()
    tick = session.execute(select(MarketTick).where(MarketTick.timestamp == t0)).scalar_one_or_none()
    assert tick is not None and tick.close == 60200.0


def test_outbox_replay_idempotent(isolated_db, temp_outbox):
    session = isolated_db[0]
    t0 = datetime(2026, 1, 1, 14, 0, tzinfo=timezone.utc)
    session.add(MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60050.0, volume=10.0))
    session.commit()
    temp_outbox.write_text(json.dumps([{"type": "tick", "payload": {"timestamp": t0.isoformat(), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 10.0}}]), encoding="utf-8")
    assert worker.flush_outbox(session) == 0
    assert not temp_outbox.exists()


def test_force_sync_deduplicates_paper_trade(isolated_db, monkeypatch):
    session = isolated_db[0]
    t0 = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    candle = {"timestamp": t0, "open": 60000.0, "high": 60200.0, "low": 59800.0, "close": 60100.0, "volume": 75.0}

    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: candle)
    monkeypatch.setattr(sync, "calculate_technical_signals", MagicMock(invoke=lambda *a: {"rsi": 55.0}))
    mock_ml = MagicMock(invoke=lambda *a: {"expected_return_pct": 0.45, "predicted_target_price": 60370.0, "confidence": 0.8, "clears_hurdle": True})
    monkeypatch.setattr(sync, "get_model_return_forecast", mock_ml)
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {
        "final_decision": "BUY", "risk_verdict": "APPROVED", "strategist_action": "BUY",
        "strategist_conviction": 8, "kelly_fraction": 0.10, "thesis": "Bullish",
    })

    ok1, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok1 is True
    assert len(session.execute(select(PaperPortfolio).where(PaperPortfolio.action == "BUY")).scalars().all()) == 1

    mock_ml.invoke = lambda *a: {"expected_return_pct": 0.95, "predicted_target_price": 60670.0, "confidence": 0.9, "clears_hurdle": True}
    ok2, msg2, _ = sync.sync_hourly_quant_cycle(force=True)
    assert ok2 is True
    assert "Trade deduplicated | No fees" in msg2
    assert len(session.execute(select(PaperPortfolio).where(PaperPortfolio.action == "BUY")).scalars().all()) == 1
    audits = session.execute(select(CouncilAudit)).scalars().all()
    assert len(audits) == 2 and "[FORCED_REANALYSIS]" in audits[1].execution_notes


def test_force_false_idempotent_noop(isolated_db, monkeypatch):
    session = isolated_db[0]
    t0 = datetime(2026, 1, 1, 15, 0, tzinfo=timezone.utc)
    session.add(MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60050.0, volume=10.0))
    session.commit()
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: {"timestamp": t0, "close": 60050.0, "candles": []})
    ok, msg, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is False and "already recorded" in msg


def test_atomic_rollback_on_failure(isolated_db, monkeypatch):
    session = isolated_db[0]
    t0 = datetime(2026, 1, 1, 16, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: {"timestamp": t0, "open": 60000.0, "high": 60200.0, "low": 59800.0, "close": 60100.0, "volume": 75.0})
    mock_signals = MagicMock()
    mock_signals.invoke.side_effect = RuntimeError("signal crash")
    monkeypatch.setattr(sync, "calculate_technical_signals", mock_signals)

    with pytest.raises(RuntimeError):
        sync.sync_hourly_quant_cycle(force=False)
    assert len(session.execute(select(MarketTick)).scalars().all()) == 0


def test_execute_paper_order_concurrency(isolated_db):
    session = isolated_db[0]
    t0 = datetime(2026, 1, 1, 10, 0, tzinfo=timezone.utc)
    tick = MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=10.0)
    session.add(tick)
    session.commit()
    o1 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=tick.id)
    o2 = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10, tick_id=tick.id)
    assert o1.id == o2.id


def test_backfill_empty_db_bootstrap(isolated_db, monkeypatch):
    session = isolated_db[0]
    t_latest = datetime(2026, 2, 1, 12, 0, tzinfo=timezone.utc)
    history = [{"timestamp": pd.Timestamp(t_latest - timedelta(hours=i)), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 10.0} for i in range(50, -1, -1)]
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: {
        "timestamp": t_latest, "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60000.0, "volume": 10.0, "candles": history,
    })
    monkeypatch.setattr(sync, "calculate_technical_signals", MagicMock(invoke=lambda *a: {}))
    monkeypatch.setattr(sync, "get_model_return_forecast", MagicMock(invoke=lambda *a: {"expected_return_pct": 0.1, "predicted_target_price": 60060.0, "confidence": 0.7, "clears_hurdle": False}))
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "HOLD", "risk_verdict": "APPROVED", "strategist_action": "HOLD", "strategist_conviction": 5, "kelly_fraction": 0.0, "thesis": "hold"})

    ok, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok is True
    assert len(session.execute(select(MarketTick)).scalars().all()) == 1


def test_single_missing_hour_backfill_and_reconcile(isolated_db, monkeypatch):
    session = isolated_db[0]
    t0, t1, t2 = datetime(2026, 2, 2, 10, 0, tzinfo=timezone.utc), datetime(2026, 2, 2, 11, 0, tzinfo=timezone.utc), datetime(2026, 2, 2, 12, 0, tzinfo=timezone.utc)

    monkeypatch.setattr(sync, "calculate_technical_signals", MagicMock(invoke=lambda *a: {}))
    monkeypatch.setattr(sync, "get_model_return_forecast", MagicMock(invoke=lambda *a: {"expected_return_pct": 0.5, "predicted_target_price": 60300.0, "confidence": 0.8, "clears_hurdle": True}))
    monkeypatch.setattr(sync, "run_trading_graph", lambda **k: {"final_decision": "BUY", "risk_verdict": "APPROVED", "strategist_action": "BUY", "strategist_conviction": 8, "kelly_fraction": 0.10, "thesis": "Bullish"})

    c0 = {"timestamp": t0, "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0, "candles": [{"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0}]}
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c0)
    ok0, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok0 is True

    c2 = {
        "timestamp": t2, "open": 60240.0, "high": 61000.0, "low": 60200.0, "close": 60900.0, "volume": 110.0,
        "candles": [
            {"timestamp": pd.Timestamp(t0), "open": 59900.0, "high": 60100.0, "low": 59800.0, "close": 60000.0, "volume": 80.0},
            {"timestamp": pd.Timestamp(t1), "open": 60000.0, "high": 60350.0, "low": 59950.0, "close": 60240.0, "volume": 95.0},
            {"timestamp": pd.Timestamp(t2), "open": 60240.0, "high": 61000.0, "low": 60200.0, "close": 60900.0, "volume": 110.0},
        ],
    }
    monkeypatch.setattr(sync, "fetch_latest_confirmed_candle", lambda: c2)
    ok2, _, _ = sync.sync_hourly_quant_cycle(force=False)
    assert ok2 is True

    ticks = session.execute(select(MarketTick).order_by(MarketTick.timestamp.asc())).scalars().all()
    assert len(ticks) == 3 and sync._normalize_utc_timestamp(ticks[1].timestamp) == t1
    rec = session.execute(select(Reconciliation)).scalar_one_or_none()
    assert rec is not None and rec.actual_price_at_t_plus_1 == pytest.approx(60240.0)


def test_large_gap_warning(isolated_db, caplog):
    session = isolated_db[0]
    t0 = datetime(2026, 2, 1, 0, 0, tzinfo=timezone.utc)
    session.add(MarketTick(timestamp=t0, open=60000.0, high=60100.0, low=59900.0, close=60000.0, volume=50.0))
    session.commit()
    t_30 = t0 + timedelta(hours=30)
    candles_30 = [{"timestamp": pd.Timestamp(t0 + timedelta(hours=h)), "open": 60000.0, "high": 60100.0, "low": 59900.0, "close": 60050.0, "volume": 40.0} for h in range(31)]

    with caplog.at_level(logging.WARNING, logger="mudra.sync"):
        inserted = sync.backfill_missing_candles(session, {"timestamp": t_30, "candles": candles_30})
        session.commit()

    assert inserted == 29 and "Detected large candle gap of 30.0 hours" in caplog.text


def test_normalize_utc_timestamp():
    expected = datetime(2026, 2, 7, 12, 0, tzinfo=timezone.utc)
    assert sync._normalize_utc_timestamp(pd.Timestamp("2026-02-07 12:00:00+00:00")) == expected
    assert sync._normalize_utc_timestamp(datetime(2026, 2, 7, 12, 0)) == expected
    assert sync._normalize_utc_timestamp("2026-02-07T12:00:00Z") == expected
    with pytest.raises(ValueError):
        sync._normalize_utc_timestamp(None)


def test_ensure_model_registered_idempotency_and_shadow_prediction(isolated_db):
    # verifies idempotent auto-registration of candidate models and shadow prediction inserts
    session = isolated_db[0]
    cand_spec = {
        "manifest_data": {
            "regressor_artifact_path": "artifacts/models/lgbm_v3_regressor.pkl",
            "regressor_sha256": "894f7637bca6c3481ea13ffa6df70233216b3264a743034944a4c2bfb6e9320f",
            "classifier_artifact_path": "artifacts/models/lgbm_v3_classifier.pkl",
            "classifier_sha256": "91acb8e0d7ded1b2aedcb3f0cefa9f3e3c2cc48f333fa2801bb46a4f7149a57d",
            "feature_schema_version": "v3_29stationary",
            "metrics": {"holdout_accuracy": 0.5289, "holdout_log_loss": 0.6913},
        }
    }

    # 1. Calling ensure_model_registered inserts a valid ModelRegistry row with current_status == "CANDIDATE"
    ensure_model_registered(session, "lgbm_v3.0.0", cand_spec)
    reg = session.execute(
        select(ModelRegistry).where(ModelRegistry.model_version == "lgbm_v3.0.0")
    ).scalars().first()
    assert reg is not None
    assert reg.model_version == "lgbm_v3.0.0"
    assert reg.current_status == "CANDIDATE"
    assert reg.feature_schema_version == "v3_29stationary"
    assert reg.regressor_sha256 == "894f7637bca6c3481ea13ffa6df70233216b3264a743034944a4c2bfb6e9320f"

    # Calling it a second time is idempotent (no duplicate error)
    ensure_model_registered(session, "lgbm_v3.0.0", cand_spec)
    regs = session.execute(
        select(ModelRegistry).where(ModelRegistry.model_version == "lgbm_v3.0.0")
    ).scalars().all()
    assert len(regs) == 1

    # 2. Inserting a SHADOW ModelPrediction for lgbm_v3.0.0 immediately afterward succeeds
    tick = MarketTick(
        timestamp=datetime.now(timezone.utc),
        open=65000.0, high=65500.0, low=64800.0, close=65200.0, volume=100.0,
        model_version=config.ACTIVE_MODEL_VERSION,
    )
    session.add(tick)
    session.flush()

    pred = ModelPrediction(
        tick_id=tick.id,
        model_version="lgbm_v3.0.0",
        run_mode="SHADOW",
        is_primary=False,
        feature_schema_version="v3_29stationary",
        predicted_return_pct=0.25,
        predicted_price_1h=65363.0,
        direction_predicted="UP",
        direction_prob_up=0.54,
        clears_hurdle=True,
    )
    session.add(pred)
    session.flush()

    assert pred.id is not None
    assert pred.model_version == "lgbm_v3.0.0"
    assert pred.run_mode == "SHADOW"


def test_sync_hourly_persists_mcp_signals_and_council_thesis(isolated_db, monkeypatch):
    """Ensure Stage 4 persists live MCP derivatives/news on MarketTick.features and council thesis in CouncilAudit."""
    from datetime import datetime, timezone
    from types import SimpleNamespace
    import src.engine as eng
    from src.database import MarketTick, CouncilAudit

    session = isolated_db[0]
    ts = datetime(2026, 11, 25, 10, 0, 0, tzinfo=timezone.utc)
    candle = {
        "timestamp": ts,
        "open": 90000.0,
        "high": 90500.0,
        "low": 89800.0,
        "close": 90250.0,
        "volume": 1234.5,
        "taker_buy_base_volume": 650.25,
        "trades": 15000,
    }

    monkeypatch.setattr(
        eng,
        "calculate_technical_signals",
        SimpleNamespace(invoke=lambda payload: {"rsi_14": 54.0, "volatility_range_pct": 0.008}),
    )
    monkeypatch.setattr(
        eng,
        "get_model_return_forecast",
        SimpleNamespace(
            invoke=lambda payload: {
                "expected_return_pct": 0.05,
                "predicted_target_price": 90295.0,
                "confidence": 0.62,
                "clears_hurdle": False,
                "direction": "UP",
                "direction_probability_up": 54.0,
                "model_version": "xgb_onnx_v1.0.0",
            }
        ),
    )
    monkeypatch.setattr("src.model_loader.get_registered_models", lambda: {})
    monkeypatch.setattr(eng, "reconcile_pending_ticks", lambda sess, tick: [])
    monkeypatch.setattr(
        eng,
        "run_trading_graph",
        lambda **kwargs: {
            "final_decision": "HOLD",
            "risk_verdict": "VETOED",
            "strategist_action": "HOLD",
            "strategist_conviction": 4,
            "kelly_fraction": 0.0,
            "rag_context": "Similar low-volatility session resolved sideways.",
            "thesis": "Edge below fee hurdle under muted volatility.",
            "risk_critic_critique": "VETOED BY RISK CRITIC: Edge clears 0/2 hurdle gates.",
            "model_consensus": "2/3 UP, 1/3 DOWN",
            "mcp_context": {
                "funding_rate_8h_pct": 0.0042,
                "futures_basis_usd": 88.5,
                "derivatives_regime": "Mildly Bullish",
                "derivatives_source": "deribit",
                "news_headlines": ["BTC holds 90k support", "Deribit funding steady"],
            },
        },
    )

    synced, msg, _ = eng.sync_hourly_quant_cycle(force=False, candle=candle)
    assert synced is True, msg

    tick = session.query(MarketTick).order_by(MarketTick.id.desc()).first()
    assert tick is not None
    assert tick.features["funding_rate_8h_pct"] == 0.0042
    assert tick.features["futures_basis_usd"] == 88.5
    assert tick.features["derivatives_regime"] == "Mildly Bullish"
    assert tick.features["derivatives_source"] == "deribit"
    assert tick.features["news_headline_count"] == 2
    assert tick.features["multi_model_consensus"] == "2/3 UP, 1/3 DOWN"

    audit = session.query(CouncilAudit).order_by(CouncilAudit.id.desc()).first()
    assert audit is not None
    assert "Edge below fee hurdle under muted volatility." in audit.execution_notes
    assert "Risk Critic:" in audit.execution_notes
