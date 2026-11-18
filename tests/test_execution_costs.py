import pytest
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from unittest.mock import MagicMock

import config
from config import (
    INITIAL_CAPITAL_USD, TAKER_FEE_RATE, SLIPPAGE_RATE, TOTAL_FRICTION_PER_SIDE_RATE,
    ROUND_TRIP_FRICTION_RATE, MIN_HURDLE_RATE, TAKER_FEE_PCT, SLIPPAGE_PCT,
    ROUND_TRIP_FRICTION_PCT, MIN_HURDLE_PCT, _parse_execution_cost_rate,
)
from src.database.models import Base
from src.engine.paper_portfolio import execute_paper_order, get_portfolio_summary
from src.backtesting import _simulate_from_signals
from scripts.experiments.sweep_threshold import run_backtest_threshold
from src.agents.council import risk_critic_node, strategist_node, run_trading_graph
import app


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


@pytest.fixture
def test_db():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(bind=engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()


def test_paper_portfolio_exact_cost_deduction(test_db):
    r_buy = execute_paper_order(test_db, "BUY", 60000.0, kelly_fraction=0.10)
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

    execute_paper_order(test_db, "SELL", 62000.0)
    summary = get_portfolio_summary(test_db, current_price=62000.0)
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


def test_tab2_scenarios_and_graph(monkeypatch):
    monkeypatch.setattr("src.agents.council._get_gemini_api_key", lambda: None)

    for scenario, expected_action in [("chop", "HOLD"), ("bullish", "BUY"), ("vol_spike", "SELL")]:
        m, t, f = app.build_council_scenario_payload(scenario, 65000.0)
        state = run_trading_graph("BTCUSDT", 65000.0, m, t, f)
        assert state["strategist_action"] == expected_action


def test_format_portfolio_rationale():
    assert f"below {MIN_HURDLE_PCT:.2f}% round-trip fee hurdle" in app.format_portfolio_rationale({"action": "HOLD", "notes": "Sub-hurdle fee drag: preserving capital."})
    assert "Sideways chop" in app.format_portfolio_rationale({"action": "HOLD", "notes": "Neutral regime"})
    assert "Risk Veto" in app.format_portfolio_rationale({"action": "HOLD", "notes": "Order vetoed due to volatility spike"})


def test_model_pedigree_markdown(monkeypatch):
    md = app.build_model_pedigree_markdown()
    assert "0.30% (Historical)" in md and f"`{ROUND_TRIP_FRICTION_PCT:.2f}%`" in md
