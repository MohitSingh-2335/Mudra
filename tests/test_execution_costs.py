# tests/test_execution_costs.py
"""
Deterministic Unit & Regression Tests for Mudra Execution Costs, Fees, Slippage & Hurdle Schema.
TASK-P1-04.2: Transaction Cost & Fee Schema Harmonization.

Covers:
1. Canonical decimal rate invariants and unit boundaries in config.py.
2. Percentage display aliases strictly equal decimal rates multiplied by 100.
3. Programmatic derivation of total per-side and round-trip execution friction.
4. Consumption of canonical config settings across paper portfolio, backtesting, and threshold sweeper.
5. Exact entry/exit fee and slippage accounting without omission or double-counting.
6. Mathematical identity between cash delta and realized P&L on closed paper trades.
7. Caller override preservation in backtesting interfaces.
8. Threshold-sweeper friction execution modeling.
9. Agent hurdle evaluation and consistent, non-contradictory risk/strategy critique texts.
10. Fail-closed validation of environment variables (valid rates, legacy pct, conflicting, negative, non-numeric).
"""

import os
import math
import importlib
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import config
from config import (
    INITIAL_CAPITAL_USD,
    TAKER_FEE_RATE,
    SLIPPAGE_RATE,
    TOTAL_FRICTION_PER_SIDE_RATE,
    ROUND_TRIP_FRICTION_RATE,
    MIN_HURDLE_RATE,
    TAKER_FEE_PCT,
    SLIPPAGE_PCT,
    TOTAL_FRICTION_PER_SIDE_PCT,
    ROUND_TRIP_FRICTION_PCT,
    MIN_HURDLE_PCT,
    _parse_execution_cost_rate,
)
from src.database.models import Base
from src.engine.paper_portfolio import (
    execute_paper_order,
    get_portfolio_summary,
    TAKER_FEE_RATE as PP_TAKER_FEE_RATE,
    SLIPPAGE_RATE as PP_SLIPPAGE_RATE,
)
from src.backtesting import (
    _simulate_from_signals,
    run_backtest,
    TAKER_FEE_RATE as BT_TAKER_FEE_RATE,
    SLIPPAGE_RATE as BT_SLIPPAGE_RATE,
)
from scripts.experiments.sweep_threshold import (
    run_backtest_threshold,
    TOTAL_FRICTION as SW_TOTAL_FRICTION,
    FEE_RATE as SW_FEE_RATE,
)
from src.agents.risk_critic_node import risk_critic_node
from src.agents.strategist_node import strategist_node
from src.agents.graph_state import AgentState


# -----------------------------------------------------------------------------
# 1. Canonical Cost Rates, Derived Invariants & Display Units
# -----------------------------------------------------------------------------

def test_canonical_cost_rates_and_decimal_units():
    """Verify that canonical cost settings are strictly decimal fractions (not percentages)."""
    assert TAKER_FEE_RATE == pytest.approx(0.0005, abs=1e-8)  # 0.05%
    assert SLIPPAGE_RATE == pytest.approx(0.0002, abs=1e-8)   # 0.02%
    assert 0.0 < TAKER_FEE_RATE < 0.01
    assert 0.0 < SLIPPAGE_RATE < 0.01


def test_derived_friction_and_hurdle_formulas():
    """Verify programmatic mathematical derivation of per-side and round-trip friction."""
    expected_per_side = TAKER_FEE_RATE + SLIPPAGE_RATE
    assert TOTAL_FRICTION_PER_SIDE_RATE == pytest.approx(expected_per_side, abs=1e-8)
    assert TOTAL_FRICTION_PER_SIDE_RATE == pytest.approx(0.0007, abs=1e-8)  # 7 bps

    expected_round_trip = 1.0 - (1.0 - TOTAL_FRICTION_PER_SIDE_RATE) ** 2
    assert ROUND_TRIP_FRICTION_RATE == pytest.approx(expected_round_trip, abs=1e-8)
    assert ROUND_TRIP_FRICTION_RATE == pytest.approx(0.00139951, abs=1e-6)  # ~14 bps

    # Minimum hurdle break-even floor strictly equals round-trip friction
    assert MIN_HURDLE_RATE == pytest.approx(ROUND_TRIP_FRICTION_RATE, abs=1e-8)


def test_percentage_display_aliases():
    """Verify that percentage display values strictly equal decimal rates multiplied by 100."""
    assert TAKER_FEE_PCT == pytest.approx(TAKER_FEE_RATE * 100.0, abs=1e-8)
    assert SLIPPAGE_PCT == pytest.approx(SLIPPAGE_RATE * 100.0, abs=1e-8)
    assert TOTAL_FRICTION_PER_SIDE_PCT == pytest.approx(TOTAL_FRICTION_PER_SIDE_RATE * 100.0, abs=1e-8)
    assert ROUND_TRIP_FRICTION_PCT == pytest.approx(ROUND_TRIP_FRICTION_RATE * 100.0, abs=1e-8)
    assert MIN_HURDLE_PCT == pytest.approx(MIN_HURDLE_RATE * 100.0, abs=1e-8)

    assert TAKER_FEE_PCT == pytest.approx(0.05, abs=1e-4)
    assert SLIPPAGE_PCT == pytest.approx(0.02, abs=1e-4)
    assert TOTAL_FRICTION_PER_SIDE_PCT == pytest.approx(0.07, abs=1e-4)
    assert ROUND_TRIP_FRICTION_PCT == pytest.approx(0.139951, abs=1e-4)


# -----------------------------------------------------------------------------
# 2. SSOT Consumer Harmonization
# -----------------------------------------------------------------------------

def test_consumers_adopt_canonical_configuration():
    """Verify that engine, backtesting, and threshold sweeper import SSOT constants."""
    assert PP_TAKER_FEE_RATE == config.TAKER_FEE_RATE
    assert PP_SLIPPAGE_RATE == config.SLIPPAGE_RATE

    assert BT_TAKER_FEE_RATE == config.TAKER_FEE_RATE
    assert BT_SLIPPAGE_RATE == config.SLIPPAGE_RATE

    assert SW_TOTAL_FRICTION == config.TOTAL_FRICTION_PER_SIDE_RATE
    assert SW_FEE_RATE == config.TAKER_FEE_RATE


# -----------------------------------------------------------------------------
# 3. Environment Variable Parsing (Valid, Legacy, Invalid & Conflicting)
# -----------------------------------------------------------------------------

def test_env_var_parsing_valid_decimal_rate(monkeypatch):
    """Verify that setting a valid rate environment variable updates configuration."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", "0.0010")
    monkeypatch.delenv("CUSTOM_TAKER_PCT", raising=False)
    rate = _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)
    assert rate == pytest.approx(0.0010)


def test_env_var_parsing_legacy_percentage(monkeypatch):
    """Verify that setting a legacy percentage variable properly divides by 100."""
    monkeypatch.delenv("CUSTOM_TAKER_RATE", raising=False)
    monkeypatch.setenv("CUSTOM_TAKER_PCT", "0.10")  # 0.10% -> 0.0010
    rate = _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)
    assert rate == pytest.approx(0.0010)


def test_env_var_parsing_consistent_both(monkeypatch):
    """Verify that setting both variables with equivalent values succeeds."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", "0.0005")
    monkeypatch.setenv("CUSTOM_TAKER_PCT", "0.05")
    rate = _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)
    assert rate == pytest.approx(0.0005)


def test_env_var_parsing_conflicting_raises(monkeypatch):
    """Verify that setting conflicting rate and percentage variables raises ValueError."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", "0.0005")
    monkeypatch.setenv("CUSTOM_TAKER_PCT", "0.10")  # 0.10% != 0.0005
    with pytest.raises(ValueError, match="Conflicting environment configuration"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)


def test_env_var_parsing_ambiguous_percentage_in_rate_var_rejected(monkeypatch):
    """Verify that passing a percentage (e.g. 0.05) to a decimal rate variable fails closed."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", "0.05")  # Looks like 0.05% mistakenly entered as 0.05 rate (>0.01 max)
    monkeypatch.delenv("CUSTOM_TAKER_PCT", raising=False)
    with pytest.raises(ValueError, match="exceeds maximum allowed rate"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005, max_rate=0.01)


def test_env_var_parsing_negative_rejected(monkeypatch):
    """Verify that negative rates or percentages fail closed."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", "-0.0005")
    monkeypatch.delenv("CUSTOM_TAKER_PCT", raising=False)
    with pytest.raises(ValueError, match="cannot be negative"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)

    monkeypatch.delenv("CUSTOM_TAKER_RATE", raising=False)
    monkeypatch.setenv("CUSTOM_TAKER_PCT", "-0.05")
    with pytest.raises(ValueError, match="cannot be negative"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)


def test_env_var_parsing_non_numeric_rejected(monkeypatch):
    """Verify that non-numeric strings fail closed."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", "invalid_number")
    monkeypatch.delenv("CUSTOM_TAKER_PCT", raising=False)
    with pytest.raises(ValueError, match="Invalid non-numeric value"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)


@pytest.mark.parametrize("val", ["nan", "inf", "-inf"])
def test_env_var_parsing_non_finite_rate_rejected(monkeypatch, val):
    """Verify that nan, inf, and -inf in decimal rate variables fail closed."""
    monkeypatch.setenv("CUSTOM_TAKER_RATE", val)
    monkeypatch.delenv("CUSTOM_TAKER_PCT", raising=False)
    with pytest.raises(ValueError, match="Invalid non-finite value for environment variable CUSTOM_TAKER_RATE"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)


@pytest.mark.parametrize("val", ["nan", "inf", "-inf"])
def test_env_var_parsing_non_finite_percentage_rejected(monkeypatch, val):
    """Verify that nan, inf, and -inf in percentage variables fail closed."""
    monkeypatch.delenv("CUSTOM_TAKER_RATE", raising=False)
    monkeypatch.setenv("CUSTOM_TAKER_PCT", val)
    with pytest.raises(ValueError, match="Invalid non-finite value for environment variable CUSTOM_TAKER_PCT"):
        _parse_execution_cost_rate("CUSTOM_TAKER_RATE", "CUSTOM_TAKER_PCT", default_rate=0.0005)


def test_hurdle_invariant_valid_override_above_friction(monkeypatch):
    """Verify that configuring a hurdle rate above estimated round-trip friction succeeds."""
    monkeypatch.setenv("MIN_HURDLE_RATE", "0.0025")
    monkeypatch.delenv("MIN_HURDLE_PCT", raising=False)
    try:
        importlib.reload(config)
        assert config.MIN_HURDLE_RATE == pytest.approx(0.0025)
        assert config.MIN_HURDLE_RATE > config.ROUND_TRIP_FRICTION_RATE
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_hurdle_invariant_default_reuses_friction_directly(monkeypatch):
    """Verify that by default (no env override), MIN_HURDLE_RATE reuses ROUND_TRIP_FRICTION_RATE directly."""
    monkeypatch.delenv("MIN_HURDLE_RATE", raising=False)
    monkeypatch.delenv("MIN_HURDLE_PCT", raising=False)
    try:
        importlib.reload(config)
        assert config.MIN_HURDLE_RATE == config.ROUND_TRIP_FRICTION_RATE
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_hurdle_invariant_default_decimal_representation_accepted(monkeypatch):
    """
    Verify that explicitly configuring the default decimal representation:
    - Decimal rate: '0.00139951' (exact mathematical base-10 value of 1 - (1 - 0.0007)^2)
    - Percentage:   '0.139951'
    is strictly accepted under the decimal configuration contract.
    """
    # 1. Decimal rate format
    monkeypatch.setenv("MIN_HURDLE_RATE", "0.00139951")
    monkeypatch.delenv("MIN_HURDLE_PCT", raising=False)
    try:
        importlib.reload(config)
        assert config.MIN_HURDLE_RATE == pytest.approx(0.00139951, abs=1e-8)
    finally:
        monkeypatch.undo()
        importlib.reload(config)

    # 2. Percentage format
    monkeypatch.delenv("MIN_HURDLE_RATE", raising=False)
    monkeypatch.setenv("MIN_HURDLE_PCT", "0.139951")
    try:
        importlib.reload(config)
        assert config.MIN_HURDLE_RATE == pytest.approx(0.00139951, abs=1e-8)
        assert config.MIN_HURDLE_PCT == pytest.approx(0.139951, abs=1e-6)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


def test_hurdle_invariant_genuinely_sub_floor_rejected(monkeypatch):
    """
    Verify that genuinely sub-floor values are strictly rejected:
    - '0.00139950' (just 1 satoshi / 0.0001 bps below floor)
    - '0.0010' (10 bps, below 14 bps floor)
    - '0.139950%'
    - '0.05%' (legacy 5 bps)
    """
    # 1. Exactly 1 satoshi / 0.00000001 below the exact floor
    monkeypatch.setenv("MIN_HURDLE_RATE", "0.00139950")
    monkeypatch.delenv("MIN_HURDLE_PCT", raising=False)
    try:
        with pytest.raises(ValueError, match="cannot be less than estimated ROUND_TRIP_FRICTION_RATE"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)

    # 2. Meaningfully below floor (0.0010)
    monkeypatch.setenv("MIN_HURDLE_RATE", "0.0010")
    monkeypatch.delenv("MIN_HURDLE_PCT", raising=False)
    try:
        with pytest.raises(ValueError, match="cannot be less than estimated ROUND_TRIP_FRICTION_RATE"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)

    # 3. Sub-floor percentage (0.139950%)
    monkeypatch.delenv("MIN_HURDLE_RATE", raising=False)
    monkeypatch.setenv("MIN_HURDLE_PCT", "0.139950")
    try:
        with pytest.raises(ValueError, match="cannot be less than estimated ROUND_TRIP_FRICTION_RATE"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)

    # 4. Legacy 0.05% (5 bps)
    monkeypatch.delenv("MIN_HURDLE_RATE", raising=False)
    monkeypatch.setenv("MIN_HURDLE_PCT", "0.05")
    try:
        with pytest.raises(ValueError, match="cannot be less than estimated ROUND_TRIP_FRICTION_RATE"):
            importlib.reload(config)
    finally:
        monkeypatch.undo()
        importlib.reload(config)


# -----------------------------------------------------------------------------
# 4. Paper Portfolio Friction Accounting & Mathematical Integrity
# -----------------------------------------------------------------------------

@pytest.fixture
def test_db():
    """Isolated, fresh in-memory SQLite session per test."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_paper_portfolio_exact_cost_deduction(test_db):
    """
    Verifies exact fee and slippage dollar deductions using canonical decimal rates:
    - On BUY of $1,000 at $60,000:
      Fee = $1,000 * 0.0005 = $0.50
      Slippage = $1,000 * 0.0002 = $0.20
      Exec price = $60,000 * (1 + 0.0002) = $60,012.00
      Net capital = $999.50
      BTC acquired = 999.50 / 60,012.00 = 0.01665499...
    """
    session = test_db
    r_buy = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)

    expected_capital = INITIAL_CAPITAL_USD * 0.10  # $1,000
    expected_fee = expected_capital * TAKER_FEE_RATE  # $0.50
    expected_slip = expected_capital * SLIPPAGE_RATE  # $0.20
    expected_exec_price = 60000.0 * (1.0 + SLIPPAGE_RATE)  # $60,012.00
    expected_btc = (expected_capital - expected_fee) / expected_exec_price

    assert r_buy.cash_balance == pytest.approx(INITIAL_CAPITAL_USD - expected_capital, abs=1e-4)
    assert r_buy.fee_deducted_usd == pytest.approx(expected_fee, abs=1e-4)
    assert r_buy.slippage_usd == pytest.approx(expected_slip, abs=1e-4)
    assert r_buy.execution_price == pytest.approx(expected_exec_price, abs=1e-4)
    assert r_buy.btc_amount == pytest.approx(expected_btc, abs=1e-7)
    assert r_buy.realized_pnl_usd == 0.0


def test_paper_portfolio_cash_delta_equals_realized_pnl(test_db):
    """
    Verifies that for a closed round-trip trade, total cash change strictly
    equals net realized P&L, proving zero fee/slippage double-counting or omission.
    """
    session = test_db

    # Buy at $60,000 ($1,000 deployed)
    r_buy = execute_paper_order(session, "BUY", 60000.0, kelly_fraction=0.10)
    # Sell at $63,000 (entire position liquidated)
    r_sell = execute_paper_order(session, "SELL", 63000.0)

    # Invariant: cash delta from starting bankroll strictly equals net realized P&L
    cash_delta = r_sell.cash_balance - INITIAL_CAPITAL_USD
    assert cash_delta == pytest.approx(r_sell.realized_pnl_usd, abs=1e-3)

    summary = get_portfolio_summary(session, current_price=63000.0)
    assert summary["realized_pnl_usd"] == pytest.approx(r_sell.realized_pnl_usd, abs=1e-2)
    assert summary["total_fees_usd"] == pytest.approx(round(r_buy.fee_deducted_usd + r_sell.fee_deducted_usd, 2), abs=1e-2)


# -----------------------------------------------------------------------------
# 5. Core Backtesting Overrides & Execution Modeling
# -----------------------------------------------------------------------------

def test_backtesting_simulation_caller_overrides():
    """Verifies that _simulate_from_signals and run_backtest accept caller overrides."""
    timestamps = pd.date_range("2026-01-01", periods=4, freq="1h")
    test_df = pd.DataFrame({
        "timestamp": timestamps,
        "close": [100.0, 105.0, 102.0, 100.0]
    })
    signals = np.array([1, 0, 0, 0])  # Buy at t=0, sell at t=1

    # Default canonical execution friction
    eq_default, tl_default = _simulate_from_signals(test_df, signals)
    assert len(tl_default) == 1
    # Total friction per side = 0.0007
    expected_entry_eq = 10000.0 * (1.0 - 0.0007)
    gross_return = (105.0 / 100.0) - 1.0  # +5%
    expected_exit_eq = expected_entry_eq * (1.0 + gross_return) * (1.0 - 0.0007)
    assert tl_default["equity_after"].iloc[0] == pytest.approx(round(expected_exit_eq, 2), abs=1e-2)

    # Caller override: higher historical fees (0.10% taker + 0.05% slippage = 0.0015 per side)
    eq_override, tl_override = _simulate_from_signals(
        test_df,
        signals,
        taker_fee_rate=0.0010,
        slippage_rate=0.0005,
    )
    expected_entry_override = 10000.0 * (1.0 - 0.0015)
    expected_exit_override = expected_entry_override * (1.0 + gross_return) * (1.0 - 0.0015)
    assert tl_override["equity_after"].iloc[0] == pytest.approx(round(expected_exit_override, 2), abs=1e-2)
    # Higher friction must strictly result in lower ending equity
    assert tl_override["equity_after"].iloc[0] < tl_default["equity_after"].iloc[0]


def test_run_backtest_interface_override():
    """Verifies that run_backtest accepts optional taker_fee_rate and slippage_rate."""
    timestamps = pd.date_range("2026-01-01", periods=3, freq="1h")
    # Include minimal required classifier features
    test_df = pd.DataFrame({
        "timestamp": timestamps,
        "close": [100.0, 105.0, 102.0],
    })
    for col in config.CLASSIFIER_FEATURES:
        test_df[col] = 1.0

    class MockModel:
        def predict(self, X):
            return np.array([1, 0, 0])

    eq_def, tl_def = run_backtest(test_df, MockModel())
    eq_ov, tl_ov = run_backtest(test_df, MockModel(), taker_fee_rate=0.002, slippage_rate=0.001)

    assert tl_ov["equity_after"].iloc[0] < tl_def["equity_after"].iloc[0]


# -----------------------------------------------------------------------------
# 6. Threshold Sweeper Modeling
# -----------------------------------------------------------------------------

def test_threshold_sweeper_incorporates_canonical_slippage():
    """Verifies that sweep_threshold.py models slippage alongside taker fees."""
    timestamps = pd.date_range("2026-01-01", periods=3, freq="1h")
    test_df = pd.DataFrame({
        "timestamp": timestamps,
        "close": [100.0, 110.0, 105.0],
    })
    for col in config.CLASSIFIER_FEATURES:
        test_df[col] = 1.0

    class MockScaler:
        def transform(self, X):
            return X

    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8], [0.8, 0.2], [0.8, 0.2]])

    eq_df, tl_df = run_backtest_threshold(test_df, MockModel(), MockScaler(), threshold=0.50)
    assert len(tl_df) == 1

    # Total friction per leg = 0.0007 (0.05% fee + 0.02% slippage)
    entry_eq = 10000.0 * (1.0 - TOTAL_FRICTION_PER_SIDE_RATE)
    exit_eq = entry_eq * (1.0 + 0.10) * (1.0 - TOTAL_FRICTION_PER_SIDE_RATE)
    expected_net_ret = (exit_eq / (entry_eq / (1.0 - TOTAL_FRICTION_PER_SIDE_RATE))) - 1.0

    assert tl_df["net_return_pct"].iloc[0] == pytest.approx(expected_net_ret * 100.0, abs=1e-4)


# -----------------------------------------------------------------------------
# 7. Agent Hurdle & Critique Non-Contradiction
# -----------------------------------------------------------------------------

def test_risk_critic_veto_text_reflects_canonical_config():
    """Verifies that risk_critic_node text accurately cites canonical round-trip break-even hurdle."""
    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {"close": 65000.0},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": {
            "expected_return_pct": 0.05,  # Below the ~0.14% round-trip hurdle
            "clears_hurdle": False,
        },
        "trade_action": "BUY",
        "conviction_score": 7,
    }

    out = risk_critic_node(state)
    assert out["risk_approved"] is False
    assert "Sub-Hurdle Fee Drag" in out["veto_reason"]
    # Check that critique explicitly quotes canonical hurdle and per-side friction
    assert f"{MIN_HURDLE_PCT:.2f}%" in out["veto_reason"]
    assert f"{ROUND_TRIP_FRICTION_PCT:.2f}%" in out["risk_critic_critique"] or f"{MIN_HURDLE_PCT:.2f}%" in out["risk_critic_critique"]
    assert f"{TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side" in out["risk_critic_critique"]
    # Verify historical 0.30% round-trip string is not present in veto text
    assert "0.30% round-trip" not in out["risk_critic_critique"]


def test_risk_critic_hold_text_reflects_canonical_config():
    """Verifies that risk_critic_node HOLD critique cites canonical friction."""
    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "trade_action": "HOLD",
        "ml_forecast": {"expected_return_pct": 0.0, "clears_hurdle": False},
        "engineered_features": {"volatility_range_pct": 1.0},
    }

    out = risk_critic_node(state)
    assert out["risk_approved"] is True
    assert f"{ROUND_TRIP_FRICTION_PCT:.2f}% round-trip friction" in out["risk_critic_critique"]
    assert f"{TOTAL_FRICTION_PER_SIDE_PCT:.2f}% per side" in out["risk_critic_critique"]
    assert "0.30% round-trip" not in out["risk_critic_critique"]


def test_strategist_heuristic_text_reflects_canonical_config():
    """Verifies that strategist_node heuristic summary cites canonical hurdle."""
    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "ml_forecast": {
            "expected_return_pct": 0.25,
            "clears_hurdle": True,
            "direction": "UP",
            "direction_badge": "BULLISH",
            "direction_probability_up": 65.0,
        },
        "engineered_features": {"volatility_range_pct": 1.0},
        "market_data": {"taker_buy_ratio": 0.55},
        "rag_context": "",
    }

    out = strategist_node(state)
    assert out["trade_action"] == "BUY"
    assert f"{MIN_HURDLE_PCT:.2f}% estimated round-trip hurdle" in out["executive_summary"]
    assert "0.05% fee hurdle" not in out["executive_summary"]
