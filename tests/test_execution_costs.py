# tests/test_execution_costs.py
"""
Deterministic Unit & Regression Tests for Mudra Execution Costs, Fees, Slippage & Hurdle Schema.
Transaction Cost & Fee Schema Harmonization.

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


# -----------------------------------------------------------------------------
# 8. Streamlit UI Live Forecast Hurdle Alignment
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "raw_pred_ret,clf_code,expected_cleared,expected_badge",
    [
        (0.0010, 1, False, "SIDEWAYS (CHOP)"),   # +0.10% > legacy 0.05%, but < MIN_HURDLE_PCT (0.139951%)
        (-0.0010, 0, False, "SIDEWAYS (CHOP)"),  # |-0.10%| > legacy 0.05%, but < MIN_HURDLE_PCT (0.139951%)
        (0.0015, 1, True, "BULLISH (UP)"),       # +0.15% >= MIN_HURDLE_PCT
        (-0.0015, 0, True, "BEARISH (DOWN)"),    # |-0.15%| >= MIN_HURDLE_PCT
    ],
)
def test_ui_generate_live_forecast_enforces_canonical_min_hurdle_pct(
    monkeypatch, raw_pred_ret, clf_code, expected_cleared, expected_badge
):
    """
    Verifies that app.generate_live_forecast evaluates cleared_hurdle against
    canonical MIN_HURDLE_PCT (0.139951%) rather than the legacy 0.05% constant.
    Specifically proves that +0.10% and -0.10% moves do NOT clear the hurdle.
    """
    from unittest.mock import MagicMock
    import app
    import src.feature_engineering as fe

    live_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=40, freq="1h", tz="UTC"),
        "open": [65000.0] * 40,
        "high": [65200.0] * 40,
        "low": [64800.0] * 40,
        "close": [65000.0] * 40,
        "volume": [100.0] * 40,
        "taker_buy_ratio": [0.55] * 40,
    })

    dummy_features = {col: 1.0 for col in config.CLASSIFIER_FEATURES}
    dummy_features["vol_1h"] = 1.0
    monkeypatch.setattr(fe, "extract_features_for_inference", lambda df: dummy_features)
    monkeypatch.setattr(
        fe,
        "validate_inference_features",
        lambda features, model_type="classifier": pd.DataFrame([dummy_features]),
    )
    monkeypatch.setattr(
        fe,
        "predict_with_validation",
        lambda model, df_in, model_type="regressor": (
            np.array([raw_pred_ret]) if model_type == "regressor" else np.array([clf_code])
        ),
    )
    monkeypatch.setattr(
        fe,
        "predict_proba_with_validation",
        lambda model, df_in: np.array([[0.25, 0.75]]) if clf_code == 1 else np.array([[0.75, 0.25]]),
    )

    res = app.generate_live_forecast(
        live_df=live_df,
        reg_model=MagicMock(),
        clf_model=MagicMock(),
        is_stale=False,
    )
    assert res is not None
    assert res["pred_ret_pct"] == pytest.approx(raw_pred_ret * 100.0, abs=1e-6)
    assert res["cleared_hurdle"] is expected_cleared
    assert res["direction_badge"] == expected_badge


@pytest.mark.parametrize(
    "raw_pred_ret,clf_code,expected_pct,expected_rel,expected_cleared,expected_badge",
    [
        (
            math.nextafter(MIN_HURDLE_RATE, 0.0),
            1,
            math.nextafter(MIN_HURDLE_PCT, 0.0),
            "below",
            False,
            "SIDEWAYS (CHOP)",
        ),
        (
            MIN_HURDLE_RATE,
            1,
            MIN_HURDLE_PCT,
            "exact",
            True,
            "BULLISH (UP)",
        ),
        (
            math.nextafter(MIN_HURDLE_RATE, math.inf),
            1,
            math.nextafter(MIN_HURDLE_PCT, math.inf),
            "above",
            True,
            "BULLISH (UP)",
        ),
        (
            math.nextafter(-MIN_HURDLE_RATE, 0.0),
            0,
            -math.nextafter(MIN_HURDLE_PCT, 0.0),
            "below",
            False,
            "SIDEWAYS (CHOP)",
        ),
        (
            -MIN_HURDLE_RATE,
            0,
            -MIN_HURDLE_PCT,
            "exact",
            True,
            "BEARISH (DOWN)",
        ),
        (
            math.nextafter(-MIN_HURDLE_RATE, -math.inf),
            0,
            -math.nextafter(MIN_HURDLE_PCT, math.inf),
            "above",
            True,
            "BEARISH (DOWN)",
        ),
    ],
)
def test_ui_generate_live_forecast_exact_min_hurdle_pct_boundary(
    monkeypatch,
    raw_pred_ret,
    clf_code,
    expected_pct,
    expected_rel,
    expected_cleared,
    expected_badge,
):
    """
    Exercises the exact IEEE-754 float64 hurdle boundary in app.generate_live_forecast()
    accounting for pred_ret_pct = raw_pred_ret * 100.0:
      - At raw_pred_ret = ±MIN_HURDLE_RATE, abs(pred_ret_pct) == MIN_HURDLE_PCT and cleared_hurdle is True.
      - At the immediately lower representable magnitude math.nextafter(±MIN_HURDLE_RATE, 0.0),
        abs(pred_ret_pct) == math.nextafter(MIN_HURDLE_PCT, 0.0) < MIN_HURDLE_PCT and cleared_hurdle is False.
      - At the immediately higher representable magnitude math.nextafter(±MIN_HURDLE_RATE, ±math.inf),
        abs(pred_ret_pct) == math.nextafter(MIN_HURDLE_PCT, math.inf) > MIN_HURDLE_PCT and cleared_hurdle is True.
    """
    from unittest.mock import MagicMock
    import app
    import src.feature_engineering as fe

    # Pre-verify that the chosen raw_pred_ret input genuinely maps to the exact target boundary
    # after app.generate_live_forecast()'s pred_ret_pct = raw_pred_ret * 100.0 conversion.
    computed_pct = raw_pred_ret * 100.0
    assert computed_pct == expected_pct
    if expected_rel == "exact":
        assert abs(computed_pct) == MIN_HURDLE_PCT
    elif expected_rel == "below":
        assert abs(computed_pct) == math.nextafter(MIN_HURDLE_PCT, 0.0)
        assert abs(computed_pct) < MIN_HURDLE_PCT
    elif expected_rel == "above":
        assert abs(computed_pct) == math.nextafter(MIN_HURDLE_PCT, math.inf)
        assert abs(computed_pct) > MIN_HURDLE_PCT

    live_df = pd.DataFrame({
        "timestamp": pd.date_range("2026-07-01", periods=40, freq="1h", tz="UTC"),
        "open": [65000.0] * 40,
        "high": [65200.0] * 40,
        "low": [64800.0] * 40,
        "close": [65000.0] * 40,
        "volume": [100.0] * 40,
        "taker_buy_ratio": [0.55] * 40,
    })

    dummy_features = {col: 1.0 for col in config.CLASSIFIER_FEATURES}
    dummy_features["vol_1h"] = 1.0
    monkeypatch.setattr(fe, "extract_features_for_inference", lambda df: dummy_features)
    monkeypatch.setattr(
        fe,
        "validate_inference_features",
        lambda features, model_type="classifier": pd.DataFrame([dummy_features]),
    )
    monkeypatch.setattr(
        fe,
        "predict_with_validation",
        lambda model, df_in, model_type="regressor": (
            np.array([raw_pred_ret], dtype=np.float64)
            if model_type == "regressor"
            else np.array([clf_code], dtype=np.int64)
        ),
    )
    monkeypatch.setattr(
        fe,
        "predict_proba_with_validation",
        lambda model, df_in: (
            np.array([[0.25, 0.75]], dtype=np.float64)
            if clf_code == 1
            else np.array([[0.75, 0.25]], dtype=np.float64)
        ),
    )

    res = app.generate_live_forecast(
        live_df=live_df,
        reg_model=MagicMock(),
        clf_model=MagicMock(),
        is_stale=False,
    )
    assert res is not None
    assert res["pred_ret_pct"] == expected_pct
    assert res["cleared_hurdle"] is expected_cleared
    assert res["direction_badge"] == expected_badge


def test_app_ui_imports_min_hurdle_pct_and_eliminates_hardcoded_005_hurdle():
    """
    Verifies that app.py imports MIN_HURDLE_PCT from config and contains no hardcoded
    'Cleared 0.05% Hurdle' or '>= 0.05' hurdle check in generate_live_forecast / Tab 1.
    """
    from pathlib import Path
    import app

    assert hasattr(app, "MIN_HURDLE_PCT")
    assert app.MIN_HURDLE_PCT == pytest.approx(config.MIN_HURDLE_PCT, abs=1e-8)

    app_source = Path(app.__file__).read_text(encoding="utf-8")
    assert "Cleared 0.05% Hurdle" not in app_source
    assert "abs(pred_ret_pct) >= 0.05" not in app_source
    assert "abs(pred_ret_pct) >= MIN_HURDLE_PCT" in app_source
    assert 'f"Cleared {MIN_HURDLE_PCT:.2f}% Hurdle [PASS]"' in app_source


# -----------------------------------------------------------------------------
# 9. Residual Cost, Risk-Boundary & Scenario Consistency
# -----------------------------------------------------------------------------

@pytest.mark.parametrize(
    "unsupported_action",
    [
        "SHORT",
        "LONG",
        " close ",
        "",
        "   ",
        None,
        123,
        "__OMIT__",
    ],
)
def test_risk_critic_rejects_unsupported_trade_actions_fail_closed(
    monkeypatch, unsupported_action
):
    """
    Verifies that risk_critic_node allows only 'HOLD', 'BUY', and 'SELL' and fails closed
    on any unsupported action ('SHORT', 'LONG', empty/whitespace string, None, non-string,
    or omitted trade_action):
      - risk_approved is False
      - recommended_position_pct == 0.0
      - stop_loss_price is None and take_profit_price is None
      - deterministic veto_reason cites 'Unsupported Trade Action'
      - never invokes the LLM approval path or treats 'SHORT' as 'SELL'
    """
    import sys
    import types

    rcn_mod = sys.modules["src.agents.risk_critic_node"]
    llm_calls = []

    class ForbiddenGenerativeModel:
        def __init__(self, model_name):
            llm_calls.append(model_name)

        def generate_content(self, prompt):
            llm_calls.append(prompt)
            raise AssertionError("LLM approval path must not be reached for unsupported actions")

    dummy_genai = types.ModuleType("google.generativeai")
    dummy_genai.configure = lambda **kwargs: None
    dummy_genai.GenerativeModel = ForbiddenGenerativeModel

    google_pkg = sys.modules.get("google")
    if google_pkg is None:
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "google", google_pkg)

    monkeypatch.setitem(sys.modules, "google.generativeai", dummy_genai)
    monkeypatch.setattr(google_pkg, "generativeai", dummy_genai, raising=False)
    monkeypatch.setattr(rcn_mod, "_get_gemini_api_key", lambda: "test-offline-api-key")

    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {"close": 65000.0},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": {
            "expected_return_pct": 0.50,
            "clears_hurdle": True,
        },
        "conviction_score": 8,
        "circuit_breaker_active": False,
    }
    if unsupported_action != "__OMIT__":
        state["trade_action"] = unsupported_action

    out = risk_critic_node(state)
    assert out["risk_approved"] is False
    assert out["recommended_position_pct"] == 0.0
    assert out["stop_loss_price"] is None
    assert out["take_profit_price"] is None
    assert out["veto_reason"] is not None
    assert "Unsupported Trade Action" in out["veto_reason"]
    assert "VETOED BY RISK CRITIC" in out["risk_critic_critique"]
    assert len(llm_calls) == 0

    # Verify Gate 0 also runs BEFORE converting malformed ancillary state fields without raising
    malformed_ancillary_state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": "malformed_price",
        "market_data": None,
        "engineered_features": {"volatility_range_pct": "malformed_vol"},
        "ml_forecast": "malformed_forecast",
        "conviction_score": "malformed_conviction",
        "circuit_breaker_active": False,
    }
    if unsupported_action != "__OMIT__":
        malformed_ancillary_state["trade_action"] = unsupported_action

    out_malformed = risk_critic_node(malformed_ancillary_state)
    assert out_malformed["risk_approved"] is False
    assert out_malformed["recommended_position_pct"] == 0.0
    assert out_malformed["stop_loss_price"] is None
    assert out_malformed["take_profit_price"] is None
    assert out_malformed["veto_reason"] is not None
    assert "Unsupported Trade Action" in out_malformed["veto_reason"]
    assert len(llm_calls) == 0


@pytest.mark.parametrize(
    "raw_action,expected_normalized,expected_ret,expected_pos_gt_zero",
    [
        ("hold", "HOLD", 0.0, False),
        ("  hold  ", "HOLD", 0.0, False),
        ("buy", "BUY", 0.25, True),
        ("  buy\n", "BUY", 0.25, True),
        ("sell", "SELL", -0.25, True),
        ("  sell  ", "SELL", -0.25, True),
    ],
)
def test_risk_critic_normalizes_lowercase_and_padded_valid_actions(
    monkeypatch,
    raw_action,
    expected_normalized,
    expected_ret,
    expected_pos_gt_zero,
):
    """
    Verifies that lowercase/whitespace-padded 'hold', 'buy', and 'sell' actions are
    normalized to 'HOLD', 'BUY', and 'SELL' and preserve their intended risk semantics.
    """
    import sys

    rcn_mod = sys.modules["src.agents.risk_critic_node"]
    monkeypatch.setattr(rcn_mod, "_get_gemini_api_key", lambda: None)

    current_price = 65000.0
    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": current_price,
        "market_data": {"close": current_price},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": {
            "expected_return_pct": expected_ret,
            "clears_hurdle": True,
        },
        "trade_action": raw_action,
        "conviction_score": 7,
    }
    out = risk_critic_node(state)
    assert out["trade_action"] == expected_normalized
    assert out["risk_approved"] is True
    assert out["veto_reason"] is None

    if expected_pos_gt_zero:
        assert out["recommended_position_pct"] > 0.0
        if expected_normalized == "BUY":
            assert out["stop_loss_price"] < current_price < out["take_profit_price"]
        else:
            assert out["take_profit_price"] < current_price < out["stop_loss_price"]

        # Verify Gate 2 (volatility veto) and Gate 3 (conviction veto) remain effective on active BUY/SELL
        vol_veto_state = dict(state, engineered_features={"volatility_range_pct": 4.5})
        vol_veto_out = risk_critic_node(vol_veto_state)
        assert vol_veto_out["risk_approved"] is False
        assert vol_veto_out["recommended_position_pct"] == 0.0
        assert vol_veto_out["stop_loss_price"] is None
        assert vol_veto_out["take_profit_price"] is None
        assert "Volatility Spike Anomaly" in vol_veto_out["veto_reason"]

        conv_veto_state = dict(state, conviction_score=3)
        conv_veto_out = risk_critic_node(conv_veto_state)
        assert conv_veto_out["risk_approved"] is False
        assert conv_veto_out["recommended_position_pct"] == 0.0
        assert conv_veto_out["stop_loss_price"] is None
        assert conv_veto_out["take_profit_price"] is None
        assert "Low Conviction Rating" in conv_veto_out["veto_reason"]
    else:
        assert out["recommended_position_pct"] == 0.0
        assert out["stop_loss_price"] is None
        assert out["take_profit_price"] is None


@pytest.mark.parametrize(
    "action,expected_ret,caller_clears,expected_approved,expected_pos_gt_zero",
    [
        # Positive & negative interior sub-hurdle returns with contradictory clears_hurdle=True
        ("BUY", 0.08, True, False, False),
        ("SELL", -0.08, True, False, False),
        ("BUY", 0.12, True, False, False),
        ("SELL", -0.12, True, False, False),
        # Immediately below threshold (1 ULP below MIN_HURDLE_PCT in magnitude)
        ("BUY", math.nextafter(MIN_HURDLE_PCT, 0.0), True, False, False),
        ("SELL", math.nextafter(-MIN_HURDLE_PCT, 0.0), True, False, False),
        # Exactly at threshold (abs(expected_return_pct) == MIN_HURDLE_PCT)
        ("BUY", MIN_HURDLE_PCT, True, True, True),
        ("SELL", -MIN_HURDLE_PCT, True, True, True),
        # Immediately above threshold (1 ULP above MIN_HURDLE_PCT in magnitude)
        ("BUY", math.nextafter(MIN_HURDLE_PCT, math.inf), True, True, True),
        ("SELL", math.nextafter(-MIN_HURDLE_PCT, -math.inf), True, True, True),
        # Super-hurdle numeric return with valid clears_hurdle=True
        ("BUY", 0.25, True, True, True),
        ("SELL", -0.25, True, True, True),
        # Super-hurdle numeric return but caller explicitly set clears_hurdle=False
        ("BUY", 0.25, False, False, False),
        ("SELL", -0.25, False, False, False),
        # Super-hurdle numeric return with missing or malformed clears_hurdle flag (must fail closed, no truthy coercion)
        ("BUY", 0.25, "__OMIT__", False, False),
        ("SELL", -0.25, "__OMIT__", False, False),
        ("BUY", 0.25, None, False, False),
        ("SELL", -0.25, None, False, False),
        ("BUY", 0.25, "false", False, False),
        ("SELL", -0.25, "false", False, False),
        ("BUY", 0.25, "true", False, False),
        ("SELL", -0.25, "true", False, False),
        ("BUY", 0.25, 1, False, False),
        ("SELL", -0.25, 1, False, False),
        ("BUY", 0.25, 0, False, False),
        ("SELL", -0.25, [True], False, False),
        # Directional sign mismatch: BUY with negative return or SELL with positive return must fail closed
        ("BUY", -MIN_HURDLE_PCT, True, False, False),
        ("SELL", MIN_HURDLE_PCT, True, False, False),
        ("BUY", -0.25, True, False, False),
        ("SELL", 0.25, True, False, False),
        ("BUY", -0.08, True, False, False),
        ("SELL", 0.08, True, False, False),
    ],
)
def test_risk_critic_numeric_hurdle_boundary_and_sign_handling(
    monkeypatch,
    action,
    expected_ret,
    caller_clears,
    expected_approved,
    expected_pos_gt_zero,
):
    """
    Verifies that risk_critic_node enforces both abs(expected_return_pct) >= MIN_HURDLE_PCT
    and a strict boolean clears_hurdle is True across positive/negative sub-hurdle values,
    exact MIN_HURDLE_PCT boundary, 1-ULP neighbors, and missing/malformed clears_hurdle flags,
    while preserving BUY and SELL stop-loss/take-profit directional orientation when approved.
    """
    import sys

    rcn_mod = sys.modules["src.agents.risk_critic_node"]
    monkeypatch.setattr(rcn_mod, "_get_gemini_api_key", lambda: None)

    current_price = 65000.0
    ml_forecast_dict = {"expected_return_pct": expected_ret}
    if caller_clears != "__OMIT__":
        ml_forecast_dict["clears_hurdle"] = caller_clears

    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": current_price,
        "market_data": {"close": current_price},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": ml_forecast_dict,
        "trade_action": action,
        "conviction_score": 7,
    }
    out = risk_critic_node(state)
    assert out["risk_approved"] is expected_approved

    if expected_pos_gt_zero:
        assert out["recommended_position_pct"] > 0.0
        assert out["veto_reason"] is None
        if action == "BUY":
            assert out["stop_loss_price"] < current_price < out["take_profit_price"]
        else:
            assert out["take_profit_price"] < current_price < out["stop_loss_price"]
    else:
        assert out["recommended_position_pct"] == 0.0
        assert out["stop_loss_price"] is None
        assert out["take_profit_price"] is None
        assert out["veto_reason"] is not None
        assert "Sub-Hurdle Fee Drag" in out["veto_reason"]


@pytest.mark.parametrize(
    "action,ml_forecast_payload",
    [
        (act, payload)
        for act in ("BUY", "SELL")
        for payload in (
            {},                                                         # Missing expected_return_pct
            {"clears_hurdle": True},                                    # Missing return with clears_hurdle=True
            {"expected_return_pct": None, "clears_hurdle": True},       # None
            {"expected_return_pct": True, "clears_hurdle": True},       # Boolean True (must not coerce to 1.0%)
            {"expected_return_pct": "malformed", "clears_hurdle": True},# Non-numeric string
            {"expected_return_pct": "", "clears_hurdle": True},         # Empty string
            {"expected_return_pct": float("nan"), "clears_hurdle": True},
            {"expected_return_pct": float("inf"), "clears_hurdle": True},
            {"expected_return_pct": float("-inf"), "clears_hurdle": True},
        )
    ],
)
def test_risk_critic_invalid_or_non_finite_return_fails_closed(action, ml_forecast_payload):
    """
    Verifies that missing, None, boolean, malformed string, NaN, +inf, and -inf
    expected_return_pct values fail closed (VETO with 0.0 allocation and no SL/TP)
    for active BUY/SELL trades without raising uncaught ValueError/TypeError.
    """
    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {"close": 65000.0},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": dict(ml_forecast_payload),
        "trade_action": action,
        "conviction_score": 7,
    }
    out = risk_critic_node(state)
    assert out["risk_approved"] is False
    assert out["recommended_position_pct"] == 0.0
    assert out["stop_loss_price"] is None
    assert out["take_profit_price"] is None
    assert out["veto_reason"] is not None
    assert "Sub-Hurdle Fee Drag" in out["veto_reason"]


@pytest.mark.parametrize(
    "ml_forecast_payload",
    [
        {},
        {"expected_return_pct": None},
        {"expected_return_pct": "not_a_number"},
        {"expected_return_pct": float("nan")},
        {"expected_return_pct": float("inf")},
        {"expected_return_pct": float("-inf")},
    ],
)
def test_risk_critic_hold_preserves_zero_allocation_with_missing_or_malformed_return(
    ml_forecast_payload,
):
    """
    Verifies that HOLD orders preserve their intended zero-allocation approval behavior
    (risk_approved=True, veto_reason=None, recommended_position_pct=0.0, no SL/TP)
    even when expected_return_pct is missing, None, malformed, or non-finite.
    """
    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {"close": 65000.0},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": dict(ml_forecast_payload),
        "trade_action": "HOLD",
        "conviction_score": 4,
    }
    out = risk_critic_node(state)
    assert out["risk_approved"] is True
    assert out["veto_reason"] is None
    assert out["recommended_position_pct"] == 0.0
    assert out["stop_loss_price"] is None
    assert out["take_profit_price"] is None
    assert "RISK CRITIC CONCURS" in out["risk_critic_critique"]

    # Also verify HOLD preserves zero-allocation approval when ancillary state fields are missing or malformed
    hold_malformed_ancillary: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": "bad_price",
        "market_data": None,
        "engineered_features": {"volatility_range_pct": "bad_vol"},
        "ml_forecast": dict(ml_forecast_payload),
        "trade_action": "  hold  ",
        "conviction_score": "bad_conviction",
    }
    out_ancillary = risk_critic_node(hold_malformed_ancillary)
    assert out_ancillary["trade_action"] == "HOLD"
    assert out_ancillary["risk_approved"] is True
    assert out_ancillary["veto_reason"] is None
    assert out_ancillary["recommended_position_pct"] == 0.0
    assert out_ancillary["stop_loss_price"] is None
    assert out_ancillary["take_profit_price"] is None


@pytest.mark.parametrize(
    "action,field_override,expected_veto_substring",
    [
        (act, override, veto_sub)
        for act in ("BUY", "SELL")
        for override, veto_sub in (
            # Malformed / non-finite / non-positive current_price
            ({"current_price": None}, "Invalid Market Price"),
            ({"current_price": "malformed_price"}, "Invalid Market Price"),
            ({"current_price": True}, "Invalid Market Price"),
            ({"current_price": 0.0}, "Invalid Market Price"),
            ({"current_price": -65000.0}, "Invalid Market Price"),
            ({"current_price": float("nan")}, "Invalid Market Price"),
            ({"current_price": float("inf")}, "Invalid Market Price"),
            ({"current_price": float("-inf")}, "Invalid Market Price"),
            # Malformed / non-finite / boolean / fractional conviction_score
            ({"conviction_score": None}, "Low Conviction Rating"),
            ({"conviction_score": "malformed_conv"}, "Low Conviction Rating"),
            ({"conviction_score": True}, "Low Conviction Rating"),
            ({"conviction_score": False}, "Low Conviction Rating"),
            ({"conviction_score": 6.5}, "Low Conviction Rating"),
            ({"conviction_score": float("nan")}, "Low Conviction Rating"),
            ({"conviction_score": float("inf")}, "Low Conviction Rating"),
            ({"conviction_score": float("-inf")}, "Low Conviction Rating"),
            # Malformed / non-finite / negative / missing volatility_range_pct
            ({"engineered_features": None}, "Volatility Spike Anomaly"),
            ({"engineered_features": {}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": None}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": "malformed_vol"}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": True}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": -0.5}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": float("nan")}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": float("inf")}}, "Volatility Spike Anomaly"),
            ({"engineered_features": {"volatility_range_pct": float("-inf")}}, "Volatility Spike Anomaly"),
        )
    ],
)
def test_risk_critic_active_trade_malformed_or_non_finite_ancillary_inputs_fail_closed(
    monkeypatch, action, field_override, expected_veto_substring
):
    """
    Verifies that for active BUY and SELL orders whose forecast return clears MIN_HURDLE_PCT,
    malformed, boolean, non-finite (NaN, +inf, -inf), or invalid current_price, conviction_score,
    or engineered_features.volatility_range_pct values fail closed without raising uncaught
    exceptions, never approve a trade, and never reach the LLM approval path.
    """
    import sys

    rcn_mod = sys.modules["src.agents.risk_critic_node"]
    monkeypatch.setattr(rcn_mod, "_get_gemini_api_key", lambda: None)

    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {"close": 65000.0},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": {
            "expected_return_pct": 0.25 if action == "BUY" else -0.25,
            "clears_hurdle": True,
        },
        "trade_action": action,
        "conviction_score": 7,
    }
    state.update(field_override)

    out = risk_critic_node(state)
    assert out["risk_approved"] is False
    assert out["recommended_position_pct"] == 0.0
    assert out["stop_loss_price"] is None
    assert out["take_profit_price"] is None
    assert out["veto_reason"] is not None
    assert expected_veto_substring in out["veto_reason"]
    assert "VETOED BY RISK CRITIC" in out["risk_critic_critique"]


def test_risk_critic_approved_llm_prompt_uses_canonical_min_hurdle_pct(monkeypatch):
    """
    Verifies that when risk_critic_node approves a trade and invokes the LLM gateway,
    the prompt sent to Gemini cites canonical MIN_HURDLE_PCT (~0.14%) rather than
    the legacy 'Cleared 0.05% hurdle' string, without requiring a live API key.
    """
    import sys
    import types
    from types import SimpleNamespace

    rcn_mod = sys.modules["src.agents.risk_critic_node"]
    captured_prompts = []

    class DummyGenerativeModel:
        def __init__(self, model_name):
            self.model_name = model_name

        def generate_content(self, prompt):
            captured_prompts.append(prompt)
            return SimpleNamespace(text="Formal CRO critique: Trade authorized under canonical hurdle.")

    dummy_genai = types.ModuleType("google.generativeai")
    dummy_genai.configure = lambda **kwargs: None
    dummy_genai.GenerativeModel = DummyGenerativeModel

    google_pkg = sys.modules.get("google")
    if google_pkg is None:
        google_pkg = types.ModuleType("google")
        google_pkg.__path__ = []
        monkeypatch.setitem(sys.modules, "google", google_pkg)

    monkeypatch.setitem(sys.modules, "google.generativeai", dummy_genai)
    monkeypatch.setattr(google_pkg, "generativeai", dummy_genai, raising=False)
    monkeypatch.setattr(rcn_mod, "_get_gemini_api_key", lambda: "test-offline-api-key")

    state: AgentState = {
        "symbol": "BTCUSDT",
        "current_price": 65000.0,
        "market_data": {"close": 65000.0},
        "engineered_features": {"volatility_range_pct": 1.0},
        "ml_forecast": {
            "expected_return_pct": 0.25,
            "clears_hurdle": True,
        },
        "trade_action": "BUY",
        "conviction_score": 7,
        "circuit_breaker_active": False,
    }

    out = risk_critic_node(state)
    assert out["risk_approved"] is True
    assert len(captured_prompts) == 1
    prompt_text = captured_prompts[0]
    assert f"(Cleared {MIN_HURDLE_PCT:.2f}% hurdle)" in prompt_text
    assert "0.05% hurdle" not in prompt_text
    assert out["risk_critic_critique"] == "Formal CRO critique: Trade authorized under canonical hurdle."


@pytest.mark.parametrize(
    "scenario_selected,expected_ret,expected_dollar_move,expected_vol_pct,expected_clears,expected_dir,expected_badge,expected_strategist_action,expected_final_decision,expected_verdict,expected_kelly_gt_zero",
    [
        ("live", 0.08, 52.0, 1.80, False, "SIDEWAYS", "SIDEWAYS [CHOP]", "HOLD", "HOLD", "APPROVED", False),
        ("chop", 0.02, 13.0, 0.40, False, "SIDEWAYS", "SIDEWAYS [CHOP]", "HOLD", "HOLD", "APPROVED", False),
        ("bullish", 0.42, 273.0, 1.50, True, "UP", "BULLISH [UP]", "BUY", "BUY", "APPROVED", True),
        ("vol_spike", -0.20, -130.0, 5.20, True, "DOWN", "BEARISH [DOWN]", "SELL", "HOLD", "VETOED", False),
    ],
)
def test_tab2_council_scenario_payload_and_graph_consistency(
    monkeypatch,
    scenario_selected,
    expected_ret,
    expected_dollar_move,
    expected_vol_pct,
    expected_clears,
    expected_dir,
    expected_badge,
    expected_strategist_action,
    expected_final_decision,
    expected_verdict,
    expected_kelly_gt_zero,
):
    """
    Verifies internal mathematical and behavioral consistency for all 4 Tab 2 scenarios:
      - predicted_dollar_move == round(current_price * expected_return_pct / 100.0, 2)
      - high > close > low > 0
      - ((high - low) / close) * 100.0 == volatility_range_pct == volatility_spread * 100.0
      - clears_hurdle, direction, direction_badge, strategist_action, final_decision,
        risk_verdict, and kelly_fraction behave as intended.
    """
    import sys
    import app
    from src.agents.trading_graph import run_trading_graph

    sn_mod = sys.modules["src.agents.strategist_node"]
    rcn_mod = sys.modules["src.agents.risk_critic_node"]
    monkeypatch.setattr(sn_mod, "_get_gemini_api_key", lambda: None)
    monkeypatch.setattr(rcn_mod, "_get_gemini_api_key", lambda: None)

    current_price = 65000.0
    m_data, t_data, f_data = app.build_council_scenario_payload(
        scenario_selected=scenario_selected,
        current_price=current_price,
    )

    assert f_data["expected_return_pct"] == pytest.approx(expected_ret, abs=1e-8)
    computed_dollar_move = current_price * (f_data["expected_return_pct"] / 100.0)
    assert f_data["predicted_dollar_move"] == pytest.approx(computed_dollar_move, abs=1e-6)
    assert f_data["predicted_dollar_move"] == pytest.approx(expected_dollar_move, abs=1e-6)

    assert m_data["high"] > m_data["close"] > m_data["low"] > 0.0
    computed_range_pct = ((m_data["high"] - m_data["low"]) / m_data["close"]) * 100.0
    assert computed_range_pct == pytest.approx(t_data["volatility_range_pct"], abs=1e-6)
    assert t_data["volatility_spread"] * 100.0 == pytest.approx(t_data["volatility_range_pct"], abs=1e-6)
    assert t_data["volatility_range_pct"] == pytest.approx(expected_vol_pct, abs=1e-6)

    assert f_data["clears_hurdle"] is expected_clears
    assert f_data["direction"] == expected_dir
    assert f_data["direction_badge"] == expected_badge

    council_state = run_trading_graph(
        symbol="BTCUSDT",
        current_price=current_price,
        market_data=m_data,
        engineered_features=t_data,
        ml_forecast=f_data,
    )
    assert council_state["strategist_action"] == expected_strategist_action
    assert council_state["final_decision"] == expected_final_decision
    assert council_state["risk_verdict"] == expected_verdict
    if expected_kelly_gt_zero:
        assert council_state["kelly_fraction"] > 0.0
    else:
        assert council_state["kelly_fraction"] == 0.0

    if scenario_selected == "vol_spike":
        assert council_state["risk_approved"] is False
        assert any("Volatility Spike Anomaly" in v for v in council_state["veto_reasons"])
    elif not expected_clears:
        veto_check = risk_critic_node({
            "symbol": "BTCUSDT",
            "current_price": current_price,
            "market_data": m_data,
            "engineered_features": {"volatility_range_pct": 1.0},
            "ml_forecast": f_data,
            "trade_action": "BUY",
            "conviction_score": 7,
        })
        assert veto_check["risk_approved"] is False
        assert "Sub-Hurdle Fee Drag" in veto_check["veto_reason"]
        assert veto_check["recommended_position_pct"] == 0.0


@pytest.mark.parametrize(
    "notes_text,expected_category",
    [
        # 1. Explicit below-hurdle / sub-hurdle rationales -> "below_hurdle"
        ("Expected return (+0.0800%) fails to clear the 0.14% estimated round-trip break-even hurdle", "below_hurdle"),
        ("Chop / sub-hurdle regime: Expected move (+0.0200%) does not justify estimated 0.14% round-trip exchange friction", "below_hurdle"),
        ("Sub-hurdle fee drag: preserving capital.", "below_hurdle"),
        ("Forecasted return < 0.14% break-even threshold", "chop"),
        ("Expected move is below hurdle; skipping entry.", "below_hurdle"),
        ("Below Fee Hurdle [SKIP]", "below_hurdle"),
        ("Market setup lacks sufficient directional momentum to clear estimated 0.14% round-trip friction", "chop"),
        # 2. Messages stating the hurdle WAS cleared on a HOLD row -> "chop" (NOT below_hurdle)
        ("Cleared 0.14% hurdle, but holding for volume confirmation.", "chop"),
        ("Expected return clears hurdle; awaiting macro catalyst.", "chop"),
        # 3. Ambiguous or deferred hurdle evaluations -> "chop"
        ("Hurdle evaluation deferred pending data", "chop"),
        # 4. Generic transaction-cost/friction notes -> "chop" or "veto"
        ("Neutral regime with normal market friction.", "chop"),
        ("Standard transaction cost model active.", "chop"),
        ("Order vetoed due to volatility spike and order-book friction.", "veto"),
    ],
)
def test_format_portfolio_rationale_explicit_conditions_and_no_overmatching(
    monkeypatch, notes_text, expected_category
):
    """
    Verifies that app.format_portfolio_rationale classifies HOLD rows as below-hurdle
    only when explicit below-hurdle/sub-hurdle phrases are present, and does not
    misclassify cleared-hurdle, deferred-evaluation, or generic friction notes.
    """
    import app

    row = {"action": "HOLD", "notes": notes_text}
    msg = app.format_portfolio_rationale(row)

    if expected_category == "below_hurdle":
        assert msg == (
            f"Capital Preserved: Expected move is below {MIN_HURDLE_PCT:.2f}% round-trip fee hurdle."
        )
    elif expected_category == "veto":
        assert msg == "Risk Veto: Trade prevented by Risk Critic to protect capital."
    else:
        assert msg == "Capital Preserved: Sideways chop; preserving cash until high-conviction setup."


@pytest.mark.parametrize(
    "configured_hurdle,notes_text,expected_category",
    [
        # 1. Custom hurdle of 0.10% and notes mentioning 0.10% vs 0.14%
        (0.10, "Move below 0.09% threshold.", "below_hurdle"),
        (0.10, "Move below 0.10% threshold.", "below_hurdle"),
        (0.10, "Forecasted return < 0.10% break-even threshold", "below_hurdle"),
        (0.10, "Move below 0.14% threshold.", "chop"),
        (0.10, "Forecasted return < 0.14% break-even threshold", "chop"),
        (0.10, "Market setup lacks sufficient directional momentum to clear estimated 0.14% round-trip friction", "chop"),
        (0.10, "Expected return (+0.0800%) fails to clear the 0.14% estimated round-trip break-even hurdle", "below_hurdle"),
        (0.10, "Expected return (+0.1400%) fails to clear the 0.14% estimated round-trip break-even hurdle", "chop"),
        # 2. Custom hurdle of 0.25% and notes mentioning 0.14% vs 0.25% vs 0.30%
        (0.25, "Move below 0.14% threshold.", "below_hurdle"),
        (0.25, "Forecasted return < 0.14% break-even threshold", "below_hurdle"),
        (0.25, "Market setup lacks sufficient directional momentum to clear estimated 0.14% round-trip friction", "below_hurdle"),
        (0.25, "Move below 0.25% threshold.", "below_hurdle"),
        (0.25, "Forecasted return < 0.25% break-even threshold", "below_hurdle"),
        (0.25, "Move below 0.30% threshold.", "chop"),
        (0.25, "Expected return (+0.1400%) fails to clear the 0.14% estimated round-trip break-even hurdle", "below_hurdle"),
        (0.25, "Expected return (+0.2800%) fails to clear the 0.25% estimated round-trip break-even hurdle", "chop"),
        # 3. Explicit below-hurdle phrases at each configured threshold -> "below_hurdle"
        (0.10, "Sub-hurdle fee drag: preserving capital.", "below_hurdle"),
        (0.10, "Expected move is below hurdle; skipping entry.", "below_hurdle"),
        (0.25, "Sub-hurdle fee drag: preserving capital.", "below_hurdle"),
        (0.25, "Expected move is below hurdle; skipping entry.", "below_hurdle"),
        # 4. Cleared-hurdle, deferred-evaluation, and generic-friction notes at each threshold -> "chop"
        (0.10, "Cleared 0.10% hurdle, but holding for volume confirmation.", "chop"),
        (0.10, "Hurdle evaluation deferred pending data", "chop"),
        (0.10, "Neutral regime with normal market friction.", "chop"),
        (0.25, "Cleared 0.25% hurdle, but holding for volume confirmation.", "chop"),
        (0.25, "Hurdle evaluation deferred pending data", "chop"),
        (0.25, "Neutral regime with normal market friction.", "chop"),
    ],
)
def test_format_portfolio_rationale_custom_min_hurdle_pct_and_tab5_provenance(
    monkeypatch, configured_hurdle, notes_text, expected_category
):
    """
    Verifies dynamic MIN_HURDLE_PCT classification in app.format_portfolio_rationale()
    across custom 0.10% and 0.25% hurdles (including notes mentioning 0.14%, explicit
    below-hurdle phrases, cleared-hurdle, deferred-evaluation, and generic-friction notes),
    confirms monkeypatch fixture restoration of app.MIN_HURDLE_PCT before override,
    and verifies app.build_model_pedigree_markdown() Tab 5 provenance.
    """
    import app

    # 5. Verify normal fixture restoration: before monkeypatching in this test invocation,
    # app.MIN_HURDLE_PCT is always restored to the canonical MIN_HURDLE_PCT (~0.139951%).
    assert app.MIN_HURDLE_PCT == pytest.approx(MIN_HURDLE_PCT, abs=1e-12)
    assert app.ROUND_TRIP_FRICTION_PCT == pytest.approx(ROUND_TRIP_FRICTION_PCT, abs=1e-12)
    assert app.TOTAL_FRICTION_PER_SIDE_PCT == pytest.approx(TOTAL_FRICTION_PER_SIDE_PCT, abs=1e-12)

    monkeypatch.setattr(app, "MIN_HURDLE_PCT", configured_hurdle)
    msg = app.format_portfolio_rationale({"action": "HOLD", "notes": notes_text})

    if expected_category == "below_hurdle":
        assert msg == (
            f"Capital Preserved: Expected move is below {configured_hurdle:.2f}% round-trip fee hurdle."
        )
    else:
        assert msg == (
            "Capital Preserved: Sideways chop; preserving cash until high-conviction setup."
        )

    rendered_default = app.build_model_pedigree_markdown()
    assert "{" not in rendered_default and "}" not in rendered_default
    assert "ROUND_TRIP_FRICTION_PCT" not in rendered_default
    assert "TOTAL_FRICTION_PER_SIDE_PCT" not in rendered_default
    assert "0.30% (Historical)" in rendered_default
    assert "+33.55 pp (Historical)" in rendered_default
    assert f"`{ROUND_TRIP_FRICTION_PCT:.2f}%` (`{TOTAL_FRICTION_PER_SIDE_PCT:.2f}%` per side)" in rendered_default
    assert (
        f"Measured under the historical `0.30%` friction assumption; not evidence of equivalent "
        f"performance under current canonical `{ROUND_TRIP_FRICTION_PCT:.2f}%` friction"
    ) in rendered_default

    monkeypatch.setattr(app, "ROUND_TRIP_FRICTION_PCT", 0.22)
    monkeypatch.setattr(app, "TOTAL_FRICTION_PER_SIDE_PCT", 0.11)
    rendered_custom = app.build_model_pedigree_markdown()
    assert "{" not in rendered_custom and "}" not in rendered_custom
    assert "`0.22%` (`0.11%` per side)" in rendered_custom
    assert "current canonical `0.22%` friction" in rendered_custom
    assert "0.30% (Historical)" in rendered_custom
    assert "+33.55 pp (Historical)" in rendered_custom