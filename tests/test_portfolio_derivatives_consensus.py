"""
Deterministic unit tests for spot profit-hold state machine, isolated margin futures
liquidation math, multi-angle price range consensus, post-mortem reconciliation narratives,
and FastMCP derivatives pulse offline fallback.
"""
from datetime import datetime, timezone, timedelta
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.database.models import Base, MarketTick
from src.engine.paper_portfolio import (
    execute_paper_order,
    evaluate_spot_hold_until_profit_decision,
    evaluate_futures_long_short_plan,
)
from src.engine.model_consensus import (
    estimate_next_hour_price_range_usd,
    evaluate_multi_angle_consensus,
)
from src.engine.reconciliation import build_plain_english_reconciliation_story
from src.mcp.mudra_mcp_server import get_live_btc_derivatives_pulse


@pytest.fixture
def mem_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    s = Session()
    try:
        yield s
    finally:
        s.close()
        Base.metadata.drop_all(bind=engine)
        engine.dispose()


def test_spot_hold_until_net_profit_lifecycle(mem_session):
    """
    Verifies the complete Spot Hold-Until-Profit state machine:
    - Flat + BUY (clears hurdle) -> BUY
    - Holding BTC + next hour UP -> HOLD (keep riding UP at $0 fee)
    - Holding BTC + next hour DOWN + price below fee-adjusted break-even -> HOLD (do not lock in fee loss)
    - Holding BTC + next hour DOWN + price above fee-adjusted break-even -> SELL (lock in net profit)
    - Holding BTC + deep crash (> 3.5% loss) -> Emergency Stop-Loss SELL
    """
    # 1. Flat -> BUY entry at $60,000
    d_entry = evaluate_spot_hold_until_profit_decision(
        mem_session,
        current_price=60000.0,
        predicted_direction="UP",
        proposed_action="BUY",
        clears_hurdle=True,
    )
    assert d_entry["final_spot_action"] == "BUY"
    assert d_entry["position_state"] == "ENTERING_LONG_SPOT"

    execute_paper_order(mem_session, "BUY", 60000.0, kelly_fraction=0.20, tick_id=1)
    mem_session.commit()

    # 2. Holding BTC + next hour predicted UP -> Keep holding ($0 fee)
    d_up = evaluate_spot_hold_until_profit_decision(
        mem_session,
        current_price=60020.0,
        predicted_direction="UP",
        proposed_action="BUY",
        clears_hurdle=True,
    )
    assert d_up["final_spot_action"] == "HOLD"
    assert d_up["position_state"] == "HOLDING_BTC_ON_UP"

    # 3. Holding BTC + next hour predicted DOWN, but price ($60,030) barely moved and would lose money after round-trip fees (~$60,084 break-even)
    d_hold_unprofitable = evaluate_spot_hold_until_profit_decision(
        mem_session,
        current_price=60030.0,
        predicted_direction="DOWN",
        proposed_action="SELL",
        clears_hurdle=True,
    )
    assert d_hold_unprofitable["final_spot_action"] == "HOLD"
    assert d_hold_unprofitable["position_state"] == "HOLDING_BTC_UNTIL_NET_PROFIT"
    assert d_hold_unprofitable["estimated_net_pnl_usd"] < 0.0

    # 4. Holding BTC + next hour predicted DOWN, and price ($60,500) clears both BUY and SELL fees -> SELL for net profit!
    d_profit_sell = evaluate_spot_hold_until_profit_decision(
        mem_session,
        current_price=60500.0,
        predicted_direction="DOWN",
        proposed_action="SELL",
        clears_hurdle=True,
    )
    assert d_profit_sell["final_spot_action"] == "SELL"
    assert d_profit_sell["position_state"] == "TAKING_NET_PROFIT_ON_DOWN"
    assert d_profit_sell["estimated_net_pnl_usd"] > 0.0

    # 5. Emergency Stop-Loss check when price drops -4.0% (to $57,500)
    d_stop = evaluate_spot_hold_until_profit_decision(
        mem_session,
        current_price=57500.0,
        predicted_direction="DOWN",
        proposed_action="SELL",
        clears_hurdle=True,
        max_stop_loss_pct=3.5,
    )
    assert d_stop["final_spot_action"] == "SELL"
    assert d_stop["position_state"] == "EMERGENCY_STOP_LOSS_EXIT"


def test_futures_long_short_liquidation_math():
    """Verifies Long, Short, and Flat Isolated Margin Futures plans and Liquidation Price boundaries."""
    long_plan = evaluate_futures_long_short_plan(
        current_price=80000.0,
        predicted_direction="UP",
        expected_return_pct=0.35,
        clears_hurdle=True,
        volatility_range_pct=1.2,
        risk_approved=True,
        conviction_score=8,
        leverage=3.0,
    )
    assert long_plan["futures_Position"] == "LONG (BUY FUTURES)"
    assert long_plan["liquidation_price"] < long_plan["stop_loss_price"] < 80000.0 < long_plan["take_profit_price"]

    short_plan = evaluate_futures_long_short_plan(
        current_price=80000.0,
        predicted_direction="DOWN",
        expected_return_pct=-0.35,
        clears_hurdle=True,
        volatility_range_pct=1.2,
        risk_approved=True,
        conviction_score=8,
        leverage=3.0,
    )
    assert short_plan["futures_Position"] == "SHORT (SELL FUTURES)"
    assert short_plan["take_profit_price"] < 80000.0 < short_plan["stop_loss_price"] < short_plan["liquidation_price"]

    flat_plan = evaluate_futures_long_short_plan(
        current_price=80000.0,
        predicted_direction="UP",
        expected_return_pct=0.04,
        clears_hurdle=False,
        volatility_range_pct=1.2,
        risk_approved=False,
        conviction_score=4,
    )
    assert flat_plan["futures_Position"] == "FLAT (NO POSITION)"
    assert flat_plan["liquidation_price"] is None


def test_multi_angle_consensus_and_price_range_estimator():
    """Verifies next-hour dollar price range estimation and 1-model / 2-model consensus voting."""
    df = pd.DataFrame({
        "high": [80400.0] * 24,
        "low": [80000.0] * 24,
        "close": [80200.0] * 24,
    })
    r_info = estimate_next_hour_price_range_usd(df, current_price=80200.0)
    assert r_info["predicted_range_usd"] == pytest.approx(400.0, abs=1.0)
    assert r_info["predicted_high_1h"] > 80200.0 > r_info["predicted_low_1h"]

    # 2-model agreement
    c2 = evaluate_multi_angle_consensus(
        current_price=80200.0,
        primary_direction="UP",
        primary_prob_up=62.0,
        primary_pred_dollar_move=210.0,
        predicted_range_usd=r_info["predicted_range_usd"],
        shadow_direction="UP",
        shadow_prob_up=66.0,
    )
    assert c2["consensus_direction"] == "UP"
    assert c2["agreement_badge"] == "2/2 Models Vote UP"
    assert c2["move_clears_fees"] is True
    assert c2["range_clears_fees"] is True


def test_plain_english_reconciliation_story_and_mcp_fallback(monkeypatch):
    """Verifies plain-English reconciliation post-mortem story generation and MCP offline fallback."""
    t0 = MarketTick(
        id=1,
        timestamp=datetime(2026, 7, 1, 10, 0, tzinfo=timezone.utc),
        open=80000.0,
        high=80200.0,
        low=79900.0,
        close=80000.0,
        volume=100.0,
    )
    t1 = MarketTick(
        id=2,
        timestamp=datetime(2026, 7, 1, 11, 0, tzinfo=timezone.utc),
        open=80000.0,
        high=80500.0,
        low=79950.0,
        close=80350.0,
        volume=120.0,
    )
    story = build_plain_english_reconciliation_story(
        pred_tick=t0,
        outcome_tick=t1,
        pred_price=80200.0,
        actual_price=80350.0,
        pred_ret_pct=0.25,
        actual_ret_pct=0.4375,
        dir_pred="UP",
        dir_real="UP",
        dir_correct=True,
        abs_err_usd=150.0,
        tag="MOMENTUM_UNDERESTIMATED",
    )
    assert "[CORRECT]" in story["headline"]
    assert "$80,350.00" in story["hypothesis_notes"]

    # Force network error to test MCP offline fallback
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    pulse = get_live_btc_derivatives_pulse()
    assert pulse["status"] == "OFFLINE_FALLBACK"
