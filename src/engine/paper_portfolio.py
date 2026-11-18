import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from sqlalchemy import desc, func, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import (
    INITIAL_CAPITAL_USD,
    MIN_HURDLE_PCT,
    PORTFOLIO_ADVISORY_LOCK_ID,
    SLIPPAGE_PCT,
    SLIPPAGE_RATE,
    TAKER_FEE_PCT,
    TAKER_FEE_RATE,
)
from src.database.connection import has_tick_id_column
from src.database.models import PaperPortfolio

logger = logging.getLogger(__name__)

__all__ = [
    "execute_paper_order",
    "get_portfolio_summary",
    "get_latest_portfolio_state",
    "get_average_entry_price",
    "evaluate_spot_hold_until_profit_decision",
    "evaluate_futures_long_short_plan",
    "INITIAL_CAPITAL_USD",
    "PORTFOLIO_ADVISORY_LOCK_ID",
    "TAKER_FEE_RATE",
    "SLIPPAGE_RATE",
    "TAKER_FEE_PCT",
    "SLIPPAGE_PCT",
]


# Calculate weighted-average entry price for open BTC inventory
def get_average_entry_price(session: Session, target_btc: float, include_fees: bool = True) -> float:
    if target_btc <= 1e-8:
        return 0.0
    last_sell = session.execute(
        select(PaperPortfolio.id).where(PaperPortfolio.action == "SELL").order_by(PaperPortfolio.id.desc()).limit(1)
    ).scalar_one_or_none()

    stmt = select(PaperPortfolio.execution_price, PaperPortfolio.btc_amount, PaperPortfolio.fee_deducted_usd).where(PaperPortfolio.action == "BUY")
    if last_sell is not None:
        stmt = stmt.where(PaperPortfolio.id > last_sell)

    records = session.execute(stmt.order_by(PaperPortfolio.id.desc())).all()
    accum_btc, accum_cost = 0.0, 0.0
    for price, amount, fee in records:
        if amount <= 0:
            continue
        needed = min(amount, target_btc - accum_btc)
        accum_cost += needed * ((price + (fee / amount)) if include_fees else price)
        accum_btc += needed
        if accum_btc >= target_btc - 1e-8:
            break
    return (accum_cost / accum_btc) if accum_btc > 0 else 0.0


# Get latest paper portfolio state
def get_latest_portfolio_state(session: Session) -> Optional[PaperPortfolio]:
    return session.execute(select(PaperPortfolio).order_by(desc(PaperPortfolio.id)).limit(1)).scalar_one_or_none()


# Baseline BTC price for buy-and-hold benchmark
def get_baseline_btc_price(session: Session) -> Optional[float]:
    rec = session.execute(select(PaperPortfolio).order_by(PaperPortfolio.id.asc()).limit(1)).scalar_one_or_none()
    return float(rec.execution_price) if rec and rec.execution_price > 0 else None


# Execute paper order with realistic fee friction and transaction ledgering
def execute_paper_order(session: Session, action: str, current_price: float, kelly_fraction: float = 0.10, notes: Optional[str] = None, tick_id: Optional[int] = None) -> PaperPortfolio:
    bind = session.get_bind()
    if bind and "postgresql" in bind.dialect.name:
        session.execute(text("SELECT pg_advisory_xact_lock(:lock_id)"), {"lock_id": PORTFOLIO_ADVISORY_LOCK_ID})

    if tick_id is not None and has_tick_id_column(session):
        existing = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none()
        if existing is not None:
            return existing

    current_price = max(1.0, float(current_price))
    latest = get_latest_portfolio_state(session)
    baseline_price = get_baseline_btc_price(session) or current_price

    if latest is None:
        init_entry = PaperPortfolio(
            timestamp=datetime.now(timezone.utc), action="INITIAL_DEPOSIT", btc_amount=0.0,
            execution_price=current_price, fee_deducted_usd=0.0, slippage_usd=0.0,
            cash_balance=INITIAL_CAPITAL_USD, btc_balance=0.0, total_equity_usd=INITIAL_CAPITAL_USD,
            realized_pnl_usd=0.0, unrealized_pnl_usd=0.0, drawdown_pct=0.0, cumulative_return_pct=0.0,
            benchmark_btc_equity=INITIAL_CAPITAL_USD, alpha_usd=0.0, alpha_pct=0.0,
            notes="Initialized paper portfolio with $10,000.00 USDT.",
        )
        session.add(init_entry)
        session.flush()
        latest = init_entry

    cash, btc = float(latest.cash_balance), float(latest.btc_balance)
    action = action.upper().strip()
    btc_delta, fee_usd, slippage_usd, realized_pnl = 0.0, 0.0, 0.0, 0.0
    exec_price = current_price

    if action == "BUY" and cash >= 10.0 and kelly_fraction > 0:
        spend = cash * min(max(kelly_fraction, 0.05), 0.20)
        exec_price = current_price * (1.0 + SLIPPAGE_RATE)
        slippage_usd = spend * SLIPPAGE_RATE
        fee_usd = spend * TAKER_FEE_RATE
        btc_bought = (spend - fee_usd) / exec_price
        cash -= spend
        btc += btc_bought
        btc_delta = btc_bought
    elif action == "SELL" and btc > 1e-6:
        exec_price = current_price * (1.0 - SLIPPAGE_RATE)
        gross = btc * exec_price
        slippage_usd = (btc * current_price) * SLIPPAGE_RATE
        fee_usd = gross * TAKER_FEE_RATE
        net_proceeds = gross - fee_usd
        avg_entry = get_average_entry_price(session, btc, include_fees=True)
        realized_pnl = net_proceeds - (btc * avg_entry)
        cash += net_proceeds
        btc_delta = -btc
        btc = 0.0
    else:
        action = "HOLD"

    open_entry = get_average_entry_price(session, btc) if btc > 0 else 0.0
    unrealized_pnl = btc * (current_price - open_entry) if btc > 0 else 0.0
    total_equity = cash + (btc * current_price)
    cum_ret = ((total_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0
    bench_eq = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD
    alpha_usd = total_equity - bench_eq
    alpha_pct = ((total_equity - bench_eq) / bench_eq) * 100.0 if bench_eq > 0 else 0.0

    ath = session.execute(select(PaperPortfolio.total_equity_usd).order_by(desc(PaperPortfolio.total_equity_usd)).limit(1)).scalar() or INITIAL_CAPITAL_USD
    drawdown_pct = ((max(float(ath), total_equity) - total_equity) / max(float(ath), total_equity)) * 100.0

    record = PaperPortfolio(
        tick_id=tick_id, timestamp=datetime.now(timezone.utc), action=action, btc_amount=btc_delta,
        execution_price=exec_price, fee_deducted_usd=fee_usd, slippage_usd=slippage_usd,
        cash_balance=cash, btc_balance=btc, total_equity_usd=total_equity,
        realized_pnl_usd=realized_pnl, unrealized_pnl_usd=unrealized_pnl, drawdown_pct=drawdown_pct,
        cumulative_return_pct=cum_ret, benchmark_btc_equity=bench_eq, alpha_usd=alpha_usd,
        alpha_pct=alpha_pct, notes=notes,
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError:
        if tick_id is not None and has_tick_id_column(session):
            existing = session.execute(select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)).scalar_one_or_none()
            if existing is not None:
                return existing
        raise
    return record


# Summary of portfolio equity, returns, and execution costs
def get_portfolio_summary(session: Session, current_price: float) -> Dict[str, Any]:
    current_price = max(1.0, float(current_price))
    latest = get_latest_portfolio_state(session)
    if latest is None:
        return {
            "cash_balance": INITIAL_CAPITAL_USD, "btc_balance": 0.0, "current_btc_price": round(current_price, 2),
            "btc_position_value_usd": 0.0, "total_equity_usd": INITIAL_CAPITAL_USD, "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": INITIAL_CAPITAL_USD, "alpha_usd": 0.0, "alpha_pct": 0.0,
            "realized_pnl_usd": 0.0, "total_fees_usd": 0.0, "drawdown_pct": 0.0, "total_trades": 0,
        }

    trade_count = int(session.execute(select(func.count(PaperPortfolio.id)).where(PaperPortfolio.action.in_(["BUY", "SELL"]))).scalar() or 0)
    pnl, fees = session.execute(select(func.coalesce(func.sum(PaperPortfolio.realized_pnl_usd), 0.0), func.coalesce(func.sum(PaperPortfolio.fee_deducted_usd), 0.0))).one()

    cash_bal = round(float(latest.cash_balance), 2)
    btc_bal = round(float(latest.btc_balance), 6)
    btc_pos_val = round(float(latest.btc_balance) * current_price, 2)
    live_equity = round(cash_bal + btc_pos_val, 2)
    cum_ret = ((live_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0

    baseline_price = get_baseline_btc_price(session) or current_price
    bench_eq = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD

    return {
        "cash_balance": cash_bal, "btc_balance": btc_bal, "current_btc_price": round(current_price, 2),
        "btc_position_value_usd": btc_pos_val, "total_equity_usd": live_equity,
        "cumulative_return_pct": round(cum_ret, 2), "benchmark_btc_equity": round(bench_eq, 2),
        "alpha_usd": round(live_equity - bench_eq, 2),
        "alpha_pct": round(((live_equity - bench_eq) / bench_eq) * 100.0, 2) if bench_eq > 0 else 0.0,
        "realized_pnl_usd": round(float(pnl), 2), "total_fees_usd": round(float(fees), 2),
        "drawdown_pct": round(float(latest.drawdown_pct), 2), "total_trades": trade_count,
    }


# Evaluate spot hold discipline to avoid selling at a fee-loss
def evaluate_spot_hold_until_profit_decision(
    session: Session, current_price: float, predicted_direction: str, proposed_action: str,
    clears_hurdle: bool = True, max_stop_loss_pct: float = 3.5,
) -> Dict[str, Any]:
    latest = get_latest_portfolio_state(session)
    btc_held = float(latest.btc_balance) if latest else 0.0
    norm_dir = str(predicted_direction or "").strip().upper()
    norm_act = str(proposed_action or "HOLD").strip().upper()
    net_exit = round(current_price * (1.0 - SLIPPAGE_RATE) * (1.0 - TAKER_FEE_RATE), 2)

    if btc_held <= 1e-8:
        if norm_act == "BUY" and clears_hurdle:
            return {"final_spot_action": "BUY", "position_state": "ENTERING_LONG_SPOT", "btc_held": 0.0, "avg_entry_with_fees": 0.0, "net_exit_price": net_exit, "estimated_net_pnl_usd": 0.0, "reason": "Spot Entry: Model predicts UP and clears fee hurdle."}
        return {"final_spot_action": "HOLD", "position_state": "FLAT_HOLDING_CASH", "btc_held": 0.0, "avg_entry_with_fees": 0.0, "net_exit_price": net_exit, "estimated_net_pnl_usd": 0.0, "reason": "Spot Flat: Keeping cash safe ($0 fee)."}

    avg_entry = get_average_entry_price(session, target_btc=btc_held, include_fees=True) or current_price
    est_pnl = (net_exit - avg_entry) * btc_held
    ret_pct = ((net_exit - avg_entry) / avg_entry) * 100.0

    if norm_dir == "UP" and norm_act != "SELL":
        return {"final_spot_action": "HOLD", "position_state": "HOLDING_BTC_ON_UP", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": "Holding BTC on UP prediction ($0 fee)."}
    if est_pnl > 0.0:
        return {"final_spot_action": "SELL", "position_state": "TAKING_NET_PROFIT_ON_DOWN", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": f"Spot Profit-Take SELL: locks in +${est_pnl:,.2f} net profit."}
    if ret_pct <= -abs(max_stop_loss_pct):
        return {"final_spot_action": "SELL", "position_state": "EMERGENCY_STOP_LOSS_EXIT", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": f"Emergency Stop-Loss SELL: breached -{abs(max_stop_loss_pct):.2f}% floor."}
    return {"final_spot_action": "HOLD", "position_state": "HOLDING_BTC_UNTIL_NET_PROFIT", "btc_held": btc_held, "avg_entry_with_fees": round(avg_entry, 2), "net_exit_price": net_exit, "estimated_net_pnl_usd": round(est_pnl, 2), "reason": "Spot Hold-Until-Profit: holding BTC to avoid locking in fee loss."}


# Futures execution planning with leverage and liquidation thresholds
def evaluate_futures_long_short_plan(
    current_price: float, predicted_direction: str, expected_return_pct: float, clears_hurdle: bool,
    volatility_range_pct: float, risk_approved: bool, conviction_score: int = 5,
    account_equity_usd: float = INITIAL_CAPITAL_USD, leverage: float = 3.0, maintenance_margin_rate: float = 0.004,
) -> Dict[str, Any]:
    if current_price <= 0.0 or not math.isfinite(current_price):
        current_price = 65000.0
    lev = max(1.0, min(float(leverage), 10.0))
    mmr = max(0.001, float(maintenance_margin_rate))
    norm_dir = str(predicted_direction or "").strip().upper()
    expected_range = round(current_price * (max(volatility_range_pct, 0.25) / 100.0), 2)

    if not risk_approved or not clears_hurdle or abs(expected_return_pct) < MIN_HURDLE_PCT or conviction_score < 5:
        return {
            "futures_Position": "FLAT (NO POSITION)", "leverage": f"{lev:.0f}x Isolated", "margin_allocated_usd": 0.0,
            "notional_exposure_usd": 0.0, "entry_price": round(current_price, 2), "liquidation_price": None,
            "stop_loss_price": None, "take_profit_price": None, "expected_1h_range_usd": expected_range,
            "summary": "FLAT (No Futures Trade) — Risk check or move magnitude does not justify leverage.",
        }

    margin_usd = round(account_equity_usd * min(0.20, max(0.05, (conviction_score - 4) * 0.04)), 2)
    notional_usd = round(margin_usd * lev, 2)
    dist = max(0.012, min(0.025, (volatility_range_pct / 100.0) * 1.2))

    if norm_dir == "UP" and expected_return_pct > 0:
        pos_side = "LONG (BUY FUTURES)"
        liq = round(current_price * (1.0 - (1.0 / lev) + mmr), 2)
        sl, tp = round(current_price * (1.0 - dist), 2), round(current_price * (1.0 + dist * 2.0), 2)
    else:
        pos_side = "SHORT (SELL FUTURES)"
        liq = round(current_price * (1.0 + (1.0 / lev) - mmr), 2)
        sl, tp = round(current_price * (1.0 + dist), 2), round(current_price * (1.0 - dist * 2.0), 2)

    return {
        "futures_Position": pos_side, "leverage": f"{lev:.0f}x Isolated", "margin_allocated_usd": margin_usd,
        "notional_exposure_usd": notional_usd, "entry_price": round(current_price, 2), "liquidation_price": liq,
        "stop_loss_price": sl, "take_profit_price": tp, "expected_1h_range_usd": expected_range,
        "summary": f"{pos_side} ({lev:.0f}x Isolated) | Margin: ${margin_usd:,.0f} | Entry: ${current_price:,.2f}",
    }
