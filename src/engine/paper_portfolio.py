"""
Forward $10,000 Paper Trading Portfolio Engine.
Manages cash, BTC inventory, transaction fees, slippage,
mark-to-market valuation, peak-to-trough drawdowns,
and excess Alpha attribution over Buy & Hold benchmark.
"""
import sys
import logging
from pathlib import Path
import math
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc, func, text
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.database.models import PaperPortfolio
from src.database.connection import has_tick_id_column
from config import (
    INITIAL_CAPITAL_USD,
    PORTFOLIO_ADVISORY_LOCK_ID,
    TAKER_FEE_RATE,
    SLIPPAGE_RATE,
    TAKER_FEE_PCT,
    SLIPPAGE_PCT,
    MIN_HURDLE_PCT,
)

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


def get_average_entry_price(session: Session, target_btc: float, include_fees: bool = True) -> float:
    """
    Computes the volume-weighted average cost basis per BTC for open BTC inventory.
    When include_fees=True (default), allocates the entry transaction fee into the unit cost basis
    so that net realized P&L accurately accounts for both entry and exit trading friction.
    Accumulates backwards over recent BUY orders strictly executed after the most recent SELL liquidation.
    """
    if target_btc <= 1e-8:
        return 0.0

    # Scope strictly to BUY orders executed after the most recent SELL liquidation
    last_sell_id = session.execute(
        select(PaperPortfolio.id)
        .where(PaperPortfolio.action == "SELL")
        .order_by(PaperPortfolio.id.desc())
        .limit(1)
    ).scalar_one_or_none()

    stmt = (
        select(PaperPortfolio.execution_price, PaperPortfolio.btc_amount, PaperPortfolio.fee_deducted_usd)
        .where(PaperPortfolio.action == "BUY")
    )
    if last_sell_id is not None:
        stmt = stmt.where(PaperPortfolio.id > last_sell_id)

    records = session.execute(stmt.order_by(PaperPortfolio.id.desc())).all()
    if not records:
        logger.warning(
            f"Cannot match open inventory ({target_btc:.6f} BTC) to eligible BUY records "
            f"after last liquidation (last_sell_id={last_sell_id}). Returning 0.0 cost basis "
            "to preserve strict cross-trade isolation."
        )
        return 0.0

    accum_btc = 0.0
    accum_cost = 0.0
    for price, amount, fee in records:
        if amount <= 0:
            continue
        needed = min(amount, target_btc - accum_btc)
        unit_cost = (price + (fee / amount)) if include_fees else price
        accum_cost += needed * unit_cost
        accum_btc += needed
        if accum_btc >= target_btc - 1e-8:
            break

    if accum_btc < target_btc - 1e-8:
        logger.warning(
            f"Eligible BUY records covered only {accum_btc:.6f} BTC of requested {target_btc:.6f} BTC "
            f"(deficit of {target_btc - accum_btc:.6f} BTC). Using available cost basis."
        )

    return (accum_cost / accum_btc) if accum_btc > 0 else 0.0


def get_latest_portfolio_state(session: Session) -> Optional[PaperPortfolio]:
    """Returns the most recent PaperPortfolio record, or None if brand new."""
    stmt = select(PaperPortfolio).order_by(desc(PaperPortfolio.id)).limit(1)
    return session.execute(stmt).scalar_one_or_none()


def get_baseline_btc_price(session: Session) -> Optional[float]:
    """Returns the initial BTC price from the very first portfolio record."""
    stmt = select(PaperPortfolio).order_by(PaperPortfolio.id.asc()).limit(1)
    first_record = session.execute(stmt).scalar_one_or_none()
    if first_record and first_record.execution_price > 0:
        return float(first_record.execution_price)
    return None


def execute_paper_order(
    session: Session,
    action: str,                # "BUY", "SELL", or "HOLD"
    current_price: float,
    kelly_fraction: float = 0.10,
    notes: Optional[str] = None,
    tick_id: Optional[int] = None,
) -> PaperPortfolio:
    """
    Executes a virtual trade order, applies execution friction (fee + slippage),
    updates cash/BTC balances, and logs the new portfolio snapshot.
    
    Parameters:
        session: Active SQLAlchemy session
        action: "BUY", "SELL", or "HOLD"
        current_price: Latest BTC/USDT price
        kelly_fraction: Fraction of cash to deploy on BUY (0.05 to 0.20)
        notes: Audit notes from Multi-Agent Council
        tick_id: Optional originating MarketTick foreign key for idempotency
        
    Returns:
        The newly committed (or deduplicated existing) PaperPortfolio record.
    """
    # 0. Concurrency Serialization: Acquire PostgreSQL transaction-scoped advisory lock
    # to serialize concurrent ledger lookups and mutations across multi-worker executions.
    bind = session.get_bind()
    if bind and "postgresql" in bind.dialect.name:
        session.execute(
            text("SELECT pg_advisory_xact_lock(:lock_id)"),
            {"lock_id": PORTFOLIO_ADVISORY_LOCK_ID},
        )

    # 1. Idempotency Guard: prevent duplicate paper executions for the same tick
    if tick_id is not None and has_tick_id_column(session):
        existing_order = session.execute(
            select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)
        ).scalar_one_or_none()
        if existing_order is not None:
            logger.info(f"[PAPER] Trade already executed for tick #{tick_id} ({existing_order.action}). Deduplicating.")
            return existing_order

    current_price = max(1.0, float(current_price))
    latest = get_latest_portfolio_state(session)
    baseline_price = get_baseline_btc_price(session) or current_price

    # 1. Initialize portfolio if empty
    if latest is None:
        init_entry = PaperPortfolio(
            timestamp=datetime.now(timezone.utc),
            action="INITIAL_DEPOSIT",
            btc_amount=0.0,
            execution_price=current_price,
            fee_deducted_usd=0.0,
            slippage_usd=0.0,
            cash_balance=INITIAL_CAPITAL_USD,
            btc_balance=0.0,
            total_equity_usd=INITIAL_CAPITAL_USD,
            realized_pnl_usd=0.0,
            unrealized_pnl_usd=0.0,
            drawdown_pct=0.0,
            cumulative_return_pct=0.0,
            benchmark_btc_equity=INITIAL_CAPITAL_USD,
            alpha_usd=0.0,
            alpha_pct=0.0,
            notes="Initialized forward testing paper portfolio with $10,000.00 USDT virtual capital.",
        )
        session.add(init_entry)
        session.flush()
        latest = init_entry

    cash = float(latest.cash_balance)
    btc = float(latest.btc_balance)
    action = action.upper().strip()
    
    btc_delta = 0.0
    fee_usd = 0.0
    slippage_usd = 0.0
    realized_pnl = 0.0
    exec_price = current_price

    # 2. Execute Order Logic
    if action == "BUY" and cash >= 10.0 and kelly_fraction > 0:
        # Deploy Kelly fraction of available cash (capped at 20%)
        size_frac = min(max(kelly_fraction, 0.05), 0.20)
        capital_to_spend = cash * size_frac
        
        # Add slippage to execution price
        exec_price = current_price * (1.0 + SLIPPAGE_RATE)
        slippage_usd = capital_to_spend * SLIPPAGE_RATE
        fee_usd = capital_to_spend * TAKER_FEE_RATE
        
        net_capital = capital_to_spend - fee_usd
        btc_bought = net_capital / exec_price
        
        cash -= capital_to_spend
        btc += btc_bought
        btc_delta = btc_bought

    elif action == "SELL" and btc > 1e-6:
        # Liquidate all BTC inventory
        exec_price = current_price * (1.0 - SLIPPAGE_RATE)
        gross_proceeds = btc * exec_price
        slippage_usd = (btc * current_price) * SLIPPAGE_RATE
        fee_usd = gross_proceeds * TAKER_FEE_RATE
        
        net_proceeds = gross_proceeds - fee_usd
        
        # Institutional weighted average cost basis accounting (including entry fee)
        avg_entry = get_average_entry_price(session, btc, include_fees=True)
        cost_basis = btc * avg_entry
        realized_pnl = net_proceeds - cost_basis
        
        cash += net_proceeds
        btc_delta = -btc
        btc = 0.0

    else:
        action = "HOLD"

    # 3. Mark to Market Valuation
    open_entry = get_average_entry_price(session, btc) if btc > 0 else 0.0
    unrealized_pnl = btc * (current_price - open_entry) if btc > 0 else 0.0
    total_equity = cash + (btc * current_price)
    cum_return_pct = ((total_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0

    # 4. Benchmark Buy & Hold & Alpha
    benchmark_equity = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD
    alpha_usd = total_equity - benchmark_equity
    alpha_pct = ((total_equity - benchmark_equity) / benchmark_equity) * 100.0 if benchmark_equity > 0 else 0.0

    # 5. Peak-to-Trough Drawdown
    stmt_ath = select(PaperPortfolio.total_equity_usd).order_by(desc(PaperPortfolio.total_equity_usd)).limit(1)
    historical_ath = session.execute(stmt_ath).scalar() or INITIAL_CAPITAL_USD
    current_ath = max(float(historical_ath), total_equity)
    drawdown_pct = ((current_ath - total_equity) / current_ath) * 100.0 if current_ath > 0 else 0.0

    # 6. Instantiate and Commit New Record
    record = PaperPortfolio(
        tick_id=tick_id,
        timestamp=datetime.now(timezone.utc),
        action=action,
        btc_amount=btc_delta,
        execution_price=exec_price,
        fee_deducted_usd=fee_usd,
        slippage_usd=slippage_usd,
        cash_balance=cash,
        btc_balance=btc,
        total_equity_usd=total_equity,
        realized_pnl_usd=realized_pnl,
        unrealized_pnl_usd=unrealized_pnl,
        drawdown_pct=drawdown_pct,
        cumulative_return_pct=cum_return_pct,
        benchmark_btc_equity=benchmark_equity,
        alpha_usd=alpha_usd,
        alpha_pct=alpha_pct,
        notes=notes,
    )
    try:
        with session.begin_nested():
            session.add(record)
            session.flush()
    except IntegrityError as exc:
        if tick_id is not None and has_tick_id_column(session):
            existing_order = session.execute(
                select(PaperPortfolio).where(PaperPortfolio.tick_id == tick_id)
            ).scalar_one_or_none()
            if existing_order is not None:
                logger.info(f"[PAPER] Concurrent trade detected for tick #{tick_id} ({existing_order.action}). Deduplicating via savepoint recovery.")
                return existing_order
        logger.error(f"[PAPER] Order insertion failed with IntegrityError: {exc}")
        raise
    return record


def get_portfolio_summary(session: Session, current_price: float) -> Dict[str, Any]:
    """
    Returns an institutional summary of current portfolio performance.
    Used by Streamlit Terminal and API endpoints.
    """
    current_price = max(1.0, float(current_price))
    latest = get_latest_portfolio_state(session)
    if latest is None:
        return {
            "cash_balance": INITIAL_CAPITAL_USD,
            "btc_balance": 0.0,
            "current_btc_price": round(current_price, 2),
            "btc_position_value_usd": 0.0,
            "total_equity_usd": INITIAL_CAPITAL_USD,
            "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": INITIAL_CAPITAL_USD,
            "alpha_usd": 0.0,
            "alpha_pct": 0.0,
            "realized_pnl_usd": 0.0,
            "total_fees_usd": 0.0,
            "drawdown_pct": 0.0,
            "total_trades": 0,
        }

    stmt_count = select(func.count(PaperPortfolio.id)).where(PaperPortfolio.action.in_(["BUY", "SELL"]))
    trade_count = int(session.execute(stmt_count).scalar() or 0)

    # Cumulative realized P&L and transaction fees across all historical ledger actions
    stmt_totals = select(
        func.coalesce(func.sum(PaperPortfolio.realized_pnl_usd), 0.0),
        func.coalesce(func.sum(PaperPortfolio.fee_deducted_usd), 0.0),
    )
    cum_realized_pnl, total_fees = session.execute(stmt_totals).one()

    # Mark current equity with live price
    cash_bal = round(float(latest.cash_balance), 2)
    btc_bal = round(float(latest.btc_balance), 6)
    btc_pos_val = round(float(latest.btc_balance) * current_price, 2)
    live_equity = round(cash_bal + btc_pos_val, 2)

    cum_ret = ((live_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0
    baseline_price = get_baseline_btc_price(session) or current_price
    bench_eq = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD
    alpha_usd = live_equity - bench_eq

    return {
        "cash_balance": cash_bal,
        "btc_balance": btc_bal,
        "current_btc_price": round(current_price, 2),
        "btc_position_value_usd": btc_pos_val,
        "total_equity_usd": live_equity,
        "cumulative_return_pct": round(cum_ret, 2),
        "benchmark_btc_equity": round(bench_eq, 2),
        "alpha_usd": round(alpha_usd, 2),
        "alpha_pct": round(((live_equity - bench_eq) / bench_eq) * 100.0, 2) if bench_eq > 0 else 0.0,
        "realized_pnl_usd": round(float(cum_realized_pnl), 2),
        "total_fees_usd": round(float(total_fees), 2),
        "drawdown_pct": round(float(latest.drawdown_pct), 2),
        "total_trades": trade_count,
    }


def evaluate_spot_hold_until_profit_decision(
    session: Session,
    current_price: float,
    predicted_direction: str,
    proposed_action: str,
    clears_hurdle: bool = True,
    max_stop_loss_pct: float = 3.5,
) -> Dict[str, Any]:
    """
    Implements the Spot 'Buy on UP, Hold Until Net Profit After Fees on DOWN'
    state machine with a Triple-Barrier emergency stop-loss floor:
    1. Flat (btc_balance == 0):
       - If proposed_action == 'BUY' and clears_hurdle: execute BUY.
       - Otherwise: HOLD cash ($0 fee).
    2. Holding BTC (btc_balance > 0):
       - Compute fee-inclusive break-even entry price via get_average_entry_price(include_fees=True)
         and net exit price after sell-side slippage (0.02%) and taker fee (0.05%).
       - If predicted_direction == 'UP' and proposed_action == 'BUY':
         Keep holding existing BTC position (HOLD_POSITION, $0 fee) or add only if cash allows.
       - If predicted_direction == 'DOWN' or proposed_action == 'SELL':
         * If net_exit_price > avg_entry_with_fees (Net Profit > $0 after all BUY+SELL fees):
           Approve SELL to lock in net profit!
         * Else if unrealized_loss_pct >= max_stop_loss_pct (Emergency Triple-Barrier Stop-Loss):
           Approve SELL to protect account from deep crash!
         * Else (Net Profit <= $0 after fees and within safe drawdown band):
           Override SELL -> HOLD ('HOLDING_UNTIL_PROFIT') so we never lock in a fee-drag loss!
    """
    latest = get_latest_portfolio_state(session)
    btc_held = float(latest.btc_balance) if latest else 0.0
    cash_bal = float(latest.cash_balance) if latest else INITIAL_CAPITAL_USD

    norm_dir = str(predicted_direction or "").strip().upper()
    norm_act = str(proposed_action or "HOLD").strip().upper()

    if btc_held <= 1e-8:
        if norm_act == "BUY" and clears_hurdle:
            return {
                "final_spot_action": "BUY",
                "position_state": "ENTERING_LONG_SPOT",
                "btc_held": 0.0,
                "avg_entry_with_fees": 0.0,
                "net_exit_price": round(current_price * (1.0 - SLIPPAGE_RATE) * (1.0 - TAKER_FEE_RATE), 2),
                "estimated_net_pnl_usd": 0.0,
                "reason": (
                    f"Spot Entry: Model predicts UP and clears the {MIN_HURDLE_PCT:.2f}% fee hurdle. "
                    f"Buying BTC at ${current_price:,.2f}."
                ),
            }
        return {
            "final_spot_action": "HOLD",
            "position_state": "FLAT_HOLDING_CASH",
            "btc_held": 0.0,
            "avg_entry_with_fees": 0.0,
            "net_exit_price": round(current_price * (1.0 - SLIPPAGE_RATE) * (1.0 - TAKER_FEE_RATE), 2),
            "estimated_net_pnl_usd": 0.0,
            "reason": "Spot Flat: No BTC held and signal does not warrant a new BUY. Keeping cash safe ($0 fee).",
        }

    # We currently hold BTC: calculate exact fee-adjusted cost basis and net exit proceeds
    avg_entry_with_fees = get_average_entry_price(session, target_btc=btc_held, include_fees=True)
    if avg_entry_with_fees <= 0.0:
        avg_entry_with_fees = current_price

    exec_sell_price = current_price * (1.0 - SLIPPAGE_RATE)
    net_exit_price = exec_sell_price * (1.0 - TAKER_FEE_RATE)
    estimated_net_pnl_usd = (net_exit_price - avg_entry_with_fees) * btc_held
    unrealized_return_pct = ((net_exit_price - avg_entry_with_fees) / avg_entry_with_fees) * 100.0

    # Case A: Next hour predicted UP -> Keep holding BTC (0 fees)
    if norm_dir == "UP" and norm_act != "SELL":
        return {
            "final_spot_action": "HOLD",
            "position_state": "HOLDING_BTC_ON_UP",
            "btc_held": btc_held,
            "avg_entry_with_fees": round(avg_entry_with_fees, 2),
            "net_exit_price": round(net_exit_price, 2),
            "estimated_net_pnl_usd": round(estimated_net_pnl_usd, 2),
            "reason": (
                f"Spot Hold (Riding UP Trend): Holding {btc_held:.6f} BTC (break-even entry "
                f"${avg_entry_with_fees:,.2f} incl. fees; current net exit ${net_exit_price:,.2f}, "
                f"net P&L ${estimated_net_pnl_usd:+,.2f}). Model predicts UP, so we keep holding at $0 fee."
            ),
        }

    # Case B: Next hour predicted DOWN (or SELL proposed) -> Check net profit after all fees
    if estimated_net_pnl_usd > 0.0:
        return {
            "final_spot_action": "SELL",
            "position_state": "TAKING_NET_PROFIT_ON_DOWN",
            "btc_held": btc_held,
            "avg_entry_with_fees": round(avg_entry_with_fees, 2),
            "net_exit_price": round(net_exit_price, 2),
            "estimated_net_pnl_usd": round(estimated_net_pnl_usd, 2),
            "reason": (
                f"Spot Profit-Take SELL: Model predicts DOWN and net exit price (${net_exit_price:,.2f} "
                f"after 0.07% sell friction) is ABOVE fee-adjusted entry (${avg_entry_with_fees:,.2f}). "
                f"Selling now locks in +${estimated_net_pnl_usd:,.2f} net profit after all fees."
            ),
        }

    # Case C: Emergency Triple-Barrier Stop-Loss check
    if unrealized_return_pct <= -abs(max_stop_loss_pct):
        return {
            "final_spot_action": "SELL",
            "position_state": "EMERGENCY_STOP_LOSS_EXIT",
            "btc_held": btc_held,
            "avg_entry_with_fees": round(avg_entry_with_fees, 2),
            "net_exit_price": round(net_exit_price, 2),
            "estimated_net_pnl_usd": round(estimated_net_pnl_usd, 2),
            "reason": (
                f"Spot Emergency Stop-Loss SELL: Position drawdown ({unrealized_return_pct:+.2f}%) breached "
                f"the -{abs(max_stop_loss_pct):.2f}% Triple-Barrier safety floor. Exiting to protect capital."
            ),
        }

    # Case D: Predicted DOWN, but selling now would lose money after fees -> Hold until profitable!
    needed_price = avg_entry_with_fees / ((1.0 - SLIPPAGE_RATE) * (1.0 - TAKER_FEE_RATE))
    return {
        "final_spot_action": "HOLD",
        "position_state": "HOLDING_BTC_UNTIL_NET_PROFIT",
        "btc_held": btc_held,
        "avg_entry_with_fees": round(avg_entry_with_fees, 2),
        "net_exit_price": round(net_exit_price, 2),
        "estimated_net_pnl_usd": round(estimated_net_pnl_usd, 2),
        "reason": (
            f"Spot Hold-Until-Profit Guard: Model predicts DOWN, but selling at ${current_price:,.2f} "
            f"(net exit ${net_exit_price:,.2f}) is below break-even entry ${avg_entry_with_fees:,.2f} "
            f"(would realize ${estimated_net_pnl_usd:+,.2f} loss). Holding BTC until price clears "
            f"${needed_price:,.2f} (safety stop at -{abs(max_stop_loss_pct):.1f}%)."
        ),
    }


def evaluate_futures_long_short_plan(
    current_price: float,
    predicted_direction: str,
    expected_return_pct: float,
    clears_hurdle: bool,
    volatility_range_pct: float,
    risk_approved: bool,
    conviction_score: int = 5,
    account_equity_usd: float = INITIAL_CAPITAL_USD,
    leverage: float = 3.0,
    maintenance_margin_rate: float = 0.004,
) -> Dict[str, Any]:
    """
    Computes an institutional Binance USDS-M Isolated Margin Long/Short Futures plan
    (including exact Isolated Margin Liquidation Price, Stop-Loss, Take-Profit, and Dollar Range)
    without mutating the Spot PaperPortfolio table.
    """
    if current_price <= 0.0 or not math.isfinite(current_price):
        current_price = 65000.0
    lev = max(1.0, min(float(leverage), 10.0))
    mmr = max(0.001, float(maintenance_margin_rate))
    norm_dir = str(predicted_direction or "").strip().upper()

    expected_range_usd = round(current_price * (max(volatility_range_pct, 0.25) / 100.0), 2)

    if not risk_approved or not clears_hurdle or abs(expected_return_pct) < MIN_HURDLE_PCT or conviction_score < 5:
        return {
            "futures_Position": "FLAT (NO POSITION)",
            "leverage": f"{lev:.0f}x Isolated",
            "margin_allocated_usd": 0.0,
            "notional_exposure_usd": 0.0,
            "entry_price": round(current_price, 2),
            "liquidation_price": None,
            "stop_loss_price": None,
            "take_profit_price": None,
            "expected_1h_range_usd": expected_range_usd,
            "summary": (
                f"FLAT (No Futures Trade) — Expected move ({expected_return_pct:+.2f}%) or risk check "
                f"does not justify opening a leveraged Long/Short position."
            ),
        }

    margin_pct = min(0.20, max(0.05, (conviction_score - 4) * 0.04))
    margin_usd = round(account_equity_usd * margin_pct, 2)
    notional_usd = round(margin_usd * lev, 2)
    stop_dist_pct = max(0.012, min(0.025, (volatility_range_pct / 100.0) * 1.2))
    tp_dist_pct = stop_dist_pct * 2.0

    if norm_dir == "UP" and expected_return_pct > 0:
        pos_side = "LONG (BUY FUTURES)"
        # Binance Isolated Margin Long Liquidation approximation: Entry * (1 - 1/L + MMR)
        liq_price = round(current_price * (1.0 - (1.0 / lev) + mmr), 2)
        sl_price = round(current_price * (1.0 - stop_dist_pct), 2)
        tp_price = round(current_price * (1.0 + tp_dist_pct), 2)
    else:
        pos_side = "SHORT (SELL FUTURES)"
        # Binance Isolated Margin Short Liquidation approximation: Entry * (1 + 1/L - MMR)
        liq_price = round(current_price * (1.0 + (1.0 / lev) - mmr), 2)
        sl_price = round(current_price * (1.0 + stop_dist_pct), 2)
        tp_price = round(current_price * (1.0 - tp_dist_pct), 2)

    return {
        "futures_Position": pos_side,
        "leverage": f"{lev:.0f}x Isolated",
        "margin_allocated_usd": margin_usd,
        "notional_exposure_usd": notional_usd,
        "entry_price": round(current_price, 2),
        "liquidation_price": liq_price,
        "stop_loss_price": sl_price,
        "take_profit_price": tp_price,
        "expected_1h_range_usd": expected_range_usd,
        "summary": (
            f"{pos_side} ({lev:.0f}x Isolated) | Margin: ${margin_usd:,.0f} "
            f"(Notional: ${notional_usd:,.0f}) | Entry: ${current_price:,.2f} | "
            f"Stop-Loss: ${sl_price:,.2f} | Take-Profit: ${tp_price:,.2f} | "
            f"Liquidation Price: ${liq_price:,.2f}"
        ),
    }


if __name__ == '__main__':
    from src.database.connection import get_db
    print("[INFO] Testing Forward Paper Trading Portfolio Engine...")
    with get_db() as session:
        state = get_latest_portfolio_state(session)
        print(f"[PORTFOLIO] Latest Ledger ID: {state.id if state else 'None'}")
        summary = get_portfolio_summary(session, current_price=65000.0)
        print(f"[PORTFOLIO] Total Equity: ${summary['total_equity_usd']:,.2f} | Cash: ${summary['cash_balance']:,.2f}")
        print(f"[PORTFOLIO] BTC Balance: {summary['btc_balance']:.6f} | Cumulative Return: {summary['cumulative_return_pct']:+.2f}%")
        print(f"[PORTFOLIO] Benchmark Buy & Hold: ${summary['benchmark_btc_equity']:,.2f} | Alpha: ${summary['alpha_usd']:+,.2f} ({summary['alpha_pct']:+.2f}%)")
    print("[SUCCESS] Paper Portfolio verification completed.")
