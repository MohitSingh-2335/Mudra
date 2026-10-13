"""
Forward $10,000 Paper Trading Portfolio Engine.
Manages cash, BTC inventory, transaction fees, slippage,
mark-to-market valuation, peak-to-trough drawdowns,
and excess Alpha attribution over Buy & Hold benchmark.
"""
import sys
import logging
from pathlib import Path
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
)

__all__ = [
    "execute_paper_order",
    "get_portfolio_summary",
    "get_latest_portfolio_state",
    "get_average_entry_price",
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
