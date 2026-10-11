"""
Forward $10,000 Paper Trading Portfolio Engine.
Manages cash, BTC inventory, transaction fees, slippage,
mark-to-market valuation, peak-to-trough drawdowns,
and excess Alpha attribution over Buy & Hold benchmark.
"""
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from src.database.models import PaperPortfolio

INITIAL_CAPITAL_USD = 10000.0
TAKER_FEE_PCT = 0.05    # 0.05% Binance VIP0 / BNB spot fee
SLIPPAGE_PCT = 0.02     # 0.02% modeled market order price impact


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
        
    Returns:
        The newly committed PaperPortfolio record.
    """
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
        exec_price = current_price * (1.0 + (SLIPPAGE_PCT / 100.0))
        slippage_usd = capital_to_spend * (SLIPPAGE_PCT / 100.0)
        fee_usd = capital_to_spend * (TAKER_FEE_PCT / 100.0)
        
        net_capital = capital_to_spend - fee_usd
        btc_bought = net_capital / exec_price
        
        cash -= capital_to_spend
        btc += btc_bought
        btc_delta = btc_bought

    elif action == "SELL" and btc > 1e-6:
        # Liquidate all BTC inventory
        exec_price = current_price * (1.0 - (SLIPPAGE_PCT / 100.0))
        gross_proceeds = btc * exec_price
        slippage_usd = (btc * current_price) * (SLIPPAGE_PCT / 100.0)
        fee_usd = gross_proceeds * (TAKER_FEE_PCT / 100.0)
        
        net_proceeds = gross_proceeds - fee_usd
        
        # Estimate realized PnL relative to initial deposit ratio
        cost_basis = INITIAL_CAPITAL_USD - latest.cash_balance
        realized_pnl = net_proceeds - cost_basis
        
        cash += net_proceeds
        btc_delta = -btc
        btc = 0.0

    else:
        action = "HOLD"

    # 3. Mark to Market Valuation
    unrealized_pnl = btc * (current_price - exec_price) if btc > 0 else 0.0
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
    session.add(record)
    session.flush()
    return record


def get_portfolio_summary(session: Session, current_price: float) -> Dict[str, Any]:
    """
    Returns an institutional summary of current portfolio performance.
    Used by Streamlit Terminal and API endpoints.
    """
    latest = get_latest_portfolio_state(session)
    if latest is None:
        return {
            "cash_balance": INITIAL_CAPITAL_USD,
            "btc_balance": 0.0,
            "total_equity_usd": INITIAL_CAPITAL_USD,
            "cumulative_return_pct": 0.0,
            "benchmark_btc_equity": INITIAL_CAPITAL_USD,
            "alpha_usd": 0.0,
            "alpha_pct": 0.0,
            "drawdown_pct": 0.0,
            "total_trades": 0,
        }

    stmt_count = select(PaperPortfolio).where(PaperPortfolio.action.in_(["BUY", "SELL"]))
    trade_count = len(session.execute(stmt_count).scalars().all())

    # Mark current equity with live price
    live_equity = float(latest.cash_balance) + (float(latest.btc_balance) * current_price)
    cum_ret = ((live_equity - INITIAL_CAPITAL_USD) / INITIAL_CAPITAL_USD) * 100.0
    baseline_price = get_baseline_btc_price(session) or current_price
    bench_eq = INITIAL_CAPITAL_USD * (current_price / baseline_price) if baseline_price > 0 else INITIAL_CAPITAL_USD
    alpha_usd = live_equity - bench_eq

    return {
        "cash_balance": round(float(latest.cash_balance), 2),
        "btc_balance": round(float(latest.btc_balance), 6),
        "total_equity_usd": round(live_equity, 2),
        "cumulative_return_pct": round(cum_ret, 2),
        "benchmark_btc_equity": round(bench_eq, 2),
        "alpha_usd": round(alpha_usd, 2),
        "alpha_pct": round(((live_equity - bench_eq) / bench_eq) * 100.0, 2) if bench_eq > 0 else 0.0,
        "drawdown_pct": round(float(latest.drawdown_pct), 2),
        "total_trades": trade_count,
    }
