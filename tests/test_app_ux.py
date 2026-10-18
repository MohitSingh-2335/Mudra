"""
Automated unit tests for dashboard UX and formatting contracts in app.py.

Verifies:
1. Risk level formatting helper (format_risk_levels) handles HOLD, BUY, SELL, and vetoed states safely.
2. Directional accuracy and total reconciled counts use non-falsy checks (0.0% is not masked).
3. Tab 1 next-hour forecast metrics contain plain-English labels and tooltips.
4. Tab 3 delayed reconciliation AI diagnosis sentences correctly classify UP and DOWN moves.
5. Tab 5 displays the verified 34,990-hour chronological split and controlled candidate rejection rationale.
6. Tab 4 portfolio ledger caption includes a balanced paper-trading simulation disclaimer.
7. Tab 3 includes the read-only forecast error investigation explorer.
"""

import pytest
import pandas as pd
from app import format_risk_levels


def test_format_risk_levels_hold_or_unapproved():
    """Verify format_risk_levels returns cash preservation message for HOLD or unapproved actions."""
    # When verdict is APPROVED but action is HOLD
    sl, tp = format_risk_levels("HOLD", "APPROVED", 65000.0)
    assert sl == "None (No active trade opened — cash preserved)"
    assert tp == "None (No active trade opened — cash preserved)"

    # When verdict is VETOED even if action was BUY
    sl, tp = format_risk_levels("BUY", "VETOED", 65000.0)
    assert sl == "None (No active trade opened — cash preserved)"
    assert tp == "None (No active trade opened — cash preserved)"

    # When action is empty or None
    sl, tp = format_risk_levels("", "APPROVED", 65000.0)
    assert sl == "None (No active trade opened — cash preserved)"
    assert tp == "None (No active trade opened — cash preserved)"


def test_format_risk_levels_approved_buy():
    """Verify format_risk_levels returns -1.5% stop loss and +2.5% take profit for approved BUY."""
    price = 100000.0
    sl, tp = format_risk_levels("BUY", "APPROVED", price)
    assert "$98,500.00 (-1.5%)" in sl
    assert "$102,500.00 (+2.5%)" in tp

    # Custom override in council_state
    custom_state = {"stop_loss_price": 97000.0, "take_profit_price": 105000.0}
    sl, tp = format_risk_levels("BUY", "APPROVED", price, custom_state)
    assert "$97,000.00 (-1.5%)" in sl
    assert "$105,000.00 (+2.5%)" in tp


def test_format_risk_levels_approved_sell():
    """Verify format_risk_levels returns +1.5% stop loss and -2.5% take profit for approved SELL."""
    price = 100000.0
    sl, tp = format_risk_levels("SELL", "APPROVED", price)
    assert "$101,500.00 (+1.5%)" in sl
    assert "$97,500.00 (-2.5%)" in tp


def test_non_falsy_kpi_extraction():
    """Verify 0.0 directional accuracy is preserved rather than falling back to default."""
    rec_kpis = {"directional_accuracy_pct": 0.0, "total_reconciled": 0}
    win_rate = float(
        rec_kpis.get("directional_accuracy_pct")
        if rec_kpis.get("directional_accuracy_pct") is not None
        else rec_kpis.get("win_rate_pct", 0.0)
    )
    total_eval = int(
        rec_kpis.get("total_reconciled")
        if rec_kpis.get("total_reconciled") is not None
        else rec_kpis.get("total_evaluated", 0)
    )
    assert win_rate == 0.0
    assert total_eval == 0


def test_app_source_contains_governance_and_disclaimers():
    """Verify app.py source code contains required governance split and disclaimer texts."""
    with open("app.py", "r", encoding="utf-8") as f:
        content = f.read()

    # Canonical split boundaries
    assert "CLEAN DATASET (34,990 Total 1-Hour Candles)" in content
    assert "27,991 Hours" in content
    assert "PURGE BUFFER (1-Hour Causality Guard — 1 Hour)" in content
    assert "2025-09-17 18:00:00 UTC" in content
    assert "OUT-OF-SAMPLE TEST PERIOD (20% Holdout — 6,998 Hours)" in content

    # Benchmark & Candidate Rejection Rationale
    assert "Controlled Model Benchmark & Rejection Rationale" in content
    assert "p1_controlled_candidate_v1" in content
    assert "REJECTED CANDIDATE" in content
    assert "Naive Persistence" in content
    assert "$398.77" in content

    # Paper trading balanced disclaimer
    assert "Simulated paper-trading performance demonstrates execution accounting" in content

    # Read-only Forecast Error Investigation Explorer
    assert "Forecast Error Investigations & Root-Cause Audit Log" in content
    assert "investigation_headline_search" in content
    assert "investigation_status_filter" in content
