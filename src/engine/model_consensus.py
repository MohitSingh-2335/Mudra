"""
Multi-Angle Model Consensus Engine for Mudra.
Combines:
- Angle 1: Direction Consensus ('UP' vs 'DOWN') across registered models (Model 1 xgb_onnx_v1.0.0
  + Model 2 xgb_v2 when present in artifacts/experiments or artifacts/models).
- Angle 2: Next-Hour Dollar Price Range ($ High - $ Low) using causal realized volatility & EMA range clustering.
- Angle 3: Expected Close-to-Close Dollar Move ($).
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from config import MIN_HURDLE_PCT

__all__ = [
    "estimate_next_hour_price_range_usd",
    "evaluate_multi_angle_consensus",
]


def estimate_next_hour_price_range_usd(
    live_df: pd.DataFrame,
    current_price: float,
) -> Dict[str, float]:
    """
    Angle 2 — Predicts next 1-hour Bitcoin High-Low price range in Dollars ($).
    Uses volatility clustering (weighted combination of last 1h range, 6h EMA range,
    and 24h ATR range) which has high autocorrelation in 1h BTC candles.
    """
    if live_df is None or live_df.empty or current_price <= 0:
        return {
            "predicted_range_usd": 0.0,
            "predicted_range_pct": 0.0,
            "predicted_high_1h": current_price,
            "predicted_low_1h": current_price,
        }

    highs = pd.to_numeric(live_df["high"], errors="coerce")
    lows = pd.to_numeric(live_df["low"], errors="coerce")
    closes = pd.to_numeric(live_df["close"], errors="coerce").replace(0, np.nan)
    range_pct_series = ((highs - lows) / closes).dropna()

    if range_pct_series.empty:
        return {
            "predicted_range_usd": 0.0,
            "predicted_range_pct": 0.0,
            "predicted_high_1h": current_price,
            "predicted_low_1h": current_price,
        }

    last_1h_pct = float(range_pct_series.iloc[-1])
    ema_6h_pct = float(range_pct_series.tail(6).ewm(span=6, adjust=False).mean().iloc[-1])
    mean_24h_pct = float(range_pct_series.tail(24).mean())

    # Volatility clustering forecast: 45% 1h + 35% 6h EMA + 20% 24h baseline
    pred_range_rate = (0.45 * last_1h_pct) + (0.35 * ema_6h_pct) + (0.20 * mean_24h_pct)
    pred_range_usd = float(current_price * pred_range_rate)

    half_range = pred_range_usd / 2.0
    return {
        "predicted_range_usd": round(pred_range_usd, 2),
        "predicted_range_pct": round(pred_range_rate * 100.0, 4),
        "predicted_high_1h": round(current_price + half_range, 2),
        "predicted_low_1h": round(max(1.0, current_price - half_range), 2),
    }


def evaluate_multi_angle_consensus(
    current_price: float,
    primary_direction: str,
    primary_prob_up: float,
    primary_pred_dollar_move: float,
    predicted_range_usd: float,
    shadow_direction: Optional[str] = None,
    shadow_prob_up: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Combines Angle 1 (Model 1 + optional Model 2 Direction Vote),
    Angle 2 (Predicted 1h Price Range in $), and Angle 3 (Predicted 1h Dollar Move)
    into a plain-English consensus report.
    """
    hurdle_dollar_min = current_price * (MIN_HURDLE_PCT / 100.0)
    votes_up = 1 if str(primary_direction).upper() == "UP" else 0
    total_models = 1

    if shadow_direction is not None and shadow_prob_up is not None:
        total_models = 2
        if str(shadow_direction).upper() == "UP":
            votes_up += 1
        avg_prob_up = (float(primary_prob_up) + float(shadow_prob_up)) / 2.0
    else:
        avg_prob_up = float(primary_prob_up)

    votes_down = total_models - votes_up
    if votes_up > votes_down:
        consensus_dir = "UP"
        agreement_text = f"{votes_up}/{total_models} Models Vote UP"
    elif votes_down > votes_up:
        consensus_dir = "DOWN"
        agreement_text = f"{votes_down}/{total_models} Models Vote DOWN"
    else:
        consensus_dir = "SPLIT"
        agreement_text = "1/2 Models Split (No Consensus -> Hold)"

    range_clears_fees = predicted_range_usd >= (hurdle_dollar_min * 1.5)
    move_clears_fees = abs(primary_pred_dollar_move) >= hurdle_dollar_min

    return {
        "consensus_direction": consensus_dir,
        "agreement_badge": agreement_text,
        "total_models_voting": total_models,
        "avg_prob_up_pct": round(avg_prob_up, 2),
        "predicted_range_usd": round(predicted_range_usd, 2),
        "predicted_dollar_move": round(primary_pred_dollar_move, 2),
        "hurdle_dollar_min": round(hurdle_dollar_min, 2),
        "range_clears_fees": bool(range_clears_fees),
        "move_clears_fees": bool(move_clears_fees),
        "plain_summary": (
            f"Direction Vote: {agreement_text} | Predicted 1h Move: ${primary_pred_dollar_move:+,.2f} "
            f"(Fee Hurdle: ${hurdle_dollar_min:,.0f}) | Predicted 1h High-Low Range: ${predicted_range_usd:,.2f}"
        ),
    }
