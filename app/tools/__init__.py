"""Tools for stock analysis - designed to be LLM-agent ready."""

from .market_data import fetch_price_bars, fetch_fundamentals, fetch_sentiment
from .indicators import calculate_vwap, calculate_ema, calculate_rsi
from .analysis import (
    find_structural_pivots,
    build_snapshot,
    generate_trade_plan,
    run_analysis,
)

__all__ = [
    # Market data tools
    "fetch_price_bars",
    "fetch_fundamentals",
    "fetch_sentiment",
    # Indicator tools
    "calculate_vwap",
    "calculate_ema",
    "calculate_rsi",
    # Analysis tools
    "find_structural_pivots",
    "build_snapshot",
    "generate_trade_plan",
    "run_analysis",
]
