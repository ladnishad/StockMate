"""Tools for stock analysis - designed to be LLM-agent ready."""

from .market_data import (
    fetch_price_bars,
    fetch_fundamentals,
    fetch_sentiment,
    fetch_news_sentiment,
)
from .indicators import (
    calculate_vwap,
    calculate_ema,
    calculate_rsi,
    analyze_volume,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    detect_divergences,
)
from .analysis import (
    find_structural_pivots,
    detect_key_levels,
    build_snapshot,
    generate_trade_plan,
    run_analysis,
)

__all__ = [
    # Market data tools (4)
    "fetch_price_bars",
    "fetch_fundamentals",
    "fetch_sentiment",
    "fetch_news_sentiment",
    # Indicator tools (8)
    "calculate_vwap",
    "calculate_ema",
    "calculate_rsi",
    "analyze_volume",
    "calculate_macd",
    "calculate_atr",
    "calculate_bollinger_bands",
    "detect_divergences",
    # Analysis tools (5)
    "find_structural_pivots",
    "detect_key_levels",
    "build_snapshot",
    "generate_trade_plan",
    "run_analysis",
]
