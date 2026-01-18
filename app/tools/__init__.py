"""Tools for stock analysis - designed to be LLM-agent ready."""

from .market_data import (
    fetch_price_bars,
    fetch_fundamentals,
    fetch_sentiment,
    fetch_news_sentiment,
    fetch_news_for_trade_style,
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
    calculate_volume_profile,
    detect_chart_patterns,
    build_snapshot,
    generate_trade_plan,
    run_analysis,
)
from .market_scanner import (
    get_market_overview,
    get_sector_performance,
    find_sector_leaders,
    run_market_scan,
)

__all__ = [
    # Market data tools (5)
    "fetch_price_bars",
    "fetch_fundamentals",
    "fetch_sentiment",
    "fetch_news_sentiment",
    "fetch_news_for_trade_style",
    # Indicator tools (8)
    "calculate_vwap",
    "calculate_ema",
    "calculate_rsi",
    "analyze_volume",
    "calculate_macd",
    "calculate_atr",
    "calculate_bollinger_bands",
    "detect_divergences",
    # Analysis tools (7)
    "find_structural_pivots",
    "detect_key_levels",
    "calculate_volume_profile",
    "detect_chart_patterns",
    "build_snapshot",
    "generate_trade_plan",
    "run_analysis",
    # Market scanner tools (4) - NEW
    "get_market_overview",
    "get_sector_performance",
    "find_sector_leaders",
    "run_market_scan",
]
