"""Tool executor for the agentic stock analyzer.

Dispatches tool calls to the appropriate SDK functions and returns results.
"""

import logging
from typing import Any, Dict, Optional

from app.agent.sdk import tools as sdk_tools
from app.agent.providers import AIProvider

logger = logging.getLogger(__name__)


async def execute_tool(
    name: str,
    arguments: Dict[str, Any],
    context: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a tool call and return the result.

    Args:
        name: The tool name to execute
        arguments: Tool arguments from the AI
        context: Execution context containing:
            - symbol: Default symbol if not in arguments
            - user_id: User ID for position lookups
            - provider: AI provider for vision analysis
            - x_provider: Provider for X/Twitter search

    Returns:
        Tool execution result as a dictionary
    """
    # Get symbol from arguments or context
    symbol = arguments.get("symbol") or context.get("symbol")
    user_id = context.get("user_id", "default")
    provider: Optional[AIProvider] = context.get("provider")
    x_provider: Optional[AIProvider] = context.get("x_provider")

    try:
        if name == "get_price":
            result = await sdk_tools.get_current_price(symbol)
            return _format_price_result(result)

        elif name == "get_market_context":
            result = await sdk_tools.get_market_context()
            return _format_market_context(result)

        elif name == "get_position":
            result = await sdk_tools.get_position_status(symbol, user_id)
            return _format_position_result(result)

        elif name == "get_and_analyze_chart":
            # Combined tool: generate chart + vision analysis in one step
            # Returns only the analysis summary (not the base64 image)
            timeframe = arguments.get("timeframe", "1d")
            trade_style = arguments.get("trade_style", "swing")
            days_back = arguments.get("days_back")

            # Default and minimum days based on timeframe
            # Intraday data needs at least 3 days to ensure data availability (weekends, holidays)
            defaults = {"5m": 3, "15m": 5, "1h": 10, "1d": 100, "1w": 365}
            minimums = {"5m": 3, "15m": 3, "1h": 5, "1d": 30, "1w": 52}

            if days_back is None:
                days_back = defaults.get(timeframe, 100)
            else:
                # Enforce minimum to ensure data availability
                days_back = max(days_back, minimums.get(timeframe, 1))

            # Generate chart
            chart_result = await sdk_tools.generate_chart(symbol, timeframe, days_back)
            chart_image = chart_result.get("chart_image_base64")

            if not chart_image:
                return {"error": "Failed to generate chart", "details": chart_result}

            if not provider:
                return {"error": "No AI provider available for vision analysis"}

            # Analyze chart with vision
            vision_result = await sdk_tools.analyze_chart_vision(
                symbol, chart_image, trade_style, provider
            )

            # Return analysis only (not the image) to keep context small
            return {
                "timeframe": timeframe,
                "trade_style": trade_style,
                "bars_analyzed": chart_result.get("bars_plotted"),
                **_format_vision_result(vision_result)
            }

        elif name == "get_technicals":
            ema_periods = arguments.get("ema_periods", [9, 21, 50])
            timeframe = arguments.get("timeframe", "1d")
            rsi_period = 14
            result = await sdk_tools.get_technical_indicators(symbol, ema_periods, rsi_period)
            return _format_technicals_result(result)

        elif name == "get_support_resistance":
            timeframe = arguments.get("timeframe", "daily")
            result = await sdk_tools.get_support_resistance(symbol, timeframe)
            return _format_sr_result(result)

        elif name == "get_fibonacci":
            trade_type = arguments.get("trade_type", "swing")
            # Get price bars first for Fibonacci calculation
            timeframe_map = {"day": "5m", "swing": "1d", "position": "1w"}
            days_map = {"day": 3, "swing": 100, "position": 365}
            bars_result = await sdk_tools.get_price_bars(
                symbol,
                timeframe_map.get(trade_type, "1d"),
                days_map.get(trade_type, 100)
            )
            bars = bars_result.get("bars", [])
            result = await sdk_tools.get_fibonacci_levels(symbol, bars, trade_type)
            return _format_fibonacci_result(result)

        elif name == "get_fundamentals":
            result = await sdk_tools.get_fundamentals(symbol)
            return _format_fundamentals_result(result)

        elif name == "get_news":
            days_back = arguments.get("days_back", 7)
            result = await sdk_tools.get_news_sentiment(symbol, days_back)
            return _format_news_result(result)

        elif name == "search_x":
            query = arguments.get("query")
            if not query:
                return {"error": "query is required"}
            if not x_provider:
                return {"error": "No X/Twitter provider available", "suggestion": "Configure Grok API for X search"}
            # Use the X provider to search
            result = await _execute_x_search(query, x_provider)
            return result

        elif name == "get_atr":
            timeframe = arguments.get("timeframe", "1d")
            days_map = {"5m": 3, "1d": 30, "1w": 52}
            bars_result = await sdk_tools.get_price_bars(
                symbol,
                timeframe,
                days_map.get(timeframe, 30)
            )
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "atr": bars_result.get("atr"),
                "atr_pct": bars_result.get("atr_pct"),
                "current_price": bars_result.get("current_price"),
                "volatility_assessment": _assess_volatility(bars_result.get("atr_pct", 0))
            }

        else:
            return {"error": f"Unknown tool: {name}"}

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return {"error": str(e), "tool": name}


async def _execute_x_search(query: str, x_provider: AIProvider) -> Dict[str, Any]:
    """Execute X/Twitter search using Grok provider."""
    try:
        # Build a simple search request
        from app.agent.providers.grok_provider import get_x_search_parameters
        from app.agent.providers import AIMessage

        search_params = get_x_search_parameters()

        response = await x_provider.create_message(
            messages=[AIMessage(
                role="user",
                content=f"""Search X/Twitter for: {query}

Analyze the sentiment and key discussions. Respond with:
1. Overall sentiment (bullish/bearish/neutral/mixed)
2. Key themes being discussed
3. Notable trader opinions or insights
4. Any breaking news or catalysts mentioned

Be specific about what traders are saying."""
            )],
            system="You are analyzing X/Twitter for stock market sentiment. Be concise and specific.",
            model_type="fast",
            max_tokens=1000,
            search_parameters=search_params,
        )

        return {
            "query": query,
            "sentiment_analysis": response.content,
            "citations": response.citations if hasattr(response, "citations") else [],
        }

    except Exception as e:
        logger.error(f"X search error: {e}")
        return {"error": f"X search failed: {str(e)}", "query": query}


def _format_price_result(result: Dict) -> Dict:
    """Format price result for AI consumption."""
    return {
        "symbol": result.get("symbol"),
        "price": result.get("price"),
        "bid": result.get("bid"),
        "ask": result.get("ask"),
        "spread_pct": result.get("spread_pct"),
        "change_today": result.get("change_pct"),
    }


def _format_market_context(result: Dict) -> Dict:
    """Format market context for AI consumption."""
    return {
        "market_direction": result.get("market_direction"),
        "bullish_indices": result.get("bullish_indices"),
        "total_indices": 4,
        "indices": result.get("indices", {}),
    }


def _format_position_result(result: Dict) -> Dict:
    """Format position result for AI consumption."""
    return {
        "has_position": result.get("has_position", False),
        "direction": result.get("direction"),
        "entry_price": result.get("entry_price"),
        "current_size": result.get("current_size"),
        "stop_loss": result.get("stop_loss"),
        "targets": result.get("targets"),
        "unrealized_pnl": result.get("unrealized_pnl"),
        "pnl_pct": result.get("pnl_pct"),
    }


def _format_vision_result(result: Dict) -> Dict:
    """Format vision analysis result for AI consumption."""
    return {
        "trend_quality": result.get("trend_quality"),
        "visual_patterns": result.get("visual_patterns", []),
        "candlestick_patterns": result.get("candlestick_patterns", []),
        "ema_structure": result.get("ema_structure"),
        "volume_confirmation": result.get("volume_confirmation"),
        "warning_signs": result.get("warning_signs", []),
        "confidence_modifier": result.get("confidence_modifier"),
        "summary": result.get("summary"),
    }


def _format_technicals_result(result: Dict) -> Dict:
    """Format technicals result for AI consumption."""
    return {
        "current_price": result.get("current_price"),
        "ema_trend": result.get("ema_trend"),
        "emas": result.get("emas", {}),
        "price_vs_emas": result.get("price_vs_emas"),
        "rsi": result.get("rsi", {}),
        "macd": result.get("macd", {}),
        "bollinger": result.get("bollinger", {}),
        "volume": result.get("volume", {}),
    }


def _format_sr_result(result: Dict) -> Dict:
    """Format support/resistance result for AI consumption."""
    return {
        "current_price": result.get("current_price"),
        "support_levels": [
            {"price": s.get("price"), "distance_pct": s.get("distance_pct")}
            for s in result.get("support", [])[:5]
        ],
        "resistance_levels": [
            {"price": r.get("price"), "distance_pct": r.get("distance_pct")}
            for r in result.get("resistance", [])[:5]
        ],
    }


def _format_fibonacci_result(result: Dict) -> Dict:
    """Format Fibonacci result for AI consumption."""
    return {
        "swing_high": result.get("swing_high"),
        "swing_low": result.get("swing_low"),
        "trend": result.get("trend"),
        "retracement_levels": result.get("retracement_levels", {}),
        "extension_levels": result.get("extension_levels", {}),
        "nearest_level": result.get("nearest_level"),
        "signal": result.get("signal"),
    }


def _format_fundamentals_result(result: Dict) -> Dict:
    """Format fundamentals result for AI consumption."""
    # Extract nested dictionaries with defaults
    valuation = result.get("valuation", {})
    growth = result.get("growth", {})
    profitability = result.get("profitability", {})
    health = result.get("health", {})
    earnings_risk = result.get("earnings_risk", {})
    price_range = result.get("price_range", {})

    return {
        "symbol": result.get("symbol"),
        "valuation": {
            "pe_ratio": valuation.get("pe_ratio"),
            "pe_forward": valuation.get("pe_forward"),
            "pb_ratio": valuation.get("pb_ratio"),
            "ps_ratio": valuation.get("ps_ratio"),
            "peg_ratio": valuation.get("peg_ratio"),
            "assessment": valuation.get("assessment"),
        },
        "growth": {
            "eps": growth.get("eps"),
            "eps_growth_yoy": growth.get("eps_growth_yoy"),
            "revenue_growth_yoy": growth.get("revenue_growth_yoy"),
            "revenue_growth_3y": growth.get("revenue_growth_3y"),
        },
        "profitability": {
            "gross_margin": profitability.get("gross_margin"),
            "operating_margin": profitability.get("operating_margin"),
            "net_margin": profitability.get("net_margin"),
            "roe": profitability.get("roe"),
            "roa": profitability.get("roa"),
        },
        "financial_health": {
            "debt_to_equity": health.get("debt_to_equity"),
            "current_ratio": health.get("current_ratio"),
            "assessment": health.get("assessment"),
        },
        "price_range": {
            "52w_high": price_range.get("52w_high"),
            "52w_low": price_range.get("52w_low"),
            "range_position_pct": price_range.get("range_position"),
        },
        "earnings_risk": {
            "has_risk": earnings_risk.get("has_risk", False),
            "days_until": earnings_risk.get("days_until"),
            "date": earnings_risk.get("date"),
            "beat_rate": earnings_risk.get("beat_rate"),
            "eps_estimate": earnings_risk.get("eps_estimate"),
        },
        "trade_guidance": result.get("trade_guidance", {}),
    }


def _format_news_result(result: Dict) -> Dict:
    """Format news result for AI consumption."""
    return {
        "sentiment": result.get("sentiment"),
        "sentiment_score": result.get("sentiment_score"),
        "article_count": result.get("article_count"),
        "has_breaking_news": result.get("has_breaking_news", False),
        "key_themes": result.get("key_themes", []),
        "headlines": result.get("headlines", [])[:5],  # Top 5 headlines
        "summary": result.get("summary"),
    }


def _assess_volatility(atr_pct: float) -> str:
    """Assess volatility for trading style suitability."""
    if atr_pct >= 3.0:
        return "High volatility - suitable for day trading"
    elif atr_pct >= 1.0:
        return "Moderate volatility - suitable for swing trading"
    elif atr_pct >= 0.5:
        return "Low volatility - suitable for position trading"
    else:
        return "Very low volatility - limited trading opportunity"
