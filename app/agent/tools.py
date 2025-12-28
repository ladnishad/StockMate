"""Agent tools - wraps existing analysis functions for Claude Agent SDK.

These tools allow the AI agent to analyze stocks using the existing
StockMate analysis capabilities.
"""

import logging
from typing import Dict, Any, List, Optional

from app.tools.analysis import (
    run_analysis,
    find_comprehensive_levels,
    calculate_volume_profile,
    detect_chart_patterns,
    detect_divergences,
)
from app.tools.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_ema,
    calculate_vwap,
    calculate_bollinger_bands,
    analyze_volume,
)
from app.tools.market_data import (
    fetch_latest_quote,
    fetch_latest_trade,
    fetch_price_bars,
    fetch_snapshots,
)
from app.storage.position_store import get_position_store

logger = logging.getLogger(__name__)


async def get_current_price(symbol: str) -> Dict[str, Any]:
    """Get the current price and quote for a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with current price, bid/ask, spread
    """
    try:
        quote = fetch_latest_quote(symbol.upper())
        trade = fetch_latest_trade(symbol.upper())

        return {
            "symbol": symbol.upper(),
            "price": trade.get("price", 0),
            "bid": quote.get("bid_price", 0),
            "ask": quote.get("ask_price", 0),
            "spread": quote.get("spread", 0),
            "spread_pct": quote.get("spread_pct", 0),
            "timestamp": trade.get("timestamp"),
        }
    except Exception as e:
        logger.error(f"Error getting price for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_key_levels(symbol: str) -> Dict[str, Any]:
    """Get support and resistance levels for a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with support and resistance levels
    """
    try:
        # Fetch price bars first
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=50)

        if not bars or len(bars) < 10:
            return {"symbol": symbol.upper(), "error": "Insufficient data", "support": [], "resistance": []}

        levels = find_comprehensive_levels(bars)

        return {
            "symbol": symbol.upper(),
            "support": [
                {"price": l["price"], "distance_pct": l.get("distance_pct", 0), "type": l.get("type", "unknown")}
                for l in levels.get("support", [])[:5]
            ],
            "resistance": [
                {"price": l["price"], "distance_pct": l.get("distance_pct", 0), "type": l.get("type", "unknown")}
                for l in levels.get("resistance", [])[:5]
            ],
        }
    except Exception as e:
        logger.error(f"Error getting levels for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e), "support": [], "resistance": []}


async def get_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Get key technical indicators for a stock.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with RSI, MACD, EMAs, volume analysis
    """
    try:
        # Fetch more bars to ensure we have enough for MACD (needs 35+)
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=100)

        if not bars or len(bars) < 35:
            return {"symbol": symbol.upper(), "error": "Insufficient data (need at least 35 bars)"}

        # Extract price series for calculations
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]

        # Calculate indicators - pass bars to functions that expect PriceBar objects
        rsi_result = calculate_rsi(bars)
        macd_result = calculate_macd(bars)
        ema_9_result = calculate_ema(bars, 9)
        ema_21_result = calculate_ema(bars, 21)
        ema_50_result = calculate_ema(bars, 50) if len(bars) >= 50 else None
        volume_result = analyze_volume(bars)
        bollinger_result = calculate_bollinger_bands(bars)

        current_price = closes[-1]

        # Extract values from Indicator objects
        rsi_value = rsi_result.value if hasattr(rsi_result, 'value') else rsi_result
        ema_9 = ema_9_result.value if hasattr(ema_9_result, 'value') else ema_9_result
        ema_21 = ema_21_result.value if hasattr(ema_21_result, 'value') else ema_21_result
        ema_50 = ema_50_result.value if ema_50_result and hasattr(ema_50_result, 'value') else ema_50_result

        return {
            "symbol": symbol.upper(),
            "price": current_price,
            "rsi": {
                "value": rsi_value,
                "signal": "oversold" if rsi_value < 30 else "overbought" if rsi_value > 70 else "neutral",
            },
            "macd": {
                "macd": macd_result.value if hasattr(macd_result, 'value') else macd_result.get("macd") if isinstance(macd_result, dict) else None,
                "signal": macd_result.metadata.get("signal_line") if hasattr(macd_result, 'metadata') else None,
                "histogram": macd_result.metadata.get("histogram") if hasattr(macd_result, 'metadata') else None,
                "crossover": macd_result.metadata.get("crossover") if hasattr(macd_result, 'metadata') else None,
            },
            "emas": {
                "ema_9": ema_9,
                "ema_21": ema_21,
                "ema_50": ema_50,
                "above_9": current_price > ema_9 if ema_9 else None,
                "above_21": current_price > ema_21 if ema_21 else None,
                "above_50": current_price > ema_50 if ema_50 else None,
            },
            "volume": {
                "current": volumes[-1],
                "average": volume_result.metadata.get("average_volume") if hasattr(volume_result, 'metadata') else None,
                "relative": volume_result.metadata.get("relative_volume") if hasattr(volume_result, 'metadata') else None,
                "trend": volume_result.interpretation if hasattr(volume_result, 'interpretation') else None,
            },
            "bollinger": {
                "upper": bollinger_result.metadata.get("upper") if hasattr(bollinger_result, 'metadata') else None,
                "middle": bollinger_result.metadata.get("middle") if hasattr(bollinger_result, 'metadata') else None,
                "lower": bollinger_result.metadata.get("lower") if hasattr(bollinger_result, 'metadata') else None,
                "position": bollinger_result.interpretation if hasattr(bollinger_result, 'interpretation') else None,
            },
        }
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def run_full_analysis(
    symbol: str,
    trader_profile: str = "swing_trader",
    account_size: float = 10000.0,
) -> Dict[str, Any]:
    """Run full stock analysis with scoring and trade plan.

    Args:
        symbol: Stock ticker symbol
        trader_profile: Trader profile (swing_trader, day_trader, etc.)
        account_size: Account size for position sizing

    Returns:
        Full analysis with score, recommendation, and trade plan
    """
    try:
        result = run_analysis(
            symbol=symbol.upper(),
            account_size=account_size,
            trader_profile=trader_profile,
            use_ai=False,
        )

        return {
            "symbol": symbol.upper(),
            "score": result.score,
            "recommendation": result.recommendation.value,
            "confidence": result.score,
            "signals": result.signals,
            "trade_plan": result.trade_plan.model_dump() if result.trade_plan else None,
            "key_reasons": result.key_reasons[:5] if result.key_reasons else [],
        }
    except Exception as e:
        logger.error(f"Error running analysis for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_volume_profile(symbol: str) -> Dict[str, Any]:
    """Get volume profile analysis (institutional positioning).

    Args:
        symbol: Stock ticker symbol

    Returns:
        Volume profile with VPOC, value area, HVN/LVN
    """
    try:
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=50)

        if not bars or len(bars) < 10:
            return {"symbol": symbol.upper(), "error": "Insufficient data"}

        profile = calculate_volume_profile(bars)

        return {
            "symbol": symbol.upper(),
            "vpoc": profile.get("vpoc"),  # Volume Point of Control
            "value_area_high": profile.get("value_area_high"),
            "value_area_low": profile.get("value_area_low"),
            "high_volume_nodes": profile.get("high_volume_nodes", [])[:3],
            "low_volume_nodes": profile.get("low_volume_nodes", [])[:3],
        }
    except Exception as e:
        logger.error(f"Error getting volume profile for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_chart_patterns(symbol: str) -> Dict[str, Any]:
    """Detect chart patterns (flags, triangles, head & shoulders, etc.).

    Args:
        symbol: Stock ticker symbol

    Returns:
        List of detected patterns with targets
    """
    try:
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=100)

        if not bars or len(bars) < 20:
            return {"symbol": symbol.upper(), "patterns": [], "error": "Insufficient data"}

        patterns_result = detect_chart_patterns(bars)

        # Extract patterns from the result dictionary
        patterns_found = patterns_result.get("patterns_found", []) if isinstance(patterns_result, dict) else []

        return {
            "symbol": symbol.upper(),
            "patterns": [
                {
                    "name": p.get("name"),
                    "type": p.get("type"),  # bullish/bearish
                    "confidence": p.get("confidence"),
                    "target": p.get("target_price"),
                    "entry": p.get("entry_price"),
                    "stop": p.get("stop_price"),
                }
                for p in patterns_found[:3]  # Top 3 patterns
            ],
            "strongest_pattern": patterns_result.get("strongest_pattern") if isinstance(patterns_result, dict) else None,
        }
    except Exception as e:
        logger.error(f"Error getting patterns for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_position_status(symbol: str, user_id: str = "default") -> Dict[str, Any]:
    """Get current position status for a stock with P&L.

    Args:
        symbol: Stock ticker symbol
        user_id: User identifier

    Returns:
        Position details with P&L or indication that no position exists
    """
    try:
        store = get_position_store()

        # Get current price for P&L calculation
        try:
            trade = fetch_latest_trade(symbol.upper())
            current_price = trade.get("price")
        except Exception:
            current_price = None

        position = await store.get_position_with_pnl(user_id, symbol.upper(), current_price)

        if not position:
            return {
                "symbol": symbol.upper(),
                "has_position": False,
                "status": "none",
            }

        result = {
            "symbol": symbol.upper(),
            "has_position": True,
            "status": position.status,
            "entry_price": position.avg_entry_price or position.entry_price,
            "current_size": position.current_size,
            "original_size": position.original_size,
            "stop_loss": position.stop_loss,
            "targets": {
                "target_1": position.target_1,
                "target_2": position.target_2,
                "target_3": position.target_3,
            },
            "targets_hit": position.targets_hit,
            "trade_type": position.trade_type,
        }

        # Add P&L data if position is active
        if position.current_size > 0:
            result["current_price"] = current_price
            result["cost_basis"] = position.cost_basis
            result["unrealized_pnl"] = position.unrealized_pnl
            result["unrealized_pnl_pct"] = position.unrealized_pnl_pct
            result["realized_pnl"] = position.realized_pnl
            result["realized_pnl_pct"] = position.realized_pnl_pct

            # Calculate risk metrics if we have entry and stop
            if position.avg_entry_price and position.stop_loss:
                risk_per_share = abs(position.avg_entry_price - position.stop_loss)
                result["risk_per_share"] = risk_per_share
                result["total_risk"] = risk_per_share * position.current_size

                # R-multiple (how many times risk is the current P&L)
                if position.unrealized_pnl is not None and result["total_risk"] > 0:
                    result["r_multiple"] = round(position.unrealized_pnl / result["total_risk"], 2)

        # Add entry/exit history counts
        result["entry_count"] = len(position.entries)
        result["exit_count"] = len(position.exits)

        return result
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_market_context() -> Dict[str, Any]:
    """Get current market context (indices, direction).

    Returns:
        Market overview with index status and direction
    """
    try:
        indices = ["SPY", "QQQ", "DIA", "IWM"]
        snapshots = fetch_snapshots(indices)

        results = {}
        bullish_count = 0

        for symbol in indices:
            snapshot = snapshots.get(symbol, {})
            daily = snapshot.get("daily_bar", {})
            prev = snapshot.get("prev_daily_bar", {})

            if daily and prev:
                current = daily.get("close", 0)
                previous = prev.get("close", 0)
                change_pct = ((current - previous) / previous * 100) if previous else 0

                results[symbol] = {
                    "price": current,
                    "change_pct": round(change_pct, 2),
                    "direction": "up" if change_pct > 0 else "down",
                }

                if change_pct > 0:
                    bullish_count += 1

        # Determine overall market direction
        if bullish_count >= 3:
            market_direction = "bullish"
        elif bullish_count <= 1:
            market_direction = "bearish"
        else:
            market_direction = "mixed"

        return {
            "indices": results,
            "market_direction": market_direction,
            "bullish_indices": bullish_count,
            "total_indices": len(indices),
        }
    except Exception as e:
        logger.error(f"Error getting market context: {e}")
        return {"error": str(e)}


def get_agent_tools() -> List[Dict[str, Any]]:
    """Get the list of tools available to the agent.

    Returns:
        List of tool definitions for Claude Agent SDK
    """
    return [
        {
            "name": "get_current_price",
            "description": "Get the current price and quote (bid/ask) for a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"],
            },
            "function": get_current_price,
        },
        {
            "name": "get_key_levels",
            "description": "Get support and resistance levels for a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"],
            },
            "function": get_key_levels,
        },
        {
            "name": "get_technical_indicators",
            "description": "Get technical indicators (RSI, MACD, EMAs, volume) for a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"],
            },
            "function": get_technical_indicators,
        },
        {
            "name": "run_full_analysis",
            "description": "Run comprehensive stock analysis with scoring and trade plan",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "trader_profile": {
                        "type": "string",
                        "description": "Trader profile",
                        "enum": ["swing_trader", "day_trader", "position_trader", "long_term_investor"],
                    },
                },
                "required": ["symbol"],
            },
            "function": run_full_analysis,
        },
        {
            "name": "get_volume_profile",
            "description": "Get volume profile analysis showing institutional positioning",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"],
            },
            "function": get_volume_profile,
        },
        {
            "name": "get_chart_patterns",
            "description": "Detect chart patterns (flags, triangles, head & shoulders)",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"}
                },
                "required": ["symbol"],
            },
            "function": get_chart_patterns,
        },
        {
            "name": "get_position_status",
            "description": "Get current position status including entry, stop, and targets",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
                    "user_id": {"type": "string", "description": "User ID", "default": "default"},
                },
                "required": ["symbol"],
            },
            "function": get_position_status,
        },
        {
            "name": "get_market_context",
            "description": "Get overall market context (indices, direction)",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
            "function": get_market_context,
        },
    ]
