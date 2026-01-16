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
    calculate_atr,
    # New institutional-grade indicators
    calculate_ichimoku,
    calculate_williams_r,
    calculate_parabolic_sar,
    calculate_cmf,
    calculate_adl,
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
    """Get support and resistance levels for a stock with touch count analysis.

    Each level includes reliability metrics based on historical touches:
    - touches: How many times price has tested this level
    - strength: Score 0-100 based on touches, recency, and hold rate
    - reliability: "weak" (1 touch), "moderate" (2-3), "strong" (4-5), "institutional" (6+)
    - last_touch_bars_ago: How recently the level was tested

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with support and resistance levels including touch count metrics
    """
    try:
        # Fetch more bars for better touch count analysis
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=100)

        if not bars or len(bars) < 10:
            return {"symbol": symbol.upper(), "error": "Insufficient data", "support": [], "resistance": []}

        levels = find_comprehensive_levels(bars)

        return {
            "symbol": symbol.upper(),
            "support": [
                {
                    "price": l["price"],
                    "distance_pct": l.get("distance_pct", 0),
                    "type": l.get("type", "unknown"),
                    "touches": l.get("touches", 0),
                    "high_volume_touches": l.get("high_volume_touches", 0),
                    "bounce_quality": l.get("bounce_quality", 0),
                    "reclaimed": l.get("reclaimed", False),
                    "strength": l.get("strength", 0),
                    "reliability": l.get("reliability", "weak"),
                    "last_touch_bars_ago": l.get("last_touch_bars_ago"),
                }
                for l in levels.get("support", [])[:5]
            ],
            "resistance": [
                {
                    "price": l["price"],
                    "distance_pct": l.get("distance_pct", 0),
                    "type": l.get("type", "unknown"),
                    "touches": l.get("touches", 0),
                    "high_volume_touches": l.get("high_volume_touches", 0),
                    "bounce_quality": l.get("bounce_quality", 0),
                    "reclaimed": l.get("reclaimed", False),
                    "strength": l.get("strength", 0),
                    "reliability": l.get("reliability", "weak"),
                    "last_touch_bars_ago": l.get("last_touch_bars_ago"),
                }
                for l in levels.get("resistance", [])[:5]
            ],
        }
    except Exception as e:
        logger.error(f"Error getting levels for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e), "support": [], "resistance": []}


async def get_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Get comprehensive technical indicators for a stock.

    Includes institutional-grade indicators for professional analysis:
    - Core: RSI, MACD, EMAs, Bollinger Bands, Volume
    - Advanced: Ichimoku Cloud, Williams %R, Parabolic SAR, CMF, ADL
    - Volatility: ATR with percentage calculation

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with full technical indicator suite
    """
    try:
        # Fetch more bars to ensure we have enough for Ichimoku (needs 52+)
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=100)

        if not bars or len(bars) < 52:
            return {"symbol": symbol.upper(), "error": "Insufficient data (need at least 52 bars)"}

        # Extract price series for calculations
        closes = [b.close for b in bars]
        volumes = [b.volume for b in bars]
        current_price = closes[-1]

        # ============================================================
        # CORE INDICATORS
        # ============================================================
        rsi_result = calculate_rsi(bars)
        macd_result = calculate_macd(bars)
        ema_9_result = calculate_ema(bars, 9)
        ema_21_result = calculate_ema(bars, 21)
        ema_50_result = calculate_ema(bars, 50)
        volume_result = analyze_volume(bars)
        bollinger_result = calculate_bollinger_bands(bars)
        atr_result = calculate_atr(bars)

        # ============================================================
        # INSTITUTIONAL-GRADE INDICATORS (Shortcomings Fix)
        # Each wrapped in try/except for resilience
        # ============================================================
        try:
            ichimoku_result = calculate_ichimoku(bars)
        except Exception as e:
            logger.warning(f"Could not calculate Ichimoku: {e}")
            ichimoku_result = None

        try:
            williams_result = calculate_williams_r(bars)
        except Exception as e:
            logger.warning(f"Could not calculate Williams %R: {e}")
            williams_result = None

        try:
            psar_result = calculate_parabolic_sar(bars)
        except Exception as e:
            logger.warning(f"Could not calculate Parabolic SAR: {e}")
            psar_result = None

        try:
            cmf_result = calculate_cmf(bars)
        except Exception as e:
            logger.warning(f"Could not calculate CMF: {e}")
            cmf_result = None

        try:
            adl_result = calculate_adl(bars)
        except Exception as e:
            logger.warning(f"Could not calculate ADL: {e}")
            adl_result = None

        # Extract values from Indicator objects
        rsi_value = rsi_result.value if hasattr(rsi_result, 'value') else rsi_result
        ema_9 = ema_9_result.value if hasattr(ema_9_result, 'value') else ema_9_result
        ema_21 = ema_21_result.value if hasattr(ema_21_result, 'value') else ema_21_result
        ema_50 = ema_50_result.value if hasattr(ema_50_result, 'value') else ema_50_result

        # Determine EMA trend alignment
        ema_bullish_count = sum([
            current_price > ema_9 if ema_9 else False,
            current_price > ema_21 if ema_21 else False,
            current_price > ema_50 if ema_50 else False,
        ])
        ema_trend = "bullish" if ema_bullish_count >= 2 else "bearish" if ema_bullish_count == 0 else "mixed"

        # ATR percentage for volatility classification
        atr_value = atr_result.value if hasattr(atr_result, 'value') else 0
        atr_pct = (atr_value / current_price * 100) if current_price > 0 else 0
        volatility_regime = "high" if atr_pct > 3.0 else "moderate" if atr_pct > 1.5 else "low"

        return {
            "symbol": symbol.upper(),
            "price": current_price,

            # Core Momentum
            "rsi": {
                "value": rsi_value,
                "signal": rsi_result.signal if hasattr(rsi_result, 'signal') else "neutral",
                "interpretation": rsi_result.metadata.get("interpretation") if hasattr(rsi_result, 'metadata') else None,
            },
            "macd": {
                "value": macd_result.value if hasattr(macd_result, 'value') else None,
                "signal": macd_result.signal if hasattr(macd_result, 'signal') else "neutral",
                "histogram": macd_result.metadata.get("histogram") if hasattr(macd_result, 'metadata') else None,
                "histogram_trend": macd_result.metadata.get("histogram_trend") if hasattr(macd_result, 'metadata') else None,
            },

            # Trend
            "emas": {
                "ema_9": ema_9,
                "ema_21": ema_21,
                "ema_50": ema_50,
                "above_9": current_price > ema_9 if ema_9 else None,
                "above_21": current_price > ema_21 if ema_21 else None,
                "above_50": current_price > ema_50 if ema_50 else None,
                "trend": ema_trend,
                "bullish_count": ema_bullish_count,
            },

            # Volume
            "volume": {
                "current": volumes[-1],
                "average": volume_result.metadata.get("average_volume") if hasattr(volume_result, 'metadata') else None,
                "relative": volume_result.metadata.get("relative_volume") if hasattr(volume_result, 'metadata') else None,
                "signal": volume_result.signal if hasattr(volume_result, 'signal') else "neutral",
            },

            # Volatility
            "bollinger": {
                "upper": bollinger_result.metadata.get("upper") if hasattr(bollinger_result, 'metadata') else None,
                "middle": bollinger_result.metadata.get("middle") if hasattr(bollinger_result, 'metadata') else None,
                "lower": bollinger_result.metadata.get("lower") if hasattr(bollinger_result, 'metadata') else None,
                "width": bollinger_result.metadata.get("bandwidth") if hasattr(bollinger_result, 'metadata') else None,
                "percent_b": bollinger_result.metadata.get("percent_b") if hasattr(bollinger_result, 'metadata') else None,
            },
            "atr": {
                "value": atr_value,
                "percentage": round(atr_pct, 2),
                "volatility_regime": volatility_regime,
                "stop_1x": atr_result.metadata.get("stop_distance_1x") if hasattr(atr_result, 'metadata') else None,
                "stop_2x": atr_result.metadata.get("stop_distance_2x") if hasattr(atr_result, 'metadata') else None,
            },

            # ============================================================
            # INSTITUTIONAL-GRADE INDICATORS
            # ============================================================

            # Ichimoku Cloud - comprehensive trend/momentum/S&R
            "ichimoku": {
                "signal": ichimoku_result.signal if ichimoku_result and hasattr(ichimoku_result, 'signal') else "N/A",
                "tenkan_sen": ichimoku_result.metadata.get("tenkan_sen") if ichimoku_result and hasattr(ichimoku_result, 'metadata') else None,
                "kijun_sen": ichimoku_result.metadata.get("kijun_sen") if ichimoku_result and hasattr(ichimoku_result, 'metadata') else None,
                "price_vs_cloud": ichimoku_result.metadata.get("price_vs_cloud") if ichimoku_result and hasattr(ichimoku_result, 'metadata') else None,
                "tk_cross": ichimoku_result.metadata.get("tk_cross") if ichimoku_result and hasattr(ichimoku_result, 'metadata') else None,
                "cloud_color": ichimoku_result.metadata.get("cloud_color") if ichimoku_result and hasattr(ichimoku_result, 'metadata') else None,
                "available": ichimoku_result is not None,
            },

            # Williams %R - overbought/oversold momentum
            "williams_r": {
                "value": williams_result.value if williams_result and hasattr(williams_result, 'value') else None,
                "signal": williams_result.signal if williams_result and hasattr(williams_result, 'signal') else "N/A",
                "overbought": williams_result.metadata.get("overbought") if williams_result and hasattr(williams_result, 'metadata') else None,
                "oversold": williams_result.metadata.get("oversold") if williams_result and hasattr(williams_result, 'metadata') else None,
                "oversold_cross": williams_result.metadata.get("oversold_cross") if williams_result and hasattr(williams_result, 'metadata') else None,
                "overbought_cross": williams_result.metadata.get("overbought_cross") if williams_result and hasattr(williams_result, 'metadata') else None,
                "available": williams_result is not None,
            },

            # Parabolic SAR - trend following with stop levels
            "parabolic_sar": {
                "value": psar_result.value if psar_result and hasattr(psar_result, 'value') else None,
                "signal": psar_result.signal if psar_result and hasattr(psar_result, 'signal') else "N/A",
                "sar_position": psar_result.metadata.get("sar_position") if psar_result and hasattr(psar_result, 'metadata') else None,
                "trend_direction": psar_result.metadata.get("trend_direction") if psar_result and hasattr(psar_result, 'metadata') else None,
                "distance_to_sar_pct": psar_result.metadata.get("distance_to_sar_pct") if psar_result and hasattr(psar_result, 'metadata') else None,
                "available": psar_result is not None,
            },

            # Chaikin Money Flow - volume-weighted buying/selling pressure
            "cmf": {
                "value": cmf_result.value if cmf_result and hasattr(cmf_result, 'value') else None,
                "signal": cmf_result.signal if cmf_result and hasattr(cmf_result, 'signal') else "N/A",
                "interpretation": cmf_result.metadata.get("interpretation") if cmf_result and hasattr(cmf_result, 'metadata') else None,
                "trend": cmf_result.metadata.get("cmf_trend") if cmf_result and hasattr(cmf_result, 'metadata') else None,
                "available": cmf_result is not None,
            },

            # Accumulation/Distribution Line - cumulative volume flow
            "adl": {
                "signal": adl_result.signal if adl_result and hasattr(adl_result, 'signal') else "N/A",
                "trend": adl_result.metadata.get("adl_trend") if adl_result and hasattr(adl_result, 'metadata') else None,
                "price_confirmation": adl_result.metadata.get("adl_vs_price") if adl_result and hasattr(adl_result, 'metadata') else None,
                "divergence": adl_result.metadata.get("adl_vs_price") == "diverging" if adl_result and hasattr(adl_result, 'metadata') else False,
                "available": adl_result is not None,
            },
        }
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def run_full_analysis(
    symbol: str,
    account_size: float = 10000.0,
) -> Dict[str, Any]:
    """Run full stock analysis with scoring and trade plan.

    The analysis automatically determines the optimal trade style
    (day/swing/position) based on technical analysis.

    Args:
        symbol: Stock ticker symbol
        account_size: Account size for position sizing

    Returns:
        Full analysis with score, recommendation, and trade plan
    """
    try:
        result = run_analysis(
            symbol=symbol.upper(),
            account_size=account_size,
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
    """Detect chart patterns with historical success rates.

    Detects classic and advanced patterns including:
    - Reversal: Head & Shoulders, Double Top/Bottom, Rising/Falling Wedge
    - Continuation: Flags, Triangles, Cup & Handle, Channels
    - Consolidation: Rectangle

    Each pattern includes historical success rate based on research data.

    Args:
        symbol: Stock ticker symbol

    Returns:
        List of detected patterns with targets and success rates
    """
    try:
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=100)

        if not bars or len(bars) < 20:
            return {"symbol": symbol.upper(), "patterns": [], "error": "Insufficient data"}

        patterns_result = detect_chart_patterns(bars)

        # Extract patterns from the result dictionary
        patterns_found = patterns_result.get("patterns_found", []) if isinstance(patterns_result, dict) else []

        # Get ATR-based tolerance that was used
        atr_tolerance = patterns_result.get("atr_tolerance") if isinstance(patterns_result, dict) else None

        return {
            "symbol": symbol.upper(),
            "atr_tolerance_used": atr_tolerance,
            "patterns": [
                {
                    "name": p.get("name"),
                    "type": p.get("type"),  # bullish/bearish/neutral
                    "confidence": p.get("confidence"),
                    "target_price": p.get("target_price"),
                    "entry_price": p.get("entry_price"),
                    "stop_price": p.get("stop_price"),
                    # Historical success rate from research (converted to percentage)
                    "success_rate": round(p.get("success_rate", 0) * 100) if p.get("success_rate") else None,
                    "failure_rate": round(100 - p.get("success_rate", 0.5) * 100) if p.get("success_rate") else None,
                }
                for p in patterns_found[:5]  # Top 5 patterns
            ],
            "strongest_pattern": patterns_result.get("strongest_pattern") if isinstance(patterns_result, dict) else None,
            "pattern_count": len(patterns_found),
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
            "description": "Run comprehensive expert stock analysis with scoring and trade plan. Automatically determines optimal trade style (day/swing/position) from technical analysis.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker symbol"},
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
