"""SDK MCP tools for the Claude Agent SDK integration.

These tools are available to sub-agents for gathering timeframe-specific data.
Each tool is parameterized so agents can request the appropriate timeframe.
"""

import logging
import base64
from typing import List, Dict, Any, Optional, Literal, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from app.agent.providers import AIProvider

from app.tools.market_data import (
    fetch_price_bars,
    fetch_latest_quote,
    fetch_latest_trade,
    fetch_snapshots,
)
from app.tools.analysis import (
    find_comprehensive_levels,
    calculate_volume_profile,
    detect_chart_patterns,
    calculate_fibonacci_levels,
)
from app.tools.indicators import detect_divergences, calculate_atr, calculate_adx
from app.tools.indicators import (
    calculate_rsi,
    calculate_macd,
    calculate_ema,
    calculate_vwap,
    calculate_bollinger_bands,
    analyze_volume,
    # Institutional-grade indicators
    calculate_ichimoku,
    calculate_williams_r,
    calculate_parabolic_sar,
    calculate_cmf,
    calculate_adl,
)
from app.storage.position_store import get_position_store

logger = logging.getLogger(__name__)


# =============================================================================
# Timeframe-Aware Data Gathering Tools
# =============================================================================


async def get_price_bars(
    symbol: str,
    timeframe: Literal["5m", "15m", "1h", "1d", "1w"],
    days_back: int,
) -> Dict[str, Any]:
    """Get price bars for specified timeframe.

    Args:
        symbol: Stock ticker symbol
        timeframe: Bar timeframe - "5m", "15m", "1h", "1d", "1w"
        days_back: How many days of data to fetch

    Returns:
        Dictionary with bars data and metadata
    """
    try:
        # fetch_price_bars expects lowercase format: "5m", "1d", "1w", etc.
        bars = fetch_price_bars(
            symbol.upper(),
            timeframe=timeframe,
            days_back=days_back,
        )

        if not bars or len(bars) < 5:
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "error": "Insufficient data",
                "bars_count": len(bars) if bars else 0,
            }

        # Extract OHLCV summary
        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]
        volumes = [b.volume for b in bars]

        # Calculate ATR for this timeframe using Wilder's smoothing
        true_ranges = []
        for i in range(1, len(bars)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i - 1])
            low_close = abs(lows[i] - closes[i - 1])
            true_ranges.append(max(high_low, high_close, low_close))

        # Use Wilder's smoothing (standard ATR method)
        if len(true_ranges) >= 14:
            # Initial ATR is simple average of first 14 TRs
            atr = sum(true_ranges[:14]) / 14
            # Apply Wilder's smoothing for remaining values
            for tr in true_ranges[14:]:
                atr = (atr * 13 + tr) / 14
        else:
            atr = sum(true_ranges) / len(true_ranges) if true_ranges else 0
        atr_pct = (atr / closes[-1] * 100) if closes[-1] > 0 else 0

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "bars_count": len(bars),
            "bars": bars,  # Include raw bars for Fibonacci calculation
            "current_price": closes[-1],
            "high": max(highs[-20:]) if len(highs) >= 20 else max(highs),
            "low": min(lows[-20:]) if len(lows) >= 20 else min(lows),
            "atr": round(atr, 2),
            "atr_pct": round(atr_pct, 2),
            "average_volume": sum(volumes) / len(volumes) if volumes else 0,
            "last_bar": {
                "open": bars[-1].open,
                "high": bars[-1].high,
                "low": bars[-1].low,
                "close": bars[-1].close,
                "volume": bars[-1].volume,
            },
        }
    except Exception as e:
        logger.error(f"Error getting {timeframe} bars for {symbol}: {e}")
        return {"symbol": symbol.upper(), "timeframe": timeframe, "error": str(e)}


async def get_technical_indicators(
    symbol: str,
    ema_periods: List[int],
    rsi_period: int = 14,
    timeframe: Literal["5m", "15m", "1h", "1d", "1w"] = "1d",
) -> Dict[str, Any]:
    """Calculate technical indicators with specified EMA periods.

    Args:
        symbol: Stock ticker symbol
        ema_periods: List of EMA periods (e.g., [5, 9, 20] for day, [9, 21, 50] for swing)
        rsi_period: RSI period (default 14)
        timeframe: Bar timeframe - "5m" for day, "1d" for swing, "1w" for position

    Returns:
        Dictionary with all technical indicators
    """
    try:
        # Need enough bars for longest EMA plus some buffer
        max_ema = max(ema_periods) if ema_periods else 50
        # Timeframe-aware data requirements
        if timeframe == "5m":
            # 5m bars: more bars per day, need fewer calendar days
            bars_needed = max(int(max_ema * 1.2) + 20, 100)
        elif timeframe == "1w":
            # Weekly bars: need 7x more calendar days to get enough bars
            # For 200 EMA: 200 * 7 * 1.5 + 100 = 2200 days (~6 years)
            bars_needed = max(int(max_ema * 7 * 1.5) + 100, 400)
        else:  # 1d, 15m, 1h
            bars_needed = max(int(max_ema * 1.5) + 30, 100)

        bars = fetch_price_bars(symbol.upper(), timeframe=timeframe, days_back=bars_needed)

        if not bars or len(bars) < max_ema:
            return {
                "symbol": symbol.upper(),
                "error": f"Insufficient data (need {max_ema}+ bars, have {len(bars) if bars else 0})",
            }

        closes = [b.close for b in bars]
        current_price = closes[-1]

        # Calculate EMAs for requested periods
        emas = {}
        ema_alignment = []
        for period in sorted(ema_periods):
            if len(bars) >= period:
                ema_result = calculate_ema(bars, period)
                ema_value = ema_result.value if hasattr(ema_result, 'value') else ema_result
                emas[f"ema_{period}"] = round(ema_value, 2) if ema_value else None
                if ema_value:
                    ema_alignment.append(current_price > ema_value)
            else:
                emas[f"ema_{period}"] = None

        # Calculate RSI
        rsi_result = calculate_rsi(bars, rsi_period)
        rsi_value = rsi_result.value if hasattr(rsi_result, 'value') else rsi_result

        # Calculate MACD
        macd_result = calculate_macd(bars)

        # Calculate Bollinger Bands
        bollinger_result = calculate_bollinger_bands(bars)

        # Volume analysis
        volume_result = analyze_volume(bars)
        volumes = [b.volume for b in bars]

        # Institutional-grade indicators (with error handling for resilience)
        try:
            ichimoku_result = calculate_ichimoku(bars) if len(bars) >= 52 else None
        except Exception:
            ichimoku_result = None

        try:
            williams_result = calculate_williams_r(bars)
        except Exception:
            williams_result = None

        try:
            psar_result = calculate_parabolic_sar(bars)
        except Exception:
            psar_result = None

        try:
            cmf_result = calculate_cmf(bars)
        except Exception:
            cmf_result = None

        try:
            adl_result = calculate_adl(bars)
        except Exception:
            adl_result = None

        # Calculate ATR (with dollar value)
        try:
            atr_result = calculate_atr(bars)
            atr_value = atr_result.value if hasattr(atr_result, 'value') else None
            atr_pct = (atr_value / current_price * 100) if atr_value and current_price > 0 else None
        except Exception:
            atr_value = None
            atr_pct = None

        # Calculate VWAP
        try:
            vwap_result = calculate_vwap(bars)
            vwap_value = vwap_result.value if hasattr(vwap_result, 'value') else None
        except Exception:
            vwap_value = None

        # Calculate ADX (trend strength)
        try:
            adx_result = calculate_adx(bars)
            adx_value = adx_result.value if hasattr(adx_result, 'value') else None
            # Determine trend strength category
            if adx_value:
                if adx_value > 50:
                    adx_strength = "very_strong"
                elif adx_value > 25:
                    adx_strength = "strong"
                elif adx_value > 20:
                    adx_strength = "moderate"
                else:
                    adx_strength = "weak"
            else:
                adx_strength = "unknown"
        except Exception:
            adx_value = None
            adx_strength = "unknown"

        # Determine EMA trend
        if len(ema_alignment) >= 2:
            if all(ema_alignment):
                ema_trend = "bullish_aligned"
            elif not any(ema_alignment):
                ema_trend = "bearish_aligned"
            else:
                ema_trend = "mixed"
        else:
            ema_trend = "unknown"

        return {
            "symbol": symbol.upper(),
            "current_price": current_price,
            "emas": emas,
            "ema_trend": ema_trend,
            "price_vs_emas": {
                f"above_ema_{p}": current_price > emas.get(f"ema_{p}", 0)
                if emas.get(f"ema_{p}")
                else None
                for p in ema_periods
            },
            "rsi": {
                "value": round(rsi_value, 2) if rsi_value else None,
                "signal": (
                    "oversold" if rsi_value and rsi_value < 30
                    else "overbought" if rsi_value and rsi_value > 70
                    else "neutral"
                ),
            },
            "macd": {
                "value": macd_result.value if hasattr(macd_result, 'value') else None,
                "signal": macd_result.signal if hasattr(macd_result, 'signal') else "neutral",
                "histogram": (
                    macd_result.metadata.get("histogram")
                    if hasattr(macd_result, 'metadata')
                    else None
                ),
            },
            "bollinger": {
                "upper": (
                    bollinger_result.metadata.get("upper_band")
                    if hasattr(bollinger_result, 'metadata')
                    else None
                ),
                "middle": (
                    bollinger_result.metadata.get("middle_band")
                    if hasattr(bollinger_result, 'metadata')
                    else None
                ),
                "lower": (
                    bollinger_result.metadata.get("lower_band")
                    if hasattr(bollinger_result, 'metadata')
                    else None
                ),
                "position": (
                    bollinger_result.metadata.get("interpretation")
                    if hasattr(bollinger_result, 'metadata')
                    else None
                ),
            },
            "volume": {
                "current": volumes[-1] if volumes else 0,
                "average": sum(volumes) / len(volumes) if volumes else 0,
                "relative": (
                    volume_result.metadata.get("relative_volume")
                    if hasattr(volume_result, 'metadata')
                    else None
                ),
                "trend": (
                    volume_result.metadata.get("interpretation")
                    if hasattr(volume_result, 'metadata')
                    else None
                ),
            },
            # Institutional-grade indicators
            "ichimoku": {
                "signal": ichimoku_result.signal if ichimoku_result and hasattr(ichimoku_result, 'signal') else "N/A",
                "price_vs_cloud": (
                    ichimoku_result.metadata.get("price_vs_cloud")
                    if ichimoku_result and hasattr(ichimoku_result, 'metadata')
                    else None
                ),
                "tk_cross": (
                    ichimoku_result.metadata.get("tk_cross")
                    if ichimoku_result and hasattr(ichimoku_result, 'metadata')
                    else None
                ),
                "cloud_color": (
                    ichimoku_result.metadata.get("cloud_color")
                    if ichimoku_result and hasattr(ichimoku_result, 'metadata')
                    else None
                ),
                "available": ichimoku_result is not None,
            } if ichimoku_result else {"available": False},
            "williams_r": {
                "value": (
                    round(williams_result.value, 2)
                    if williams_result and hasattr(williams_result, 'value') and williams_result.value is not None
                    else None
                ),
                "signal": williams_result.signal if williams_result and hasattr(williams_result, 'signal') else "N/A",
                "available": williams_result is not None,
            } if williams_result else {"available": False},
            "parabolic_sar": {
                "value": (
                    round(psar_result.value, 2)
                    if psar_result and hasattr(psar_result, 'value') and psar_result.value is not None
                    else None
                ),
                "trend_direction": (
                    psar_result.metadata.get("trend_direction")
                    if psar_result and hasattr(psar_result, 'metadata')
                    else None
                ),
                "signal": psar_result.signal if psar_result and hasattr(psar_result, 'signal') else "N/A",
                "available": psar_result is not None,
            } if psar_result else {"available": False},
            "cmf": {
                "value": (
                    round(cmf_result.value, 4)
                    if cmf_result and hasattr(cmf_result, 'value') and cmf_result.value is not None
                    else None
                ),
                "signal": cmf_result.signal if cmf_result and hasattr(cmf_result, 'signal') else "N/A",
                "interpretation": (
                    cmf_result.metadata.get("interpretation")
                    if cmf_result and hasattr(cmf_result, 'metadata')
                    else None
                ),
                "available": cmf_result is not None,
            } if cmf_result else {"available": False},
            "adl": {
                "signal": adl_result.signal if adl_result and hasattr(adl_result, 'signal') else "N/A",
                "trend": (
                    adl_result.metadata.get("adl_trend")
                    if adl_result and hasattr(adl_result, 'metadata')
                    else None
                ),
                "divergence": (
                    adl_result.metadata.get("adl_vs_price") == "diverging"
                    if adl_result and hasattr(adl_result, 'metadata')
                    else False
                ),
                "available": adl_result is not None,
            } if adl_result else {"available": False},
            # Enhanced data points for better AI analysis
            "atr": {
                "value": round(atr_value, 2) if atr_value else None,
                "pct": round(atr_pct, 2) if atr_pct else None,
                "volatility_regime": (
                    "high" if atr_pct and atr_pct > 3.0
                    else "moderate" if atr_pct and atr_pct > 1.5
                    else "low" if atr_pct else "unknown"
                ),
            },
            "vwap": {
                "value": round(vwap_value, 2) if vwap_value else None,
                "price_vs_vwap": (
                    "above" if vwap_value and current_price > vwap_value
                    else "below" if vwap_value and current_price < vwap_value
                    else "at" if vwap_value else "unknown"
                ),
                "distance_pct": (
                    round((current_price - vwap_value) / vwap_value * 100, 2)
                    if vwap_value and vwap_value > 0 else None
                ),
            },
            "adx": {
                "value": round(adx_value, 2) if adx_value else None,
                "strength": adx_strength,
                "trending": adx_value is not None and adx_value > 25,
                "interpretation": (
                    "Very strong trend - consider riding the trend" if adx_strength == "very_strong"
                    else "Strong trend - favor trend-following strategies" if adx_strength == "strong"
                    else "Moderate trend - trend emerging" if adx_strength == "moderate"
                    else "Weak/no trend - favor range-bound strategies" if adx_strength == "weak"
                    else "Unable to determine trend strength"
                ),
            },
        }
    except Exception as e:
        logger.error(f"Error getting indicators for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_support_resistance(
    symbol: str,
    timeframe: Literal["intraday", "daily", "weekly"],
) -> Dict[str, Any]:
    """Get support and resistance levels for specified timeframe.

    Args:
        symbol: Stock ticker symbol
        timeframe: Level timeframe - "intraday", "daily", or "weekly"

    Returns:
        Dictionary with support and resistance levels
    """
    try:
        # Different lookback periods for different timeframes
        # Note: Calendar days, not trading days. ~1.5x multiplier for weekends
        lookback_map = {
            "intraday": 20,   # 20 calendar days (~13 trading days) for intraday levels
            "daily": 50,      # 50 days for swing levels
            "weekly": 200,    # 200 days for position levels
        }
        days = lookback_map.get(timeframe, 50)

        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=days)

        if not bars or len(bars) < 10:
            return {
                "symbol": symbol.upper(),
                "timeframe": timeframe,
                "error": "Insufficient data",
            }

        levels = find_comprehensive_levels(bars)

        # Get current price for distance calculation
        current_price = bars[-1].close

        # Format levels with distance and institutional metrics
        def format_levels(level_list, max_count=5):
            formatted = []
            for level in level_list[:max_count]:
                price = level.get("price", 0)
                distance_pct = ((price - current_price) / current_price * 100) if current_price > 0 else 0
                formatted.append({
                    "price": round(price, 2),
                    "distance_pct": round(distance_pct, 2),
                    "type": level.get("type", "unknown"),
                    "strength": level.get("strength", 0),
                    # Institutional-grade metrics
                    "touches": level.get("touches", 0),
                    "high_volume_touches": level.get("high_volume_touches", 0),
                    "bounce_quality": level.get("bounce_quality", 0),
                    "reclaimed": level.get("reclaimed", False),
                    "reliability": level.get("reliability", "weak"),
                    "last_touch_bars_ago": level.get("last_touch_bars_ago"),
                })
            return formatted

        result = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "current_price": current_price,
            "support": format_levels(levels.get("support", [])),
            "resistance": format_levels(levels.get("resistance", [])),
        }

        # Add timeframe-specific levels
        if timeframe == "intraday":
            # Add prior day levels for intraday
            if len(bars) >= 2:
                prior_day = bars[-2]
                result["prior_day"] = {
                    "high": prior_day.high,
                    "low": prior_day.low,
                    "close": prior_day.close,
                }

        elif timeframe == "weekly":
            # Add all-time context for position trades
            all_highs = [b.high for b in bars]
            all_lows = [b.low for b in bars]
            result["range"] = {
                "period_high": max(all_highs),
                "period_low": min(all_lows),
                "current_position_pct": (
                    (current_price - min(all_lows)) / (max(all_highs) - min(all_lows)) * 100
                    if max(all_highs) != min(all_lows)
                    else 50
                ),
            }

        return result
    except Exception as e:
        logger.error(f"Error getting S/R for {symbol}: {e}")
        return {"symbol": symbol.upper(), "timeframe": timeframe, "error": str(e)}


async def get_fibonacci_levels(
    symbol: str,
    price_bars: List,
    trade_type: Literal["day", "swing", "position"] = "swing",
) -> Dict[str, Any]:
    """Calculate Fibonacci retracement and extension levels for the given price bars.

    Each trade style uses different lookback periods for swing detection:
    - Day trade: 10 bars (recent intraday swings)
    - Swing trade: 30 bars (multi-day swings)
    - Position trade: 50 bars (major trend swings)

    Args:
        symbol: Stock ticker symbol
        price_bars: Price bars already fetched for this timeframe
        trade_type: "day", "swing", or "position"

    Returns:
        Dict with retracement_levels, extension_levels, signal, trend, etc.
    """
    try:
        if not price_bars or len(price_bars) < 10:
            logger.warning(f"Insufficient bars for Fibonacci: {len(price_bars) if price_bars else 0}")
            return {
                "symbol": symbol.upper(),
                "error": "Insufficient data for Fibonacci",
                "trade_type": trade_type,
                "swing_high": None,
                "swing_low": None,
                "swing_range": 0,
                "trend": "unknown",
                "signal": "neutral",
                "current_price": None,
                "nearest_level": None,
                "retracement_levels": {},
                "extension_levels": {},
            }

        # Different lookback periods for different trade styles
        lookback_map = {"day": 15, "swing": 30, "position": 50}
        swing_lookback = lookback_map.get(trade_type, 30)

        # Calculate Fibonacci levels using the core analysis function
        fib_indicator = calculate_fibonacci_levels(
            price_bars,
            swing_lookback=min(swing_lookback, len(price_bars))
        )

        metadata = fib_indicator.metadata
        swing_high = metadata.get("swing_high")
        swing_low = metadata.get("swing_low")
        current_price = metadata.get("current_price", price_bars[-1].close if price_bars else 0)

        result = {
            "symbol": symbol.upper(),
            "trade_type": trade_type,
            "swing_high": swing_high,
            "swing_low": swing_low,
            "swing_range": (swing_high - swing_low) if swing_high and swing_low else 0,
            "trend": metadata.get("trend", "unknown"),
            "signal": fib_indicator.signal,
            "current_price": current_price,
            "nearest_level": metadata.get("nearest_level"),
            "retracement_levels": metadata.get("retracement", {}),
            "extension_levels": metadata.get("extension", {}),
        }

        logger.info(
            f"Fibonacci for {symbol} ({trade_type}): "
            f"High=${swing_high:.2f}, Low=${swing_low:.2f}, "
            f"Trend={result['trend']}, Signal={result['signal']}"
            if swing_high and swing_low else f"Fibonacci for {symbol}: No clear swings"
        )

        return result

    except Exception as e:
        logger.error(f"Error calculating Fibonacci for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "trade_type": trade_type,
            "swing_high": None,
            "swing_low": None,
            "swing_range": 0,
            "trend": "unknown",
            "signal": "neutral",
            "current_price": None,
            "nearest_level": None,
            "retracement_levels": {},
            "extension_levels": {},
        }


async def get_volume_profile(
    symbol: str,
    days_back: int,
) -> Dict[str, Any]:
    """Get volume profile analysis for specified lookback period.

    Args:
        symbol: Stock ticker symbol
        days_back: Number of days to analyze (1 for intraday, 50 for swing, 200 for position)

    Returns:
        Dictionary with volume profile data
    """
    try:
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=days_back)

        if not bars or len(bars) < 10:
            return {"symbol": symbol.upper(), "error": "Insufficient data"}

        profile = calculate_volume_profile(bars)

        return {
            "symbol": symbol.upper(),
            "lookback_days": days_back,
            "bars_analyzed": len(bars),
            "vpoc": profile.get("vpoc"),  # Volume Point of Control
            "value_area_high": profile.get("value_area_high"),
            "value_area_low": profile.get("value_area_low"),
            "high_volume_nodes": profile.get("high_volume_nodes", [])[:3],
            "low_volume_nodes": profile.get("low_volume_nodes", [])[:3],
        }
    except Exception as e:
        logger.error(f"Error getting volume profile for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_chart_patterns(
    symbol: str,
    days_back: int,
) -> Dict[str, Any]:
    """Get chart patterns with historical success rates.

    Detects classic and advanced patterns including:
    - Reversal: Head & Shoulders, Double Top/Bottom, Rising/Falling Wedge
    - Continuation: Flags, Triangles, Cup & Handle, Channels
    - Consolidation: Rectangle

    Each pattern includes historical success rate based on research data.

    Args:
        symbol: Stock ticker symbol
        days_back: Lookback period (20 for day, 100 for swing, 200 for position)

    Returns:
        Detected patterns with targets and success rates
    """
    try:
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=days_back)

        if not bars or len(bars) < 20:
            return {"symbol": symbol.upper(), "patterns": [], "error": "Insufficient data"}

        patterns_result = detect_chart_patterns(bars)
        patterns_found = patterns_result.get("patterns_found", []) if isinstance(patterns_result, dict) else []

        return {
            "symbol": symbol.upper(),
            "lookback_days": days_back,
            "atr_tolerance_used": patterns_result.get("atr_tolerance") if isinstance(patterns_result, dict) else None,
            "patterns": [
                {
                    "name": p.get("name"),
                    "type": p.get("type"),  # bullish/bearish/neutral
                    "confidence": p.get("confidence"),
                    "target_price": p.get("target_price"),
                    "entry_price": p.get("entry_price"),
                    "stop_price": p.get("stop_price"),
                    "success_rate": round(p.get("success_rate", 0) * 100) if p.get("success_rate") else None,
                }
                for p in patterns_found[:5]
            ],
            "strongest_pattern": patterns_result.get("strongest_pattern") if isinstance(patterns_result, dict) else None,
            "pattern_count": len(patterns_found),
        }
    except Exception as e:
        logger.error(f"Error getting chart patterns for {symbol}: {e}")
        return {"symbol": symbol.upper(), "patterns": [], "error": str(e)}


async def generate_chart(
    symbol: str,
    timeframe: Literal["5m", "15m", "1d", "1w"],
    days_back: int,
) -> Dict[str, Any]:
    """Generate candlestick chart for specified timeframe.

    Args:
        symbol: Stock ticker symbol
        timeframe: Chart timeframe
        days_back: Days of data to show

    Returns:
        Dictionary with base64-encoded chart image
    """
    try:
        import io
        import mplfinance as mpf
        import pandas as pd

        # fetch_price_bars expects lowercase format: "5m", "1d", "1w", etc.
        bars = fetch_price_bars(symbol.upper(), timeframe=timeframe, days_back=days_back)

        if not bars or len(bars) < 10:
            return {"symbol": symbol.upper(), "error": "Insufficient data for chart"}

        # Create DataFrame for mplfinance
        data = {
            "Date": [b.timestamp for b in bars],
            "Open": [b.open for b in bars],
            "High": [b.high for b in bars],
            "Low": [b.low for b in bars],
            "Close": [b.close for b in bars],
            "Volume": [b.volume for b in bars],
        }
        df = pd.DataFrame(data)
        df.set_index("Date", inplace=True)
        df.index = pd.to_datetime(df.index)

        # Determine EMAs based on timeframe
        ema_periods = {
            "5m": [5, 9, 20],
            "15m": [5, 9, 20],
            "1d": [9, 21, 50],
            "1w": [21, 50, 200],
        }
        periods = ema_periods.get(timeframe, [9, 21, 50])

        # Calculate EMAs that fit the data
        ema_lines = []
        for period in periods:
            if len(df) >= period:
                ema = df["Close"].ewm(span=period, adjust=False).mean()
                ema_lines.append(mpf.make_addplot(ema, color="blue" if period == periods[0] else "orange" if period == periods[1] else "purple"))

        # Create chart
        buf = io.BytesIO()
        mc = mpf.make_marketcolors(
            up="green",
            down="red",
            edge="inherit",
            wick="inherit",
            volume="in",
        )
        style = mpf.make_mpf_style(
            marketcolors=mc,
            gridstyle="-",
            gridcolor="gray",
            facecolor="white",
        )

        # Plot with EMAs if available
        kwargs = {
            "type": "candle",
            "style": style,
            "volume": True,
            "title": f"{symbol} {timeframe.upper()} Chart",
            "figsize": (12, 8),
            "savefig": dict(fname=buf, format="png", dpi=100),
        }
        if ema_lines:
            kwargs["addplot"] = ema_lines

        mpf.plot(df, **kwargs)
        buf.seek(0)

        # Encode to base64
        chart_base64 = base64.b64encode(buf.read()).decode("utf-8")

        return {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "chart_image_base64": chart_base64,
            "bars_plotted": len(df),
        }
    except Exception as e:
        logger.error(f"Error generating chart for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def analyze_chart_vision(
    symbol: str,
    chart_image_base64: str,
    trade_style: Literal["day", "swing", "position"],
    provider: "AIProvider",
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze chart image using the user's selected AI provider.

    Args:
        symbol: Stock ticker symbol
        chart_image_base64: Base64-encoded chart image
        trade_style: Trade style for analysis context
        provider: AI provider instance (Claude or Grok) for vision analysis
        user_id: User ID for usage tracking (optional)

    Returns:
        Dictionary with vision analysis results
    """
    try:
        import json

        # Style-specific prompts
        style_focus = {
            "day": "Focus on intraday patterns, VWAP position, opening range, momentum candles, and volume spikes.",
            "swing": "Focus on multi-day patterns (flags, triangles, bases), daily S/R, EMA alignment, and volume on breakouts.",
            "position": "Focus on weekly trend structure, major support/resistance, 50/200 EMA positioning, and accumulation patterns.",
        }

        prompt = f"""Analyze this {trade_style} trading chart for {symbol}.

{style_focus.get(trade_style, "")}

Identify:
1. **Trend Quality**: Is the trend clean, moderate, or choppy?
2. **Patterns**: Any chart patterns visible (flags, triangles, H&S, etc.)?
3. **Candlestick Patterns**: Recent candlestick patterns (engulfing, hammer, doji)?
4. **EMA Structure**: How is price positioned relative to the EMAs?
5. **Volume**: Volume confirmation or divergence?
6. **Warning Signs**: Any concerning patterns (divergences, exhaustion, failed breakouts)?

Respond with JSON:
{{
    "trend_quality": "clean" | "moderate" | "choppy",
    "visual_patterns": ["pattern1", "pattern2"],
    "candlestick_patterns": ["pattern1"],
    "ema_structure": "description of EMA positioning",
    "volume_confirmation": "description of volume pattern",
    "warning_signs": ["warning1"],
    "confidence_modifier": -20 to +20,
    "summary": "One sentence summary"
}}
"""

        # Use the provider's analyze_image method (works for both Claude and Grok)
        response = await provider.analyze_image(
            image_base64=chart_image_base64,
            prompt=prompt,
            model_type="planning",
        )

        # Track usage for vision analysis
        if user_id:
            try:
                from app.services.usage_tracker import get_usage_tracker
                from app.models.usage import OperationType, ModelProvider as UsageModelProvider
                tracker = get_usage_tracker()
                usage_provider = UsageModelProvider.GROK if provider.supports_x_search else UsageModelProvider.CLAUDE
                await tracker.track_ai_response(
                    user_id=user_id,
                    provider=usage_provider,
                    model=provider.get_model("planning"),
                    operation_type=OperationType.IMAGE_ANALYSIS,
                    response=response,
                    symbol=symbol,
                    endpoint=f"/create-plan-v2-stream/vision-{trade_style}",
                )
            except Exception as track_err:
                logger.warning(f"Failed to track vision usage for {symbol}: {track_err}")

        # Parse response from AIResponse.content
        response_text = response.content

        # Try to extract JSON
        try:
            # Find JSON in response
            json_start = response_text.find("{")
            json_end = response_text.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                result = json.loads(json_str)
                result["symbol"] = symbol.upper()
                result["trade_style"] = trade_style
                return result
        except json.JSONDecodeError:
            pass

        # Fallback if JSON parsing fails
        return {
            "symbol": symbol.upper(),
            "trade_style": trade_style,
            "trend_quality": "moderate",
            "visual_patterns": [],
            "candlestick_patterns": [],
            "ema_structure": "Unable to parse",
            "volume_confirmation": "Unable to parse",
            "warning_signs": [],
            "confidence_modifier": 0,
            "summary": response_text[:200] if response_text else "No response",
            "raw_response": response_text,
        }
    except Exception as e:
        logger.error(f"Error analyzing chart for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "trade_style": trade_style,
            "error": str(e),
            "confidence_modifier": 0,
        }


# =============================================================================
# Common Tools (Shared Context)
# =============================================================================


async def get_position_status(symbol: str, user_id: str = "default") -> Dict[str, Any]:
    """Get current position status for a stock.

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
                "direction": None,
                "status": "none",
            }

        direction = "long" if position.current_size > 0 else "short" if position.current_size < 0 else None

        return {
            "symbol": symbol.upper(),
            "has_position": True,
            "direction": direction,
            "status": position.status,
            "entry_price": position.avg_entry_price or position.entry_price,
            "current_size": abs(position.current_size),
            "stop_loss": position.stop_loss,
            "targets": {
                "target_1": position.target_1,
                "target_2": position.target_2,
                "target_3": position.target_3,
            },
            "unrealized_pnl": position.unrealized_pnl,
            "unrealized_pnl_pct": position.unrealized_pnl_pct,
            "trade_type": position.trade_type,
        }
    except Exception as e:
        logger.error(f"Error getting position for {symbol}: {e}")
        return {"symbol": symbol.upper(), "error": str(e)}


async def get_market_context() -> Dict[str, Any]:
    """Get overall market context (indices, direction).

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
            "timestamp": datetime.utcnow().isoformat(),
        }
    except Exception as e:
        logger.error(f"Error getting market context: {e}")
        return {"error": str(e)}


async def get_current_price(symbol: str) -> Dict[str, Any]:
    """Get current price and quote for a stock.

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


async def get_news_sentiment(
    symbol: str,
    days_back: int = 7,
) -> Dict[str, Any]:
    """Get news sentiment and recent headlines for a stock from Finnhub.

    This uses the Finnhub News API which is included in the free tier
    (60 API calls/minute).

    NEWS INFLUENCE ON TRADE DECISIONS:
    - For DAY TRADES: News has LOW weight (20%) - intraday momentum dominates
    - For SWING TRADES: News has MODERATE weight (50%) - catalysts affect 2-10 day holds
    - For POSITION TRADES: News has HIGH weight (100%) - major news affects weeks/months

    Args:
        symbol: Stock ticker symbol
        days_back: Number of days of news to fetch

    Returns:
        Dictionary with:
        - sentiment: Overall sentiment label (bullish/neutral/bearish)
        - sentiment_score: Numeric score from -1 to 1
        - article_count: Number of articles analyzed
        - headlines: List of recent article titles
        - has_breaking_news: Whether there's very recent news (< 2 hours)
        - key_themes: Common topics from article tags
        - summary: Brief summary of news context
    """
    from app.tools.market_data import fetch_news_sentiment

    try:
        # fetch_news_sentiment is now async and returns NewsContext
        news_context = await fetch_news_sentiment(symbol.upper(), days_back=days_back)

        # Extract headlines from articles
        headlines = [article.title for article in news_context.articles[:5]]

        # Build summary
        summary_parts = []
        if news_context.article_count > 0:
            summary_parts.append(f"{news_context.article_count} articles analyzed")
        if news_context.has_breaking_news:
            summary_parts.append("BREAKING NEWS detected")
        if news_context.key_themes:
            summary_parts.append(f"Topics: {', '.join(news_context.key_themes[:3])}")

        return {
            "symbol": symbol.upper(),
            "sentiment": news_context.overall_sentiment,
            "sentiment_score": news_context.sentiment_score,
            "article_count": news_context.article_count,
            "headlines": headlines,
            "has_breaking_news": news_context.has_breaking_news,
            "key_themes": news_context.key_themes,
            "summary": ". ".join(summary_parts) if summary_parts else "No recent news",
            "data_source": news_context.data_source,
        }
    except Exception as e:
        logger.warning(f"Error getting news for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "sentiment": "neutral",
            "sentiment_score": 0,
            "article_count": 0,
            "headlines": [],
            "has_breaking_news": False,
            "key_themes": [],
            "summary": "News unavailable",
            "data_source": "error",
        }


async def get_fundamentals(symbol: str) -> Dict[str, Any]:
    """Get fundamental financial data for a stock.

    This tool retrieves comprehensive fundamental data including:
    - Valuation metrics (P/E, P/B, P/S, PEG)
    - Growth metrics (EPS growth, revenue growth)
    - Profitability (margins, ROE, ROA)
    - Financial health (debt ratios, liquidity)
    - Earnings calendar (CRITICAL for risk assessment)

    IMPORTANT FOR TRADE DECISIONS:
    - For DAY TRADES: Fundamentals have LOW weight (momentum dominates)
    - For SWING TRADES: Fundamentals have MODERATE weight (1-2 week exposure)
    - For POSITION TRADES: Fundamentals have HIGH weight (weeks-months exposure)

    EARNINGS WARNING: If earnings are within 7 days, this is HIGH RISK
    for swing and position trades due to gap risk.

    Args:
        symbol: Stock ticker symbol

    Returns:
        Dictionary with comprehensive fundamental data and risk flags
    """
    from app.tools.market_data import fetch_fundamentals

    try:
        fundamentals = await fetch_fundamentals(symbol.upper())

        # Build response dict
        result = {
            "symbol": symbol.upper(),
            # Valuation Summary
            "valuation": {
                "pe_ratio": fundamentals.pe_ratio,
                "pe_forward": fundamentals.pe_forward,
                "pb_ratio": fundamentals.pb_ratio,
                "ps_ratio": fundamentals.ps_ratio,
                "peg_ratio": fundamentals.peg_ratio,
                "assessment": fundamentals.get_valuation_assessment(),
            },
            # Growth Summary
            "growth": {
                "eps": fundamentals.eps,
                "eps_growth_yoy": fundamentals.eps_growth_yoy,
                "revenue_growth_yoy": fundamentals.revenue_growth_yoy,
                "revenue_growth_3y": fundamentals.revenue_growth_3y,
            },
            # Profitability Summary
            "profitability": {
                "gross_margin": fundamentals.gross_margin,
                "operating_margin": fundamentals.operating_margin,
                "net_margin": fundamentals.net_margin,
                "roe": fundamentals.roe,
                "roa": fundamentals.roa,
            },
            # Financial Health Summary
            "health": {
                "debt_to_equity": fundamentals.debt_to_equity,
                "current_ratio": fundamentals.current_ratio,
                "quick_ratio": fundamentals.quick_ratio,
                "assessment": fundamentals.get_financial_health_score(),
            },
            # 52-Week Range
            "price_range": {
                "52w_high": fundamentals.fifty_two_week_high,
                "52w_low": fundamentals.fifty_two_week_low,
                "range_position": fundamentals.fifty_two_week_range_position,
            },
            # CRITICAL: Earnings Risk
            "earnings_risk": {
                "has_risk": fundamentals.has_earnings_risk(),
                "days_until": (
                    fundamentals.next_earnings.days_until
                    if fundamentals.next_earnings
                    else None
                ),
                "date": (
                    fundamentals.next_earnings.date
                    if fundamentals.next_earnings
                    else None
                ),
                "eps_estimate": (
                    fundamentals.next_earnings.eps_estimate
                    if fundamentals.next_earnings
                    else None
                ),
                "beat_rate": fundamentals.earnings_beat_rate,
                "avg_surprise": fundamentals.avg_earnings_surprise,
            },
            # Trade Style Guidance
            "trade_guidance": {
                "day_trade_weight": "low",
                "swing_trade_weight": (
                    "moderate" if not fundamentals.has_earnings_risk() else "high_risk"
                ),
                "position_trade_weight": (
                    "high" if not fundamentals.has_earnings_risk() else "extreme_risk"
                ),
            },
        }

        return result

    except Exception as e:
        logger.error(f"Error getting fundamentals for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "earnings_risk": {"has_risk": False},
        }


async def get_divergences(
    symbol: str,
    lookback: int = 75,
) -> Dict[str, Any]:
    """Detect RSI and MACD divergences for reversal signals.

    Divergences are powerful leading indicators:
    - Bullish divergence: Price makes lower low, indicator makes higher low (reversal up)
    - Bearish divergence: Price makes higher high, indicator makes lower high (reversal down)

    Args:
        symbol: Stock ticker symbol
        lookback: Number of calendar days of data to analyze (default: 75).
                  Note: 75 calendar days ≈ 50 trading days, ensuring enough bars for MACD.

    Returns:
        Dictionary with RSI and MACD divergence detection results
    """
    try:
        bars = fetch_price_bars(symbol.upper(), timeframe="1d", days_back=lookback)

        # MACD requires 35 bars (slow_period=26 + signal_period=9)
        if not bars or len(bars) < 35:
            return {
                "symbol": symbol.upper(),
                "error": "Insufficient data for divergence detection",
            }

        # Detect RSI divergences
        rsi_result = {"detected": False, "type": "neutral", "interpretation": "None"}
        try:
            rsi_div = detect_divergences(bars, indicator_type="rsi")
            if rsi_div.metadata.get("regular_bullish"):
                rsi_result = {
                    "detected": True,
                    "type": "bullish",
                    "interpretation": "Price making lower lows but RSI making higher lows - potential reversal up",
                }
            elif rsi_div.metadata.get("regular_bearish"):
                rsi_result = {
                    "detected": True,
                    "type": "bearish",
                    "interpretation": "Price making higher highs but RSI making lower highs - potential reversal down",
                }
            elif rsi_div.metadata.get("hidden_bullish"):
                rsi_result = {
                    "detected": True,
                    "type": "hidden_bullish",
                    "interpretation": "Hidden bullish divergence - trend continuation signal",
                }
            elif rsi_div.metadata.get("hidden_bearish"):
                rsi_result = {
                    "detected": True,
                    "type": "hidden_bearish",
                    "interpretation": "Hidden bearish divergence - downtrend continuation signal",
                }
        except Exception as e:
            logger.warning(f"RSI divergence detection failed: {e}")

        # Detect MACD divergences
        macd_result = {"detected": False, "type": "neutral", "interpretation": "None"}
        try:
            macd_div = detect_divergences(bars, indicator_type="macd")
            if macd_div.metadata.get("regular_bullish"):
                macd_result = {
                    "detected": True,
                    "type": "bullish",
                    "interpretation": "Price making lower lows but MACD making higher lows - strong reversal signal",
                }
            elif macd_div.metadata.get("regular_bearish"):
                macd_result = {
                    "detected": True,
                    "type": "bearish",
                    "interpretation": "Price making higher highs but MACD making lower highs - strong reversal signal",
                }
            elif macd_div.metadata.get("hidden_bullish"):
                macd_result = {
                    "detected": True,
                    "type": "hidden_bullish",
                    "interpretation": "Hidden bullish MACD divergence - trend continuation",
                }
            elif macd_div.metadata.get("hidden_bearish"):
                macd_result = {
                    "detected": True,
                    "type": "hidden_bearish",
                    "interpretation": "Hidden bearish MACD divergence - downtrend continuation",
                }
        except Exception as e:
            logger.warning(f"MACD divergence detection failed: {e}")

        # Overall divergence assessment
        has_bullish = rsi_result["type"] in ["bullish", "hidden_bullish"] or macd_result["type"] in ["bullish", "hidden_bullish"]
        has_bearish = rsi_result["type"] in ["bearish", "hidden_bearish"] or macd_result["type"] in ["bearish", "hidden_bearish"]

        return {
            "symbol": symbol.upper(),
            "rsi_divergence": rsi_result,
            "macd_divergence": macd_result,
            "has_divergence": rsi_result["detected"] or macd_result["detected"],
            "overall_signal": "bullish" if has_bullish else "bearish" if has_bearish else "neutral",
        }
    except Exception as e:
        logger.error(f"Error detecting divergences for {symbol}: {e}")
        return {
            "symbol": symbol.upper(),
            "error": str(e),
            "rsi_divergence": {"detected": False, "type": "neutral", "interpretation": "Error"},
            "macd_divergence": {"detected": False, "type": "neutral", "interpretation": "Error"},
            "has_divergence": False,
            "overall_signal": "neutral",
        }
