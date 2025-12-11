"""Analysis and trading logic tools.

All functions are designed to be used as LLM agent tools with clear
signatures, comprehensive docstrings, and structured input/output.
"""

from typing import List, Literal, Optional
from datetime import datetime
import logging
import numpy as np

from app.models.data import (
    PriceBar,
    Fundamentals,
    Sentiment,
    Indicator,
    StructuralPivot,
    MarketSnapshot,
)
from app.models.response import AnalysisResponse, TradePlan
from app.tools.market_data import fetch_price_bars, fetch_fundamentals, fetch_sentiment, fetch_latest_quote
from app.tools.indicators import (
    calculate_vwap,
    calculate_ema,
    calculate_rsi,
    analyze_volume,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    detect_divergences,
)

logger = logging.getLogger(__name__)


def find_structural_pivots(
    price_bars: List[PriceBar],
    lookback: int = 20,
    min_touches: int = 2,
) -> List[StructuralPivot]:
    """Find structural pivot points (support and resistance levels).

    This tool identifies key price levels where the stock has repeatedly
    bounced (support) or stalled (resistance). These levels are useful
    for setting entry points, stop losses, and price targets.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        lookback: Number of bars to look back for pivot detection
        min_touches: Minimum number of touches required to confirm a level

    Returns:
        List of StructuralPivot objects sorted by strength (descending)

    Raises:
        ValueError: If price_bars is empty or parameters are invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 100)
        >>> pivots = find_structural_pivots(bars, lookback=20, min_touches=2)
        >>> for pivot in pivots[:3]:  # Top 3 strongest levels
        ...     print(f"{pivot.type}: ${pivot.price:.2f} (strength: {pivot.strength:.0f})")
    """
    logger.info(f"Finding structural pivots in {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < lookback:
        raise ValueError(f"Need at least {lookback} bars for pivot detection")

    pivots = []

    # Extract highs and lows
    highs = [bar.high for bar in price_bars]
    lows = [bar.low for bar in price_bars]
    current_price = price_bars[-1].close

    # Find resistance levels (local maxima)
    for i in range(lookback, len(highs) - lookback):
        window_highs = highs[i - lookback:i + lookback + 1]
        if highs[i] == max(window_highs):
            # Found a local maximum
            level = highs[i]

            # Count touches within 1% of this level
            touches = sum(
                1 for h in highs
                if abs(h - level) / level <= 0.01
            )

            if touches >= min_touches:
                # Calculate strength based on touches and proximity to current price
                distance_pct = abs(level - current_price) / current_price * 100
                strength = min(100, (touches * 20) - (distance_pct * 2))

                pivots.append(StructuralPivot(
                    price=round(level, 2),
                    type="resistance",
                    strength=max(0, round(strength, 1)),
                    touches=touches,
                ))

    # Find support levels (local minima)
    for i in range(lookback, len(lows) - lookback):
        window_lows = lows[i - lookback:i + lookback + 1]
        if lows[i] == min(window_lows):
            # Found a local minimum
            level = lows[i]

            # Count touches within 1% of this level
            touches = sum(
                1 for l in lows
                if abs(l - level) / level <= 0.01
            )

            if touches >= min_touches:
                # Calculate strength based on touches and proximity to current price
                distance_pct = abs(level - current_price) / current_price * 100
                strength = min(100, (touches * 20) - (distance_pct * 2))

                pivots.append(StructuralPivot(
                    price=round(level, 2),
                    type="support",
                    strength=max(0, round(strength, 1)),
                    touches=touches,
                ))

    # Remove duplicates (levels within 0.5% of each other)
    unique_pivots = []
    for pivot in sorted(pivots, key=lambda x: x.strength, reverse=True):
        if not any(
            abs(pivot.price - existing.price) / existing.price <= 0.005
            for existing in unique_pivots
        ):
            unique_pivots.append(pivot)

    logger.info(f"Found {len(unique_pivots)} structural pivots")
    return sorted(unique_pivots, key=lambda x: x.strength, reverse=True)


def detect_key_levels(
    price_bars: List[PriceBar],
    current_price: Optional[float] = None,
) -> dict:
    """Detect key psychological and technical price levels.

    Identifies important price levels that traders watch:
    - Round numbers (100.00, 150.00, etc.) - psychological levels
    - Gap levels - overnight gaps that often get filled
    - Previous period highs/lows - daily, weekly, monthly resistance/support
    - Opening prices from significant periods

    These levels often act as support/resistance due to trader psychology
    and institutional order placement.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        current_price: Current price (optional, uses last bar if not provided)

    Returns:
        Dictionary containing categorized key levels with metadata

    Raises:
        ValueError: If price_bars is empty

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 60)
        >>> levels = detect_key_levels(bars)
        >>> print(f"Round numbers nearby: {levels['round_numbers']}")
        >>> print(f"Unfilled gaps: {levels['unfilled_gaps']}")
        >>> print(f"Previous day high: ${levels['previous_day_high']:.2f}")
    """
    logger.info(f"Detecting key levels for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if current_price is None:
        current_price = price_bars[-1].close

    key_levels = {
        "round_numbers": [],
        "unfilled_gaps": [],
        "previous_day_high": None,
        "previous_day_low": None,
        "previous_week_high": None,
        "previous_week_low": None,
        "previous_month_high": None,
        "previous_month_low": None,
        "all_levels": [],  # Consolidated list with types
    }

    # 1. Detect Round Number Levels (psychological levels)
    # Find nearest round numbers above and below current price
    price_magnitude = 10 ** (len(str(int(current_price))) - 1)  # e.g., 100 for $150

    # Major round numbers (e.g., 100, 150, 200)
    major_increment = price_magnitude / 2 if price_magnitude >= 10 else 10
    minor_increment = price_magnitude / 10 if price_magnitude >= 10 else 5

    # Find round numbers within +/- 20% of current price
    search_range = current_price * 0.20

    # Major levels (e.g., $100, $150, $200)
    major_levels = []
    test_price = (current_price // major_increment) * major_increment
    for i in range(-5, 6):
        level = test_price + (i * major_increment)
        if abs(level - current_price) <= search_range and level > 0:
            distance_pct = ((level - current_price) / current_price) * 100
            major_levels.append({
                "price": round(level, 2),
                "type": "round_major",
                "distance_pct": round(distance_pct, 2),
                "significance": "high" if level % price_magnitude == 0 else "medium",
            })

    # Half-levels (e.g., $125, $175)
    half_levels = []
    test_price = (current_price // minor_increment) * minor_increment
    for i in range(-10, 11):
        level = test_price + (i * minor_increment)
        if abs(level - current_price) <= search_range and level > 0:
            # Skip if already in major levels
            if not any(abs(m["price"] - level) < 1 for m in major_levels):
                distance_pct = ((level - current_price) / current_price) * 100
                half_levels.append({
                    "price": round(level, 2),
                    "type": "round_minor",
                    "distance_pct": round(distance_pct, 2),
                    "significance": "low",
                })

    key_levels["round_numbers"] = sorted(
        major_levels + half_levels,
        key=lambda x: abs(x["distance_pct"])
    )[:10]  # Keep 10 closest

    # 2. Detect Gap Levels (unfilled gaps)
    gaps = []
    for i in range(1, len(price_bars)):
        prev_bar = price_bars[i - 1]
        curr_bar = price_bars[i]

        # Gap up: current low > previous high
        if curr_bar.low > prev_bar.high:
            gap_size = curr_bar.low - prev_bar.high
            gap_size_pct = (gap_size / prev_bar.high) * 100

            # Check if gap has been filled
            filled = False
            for j in range(i + 1, len(price_bars)):
                if price_bars[j].low <= prev_bar.high:
                    filled = True
                    break

            if not filled and gap_size_pct >= 0.5:  # Only significant gaps (>0.5%)
                gaps.append({
                    "gap_high": round(curr_bar.low, 2),
                    "gap_low": round(prev_bar.high, 2),
                    "gap_size": round(gap_size, 2),
                    "gap_size_pct": round(gap_size_pct, 2),
                    "direction": "up",
                    "bar_index": i,
                    "filled": False,
                    "distance_from_current": round(
                        ((prev_bar.high - current_price) / current_price) * 100, 2
                    ),
                })

        # Gap down: current high < previous low
        elif curr_bar.high < prev_bar.low:
            gap_size = prev_bar.low - curr_bar.high
            gap_size_pct = (gap_size / prev_bar.low) * 100

            # Check if gap has been filled
            filled = False
            for j in range(i + 1, len(price_bars)):
                if price_bars[j].high >= prev_bar.low:
                    filled = True
                    break

            if not filled and gap_size_pct >= 0.5:  # Only significant gaps (>0.5%)
                gaps.append({
                    "gap_high": round(prev_bar.low, 2),
                    "gap_low": round(curr_bar.high, 2),
                    "gap_size": round(gap_size, 2),
                    "gap_size_pct": round(gap_size_pct, 2),
                    "direction": "down",
                    "bar_index": i,
                    "filled": False,
                    "distance_from_current": round(
                        ((prev_bar.low - current_price) / current_price) * 100, 2
                    ),
                })

    key_levels["unfilled_gaps"] = sorted(
        gaps,
        key=lambda x: abs(x["distance_from_current"])
    )[:5]  # Keep 5 closest gaps

    # 3. Previous Period Highs/Lows
    if len(price_bars) >= 2:
        # Previous day
        prev_day = price_bars[-2]
        key_levels["previous_day_high"] = round(prev_day.high, 2)
        key_levels["previous_day_low"] = round(prev_day.low, 2)
        key_levels["previous_day_close"] = round(prev_day.close, 2)

    if len(price_bars) >= 7:
        # Previous week (last 5 trading days)
        week_bars = price_bars[-7:-2]
        key_levels["previous_week_high"] = round(max(bar.high for bar in week_bars), 2)
        key_levels["previous_week_low"] = round(min(bar.low for bar in week_bars), 2)

    if len(price_bars) >= 30:
        # Previous month (last 20 trading days)
        month_bars = price_bars[-30:-2]
        key_levels["previous_month_high"] = round(max(bar.high for bar in month_bars), 2)
        key_levels["previous_month_low"] = round(min(bar.low for bar in month_bars), 2)

    # 4. Create consolidated list of all levels for easy access
    all_levels = []

    # Add period highs/lows
    if key_levels["previous_day_high"]:
        all_levels.append({
            "price": key_levels["previous_day_high"],
            "type": "previous_day_high",
            "significance": "high",
        })
        all_levels.append({
            "price": key_levels["previous_day_low"],
            "type": "previous_day_low",
            "significance": "high",
        })

    if key_levels["previous_week_high"]:
        all_levels.append({
            "price": key_levels["previous_week_high"],
            "type": "previous_week_high",
            "significance": "medium",
        })
        all_levels.append({
            "price": key_levels["previous_week_low"],
            "type": "previous_week_low",
            "significance": "medium",
        })

    if key_levels["previous_month_high"]:
        all_levels.append({
            "price": key_levels["previous_month_high"],
            "type": "previous_month_high",
            "significance": "medium",
        })
        all_levels.append({
            "price": key_levels["previous_month_low"],
            "type": "previous_month_low",
            "significance": "medium",
        })

    # Add round numbers (top 5)
    for rn in key_levels["round_numbers"][:5]:
        all_levels.append({
            "price": rn["price"],
            "type": rn["type"],
            "significance": rn["significance"],
        })

    # Add gap levels
    for gap in key_levels["unfilled_gaps"]:
        all_levels.append({
            "price": gap["gap_high"],
            "type": f"gap_{gap['direction']}_high",
            "significance": "medium",
        })
        all_levels.append({
            "price": gap["gap_low"],
            "type": f"gap_{gap['direction']}_low",
            "significance": "medium",
        })

    # Sort by proximity to current price
    key_levels["all_levels"] = sorted(
        all_levels,
        key=lambda x: abs(x["price"] - current_price)
    )

    # Calculate nearest support and resistance from all levels
    support_levels = [lvl for lvl in all_levels if lvl["price"] < current_price]
    resistance_levels = [lvl for lvl in all_levels if lvl["price"] > current_price]

    key_levels["nearest_support"] = (
        sorted(support_levels, key=lambda x: x["price"], reverse=True)[0]
        if support_levels else None
    )
    key_levels["nearest_resistance"] = (
        sorted(resistance_levels, key=lambda x: x["price"])[0]
        if resistance_levels else None
    )

    logger.info(
        f"Key Levels: {len(key_levels['round_numbers'])} round numbers, "
        f"{len(key_levels['unfilled_gaps'])} unfilled gaps, "
        f"Nearest support: {key_levels['nearest_support']['price'] if key_levels['nearest_support'] else 'N/A'}, "
        f"Nearest resistance: {key_levels['nearest_resistance']['price'] if key_levels['nearest_resistance'] else 'N/A'}"
    )

    return key_levels


def calculate_volume_profile(
    price_bars: List[PriceBar],
    num_bins: int = 50,
    value_area_percent: float = 70.0,
) -> dict:
    """Calculate Volume Profile with VPOC, Value Area, and HVN/LVN detection.

    Volume Profile is an institutional-grade tool that shows where the majority of
    trading activity occurred at specific price levels. It reveals:
    - Where institutions are positioned (high volume = strong hands)
    - Support/resistance levels based on actual trading activity
    - Low volume areas where price moves fast (liquidity gaps)

    Key Concepts:
    - VPOC (Volume Point of Control): Price level with highest traded volume
    - Value Area: Price range containing 70% of total volume (where market accepted price)
    - HVN (High Volume Nodes): Strong support/resistance from institutional positioning
    - LVN (Low Volume Nodes): Price levels with low activity - expect fast moves through these

    Technical Implementation Note:
    This implementation uses OHLCV (bar) data and distributes each bar's volume equally
    across all price levels touched by that bar (from low to high). This is the standard
    approach for volume profile calculation with bar data.

    For absolute precision, institutional traders use tick-by-tick data (Time Price
    Opportunity / TPO) which shows exact prices where trades occurred. However, OHLCV-based
    volume profile is widely used and provides sufficient accuracy for:
    - Daily/hourly/minute timeframes
    - Retail and professional trading
    - Identifying major support/resistance zones

    The equal distribution assumption works well because:
    1. It's neutral (doesn't favor any price in the bar's range)
    2. Over many bars, the distribution approximates actual volume
    3. Major volume clusters still emerge clearly at significant price levels

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        num_bins: Number of price levels to create (default: 50)
        value_area_percent: Percentage of volume for value area (default: 70%)

    Returns:
        Dictionary containing:
        - vpoc: Volume Point of Control (price with highest volume)
        - value_area_high: Upper bound of value area
        - value_area_low: Lower bound of value area
        - high_volume_nodes: List of HVN levels
        - low_volume_nodes: List of LVN levels
        - volume_distribution: Full volume by price level
        - poc_strength: How dominant the VPOC is (concentration metric)

    Raises:
        ValueError: If price_bars is empty or parameters invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 60)
        >>> vp = calculate_volume_profile(bars, num_bins=50)
        >>> print(f"VPOC: ${vp['vpoc']:.2f} (institutions positioned here)")
        >>> print(f"Value Area: ${vp['value_area_low']:.2f} - ${vp['value_area_high']:.2f}")
        >>> print(f"High Volume Nodes: {vp['high_volume_nodes']}")
    """
    logger.info(f"Calculating Volume Profile for {len(price_bars)} bars (bins: {num_bins})")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if num_bins < 10:
        raise ValueError("num_bins must be at least 10")

    if not 50 <= value_area_percent <= 90:
        raise ValueError("value_area_percent must be between 50 and 90")

    # Extract price and volume data
    highs = np.array([bar.high for bar in price_bars])
    lows = np.array([bar.low for bar in price_bars])
    closes = np.array([bar.close for bar in price_bars])
    volumes = np.array([bar.volume for bar in price_bars])

    # Determine price range
    price_min = np.min(lows)
    price_max = np.max(highs)
    price_range = price_max - price_min

    if price_range == 0:
        raise ValueError("Price range is zero - cannot calculate volume profile")

    # Create price bins (levels)
    bin_size = price_range / num_bins
    price_levels = np.linspace(price_min, price_max, num_bins + 1)

    # Initialize volume at each price level
    volume_at_price = np.zeros(num_bins)

    # Distribute volume across price levels
    # For each bar, distribute its volume proportionally across the price levels it touched
    for i in range(len(price_bars)):
        bar_low = lows[i]
        bar_high = highs[i]
        bar_volume = volumes[i]

        # Find which bins this bar touched
        low_bin = int((bar_low - price_min) / bin_size)
        high_bin = int((bar_high - price_min) / bin_size)

        # Clamp to valid range
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))

        # Distribute volume across touched bins
        bins_touched = high_bin - low_bin + 1
        if bins_touched > 0:
            volume_per_bin = bar_volume / bins_touched
            for bin_idx in range(low_bin, high_bin + 1):
                volume_at_price[bin_idx] += volume_per_bin

    # Calculate bin center prices
    bin_centers = price_levels[:-1] + (bin_size / 2)

    # Find VPOC (Volume Point of Control) - highest volume price
    vpoc_idx = np.argmax(volume_at_price)
    vpoc = bin_centers[vpoc_idx]
    vpoc_volume = volume_at_price[vpoc_idx]

    # Calculate POC strength (how concentrated is volume at VPOC)
    total_volume = np.sum(volume_at_price)
    poc_strength = (vpoc_volume / total_volume * 100) if total_volume > 0 else 0

    # Calculate Value Area (price range containing value_area_percent of total volume)
    target_volume = total_volume * (value_area_percent / 100)

    # Start from VPOC and expand up/down until we reach target volume
    value_area_volume = volume_at_price[vpoc_idx]
    va_low_idx = vpoc_idx
    va_high_idx = vpoc_idx

    while value_area_volume < target_volume:
        # Check which direction has more volume
        vol_above = volume_at_price[va_high_idx + 1] if va_high_idx < num_bins - 1 else 0
        vol_below = volume_at_price[va_low_idx - 1] if va_low_idx > 0 else 0

        if vol_above >= vol_below and va_high_idx < num_bins - 1:
            va_high_idx += 1
            value_area_volume += volume_at_price[va_high_idx]
        elif vol_below > 0 and va_low_idx > 0:
            va_low_idx -= 1
            value_area_volume += volume_at_price[va_low_idx]
        else:
            # Can't expand anymore
            break

    value_area_high = bin_centers[va_high_idx]
    value_area_low = bin_centers[va_low_idx]

    # Identify High Volume Nodes (HVN) - price levels with significantly high volume
    # Use 1.5x average volume as threshold
    avg_volume = np.mean(volume_at_price)
    hvn_threshold = avg_volume * 1.5

    high_volume_nodes = []
    for i, vol in enumerate(volume_at_price):
        if vol >= hvn_threshold:
            # Check if this is a local maximum (peak)
            is_peak = True
            for j in range(max(0, i - 2), min(num_bins, i + 3)):
                if j != i and volume_at_price[j] > vol:
                    is_peak = False
                    break

            if is_peak:
                high_volume_nodes.append({
                    "price": round(bin_centers[i], 2),
                    "volume": round(float(vol), 0),
                    "strength": round((vol / avg_volume), 2),
                })

    # Sort HVNs by volume (strongest first)
    high_volume_nodes = sorted(high_volume_nodes, key=lambda x: x["volume"], reverse=True)[:10]

    # Identify Low Volume Nodes (LVN) - areas where price moves fast
    # Use 0.5x average volume as threshold
    lvn_threshold = avg_volume * 0.5

    low_volume_nodes = []
    for i, vol in enumerate(volume_at_price):
        if vol <= lvn_threshold and vol > 0:
            # Check if this is a local minimum (valley)
            is_valley = True
            for j in range(max(0, i - 2), min(num_bins, i + 3)):
                if j != i and volume_at_price[j] < vol:
                    is_valley = False
                    break

            if is_valley:
                low_volume_nodes.append({
                    "price": round(bin_centers[i], 2),
                    "volume": round(float(vol), 0),
                    "weakness": round((avg_volume / vol if vol > 0 else 999), 2),
                })

    # Sort LVNs by weakness (most significant gaps first)
    low_volume_nodes = sorted(low_volume_nodes, key=lambda x: x["weakness"], reverse=True)[:10]

    # Create volume distribution for full profile
    volume_distribution = [
        {
            "price": round(bin_centers[i], 2),
            "volume": round(float(volume_at_price[i]), 0),
            "percentage": round((volume_at_price[i] / total_volume * 100), 2) if total_volume > 0 else 0,
        }
        for i in range(num_bins)
    ]

    # Determine current price position relative to value area
    current_price = closes[-1]
    if current_price > value_area_high:
        position = "above_value_area"
    elif current_price < value_area_low:
        position = "below_value_area"
    else:
        position = "within_value_area"

    # Calculate distance to VPOC (institutional positioning)
    distance_to_vpoc = ((current_price - vpoc) / vpoc) * 100

    logger.info(
        f"Volume Profile: VPOC=${vpoc:.2f} (strength: {poc_strength:.1f}%), "
        f"Value Area: ${value_area_low:.2f}-${value_area_high:.2f}, "
        f"HVNs: {len(high_volume_nodes)}, LVNs: {len(low_volume_nodes)}, "
        f"Current: {position}"
    )

    return {
        "vpoc": round(vpoc, 2),
        "vpoc_volume": round(float(vpoc_volume), 0),
        "poc_strength": round(poc_strength, 2),
        "value_area_high": round(value_area_high, 2),
        "value_area_low": round(value_area_low, 2),
        "value_area_volume_percent": round(value_area_percent, 1),
        "high_volume_nodes": high_volume_nodes,
        "low_volume_nodes": low_volume_nodes,
        "volume_distribution": volume_distribution,
        "current_price_position": position,
        "distance_to_vpoc_percent": round(distance_to_vpoc, 2),
        "total_volume": round(float(total_volume), 0),
        "num_bins": num_bins,
    }


def detect_chart_patterns(
    price_bars: List[PriceBar],
    min_pattern_bars: int = 20,
    tolerance: float = 0.03,
) -> dict:
    """Detect major chart patterns for reversal and continuation signals.

    Implements geometric pattern recognition for patterns that professional traders use:
    - Reversal Patterns: H&S, Inverse H&S, Double Tops/Bottoms, Wedges
    - Continuation Patterns: Flags, Triangles

    Each pattern is validated using strict geometric rules and provides:
    - Pattern type and reliability
    - Entry/exit levels
    - Expected price target
    - Pattern strength score

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        min_pattern_bars: Minimum bars required for pattern (default: 20)
        tolerance: Price level tolerance as percentage (default: 3%)

    Returns:
        Dictionary containing:
        - patterns_found: List of detected patterns with metadata
        - strongest_pattern: Most reliable pattern if any
        - bullish_patterns: Count of bullish patterns
        - bearish_patterns: Count of bearish patterns

    Raises:
        ValueError: If price_bars is empty or parameters invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 60)
        >>> patterns = detect_chart_patterns(bars)
        >>> if patterns['strongest_pattern']:
        >>>     p = patterns['strongest_pattern']
        >>>     print(f"Pattern: {p['name']} (confidence: {p['confidence']:.1f}%)")
        >>>     print(f"Target: ${p['target_price']:.2f}")
    """
    logger.info(f"Detecting chart patterns in {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < min_pattern_bars:
        raise ValueError(f"Need at least {min_pattern_bars} bars for pattern detection")

    # Extract OHLC data
    highs = np.array([bar.high for bar in price_bars])
    lows = np.array([bar.low for bar in price_bars])
    closes = np.array([bar.close for bar in price_bars])
    volumes = np.array([bar.volume for bar in price_bars])

    current_price = closes[-1]
    patterns_found = []

    # Helper: Find swing points (peaks and troughs)
    def find_swing_points(data, window=5):
        """Find local maxima (peaks) and minima (troughs).

        Uses strict inequality to avoid detecting flat tops/bottoms as swing points.
        This reduces false pattern detection in consolidating markets.
        """
        peaks = []
        troughs = []

        for i in range(window, len(data) - window):
            # Peak: strictly higher than surrounding points (avoids flat tops)
            if all(data[i] > data[i-j] for j in range(1, window+1)) and \
               all(data[i] > data[i+j] for j in range(1, window+1)):
                peaks.append((i, data[i]))

            # Trough: strictly lower than surrounding points (avoids flat bottoms)
            if all(data[i] < data[i-j] for j in range(1, window+1)) and \
               all(data[i] < data[i+j] for j in range(1, window+1)):
                troughs.append((i, data[i]))

        return peaks, troughs

    peaks, troughs = find_swing_points(highs, window=5)

    # Pattern 1: Head and Shoulders (Bearish Reversal)
    if len(peaks) >= 3 and len(troughs) >= 2:
        for i in range(len(peaks) - 2):
            left_shoulder = peaks[i]
            head = peaks[i + 1]
            right_shoulder = peaks[i + 2]

            # Validate H&S geometry
            # 1. Head must be higher than both shoulders
            if head[1] > left_shoulder[1] and head[1] > right_shoulder[1]:
                # 2. Shoulders should be roughly equal height (within tolerance)
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                if shoulder_diff <= tolerance:
                    # 3. Find neckline (lows between shoulders)
                    neckline_lows = [t for t in troughs if left_shoulder[0] < t[0] < right_shoulder[0]]
                    if len(neckline_lows) >= 1:
                        neckline = np.mean([t[1] for t in neckline_lows])

                        # Calculate target: Head to neckline distance projected down
                        pattern_height = head[1] - neckline
                        target_price = neckline - pattern_height

                        # Calculate confidence based on geometry perfection
                        shoulder_symmetry = 1 - shoulder_diff
                        confidence = min(85, shoulder_symmetry * 100)

                        patterns_found.append({
                            "name": "Head and Shoulders",
                            "type": "bearish_reversal",
                            "confidence": round(confidence, 1),
                            "neckline": round(neckline, 2),
                            "target_price": round(target_price, 2),
                            "current_price": round(current_price, 2),
                            "distance_to_entry_pct": round(((neckline - current_price) / current_price * 100), 2),
                            "expected_move_pct": round((pattern_height / neckline * 100), 2),
                            "pattern_complete": current_price < neckline,
                        })

    # Pattern 2: Inverse Head and Shoulders (Bullish Reversal)
    if len(troughs) >= 3 and len(peaks) >= 2:
        for i in range(len(troughs) - 2):
            left_shoulder = troughs[i]
            head = troughs[i + 1]
            right_shoulder = troughs[i + 2]

            # Validate inverse H&S geometry
            if head[1] < left_shoulder[1] and head[1] < right_shoulder[1]:
                shoulder_diff = abs(left_shoulder[1] - right_shoulder[1]) / left_shoulder[1]
                if shoulder_diff <= tolerance:
                    # Find neckline (highs between shoulders)
                    neckline_highs = [p for p in peaks if left_shoulder[0] < p[0] < right_shoulder[0]]
                    if len(neckline_highs) >= 1:
                        neckline = np.mean([p[1] for p in neckline_highs])

                        pattern_height = neckline - head[1]
                        target_price = neckline + pattern_height

                        shoulder_symmetry = 1 - shoulder_diff
                        confidence = min(85, shoulder_symmetry * 100)

                        patterns_found.append({
                            "name": "Inverse Head and Shoulders",
                            "type": "bullish_reversal",
                            "confidence": round(confidence, 1),
                            "neckline": round(neckline, 2),
                            "target_price": round(target_price, 2),
                            "current_price": round(current_price, 2),
                            "distance_to_entry_pct": round(((current_price - neckline) / neckline * 100), 2),
                            "expected_move_pct": round((pattern_height / neckline * 100), 2),
                            "pattern_complete": current_price > neckline,
                        })

    # Pattern 3: Double Top (Bearish Reversal)
    if len(peaks) >= 2:
        for i in range(len(peaks) - 1):
            first_peak = peaks[i]
            second_peak = peaks[i + 1]

            # Peaks should be similar height
            peak_diff = abs(first_peak[1] - second_peak[1]) / first_peak[1]
            if peak_diff <= tolerance:
                # Find the trough between peaks
                troughs_between = [t for t in troughs if first_peak[0] < t[0] < second_peak[0]]
                if troughs_between:
                    support_level = min([t[1] for t in troughs_between])

                    # Target: Distance from peaks to support
                    pattern_height = np.mean([first_peak[1], second_peak[1]]) - support_level
                    target_price = support_level - pattern_height

                    peak_symmetry = 1 - peak_diff
                    confidence = min(80, peak_symmetry * 95)

                    patterns_found.append({
                        "name": "Double Top",
                        "type": "bearish_reversal",
                        "confidence": round(confidence, 1),
                        "support_level": round(support_level, 2),
                        "target_price": round(target_price, 2),
                        "current_price": round(current_price, 2),
                        "distance_to_entry_pct": round(((support_level - current_price) / current_price * 100), 2),
                        "expected_move_pct": round((pattern_height / support_level * 100), 2),
                        "pattern_complete": current_price < support_level,
                    })

    # Pattern 4: Double Bottom (Bullish Reversal)
    if len(troughs) >= 2:
        for i in range(len(troughs) - 1):
            first_trough = troughs[i]
            second_trough = troughs[i + 1]

            trough_diff = abs(first_trough[1] - second_trough[1]) / first_trough[1]
            if trough_diff <= tolerance:
                peaks_between = [p for p in peaks if first_trough[0] < p[0] < second_trough[0]]
                if peaks_between:
                    resistance_level = max([p[1] for p in peaks_between])

                    pattern_height = resistance_level - np.mean([first_trough[1], second_trough[1]])
                    target_price = resistance_level + pattern_height

                    trough_symmetry = 1 - trough_diff
                    confidence = min(80, trough_symmetry * 95)

                    patterns_found.append({
                        "name": "Double Bottom",
                        "type": "bullish_reversal",
                        "confidence": round(confidence, 1),
                        "resistance_level": round(resistance_level, 2),
                        "target_price": round(target_price, 2),
                        "current_price": round(current_price, 2),
                        "distance_to_entry_pct": round(((current_price - resistance_level) / resistance_level * 100), 2),
                        "expected_move_pct": round((pattern_height / resistance_level * 100), 2),
                        "pattern_complete": current_price > resistance_level,
                    })

    # Pattern 5: Ascending Triangle (Bullish Continuation)
    # Look for horizontal resistance with rising support
    if len(peaks) >= 2 and len(troughs) >= 2:
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
        recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs

        # Check if peaks are roughly horizontal (resistance)
        if len(recent_peaks) >= 2:
            peak_prices = [p[1] for p in recent_peaks]
            peak_variance = np.std(peak_prices) / np.mean(peak_prices)

            if peak_variance < tolerance:  # Horizontal resistance
                # Check if troughs are rising
                if len(recent_troughs) >= 2:
                    trough_rising = all(recent_troughs[i][1] < recent_troughs[i+1][1]
                                      for i in range(len(recent_troughs)-1))

                    if trough_rising:
                        resistance = np.mean(peak_prices)
                        pattern_height = resistance - recent_troughs[0][1]
                        target_price = resistance + pattern_height

                        confidence = 75  # Moderate confidence for triangles

                        patterns_found.append({
                            "name": "Ascending Triangle",
                            "type": "bullish_continuation",
                            "confidence": round(confidence, 1),
                            "resistance": round(resistance, 2),
                            "target_price": round(target_price, 2),
                            "current_price": round(current_price, 2),
                            "distance_to_breakout_pct": round(((resistance - current_price) / current_price * 100), 2),
                            "expected_move_pct": round((pattern_height / resistance * 100), 2),
                            "pattern_complete": current_price > resistance,
                        })

    # Pattern 6: Descending Triangle (Bearish Continuation)
    if len(peaks) >= 2 and len(troughs) >= 2:
        recent_peaks = peaks[-3:] if len(peaks) >= 3 else peaks
        recent_troughs = troughs[-3:] if len(troughs) >= 3 else troughs

        # Check if troughs are roughly horizontal (support)
        if len(recent_troughs) >= 2:
            trough_prices = [t[1] for t in recent_troughs]
            trough_variance = np.std(trough_prices) / np.mean(trough_prices)

            if trough_variance < tolerance:  # Horizontal support
                # Check if peaks are falling
                if len(recent_peaks) >= 2:
                    peaks_falling = all(recent_peaks[i][1] > recent_peaks[i+1][1]
                                      for i in range(len(recent_peaks)-1))

                    if peaks_falling:
                        support = np.mean(trough_prices)
                        pattern_height = recent_peaks[0][1] - support
                        target_price = support - pattern_height

                        confidence = 75

                        patterns_found.append({
                            "name": "Descending Triangle",
                            "type": "bearish_continuation",
                            "confidence": round(confidence, 1),
                            "support": round(support, 2),
                            "target_price": round(target_price, 2),
                            "current_price": round(current_price, 2),
                            "distance_to_breakdown_pct": round(((support - current_price) / current_price * 100), 2),
                            "expected_move_pct": round((pattern_height / support * 100), 2),
                            "pattern_complete": current_price < support,
                        })

    # Pattern 7: Bull Flag (Bullish Continuation)
    # Look for strong uptrend followed by consolidation with slight downward slope
    if len(price_bars) >= 30:
        # Check for prior uptrend (flagpole)
        lookback = min(20, len(closes) - 10)
        flagpole_start = closes[-lookback]
        flag_start = closes[-10]

        # Strong prior move up (>5%)
        if (flag_start - flagpole_start) / flagpole_start > 0.05:
            # Recent consolidation/slight pullback
            recent_closes = closes[-10:]
            consolidation_range = (np.max(recent_closes) - np.min(recent_closes)) / np.mean(recent_closes)

            # Calculate slope of consolidation (should be slightly downward for bull flag)
            flag_slope = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]

            # Tight consolidation (<3%) with characteristic downward slope
            # True bull flag: -5% < slope < 0 (slight downward drift)
            if consolidation_range < 0.03 and -0.05 < flag_slope < 0.01:
                flagpole_height = flag_start - flagpole_start
                target_price = current_price + flagpole_height

                # Higher confidence if slope is ideal (-1% to 0%)
                if -0.01 <= flag_slope < 0:
                    confidence = 75  # Ideal slope
                else:
                    confidence = 68  # Acceptable but not perfect

                patterns_found.append({
                    "name": "Bull Flag",
                    "type": "bullish_continuation",
                    "confidence": round(confidence, 1),
                    "flag_high": round(np.max(recent_closes), 2),
                    "flag_slope_pct": round(flag_slope * 100, 2),
                    "target_price": round(target_price, 2),
                    "current_price": round(current_price, 2),
                    "flagpole_move_pct": round((flagpole_height / flagpole_start * 100), 2),
                    "expected_move_pct": round((flagpole_height / current_price * 100), 2),
                    "pattern_complete": False,  # Continuation patterns are anticipatory
                })

    # Calculate statistics
    bullish_patterns = [p for p in patterns_found if 'bullish' in p['type']]
    bearish_patterns = [p for p in patterns_found if 'bearish' in p['type']]

    # Find strongest pattern (highest confidence)
    strongest_pattern = None
    if patterns_found:
        strongest_pattern = max(patterns_found, key=lambda x: x['confidence'])

    logger.info(
        f"Chart Patterns: Found {len(patterns_found)} patterns - "
        f"{len(bullish_patterns)} bullish, {len(bearish_patterns)} bearish"
    )
    if strongest_pattern:
        logger.info(f"Strongest: {strongest_pattern['name']} ({strongest_pattern['confidence']:.1f}%)")

    return {
        "patterns_found": patterns_found,
        "pattern_count": len(patterns_found),
        "bullish_patterns": len(bullish_patterns),
        "bearish_patterns": len(bearish_patterns),
        "strongest_pattern": strongest_pattern,
        "net_sentiment": len(bullish_patterns) - len(bearish_patterns),
    }


def build_snapshot(symbol: str) -> MarketSnapshot:
    """Build a complete market snapshot for a stock.

    This tool orchestrates data collection from multiple sources and timeframes
    to create a comprehensive snapshot for analysis.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        MarketSnapshot object containing all market data and calculated indicators

    Raises:
        ValueError: If symbol is invalid or data fetch fails
        Exception: If any component fails

    Example:
        >>> snapshot = build_snapshot("AAPL")
        >>> print(f"Symbol: {snapshot.symbol}")
        >>> print(f"Current Price: ${snapshot.current_price:.2f}")
        >>> print(f"Indicators: {len(snapshot.indicators)}")
        >>> print(f"Pivots: {len(snapshot.pivots)}")
    """
    logger.info(f"Building market snapshot for {symbol}")

    try:
        # Fetch price data for multiple timeframes
        price_bars_1d = fetch_price_bars(symbol, timeframe="1d", days_back=100)

        # Get current price from latest bar
        current_price = price_bars_1d[-1].close

        # Fetch hourly data for intraday analysis (last 30 days)
        try:
            price_bars_1h = fetch_price_bars(symbol, timeframe="1h", days_back=30)
        except Exception as e:
            logger.warning(f"Could not fetch 1h bars: {e}")
            price_bars_1h = None

        # Fetch 15-minute data for day trading analysis (last 7 days)
        try:
            price_bars_15m = fetch_price_bars(symbol, timeframe="15m", days_back=7)
        except Exception as e:
            logger.warning(f"Could not fetch 15m bars: {e}")
            price_bars_15m = None

        # Fetch fundamentals
        fundamentals = fetch_fundamentals(symbol)

        # Fetch sentiment
        sentiment = fetch_sentiment(symbol)

        # Calculate indicators on daily timeframe
        indicators = []

        # VWAP
        try:
            vwap = calculate_vwap(price_bars_1d)
            indicators.append(vwap)
        except Exception as e:
            logger.warning(f"Could not calculate VWAP: {e}")

        # EMAs (multiple periods)
        for period in [9, 20, 50]:
            try:
                ema = calculate_ema(price_bars_1d, period=period)
                indicators.append(ema)
            except Exception as e:
                logger.warning(f"Could not calculate EMA({period}): {e}")

        # RSI
        try:
            rsi = calculate_rsi(price_bars_1d, period=14)
            indicators.append(rsi)
        except Exception as e:
            logger.warning(f"Could not calculate RSI: {e}")

        # Volume Analysis - CRITICAL
        try:
            volume = analyze_volume(price_bars_1d, sma_period=20)
            indicators.append(volume)
        except Exception as e:
            logger.warning(f"Could not analyze volume: {e}")

        # MACD
        try:
            macd = calculate_macd(price_bars_1d)
            indicators.append(macd)
        except Exception as e:
            logger.warning(f"Could not calculate MACD: {e}")

        # ATR
        try:
            atr = calculate_atr(price_bars_1d, period=14)
            indicators.append(atr)
        except Exception as e:
            logger.warning(f"Could not calculate ATR: {e}")

        # Bollinger Bands
        try:
            bb = calculate_bollinger_bands(price_bars_1d, period=20)
            indicators.append(bb)
        except Exception as e:
            logger.warning(f"Could not calculate Bollinger Bands: {e}")

        # Divergence Detection (RSI and MACD)
        try:
            rsi_divergence = detect_divergences(
                price_bars_1d,
                indicator_type="rsi",
                lookback_period=14,
                swing_detection_window=5
            )
            indicators.append(rsi_divergence)
        except Exception as e:
            logger.warning(f"Could not detect RSI divergences: {e}")

        try:
            macd_divergence = detect_divergences(
                price_bars_1d,
                indicator_type="macd",
                lookback_period=14,
                swing_detection_window=5
            )
            indicators.append(macd_divergence)
        except Exception as e:
            logger.warning(f"Could not detect MACD divergences: {e}")

        # Find structural pivots
        pivots = find_structural_pivots(price_bars_1d, lookback=20, min_touches=2)

        # Build snapshot
        snapshot = MarketSnapshot(
            symbol=symbol,
            current_price=current_price,
            price_bars_1d=price_bars_1d,
            price_bars_1h=price_bars_1h,
            price_bars_15m=price_bars_15m,
            fundamentals=fundamentals,
            sentiment=sentiment,
            indicators=indicators,
            pivots=pivots,
            timestamp=datetime.utcnow(),
        )

        logger.info(f"Successfully built snapshot for {symbol}")
        return snapshot

    except Exception as e:
        logger.error(f"Error building snapshot for {symbol}: {str(e)}")
        raise


def generate_trade_plan(
    snapshot: MarketSnapshot,
    account_size: float,
    risk_percentage: float = 1.0,
    use_live_quote: bool = True,
    profile: Optional["TraderProfile"] = None,
) -> Optional[TradePlan]:
    """Generate a trade plan based on market snapshot.

    This tool analyzes the market snapshot and generates a detailed trade plan
    including entry, stop loss, targets, and position sizing.

    With AlgoTrader Plus, this function can optionally fetch real-time bid/ask
    quotes for more accurate entry pricing and spread analysis.

    When a trader profile is provided, the trade plan is customized:
    - Stop method: ATR-based, structure-based, or percentage-based
    - Target method: R:R ratios, resistance levels, or Fibonacci extensions
    - Risk parameters: Profile-specific risk % and position limits
    - Trade type: Aligned with profile's allowed trade types

    Args:
        snapshot: MarketSnapshot containing all market data
        account_size: Total account size in dollars
        risk_percentage: Percentage of account to risk per trade (default: 1%)
        use_live_quote: Whether to fetch real-time quote for entry price (default: True)
        profile: Optional TraderProfile for customized trade planning

    Returns:
        TradePlan object if conditions are favorable, None otherwise

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> snapshot = build_snapshot("AAPL")
        >>> trade_plan = generate_trade_plan(snapshot, account_size=10000, use_live_quote=True)
        >>> if trade_plan:
        ...     print(f"Trade Type: {trade_plan.trade_type}")
        ...     print(f"Entry: ${trade_plan.entry_price:.2f}")
        ...     print(f"Stop: ${trade_plan.stop_loss:.2f}")
        ...     print(f"Position Size: {trade_plan.position_size} shares")
    """
    profile_name = profile.name if profile else "default"
    logger.info(f"Generating trade plan for {snapshot.symbol} (profile: {profile_name})")

    if account_size <= 0:
        raise ValueError("account_size must be > 0")

    if not 0 < risk_percentage <= 100:
        raise ValueError("risk_percentage must be between 0 and 100")

    current_price = snapshot.current_price
    live_quote = None
    spread_warning = None

    # Try to fetch real-time quote for better entry pricing
    if use_live_quote:
        try:
            live_quote = fetch_latest_quote(snapshot.symbol)
            if live_quote:
                # Use mid-price as reference, ask price for entry
                current_price = live_quote["mid_price"]

                # Check spread and warn if too wide
                spread_pct = live_quote["spread_pct"]
                if spread_pct > 0.5:  # Spread > 0.5%
                    spread_warning = f"Wide spread ({spread_pct:.2f}%)"
                    logger.warning(f"{snapshot.symbol} has wide spread: {spread_pct:.2f}%")

                logger.info(f"Using live quote for {snapshot.symbol}: bid=${live_quote['bid_price']:.2f} ask=${live_quote['ask_price']:.2f} spread={spread_pct:.3f}%")
        except Exception as e:
            logger.warning(f"Could not fetch live quote for {snapshot.symbol}: {e}. Using historical price.")
            live_quote = None

    # Find nearest support and resistance levels
    supports = [p for p in snapshot.pivots if p.type == "support" and p.price < current_price]
    resistances = [p for p in snapshot.pivots if p.type == "resistance" and p.price > current_price]

    # Sort by proximity to current price
    supports.sort(key=lambda x: current_price - x.price)
    resistances.sort(key=lambda x: x.price - current_price)

    # Determine trade type based on profile or market structure
    # Get EMAs for trend analysis
    ema_9 = next((i for i in snapshot.indicators if i.name == "EMA_9"), None)
    ema_20 = next((i for i in snapshot.indicators if i.name == "EMA_20"), None)
    ema_50 = next((i for i in snapshot.indicators if i.name == "EMA_50"), None)

    # If profile specified, use profile's preferred trade type
    if profile:
        # Use the first allowed trade type from the profile
        trade_type = profile.allowed_trade_types[0]
    else:
        # Determine trend and trade type from market structure
        if ema_9 and ema_20 and ema_50:
            if ema_9.value > ema_20.value > ema_50.value:
                # Strong uptrend - prefer swing/long
                if current_price > ema_9.value * 1.02:
                    trade_type = "swing"
                else:
                    trade_type = "long"
            elif ema_9.value > ema_20.value:
                # Moderate uptrend - swing trade
                trade_type = "swing"
            else:
                # Short-term momentum - day trade
                trade_type = "day"
        else:
            # Default to swing if we can't determine trend
            trade_type = "swing"

    # Get profile-specific parameters or defaults
    if profile:
        stop_method = profile.risk.stop_method
        atr_multiplier = profile.risk.atr_multiplier
        max_position_pct = profile.risk.max_position_percent
        target_method = profile.targets.method
        rr_ratios = profile.targets.rr_ratios
        validate_targets = profile.targets.validate_against_resistance
        use_fib_extensions = profile.targets.use_fibonacci_extensions
    else:
        stop_method = "atr"
        atr_multiplier = 2.0
        max_position_pct = 20.0
        target_method = "rr_ratio"
        rr_ratios = [1.5, 2.5, 3.5]
        validate_targets = True
        use_fib_extensions = False

    # Set entry price based on live quote or historical data
    if live_quote:
        # Use ask price for market buy, or mid-price minus small buffer for limit order
        entry_price = live_quote["ask_price"]  # Use ask for guaranteed fill
    else:
        # Fallback: use current price minus small buffer
        entry_price = current_price * 0.998  # 0.2% below current

    # Get ATR for volatility-based stop loss
    atr_indicator = next((i for i in snapshot.indicators if i.name.startswith("ATR")), None)

    # Set stop loss based on profile's stop method
    if stop_method == "atr" and atr_indicator:
        # Use ATR-based stop loss with profile's multiplier
        atr_value = atr_indicator.value
        atr_stop_distance = atr_value * atr_multiplier
        stop_loss = entry_price - atr_stop_distance

        # If we have support nearby, use the better of the two
        if supports and supports[0].price > stop_loss:
            stop_loss = supports[0].price * 0.995  # Support with buffer
    elif stop_method == "structure" and supports:
        # Structure-based: use nearest support level
        stop_loss = supports[0].price * 0.995  # 0.5% below support
    elif stop_method == "percentage":
        # Percentage-based stop
        stop_pct = 0.02 if trade_type == "day" else 0.03 if trade_type == "swing" else 0.05
        stop_loss = entry_price * (1 - stop_pct)
    elif atr_indicator:
        # Default fallback to ATR if available
        atr_value = atr_indicator.value
        atr_stop_distance = atr_value * atr_multiplier
        stop_loss = entry_price - atr_stop_distance
    else:
        # Final fallback: use support or percentage-based
        if supports:
            stop_loss = supports[0].price * 0.995
        else:
            stop_pct = 0.02 if trade_type == "day" else 0.03
            stop_loss = entry_price * (1 - stop_pct)

    # Ensure stop loss is reasonable (not too tight or too wide)
    risk_per_share = entry_price - stop_loss
    if risk_per_share < entry_price * 0.01:  # Less than 1%
        stop_loss = entry_price * 0.99
        risk_per_share = entry_price - stop_loss
    elif risk_per_share > entry_price * 0.08:  # More than 8% (widened for ATR)
        stop_loss = entry_price * 0.92
        risk_per_share = entry_price - stop_loss

    # Calculate position size based on risk
    risk_amount = account_size * (risk_percentage / 100)
    position_size = int(risk_amount / risk_per_share)

    # Ensure position size is reasonable using profile's max position
    max_position_value = account_size * (max_position_pct / 100)
    max_shares = int(max_position_value / entry_price)
    position_size = min(position_size, max_shares)

    if position_size < 1:
        logger.warning(f"Position size too small for {snapshot.symbol}")
        return None

    # Set price targets based on profile's target method
    targets = []

    # Calculate Fibonacci levels if needed for targets
    fib_indicator = None
    if use_fib_extensions or target_method == "fibonacci":
        try:
            fib_indicator = calculate_fibonacci_levels(snapshot.price_bars_1d)
        except Exception as e:
            logger.warning(f"Could not calculate Fibonacci for targets: {e}")

    if target_method == "fibonacci" and fib_indicator:
        # Use Fibonacci extension levels
        extensions = fib_indicator.metadata.get("extension", {})
        for level_name in ["1.272", "1.618", "2.000"]:
            if level_name in extensions and extensions[level_name] > entry_price:
                targets.append(extensions[level_name])
    elif target_method == "structure" and resistances:
        # Use resistance levels
        for resistance in resistances[:3]:
            if resistance.price > entry_price:
                targets.append(resistance.price)

    # Fill in targets with R:R ratios if we don't have enough levels
    for i, rr in enumerate(rr_ratios):
        if len(targets) <= i:
            target = entry_price + (risk_per_share * rr)
            targets.append(target)

    # Validate targets against resistance if configured
    if validate_targets and resistances:
        try:
            validation = validate_targets_against_structure(
                targets=targets,
                current_price=entry_price,
                resistances=list(resistances),
                fibonacci_indicator=fib_indicator,
            )
            if validation["warnings"]:
                for warning in validation["warnings"]:
                    logger.warning(warning)
        except Exception as e:
            logger.warning(f"Could not validate targets: {e}")

    trade_plan = TradePlan(
        trade_type=trade_type,
        entry_price=round(entry_price, 2),
        stop_loss=round(stop_loss, 2),
        target_1=round(targets[0], 2) if len(targets) > 0 else round(entry_price * 1.03, 2),
        target_2=round(targets[1], 2) if len(targets) > 1 else None,
        target_3=round(targets[2], 2) if len(targets) > 2 else None,
        position_size=position_size,
        risk_amount=round(risk_amount, 2),
        risk_percentage=risk_percentage,
    )

    quote_source = "live quote" if live_quote else "historical"
    spread_info = f" [{spread_warning}]" if spread_warning else ""
    logger.info(f"Generated trade plan for {snapshot.symbol}: {trade_type} trade, {position_size} shares (entry from {quote_source}){spread_info}")
    return trade_plan


def run_analysis(
    symbol: str,
    account_size: float,
    use_ai: bool = False,
    trader_profile: Optional[str] = None,
) -> AnalysisResponse:
    """Run complete stock analysis and generate trading recommendation.

    This is the main orchestration function that ties together all analysis tools
    to produce a final BUY or NO_BUY recommendation with detailed trade plan.

    When a trader profile is specified, the analysis is customized:
    - Scoring weights are adjusted based on the profile's trading style
    - Profile-specific indicators (ADX, Stochastic, Fibonacci) are included
    - Risk parameters and thresholds are profile-specific
    - Trade plans use profile-appropriate stop/target methods

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        account_size: Total account size in dollars
        use_ai: Whether to use AI-enhanced analysis (future feature)
        trader_profile: Optional profile name ('day_trader', 'swing_trader',
                        'position_trader', 'long_term_investor')

    Returns:
        AnalysisResponse object with recommendation and trade plan

    Raises:
        ValueError: If parameters are invalid
        Exception: If analysis fails

    Example:
        >>> result = run_analysis("AAPL", account_size=10000, trader_profile="swing_trader")
        >>> print(f"Recommendation: {result.recommendation}")
        >>> print(f"Confidence: {result.confidence:.1f}%")
        >>> if result.trade_plan:
        ...     print(f"Entry: ${result.trade_plan.entry_price:.2f}")
    """
    logger.info(
        f"Running analysis for {symbol} (account: ${account_size:.2f}, "
        f"profile: {trader_profile or 'default'})"
    )

    try:
        # Load profile if specified
        profile = None
        if trader_profile:
            from app.models.profile_presets import get_profile
            profile = get_profile(trader_profile)
            logger.info(f"Using trader profile: {profile.name}")

        # Build market snapshot
        snapshot = build_snapshot(symbol)

        # Initialize scoring system (0-100)
        # True percentage-based weights that sum to 100%.
        # Each factor contributes a percentage of the final score.
        # Starting from 0, factors add/subtract their weighted percentages.
        # Final score clamped to [0,100] with threshold triggering BUY recommendation.
        score = 0  # Start at 0
        reasons = []

        # Get weights from profile or use defaults
        if profile:
            WEIGHTS = {
                "sentiment": profile.weights.sentiment,
                "ema_trend": profile.weights.ema_trend,
                "rsi": profile.weights.rsi,
                "vwap": profile.weights.vwap,
                "volume": profile.weights.volume,
                "macd": profile.weights.macd,
                "bollinger": profile.weights.bollinger,
                "multi_tf": profile.weights.multi_tf,
                "support_resistance": profile.weights.support_resistance,
                "divergence": profile.weights.divergence,
                "volume_profile": profile.weights.volume_profile,
                "chart_patterns": profile.weights.chart_patterns,
                "fibonacci": profile.weights.fibonacci,
                "adx": profile.weights.adx,
                "stochastic": profile.weights.stochastic,
            }
            confidence_threshold = profile.buy_confidence_threshold
            rsi_overbought = profile.rsi_overbought
            rsi_oversold = profile.rsi_oversold
            adx_threshold = profile.adx_trend_threshold
        else:
            # Default weights (percentages that sum to 100%)
            # Total: 100.0% distributed across 12 active factors
            WEIGHTS = {
                "sentiment": 10.26,      # Sentiment strength
                "ema_trend": 12.82,      # Trend alignment (most important)
                "rsi": 7.69,             # Momentum oscillator
                "vwap": 7.69,            # Institutional positioning
                "volume": 10.26,         # Volume confirmation (critical)
                "macd": 10.26,           # Momentum and crossovers
                "bollinger": 7.69,       # Volatility bands
                "multi_tf": 7.69,        # Multi-timeframe confluence
                "support_resistance": 5.13,  # Key levels
                "divergence": 7.69,      # Reversal signals
                "volume_profile": 5.13,  # Institutional volume
                "chart_patterns": 7.69,  # Pattern recognition
                "fibonacci": 0.0,        # Not used in default
                "adx": 0.0,              # Not used in default
                "stochastic": 0.0,       # Not used in default
            }
            confidence_threshold = 65.0
            rsi_overbought = 70.0
            rsi_oversold = 30.0
            adx_threshold = 25.0
        # Note: ATR excluded (0%) - informational only, non-directional

        # 1. Sentiment Analysis (weight: 10.26%)
        if snapshot.sentiment.label == "bullish":
            score += WEIGHTS["sentiment"]
            reasons.append(f"Bullish sentiment (score: {snapshot.sentiment.score:.2f})")
        elif snapshot.sentiment.label == "bearish":
            score -= WEIGHTS["sentiment"]
            reasons.append(f"Bearish sentiment (score: {snapshot.sentiment.score:.2f})")
        # else: neutral = 0 points

        # 2. Trend Analysis - EMAs (weight: 12.82%)
        ema_signals = [i for i in snapshot.indicators if i.name.startswith("EMA")]
        if ema_signals:
            bullish_emas = sum(1 for ema in ema_signals if ema.signal == "bullish")
            bearish_emas = sum(1 for ema in ema_signals if ema.signal == "bearish")

            # Scale by alignment: +100% if all bullish, -100% if all bearish
            ema_alignment = (bullish_emas - bearish_emas) / len(ema_signals)
            ema_score = ema_alignment * WEIGHTS["ema_trend"]
            score += ema_score

            if ema_score > 0:
                reasons.append(f"Price above key EMAs ({bullish_emas}/{len(ema_signals)} bullish)")
            elif ema_score < 0:
                reasons.append(f"Price below key EMAs ({bearish_emas}/{len(ema_signals)} bearish)")

        # 3. RSI Analysis (weight varies by profile)
        rsi_indicator = next((i for i in snapshot.indicators if i.name.startswith("RSI")), None)
        if rsi_indicator and WEIGHTS["rsi"] > 0:
            rsi_value = rsi_indicator.value

            if 40 <= rsi_value <= rsi_overbought:
                # Sweet spot - bullish momentum without overbought
                score += WEIGHTS["rsi"]
                reasons.append(f"RSI in bullish zone ({rsi_value:.1f})")
            elif rsi_value < rsi_oversold:
                # Oversold - potential bounce (partial weight)
                score += WEIGHTS["rsi"] * 0.67
                reasons.append(f"RSI oversold ({rsi_value:.1f}) - potential bounce")
            elif rsi_value > rsi_overbought:
                # Overbought - risky
                score -= WEIGHTS["rsi"]
                reasons.append(f"RSI overbought ({rsi_value:.1f}) - caution")

        # 4. VWAP Analysis (weight: 7.69%)
        vwap_indicator = next((i for i in snapshot.indicators if i.name == "VWAP"), None)
        if vwap_indicator:
            if vwap_indicator.signal == "bullish":
                score += WEIGHTS["vwap"]
                reasons.append(f"Price above VWAP (${vwap_indicator.value:.2f})")
            elif vwap_indicator.signal == "bearish":
                score -= WEIGHTS["vwap"]
                reasons.append(f"Price below VWAP (${vwap_indicator.value:.2f})")

        # 5. Volume Analysis (weight: 10.26%) - CRITICAL
        volume_indicator = next((i for i in snapshot.indicators if i.name == "Volume"), None)
        if volume_indicator:
            if volume_indicator.signal == "bullish":
                score += WEIGHTS["volume"]
                rel_vol = volume_indicator.metadata.get("relative_volume", 1.0)
                reasons.append(f"Strong volume confirmation ({rel_vol:.1f}x avg)")
            elif volume_indicator.signal == "bearish":
                score -= WEIGHTS["volume"] * 0.75  # Slightly less penalty
                reasons.append("High volume distribution - bearish")
            elif volume_indicator.metadata.get("interpretation") == "low_volume":
                score -= WEIGHTS["volume"] * 0.5  # Half penalty for low volume
                reasons.append("Low volume - unreliable move")

        # 6. MACD Analysis (weight: 10.26%)
        macd_indicator = next((i for i in snapshot.indicators if i.name == "MACD"), None)
        if macd_indicator:
            if macd_indicator.metadata.get("bullish_crossover"):
                score += WEIGHTS["macd"]
                reasons.append("MACD bullish crossover - strong entry signal")
            elif macd_indicator.metadata.get("bearish_crossover"):
                score -= WEIGHTS["macd"]
                reasons.append("MACD bearish crossover - exit signal")
            elif macd_indicator.signal == "bullish":
                score += WEIGHTS["macd"] * 0.5  # Half weight for momentum vs crossover
                reasons.append("MACD bullish momentum")
            elif macd_indicator.signal == "bearish":
                score -= WEIGHTS["macd"] * 0.5
                reasons.append("MACD bearish momentum")

        # 7. Bollinger Bands (weight: 7.69%)
        bb_indicator = next((i for i in snapshot.indicators if i.name == "BollingerBands"), None)
        if bb_indicator:
            interpretation = bb_indicator.metadata.get("interpretation")
            if interpretation == "oversold":
                score += WEIGHTS["bollinger"]
                reasons.append("Price below Bollinger Bands - oversold bounce")
            elif interpretation == "overbought":
                score -= WEIGHTS["bollinger"]
                reasons.append("Price above Bollinger Bands - overbought")
            elif bb_indicator.metadata.get("squeeze"):
                score += WEIGHTS["bollinger"] * 0.33  # Partial weight for squeeze
                reasons.append("Bollinger squeeze - potential breakout setup")

        # 8. Multi-Timeframe Confluence (weight: 7.69%)
        try:
            confluence = analyze_multi_timeframe_confluence(snapshot)
            if confluence["score"] >= 70:
                score += WEIGHTS["multi_tf"]
                reasons.append(f"Strong timeframe alignment ({confluence['score']:.0f}%)")
            elif confluence["score"] >= 50:
                score += WEIGHTS["multi_tf"] * 0.53  # Moderate alignment
                reasons.append(f"Moderate timeframe alignment ({confluence['score']:.0f}%)")
            elif confluence["score"] < 30:
                score -= WEIGHTS["multi_tf"] * 0.67  # Penalty for divergence
                reasons.append("Timeframe divergence - conflicting signals")
        except Exception as e:
            logger.warning(f"Could not analyze confluence: {e}")

        # 9. Support/Resistance (weight: 5.13%)
        current_price = snapshot.current_price

        # Detect key levels for enhanced support/resistance analysis
        try:
            key_levels = detect_key_levels(snapshot.price_bars_1d, current_price)

            # Check for nearby key levels
            nearest_support = key_levels.get("nearest_support")
            nearest_resistance = key_levels.get("nearest_resistance")

            if nearest_support and nearest_resistance:
                support_distance = abs(current_price - nearest_support["price"]) / current_price
                resistance_distance = abs(nearest_resistance["price"] - current_price) / current_price

                # Better risk/reward if support is closer than resistance
                if support_distance < 0.01 and resistance_distance > 0.03:
                    score += WEIGHTS["support_resistance"]
                    reasons.append(
                        f"Near strong support (${nearest_support['price']:.2f}) "
                        f"with room to resistance"
                    )
                elif resistance_distance < 0.01:
                    score -= WEIGHTS["support_resistance"]
                    reasons.append(f"Near resistance (${nearest_resistance['price']:.2f}) - limited upside")

            # Check for unfilled gaps nearby (gap fill tendency)
            unfilled_gaps = key_levels.get("unfilled_gaps", [])
            if unfilled_gaps:
                closest_gap = unfilled_gaps[0]
                gap_distance = abs(closest_gap["distance_from_current"])
                if gap_distance < 2:  # Within 2%
                    if closest_gap["direction"] == "up" and closest_gap["distance_from_current"] < 0:
                        score += WEIGHTS["support_resistance"] * 0.5
                        reasons.append("Unfilled gap below - potential support")
                    elif closest_gap["direction"] == "down" and closest_gap["distance_from_current"] > 0:
                        score -= WEIGHTS["support_resistance"] * 0.5
                        reasons.append("Unfilled gap above - potential resistance")
        except Exception as e:
            logger.warning(f"Could not analyze key levels: {e}")
            # Fallback to basic pivot analysis
            nearby_resistance = [
                p for p in snapshot.pivots
                if p.type == "resistance" and 0 < (p.price - current_price) / current_price < 0.02
            ]
            nearby_support = [
                p for p in snapshot.pivots
                if p.type == "support" and 0 < (current_price - p.price) / current_price < 0.02
            ]

            if nearby_support and not nearby_resistance:
                score += WEIGHTS["support_resistance"]
                reasons.append("Near strong support with room to resistance")
            elif nearby_resistance and not nearby_support:
                score -= WEIGHTS["support_resistance"]
                reasons.append("Near strong resistance - limited upside")

        # 10. Divergence Detection (weight: 7.69%)
        rsi_divergence = next(
            (i for i in snapshot.indicators if i.name == "Divergence_RSI"),
            None
        )
        macd_divergence = next(
            (i for i in snapshot.indicators if i.name == "Divergence_MACD"),
            None
        )

        divergence_signals = []
        if rsi_divergence:
            if rsi_divergence.metadata.get("regular_bullish"):
                score += WEIGHTS["divergence"]
                divergence_signals.append("RSI bullish divergence")
            elif rsi_divergence.metadata.get("regular_bearish"):
                score -= WEIGHTS["divergence"]
                divergence_signals.append("RSI bearish divergence")
            elif rsi_divergence.metadata.get("hidden_bullish"):
                score += WEIGHTS["divergence"] * 0.53  # Hidden divergence less strong
                divergence_signals.append("Hidden RSI bullish divergence")

        if macd_divergence:
            if macd_divergence.metadata.get("regular_bullish"):
                score += WEIGHTS["divergence"]
                divergence_signals.append("MACD bullish divergence")
            elif macd_divergence.metadata.get("regular_bearish"):
                score -= WEIGHTS["divergence"]
                divergence_signals.append("MACD bearish divergence")

        if divergence_signals:
            reasons.append(" + ".join(divergence_signals) + " - strong reversal signal")

        # 11. ATR Volatility Check (contribution: informational only, 0 points)
        # Note: ATR signals are now neutral (non-directional). High ATR doesn't mean bearish;
        # it can occur in strong uptrends. We log it for informational purposes but don't score it.
        atr_indicator = next((i for i in snapshot.indicators if i.name.startswith("ATR")), None)
        if atr_indicator:
            volatility = atr_indicator.metadata.get("volatility")
            atr_pct = atr_indicator.metadata.get("atr_percentage", 0)
            # Just log volatility for context - don't adjust score
            if volatility == "high":
                reasons.append(f"High volatility ({atr_pct:.1f}% ATR) - use wider stops")
            elif volatility == "low":
                reasons.append(f"Low volatility ({atr_pct:.1f}% ATR) - stable price action")

        # 12. Volume Profile Analysis (weight: 5.13%)
        try:
            volume_profile = calculate_volume_profile(snapshot.price_bars_1d, num_bins=50)

            # Check position relative to value area
            position = volume_profile["current_price_position"]
            distance_to_vpoc = volume_profile["distance_to_vpoc_percent"]

            if position == "within_value_area":
                # Price in accepted value area - neutral to bullish
                score += WEIGHTS["volume_profile"] * 0.5
                reasons.append(f"Price within value area (market acceptance)")
            elif position == "below_value_area" and abs(distance_to_vpoc) < 5:
                # Price below value but near VPOC - potential bounce
                score += WEIGHTS["volume_profile"]
                reasons.append(f"Price near VPOC ${volume_profile['vpoc']:.2f} (institutional support)")
            elif position == "above_value_area":
                # Price above value area - bullish but extended
                score += WEIGHTS["volume_profile"] * 0.3
                reasons.append("Price above value area (bullish but extended)")

            # Check for High Volume Node support
            hvn_support = [
                hvn for hvn in volume_profile["high_volume_nodes"]
                if hvn["price"] < current_price and (current_price - hvn["price"]) / current_price < 0.02
            ]
            if hvn_support:
                score += WEIGHTS["volume_profile"] * 0.5
                strongest_hvn = max(hvn_support, key=lambda x: x["strength"])
                reasons.append(f"HVN support at ${strongest_hvn['price']:.2f} (institutional positioning)")

        except Exception as e:
            logger.warning(f"Could not analyze volume profile: {e}")

        # 13. Chart Pattern Recognition (weight: 7.69%)
        try:
            chart_patterns = detect_chart_patterns(snapshot.price_bars_1d, min_pattern_bars=20)

            if chart_patterns["pattern_count"] > 0:
                strongest = chart_patterns["strongest_pattern"]

                if strongest:
                    pattern_type = strongest["type"]
                    confidence_score = strongest["confidence"]

                    if "bullish" in pattern_type:
                        # Bullish pattern detected - scale by confidence
                        pattern_contribution = WEIGHTS["chart_patterns"] * (confidence_score / 100)
                        score += pattern_contribution
                        reasons.append(
                            f"{strongest['name']} pattern ({confidence_score:.0f}% confidence) "
                            f"- target ${strongest['target_price']:.2f}"
                        )
                    elif "bearish" in pattern_type:
                        # Bearish pattern detected - scale by confidence
                        pattern_contribution = WEIGHTS["chart_patterns"] * (confidence_score / 100)
                        score -= pattern_contribution
                        reasons.append(
                            f"{strongest['name']} pattern ({confidence_score:.0f}% confidence) "
                            f"- bearish signal"
                        )

                # Net sentiment from multiple patterns
                net_sentiment = chart_patterns["net_sentiment"]
                if abs(net_sentiment) >= 2:
                    if net_sentiment > 0:
                        score += WEIGHTS["chart_patterns"] * 0.33  # Bonus for multiple patterns
                        reasons.append(f"Multiple bullish patterns ({chart_patterns['bullish_patterns']} found)")
                    else:
                        score -= WEIGHTS["chart_patterns"] * 0.33
                        reasons.append(f"Multiple bearish patterns ({chart_patterns['bearish_patterns']} found)")

        except Exception as e:
            logger.warning(f"Could not detect chart patterns: {e}")

        # 14. Fibonacci Analysis (profile-specific, weight varies)
        if WEIGHTS.get("fibonacci", 0) > 0:
            try:
                fib_indicator = calculate_fibonacci_levels(snapshot.price_bars_1d)
                if fib_indicator.signal == "bullish" and fib_indicator.metadata.get("at_entry_level"):
                    score += WEIGHTS["fibonacci"]
                    reasons.append(
                        f"At Fibonacci {fib_indicator.metadata['nearest_level']} entry level"
                    )
                elif fib_indicator.signal == "bearish":
                    score -= WEIGHTS["fibonacci"] * 0.5
                elif fib_indicator.metadata.get("near_fib_level"):
                    # Near a Fib level but not ideal entry
                    score += WEIGHTS["fibonacci"] * 0.3
                    reasons.append(f"Near Fibonacci {fib_indicator.metadata['nearest_level']} level")
            except Exception as e:
                logger.warning(f"Could not calculate Fibonacci: {e}")

        # 15. ADX Trend Strength (profile-specific, weight varies)
        if WEIGHTS.get("adx", 0) > 0:
            try:
                from app.tools.indicators import calculate_adx
                adx = calculate_adx(snapshot.price_bars_1d)

                if adx.value >= adx_threshold:
                    if adx.metadata.get("trend_direction") == "bullish":
                        score += WEIGHTS["adx"]
                        reasons.append(f"Strong uptrend (ADX: {adx.value:.0f})")
                    elif adx.metadata.get("trend_direction") == "bearish":
                        score -= WEIGHTS["adx"]
                        reasons.append(f"Strong downtrend (ADX: {adx.value:.0f})")
                else:
                    # Ranging market - not ideal for trend-following
                    score -= WEIGHTS["adx"] * 0.3
                    reasons.append(f"Weak trend (ADX: {adx.value:.0f}) - ranging market")
            except Exception as e:
                logger.warning(f"Could not calculate ADX: {e}")

        # 16. Stochastic Oscillator (profile-specific, weight varies)
        if WEIGHTS.get("stochastic", 0) > 0:
            try:
                from app.tools.indicators import calculate_stochastic
                stoch = calculate_stochastic(snapshot.price_bars_1d)

                if stoch.metadata.get("bullish_crossover") and stoch.metadata.get("is_oversold"):
                    score += WEIGHTS["stochastic"]
                    reasons.append("Stochastic bullish crossover from oversold")
                elif stoch.metadata.get("bearish_crossover") and stoch.metadata.get("is_overbought"):
                    score -= WEIGHTS["stochastic"]
                    reasons.append("Stochastic bearish crossover from overbought")
                elif stoch.signal == "bullish":
                    score += WEIGHTS["stochastic"] * 0.5
                elif stoch.signal == "bearish":
                    score -= WEIGHTS["stochastic"] * 0.5
            except Exception as e:
                logger.warning(f"Could not calculate Stochastic: {e}")

        # Normalize score to 0-100
        # Score already calculated as percentage (0-100 range), just clamp
        confidence = max(0, min(100, score))

        # Shift to make 50% neutral baseline for comparison
        # (all bullish = ~100%, all bearish = ~0%, neutral mix = ~50%)
        confidence = max(0, min(100, score + 50))

        # Generate recommendation using profile-specific threshold
        if confidence >= confidence_threshold:
            recommendation = "BUY"
            # Generate trade plan with profile-specific parameters
            risk_pct = profile.risk.risk_percentage if profile else 1.0
            trade_plan = generate_trade_plan(
                snapshot,
                account_size,
                risk_percentage=risk_pct,
                profile=profile,
            )
        else:
            recommendation = "NO_BUY"
            trade_plan = None
            reasons.append(
                f"Confidence too low ({confidence:.1f}%) - "
                f"threshold: {confidence_threshold}%"
            )

        # Build reasoning text
        reasoning = " | ".join(reasons[:5])  # Top 5 reasons

        response = AnalysisResponse(
            symbol=symbol,
            recommendation=recommendation,
            confidence=round(confidence, 1),
            trade_plan=trade_plan,
            reasoning=reasoning,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )

        logger.info(
            f"Analysis complete for {symbol}: {recommendation} "
            f"(confidence: {confidence:.1f}%)"
        )

        return response

    except Exception as e:
        logger.error(f"Error running analysis for {symbol}: {str(e)}")
        raise


def calculate_fibonacci_levels(
    price_bars: List[PriceBar],
    swing_lookback: int = 20,
) -> Indicator:
    """Calculate Fibonacci retracement and extension levels.

    Fibonacci levels are critical for swing traders - the market respects these
    levels consistently, making them self-fulfilling prophecies.

    Key levels for entries:
    - 0.382 (38.2%): Shallow retracement, strong trend
    - 0.500 (50%): Common retracement level
    - 0.618 (61.8%): Golden ratio, strongest level
    - 0.786 (78.6%): Deep retracement, last chance

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        swing_lookback: Period to identify swing high and low

    Returns:
        Indicator object with Fibonacci analysis and signal for scoring

    Raises:
        ValueError: If price_bars is empty or insufficient data

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 100)
        >>> fib = calculate_fibonacci_levels(bars)
        >>> if fib.signal == "bullish" and fib.metadata['at_entry_level']:
        >>>     print(f"Entry at {fib.metadata['nearest_level']} level")
    """
    logger.info(f"Calculating Fibonacci levels for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < swing_lookback:
        raise ValueError(f"Need at least {swing_lookback} bars for Fibonacci calculation")

    # Find swing high and low in recent period
    recent_bars = price_bars[-swing_lookback:]
    swing_high = max(bar.high for bar in recent_bars)
    swing_low = min(bar.low for bar in recent_bars)

    current_price = price_bars[-1].close

    # Determine trend direction (for retracement vs extension)
    if current_price > (swing_high + swing_low) / 2:
        trend = "uptrend"
        # Retracement from swing high down to swing low
        retracement_base = swing_high
        retracement_target = swing_low
    else:
        trend = "downtrend"
        # Retracement from swing low up to swing high
        retracement_base = swing_low
        retracement_target = swing_high

    # Calculate Fibonacci retracement levels
    diff = retracement_base - retracement_target
    retracement_levels = {
        "0.000": retracement_base,
        "0.236": retracement_base - (diff * 0.236),
        "0.382": retracement_base - (diff * 0.382),
        "0.500": retracement_base - (diff * 0.500),
        "0.618": retracement_base - (diff * 0.618),
        "0.786": retracement_base - (diff * 0.786),
        "1.000": retracement_target,
    }

    # Calculate Fibonacci extension levels
    extension_levels = {
        "1.272": retracement_base + (diff * 0.272),
        "1.618": retracement_base + (diff * 0.618),
        "2.000": retracement_base + diff,
        "2.618": retracement_base + (diff * 1.618),
    }

    # Find nearest Fibonacci level to current price
    all_levels = {**retracement_levels, **extension_levels}
    nearest_level = min(all_levels.items(), key=lambda x: abs(x[1] - current_price))

    # Check if price is near a Fibonacci level (within 1%)
    near_fib = abs(nearest_level[1] - current_price) / current_price < 0.01

    # Determine if at a favorable entry level
    key_entry_levels = ["0.382", "0.500", "0.618", "0.786"]
    at_entry_level = nearest_level[0] in key_entry_levels and near_fib

    # Calculate midpoint for position check
    midpoint = (swing_high + swing_low) / 2

    # Determine signal for scoring
    if trend == "uptrend" and at_entry_level and current_price < midpoint * 1.05:
        # Good pullback entry in uptrend
        signal = "bullish"
    elif trend == "downtrend" and at_entry_level and current_price > midpoint * 0.95:
        # Good bounce entry in downtrend (potential reversal)
        signal = "bearish"
    elif near_fib:
        # Near a Fib level but not ideal entry
        signal = "neutral"
    else:
        signal = "neutral"

    logger.info(
        f"Fibonacci: Swing High={swing_high:.2f}, Swing Low={swing_low:.2f}, "
        f"Trend={trend}, Near Level={nearest_level[0]}, Signal={signal}"
    )

    return Indicator(
        name="Fibonacci",
        value=round(current_price, 2),
        signal=signal,
        metadata={
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "trend": trend,
            "retracement": {k: round(v, 2) for k, v in retracement_levels.items()},
            "extension": {k: round(v, 2) for k, v in extension_levels.items()},
            "nearest_level": nearest_level[0],
            "nearest_price": round(nearest_level[1], 2),
            "near_fib_level": near_fib,
            "at_entry_level": at_entry_level,
            "current_price": round(current_price, 2),
        },
    )


def validate_targets_against_structure(
    targets: List[float],
    current_price: float,
    resistances: List[StructuralPivot],
    fibonacci_indicator: Optional[Indicator] = None,
) -> dict:
    """Validate price targets against known resistance levels.

    Checks if proposed targets will hit major resistance levels before
    being reached, and provides warnings/adjustments. This helps ensure
    targets are realistic and achievable.

    Args:
        targets: List of proposed target prices
        current_price: Current stock price
        resistances: List of StructuralPivot resistance levels
        fibonacci_indicator: Optional Fibonacci indicator for extension validation

    Returns:
        Dictionary with validated targets and warnings

    Example:
        >>> validated = validate_targets_against_structure(
        ...     targets=[185.0, 190.0, 195.0],
        ...     current_price=180.0,
        ...     resistances=pivots
        ... )
        >>> if validated['warnings']:
        ...     print(f"Target concerns: {validated['warnings']}")
    """
    validated_targets = []
    warnings = []

    # Sort resistances by price (ascending), only those above current price
    sorted_resistances = sorted(
        [r for r in resistances if r.price > current_price],
        key=lambda x: x.price,
    )

    for i, target in enumerate(targets):
        target_info = {
            "original_target": round(target, 2),
            "validated_target": round(target, 2),
            "blocked_by_resistance": False,
            "resistance_level": None,
            "resistance_strength": None,
            "suggested_target": None,
            "near_fib_extension": None,
        }

        # Check if target is beyond any strong resistance
        for resistance in sorted_resistances:
            if resistance.price < target:
                # Target is beyond this resistance
                if resistance.strength >= 60:  # Strong resistance
                    target_info["blocked_by_resistance"] = True
                    target_info["resistance_level"] = round(resistance.price, 2)
                    target_info["resistance_strength"] = round(resistance.strength, 1)

                    # Suggest adjusted target just below resistance
                    suggested_target = resistance.price * 0.995
                    target_info["suggested_target"] = round(suggested_target, 2)

                    warnings.append(
                        f"Target {i + 1} (${target:.2f}) may be blocked by "
                        f"resistance at ${resistance.price:.2f} (strength: {resistance.strength:.0f})"
                    )
                    break

        # Also validate against Fibonacci extensions if provided
        if fibonacci_indicator and "extension" in fibonacci_indicator.metadata:
            for level_name, level_price in fibonacci_indicator.metadata["extension"].items():
                if abs(target - level_price) / target < 0.02:  # Within 2%
                    target_info["near_fib_extension"] = level_name

        validated_targets.append(target_info)

    return {
        "targets": validated_targets,
        "warnings": warnings,
        "has_blocked_targets": any(t["blocked_by_resistance"] for t in validated_targets),
        "resistance_count_ahead": len(sorted_resistances),
    }


def analyze_multi_timeframe_confluence(
    snapshot: MarketSnapshot,
) -> dict:
    """Analyze confluence across multiple timeframes.

    Highest probability trades occur when multiple timeframes align. This function
    scores the alignment of signals across daily, hourly, and 15-minute timeframes.

    Args:
        snapshot: MarketSnapshot containing multi-timeframe data

    Returns:
        Dictionary with confluence score and detailed breakdown

    Example:
        >>> snapshot = build_snapshot("AAPL")
        >>> confluence = analyze_multi_timeframe_confluence(snapshot)
        >>> print(f"Confluence Score: {confluence['score']}/100")
        >>> print(f"Alignment: {confluence['alignment']}")
    """
    logger.info(f"Analyzing multi-timeframe confluence for {snapshot.symbol}")

    scores = {
        "daily": 0,
        "hourly": 0,
        "minute": 0,
    }
    
    signals = {
        "daily": {"trend": None, "momentum": None, "volume": None},
        "hourly": {"trend": None, "momentum": None, "volume": None},
        "minute": {"trend": None, "momentum": None, "volume": None},
    }

    # Analyze daily timeframe (from main snapshot)
    if snapshot.price_bars_1d and len(snapshot.price_bars_1d) >= 50:
        daily_bars = snapshot.price_bars_1d
        
        # Trend (EMAs)
        emas = [i for i in snapshot.indicators if i.name.startswith("EMA")]
        if emas:
            bullish_emas = sum(1 for ema in emas if ema.signal == "bullish")
            if bullish_emas >= len(emas) * 0.66:  # 2/3 bullish
                signals["daily"]["trend"] = "bullish"
                scores["daily"] += 35
            elif bullish_emas <= len(emas) * 0.33:  # 1/3 bullish
                signals["daily"]["trend"] = "bearish"
            else:
                signals["daily"]["trend"] = "neutral"
                scores["daily"] += 15
        
        # Momentum (RSI, MACD)
        rsi = next((i for i in snapshot.indicators if i.name.startswith("RSI")), None)
        macd = next((i for i in snapshot.indicators if i.name == "MACD"), None)
        
        momentum_bullish = 0
        momentum_total = 0
        if rsi:
            momentum_total += 1
            if rsi.signal == "bullish":
                momentum_bullish += 1
        if macd:
            momentum_total += 1
            if macd.signal == "bullish":
                momentum_bullish += 1
        
        if momentum_total > 0:
            if momentum_bullish >= momentum_total * 0.66:
                signals["daily"]["momentum"] = "bullish"
                scores["daily"] += 35
            elif momentum_bullish <= momentum_total * 0.33:
                signals["daily"]["momentum"] = "bearish"
            else:
                signals["daily"]["momentum"] = "neutral"
                scores["daily"] += 15
        
        # Volume
        volume_ind = next((i for i in snapshot.indicators if i.name == "Volume"), None)
        if volume_ind:
            if volume_ind.signal == "bullish":
                signals["daily"]["volume"] = "bullish"
                scores["daily"] += 30
            elif volume_ind.signal == "bearish":
                signals["daily"]["volume"] = "bearish"
            else:
                signals["daily"]["volume"] = "neutral"
                scores["daily"] += 10

    # Analyze hourly timeframe (if available)
    if snapshot.price_bars_1h and len(snapshot.price_bars_1h) >= 20:
        from app.tools.indicators import calculate_ema, calculate_rsi
        
        hourly_bars = snapshot.price_bars_1h
        
        # Quick trend check with EMA
        try:
            ema_20 = calculate_ema(hourly_bars, period=20)
            if ema_20.signal == "bullish":
                signals["hourly"]["trend"] = "bullish"
                scores["hourly"] += 30
            elif ema_20.signal == "bearish":
                signals["hourly"]["trend"] = "bearish"
            else:
                signals["hourly"]["trend"] = "neutral"
                scores["hourly"] += 15
        except:
            pass
        
        # Momentum check
        try:
            rsi = calculate_rsi(hourly_bars, period=14)
            if rsi.signal == "bullish":
                signals["hourly"]["momentum"] = "bullish"
                scores["hourly"] += 30
            elif rsi.signal == "bearish":
                signals["hourly"]["momentum"] = "bearish"
            else:
                signals["hourly"]["momentum"] = "neutral"
                scores["hourly"] += 15
        except:
            pass

    # Analyze 15-minute timeframe (if available)
    if snapshot.price_bars_15m and len(snapshot.price_bars_15m) >= 20:
        from app.tools.indicators import calculate_ema, calculate_rsi
        
        minute_bars = snapshot.price_bars_15m
        
        # Quick trend check
        try:
            ema_20 = calculate_ema(minute_bars, period=20)
            if ema_20.signal == "bullish":
                signals["minute"]["trend"] = "bullish"
                scores["minute"] += 30
            elif ema_20.signal == "bearish":
                signals["minute"]["trend"] = "bearish"
            else:
                signals["minute"]["trend"] = "neutral"
                scores["minute"] += 15
        except:
            pass

    # Calculate overall confluence score
    total_score = sum(scores.values())
    max_possible = 100 * 3  # 100 points per timeframe
    confluence_score = (total_score / max_possible) * 100 if max_possible > 0 else 0

    # Determine alignment
    if confluence_score >= 70:
        alignment = "strong"
    elif confluence_score >= 50:
        alignment = "moderate"
    elif confluence_score >= 30:
        alignment = "weak"
    else:
        alignment = "divergent"

    # Count aligned signals
    all_signals = []
    for tf in signals.values():
        for signal in tf.values():
            if signal:
                all_signals.append(signal)
    
    bullish_count = all_signals.count("bullish")
    bearish_count = all_signals.count("bearish")
    neutral_count = all_signals.count("neutral")

    logger.info(
        f"Multi-timeframe confluence: {confluence_score:.1f}/100 ({alignment}), "
        f"Bullish: {bullish_count}, Bearish: {bearish_count}, Neutral: {neutral_count}"
    )

    return {
        "score": round(confluence_score, 1),
        "alignment": alignment,
        "timeframe_scores": scores,
        "signals": signals,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": neutral_count,
    }


def detect_candlestick_patterns(
    price_bars: List[PriceBar],
    lookback: int = 5,
) -> List[dict]:
    """Detect common candlestick patterns.

    Identifies reversal and continuation patterns that have statistical edge.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        lookback: Number of recent bars to analyze

    Returns:
        List of detected patterns with type and strength

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> patterns = detect_candlestick_patterns(bars)
        >>> for pattern in patterns:
        ...     print(f"{pattern['name']}: {pattern['signal']} ({pattern['strength']})")
    """
    logger.info(f"Detecting candlestick patterns in last {lookback} bars")

    if len(price_bars) < lookback:
        return []

    recent_bars = price_bars[-lookback:]
    patterns = []

    for i in range(1, len(recent_bars)):
        bar = recent_bars[i]
        prev_bar = recent_bars[i - 1] if i > 0 else None

        body = abs(bar.close - bar.open)
        range_size = bar.high - bar.low
        upper_wick = bar.high - max(bar.open, bar.close)
        lower_wick = min(bar.open, bar.close) - bar.low

        # Doji (indecision)
        if body < range_size * 0.1 and range_size > 0:
            patterns.append({
                "name": "Doji",
                "signal": "neutral",
                "strength": "medium",
                "description": "Indecision, potential reversal",
            })

        # Hammer (bullish reversal)
        if (lower_wick > body * 2 and 
            upper_wick < body * 0.3 and 
            bar.close < prev_bar.close if prev_bar else False):
            patterns.append({
                "name": "Hammer",
                "signal": "bullish",
                "strength": "strong",
                "description": "Bullish reversal after downtrend",
            })

        # Shooting Star (bearish reversal)
        if (upper_wick > body * 2 and 
            lower_wick < body * 0.3 and 
            bar.close > prev_bar.close if prev_bar else False):
            patterns.append({
                "name": "Shooting Star",
                "signal": "bearish",
                "strength": "strong",
                "description": "Bearish reversal after uptrend",
            })

        # Bullish Engulfing
        if (prev_bar and 
            prev_bar.close < prev_bar.open and  # Previous red
            bar.close > bar.open and  # Current green
            bar.open < prev_bar.close and 
            bar.close > prev_bar.open):
            patterns.append({
                "name": "Bullish Engulfing",
                "signal": "bullish",
                "strength": "strong",
                "description": "Strong bullish reversal",
            })

        # Bearish Engulfing
        if (prev_bar and 
            prev_bar.close > prev_bar.open and  # Previous green
            bar.close < bar.open and  # Current red
            bar.open > prev_bar.close and 
            bar.close < prev_bar.open):
            patterns.append({
                "name": "Bearish Engulfing",
                "signal": "bearish",
                "strength": "strong",
                "description": "Strong bearish reversal",
            })

    logger.info(f"Detected {len(patterns)} candlestick patterns")
    return patterns
