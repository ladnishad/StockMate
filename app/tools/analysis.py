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
from app.tools.market_data import fetch_price_bars, fetch_fundamentals, fetch_sentiment
from app.tools.indicators import calculate_vwap, calculate_ema, calculate_rsi

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
) -> Optional[TradePlan]:
    """Generate a trade plan based on market snapshot.

    This tool analyzes the market snapshot and generates a detailed trade plan
    including entry, stop loss, targets, and position sizing.

    Args:
        snapshot: MarketSnapshot containing all market data
        account_size: Total account size in dollars
        risk_percentage: Percentage of account to risk per trade (default: 1%)

    Returns:
        TradePlan object if conditions are favorable, None otherwise

    Raises:
        ValueError: If parameters are invalid

    Example:
        >>> snapshot = build_snapshot("AAPL")
        >>> trade_plan = generate_trade_plan(snapshot, account_size=10000)
        >>> if trade_plan:
        ...     print(f"Trade Type: {trade_plan.trade_type}")
        ...     print(f"Entry: ${trade_plan.entry_price:.2f}")
        ...     print(f"Stop: ${trade_plan.stop_loss:.2f}")
        ...     print(f"Position Size: {trade_plan.position_size} shares")
    """
    logger.info(f"Generating trade plan for {snapshot.symbol}")

    if account_size <= 0:
        raise ValueError("account_size must be > 0")

    if not 0 < risk_percentage <= 100:
        raise ValueError("risk_percentage must be between 0 and 100")

    current_price = snapshot.current_price

    # Find nearest support and resistance levels
    supports = [p for p in snapshot.pivots if p.type == "support" and p.price < current_price]
    resistances = [p for p in snapshot.pivots if p.type == "resistance" and p.price > current_price]

    # Sort by proximity to current price
    supports.sort(key=lambda x: current_price - x.price)
    resistances.sort(key=lambda x: x.price - current_price)

    # Determine trade type based on market structure and indicators
    # Get EMAs
    ema_9 = next((i for i in snapshot.indicators if i.name == "EMA_9"), None)
    ema_20 = next((i for i in snapshot.indicators if i.name == "EMA_20"), None)
    ema_50 = next((i for i in snapshot.indicators if i.name == "EMA_50"), None)

    # Determine trend and trade type
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

    # Set entry price (current price or slightly below for better entry)
    entry_price = current_price * 0.998  # 0.2% below current

    # Set stop loss based on nearest support or ATR
    if supports:
        # Use nearest strong support as stop loss (with small buffer)
        stop_loss = supports[0].price * 0.995
    else:
        # Use percentage-based stop (2-3% depending on trade type)
        stop_pct = 0.02 if trade_type == "day" else 0.03
        stop_loss = entry_price * (1 - stop_pct)

    # Ensure stop loss is reasonable (not too tight or too wide)
    risk_per_share = entry_price - stop_loss
    if risk_per_share < entry_price * 0.01:  # Less than 1%
        stop_loss = entry_price * 0.99
        risk_per_share = entry_price - stop_loss
    elif risk_per_share > entry_price * 0.05:  # More than 5%
        stop_loss = entry_price * 0.95
        risk_per_share = entry_price - stop_loss

    # Calculate position size based on risk
    risk_amount = account_size * (risk_percentage / 100)
    position_size = int(risk_amount / risk_per_share)

    # Ensure position size is reasonable
    max_position_value = account_size * 0.2  # Max 20% of account
    max_shares = int(max_position_value / entry_price)
    position_size = min(position_size, max_shares)

    if position_size < 1:
        logger.warning(f"Position size too small for {snapshot.symbol}")
        return None

    # Set price targets based on resistance levels or R:R ratios
    targets = []

    if resistances:
        # Use actual resistance levels
        for i, resistance in enumerate(resistances[:3]):
            if resistance.price > entry_price:
                targets.append(resistance.price)

    # Fill in targets with R:R ratios if we don't have enough resistance levels
    risk_reward_ratios = [1.5, 2.5, 3.5]
    for i, rr in enumerate(risk_reward_ratios):
        if len(targets) <= i:
            target = entry_price + (risk_per_share * rr)
            targets.append(target)

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

    logger.info(f"Generated trade plan for {snapshot.symbol}: {trade_type} trade, {position_size} shares")
    return trade_plan


def run_analysis(symbol: str, account_size: float, use_ai: bool = False) -> AnalysisResponse:
    """Run complete stock analysis and generate trading recommendation.

    This is the main orchestration function that ties together all analysis tools
    to produce a final BUY or NO_BUY recommendation with detailed trade plan.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        account_size: Total account size in dollars
        use_ai: Whether to use AI-enhanced analysis (future feature)

    Returns:
        AnalysisResponse object with recommendation and trade plan

    Raises:
        ValueError: If parameters are invalid
        Exception: If analysis fails

    Example:
        >>> result = run_analysis("AAPL", account_size=10000)
        >>> print(f"Recommendation: {result.recommendation}")
        >>> print(f"Confidence: {result.confidence:.1f}%")
        >>> if result.trade_plan:
        ...     print(f"Entry: ${result.trade_plan.entry_price:.2f}")
    """
    logger.info(f"Running analysis for {symbol} (account: ${account_size:.2f})")

    try:
        # Build market snapshot
        snapshot = build_snapshot(symbol)

        # Initialize scoring system (0-100)
        score = 50  # Start neutral
        reasons = []

        # 1. Sentiment Analysis (weight: 20%)
        sentiment_weight = 20
        if snapshot.sentiment.label == "bullish":
            score += sentiment_weight
            reasons.append(f"Bullish sentiment (score: {snapshot.sentiment.score:.2f})")
        elif snapshot.sentiment.label == "bearish":
            score -= sentiment_weight
            reasons.append(f"Bearish sentiment (score: {snapshot.sentiment.score:.2f})")

        # 2. Trend Analysis - EMAs (weight: 25%)
        ema_signals = [i for i in snapshot.indicators if i.name.startswith("EMA")]
        if ema_signals:
            bullish_emas = sum(1 for ema in ema_signals if ema.signal == "bullish")
            bearish_emas = sum(1 for ema in ema_signals if ema.signal == "bearish")

            ema_score = (bullish_emas - bearish_emas) / len(ema_signals) * 25
            score += ema_score

            if ema_score > 0:
                reasons.append(f"Price above key EMAs ({bullish_emas}/{len(ema_signals)} bullish)")
            else:
                reasons.append(f"Price below key EMAs ({bearish_emas}/{len(ema_signals)} bearish)")

        # 3. RSI Analysis (weight: 15%)
        rsi_indicator = next((i for i in snapshot.indicators if i.name.startswith("RSI")), None)
        if rsi_indicator:
            rsi_value = rsi_indicator.value

            if 40 <= rsi_value <= 70:
                # Sweet spot - bullish momentum without overbought
                score += 15
                reasons.append(f"RSI in bullish zone ({rsi_value:.1f})")
            elif rsi_value < 30:
                # Oversold - potential bounce
                score += 10
                reasons.append(f"RSI oversold ({rsi_value:.1f}) - potential bounce")
            elif rsi_value > 80:
                # Overbought - risky
                score -= 15
                reasons.append(f"RSI overbought ({rsi_value:.1f}) - caution")

        # 4. VWAP Analysis (weight: 15%)
        vwap_indicator = next((i for i in snapshot.indicators if i.name == "VWAP"), None)
        if vwap_indicator:
            if vwap_indicator.signal == "bullish":
                score += 15
                reasons.append(f"Price above VWAP (${vwap_indicator.value:.2f})")
            elif vwap_indicator.signal == "bearish":
                score -= 15
                reasons.append(f"Price below VWAP (${vwap_indicator.value:.2f})")

        # 5. Support/Resistance (weight: 15%)
        current_price = snapshot.current_price
        nearby_resistance = [
            p for p in snapshot.pivots
            if p.type == "resistance" and 0 < (p.price - current_price) / current_price < 0.02
        ]
        nearby_support = [
            p for p in snapshot.pivots
            if p.type == "support" and 0 < (current_price - p.price) / current_price < 0.02
        ]

        if nearby_support and not nearby_resistance:
            score += 15
            reasons.append("Near strong support with room to resistance")
        elif nearby_resistance and not nearby_support:
            score -= 15
            reasons.append("Near strong resistance - limited upside")

        # 6. 52-Week Range (weight: 10%)
        if snapshot.fundamentals.fifty_two_week_high and snapshot.fundamentals.fifty_two_week_low:
            range_position = (
                (current_price - snapshot.fundamentals.fifty_two_week_low) /
                (snapshot.fundamentals.fifty_two_week_high - snapshot.fundamentals.fifty_two_week_low)
            )

            if 0.3 <= range_position <= 0.7:
                # Mid-range - good risk/reward
                score += 10
                reasons.append(f"Mid-range position ({range_position:.1%} of 52w range)")
            elif range_position < 0.3:
                # Near lows - potential value
                score += 5
                reasons.append(f"Near 52-week low ({range_position:.1%} of range)")
            elif range_position > 0.9:
                # Near highs - risky
                score -= 10
                reasons.append(f"Near 52-week high ({range_position:.1%} of range) - risky")

        # Normalize score to 0-100
        confidence = max(0, min(100, score))

        # Generate recommendation
        if confidence >= 65:
            recommendation = "BUY"
            # Generate trade plan
            trade_plan = generate_trade_plan(snapshot, account_size, risk_percentage=1.0)
        else:
            recommendation = "NO_BUY"
            trade_plan = None
            reasons.append(f"Confidence too low ({confidence:.1f}%) - wait for better setup")

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
