"""Technical indicator calculation tools.

All functions are designed to be used as LLM agent tools with clear
signatures, comprehensive docstrings, and structured input/output.
"""

from typing import List, Dict, Any
import logging
import pandas as pd
import numpy as np

from app.models.data import PriceBar, Indicator

logger = logging.getLogger(__name__)


def calculate_vwap(price_bars: List[PriceBar]) -> Indicator:
    """Calculate Volume Weighted Average Price (VWAP).

    VWAP is the average price weighted by volume, commonly used by institutional
    traders to assess entry/exit quality. Price above VWAP is generally bullish.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)

    Returns:
        Indicator object with VWAP value and trading signal

    Raises:
        ValueError: If price_bars is empty or invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 30)
        >>> vwap = calculate_vwap(bars)
        >>> print(f"VWAP: ${vwap.value:.2f}, Signal: {vwap.signal}")
    """
    logger.info(f"Calculating VWAP for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    # Convert to DataFrame for easier calculation
    df = pd.DataFrame([
        {
            "timestamp": bar.timestamp,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in price_bars
    ])

    # Calculate typical price
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3

    # Calculate VWAP
    df["tpv"] = df["typical_price"] * df["volume"]
    df["cumulative_tpv"] = df["tpv"].cumsum()
    df["cumulative_volume"] = df["volume"].cumsum()

    # Validate cumulative volume is not zero (prevents division by zero)
    if df["cumulative_volume"].iloc[-1] == 0:
        raise ValueError("Total volume is zero - cannot calculate VWAP")

    df["vwap"] = df["cumulative_tpv"] / df["cumulative_volume"]

    current_vwap = float(df["vwap"].iloc[-1])
    current_price = float(df["close"].iloc[-1])

    # Determine signal
    price_to_vwap_ratio = current_price / current_vwap
    if price_to_vwap_ratio > 1.02:  # Price > 2% above VWAP
        signal = "bullish"
    elif price_to_vwap_ratio < 0.98:  # Price > 2% below VWAP
        signal = "bearish"
    else:
        signal = "neutral"

    logger.info(f"VWAP: {current_vwap:.2f}, Signal: {signal}")

    return Indicator(
        name="VWAP",
        value=round(current_vwap, 2),
        signal=signal,
        metadata={
            "current_price": round(current_price, 2),
            "price_to_vwap_ratio": round(price_to_vwap_ratio, 4),
        }
    )


def calculate_ema(
    price_bars: List[PriceBar],
    period: int = 20,
) -> Indicator:
    """Calculate Exponential Moving Average (EMA).

    EMA gives more weight to recent prices, making it more responsive to
    price changes than Simple Moving Average (SMA).

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: EMA period (default: 20)

    Returns:
        Indicator object with EMA value and trading signal

    Raises:
        ValueError: If price_bars is empty or period is invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> ema_20 = calculate_ema(bars, period=20)
        >>> ema_50 = calculate_ema(bars, period=50)
        >>> print(f"EMA(20): ${ema_20.value:.2f}, Signal: {ema_20.signal}")
    """
    logger.info(f"Calculating EMA({period}) for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if period < 1:
        raise ValueError("period must be >= 1")

    if len(price_bars) < period:
        raise ValueError(f"Need at least {period} bars for EMA({period})")

    # Extract closing prices
    closes = [bar.close for bar in price_bars]

    # Calculate EMA using pandas
    ema_series = pd.Series(closes).ewm(span=period, adjust=False).mean()
    current_ema = float(ema_series.iloc[-1])
    current_price = closes[-1]

    # Determine signal based on price relative to EMA
    if current_price > current_ema * 1.01:  # Price > 1% above EMA
        signal = "bullish"
    elif current_price < current_ema * 0.99:  # Price > 1% below EMA
        signal = "bearish"
    else:
        signal = "neutral"

    # Calculate EMA slope (trend direction)
    if len(ema_series) >= 5:
        recent_slope = (ema_series.iloc[-1] - ema_series.iloc[-5]) / ema_series.iloc[-5] * 100
    else:
        recent_slope = 0

    logger.info(f"EMA({period}): {current_ema:.2f}, Signal: {signal}")

    return Indicator(
        name=f"EMA_{period}",
        value=round(current_ema, 2),
        signal=signal,
        metadata={
            "period": period,
            "current_price": round(current_price, 2),
            "slope_5bar_pct": round(recent_slope, 2),
        }
    )


def calculate_ema_series(
    price_bars: List[PriceBar],
    period: int = 20,
) -> List[float]:
    """Calculate EMA series for all bars (for chart overlay).

    Unlike calculate_ema() which returns just the latest value, this function
    returns the full EMA series for use in chart overlays.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: EMA period (default: 20)

    Returns:
        List of EMA values, one for each input bar.
        Early values (before period) will still be calculated using available data.

    Raises:
        ValueError: If price_bars is empty

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 100)
        >>> ema_9 = calculate_ema_series(bars, period=9)
        >>> ema_21 = calculate_ema_series(bars, period=21)
        >>> # Use for chart overlay - each value corresponds to a bar
        >>> print(f"EMA values count: {len(ema_9)} (same as bars)")
    """
    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if period < 1:
        raise ValueError("period must be >= 1")

    # Extract closing prices
    closes = [bar.close for bar in price_bars]

    # Calculate EMA series using pandas
    ema_series = pd.Series(closes).ewm(span=period, adjust=False).mean()

    # Round to 2 decimal places for cleaner output
    return [round(v, 2) for v in ema_series.tolist()]


def calculate_rsi(
    price_bars: List[PriceBar],
    period: int = 14,
) -> Indicator:
    """Calculate Relative Strength Index (RSI).

    RSI measures momentum on a scale of 0-100. Traditional interpretation:
    - RSI > 70: Overbought (potential reversal down)
    - RSI < 30: Oversold (potential reversal up)
    - RSI > 50: Bullish momentum
    - RSI < 50: Bearish momentum

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: RSI period (default: 14)

    Returns:
        Indicator object with RSI value and trading signal

    Raises:
        ValueError: If price_bars is empty or period is invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 30)
        >>> rsi = calculate_rsi(bars, period=14)
        >>> print(f"RSI: {rsi.value:.1f}, Signal: {rsi.signal}")
    """
    logger.info(f"Calculating RSI({period}) for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if period < 2:
        raise ValueError("period must be >= 2")

    if len(price_bars) < period + 1:
        raise ValueError(f"Need at least {period + 1} bars for RSI({period})")

    # Extract closing prices
    closes = np.array([bar.close for bar in price_bars])

    # Calculate price changes
    deltas = np.diff(closes)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate average gains and losses
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate subsequent values using smoothed method
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RSI
    if avg_loss == 0:
        rsi_value = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_value = 100 - (100 / (1 + rs))

    # Determine signal
    if rsi_value > 70:
        signal = "bearish"  # Overbought
        interpretation = "overbought"
    elif rsi_value < 30:
        signal = "bullish"  # Oversold
        interpretation = "oversold"
    elif rsi_value > 50:
        signal = "bullish"  # Bullish momentum
        interpretation = "bullish_momentum"
    else:
        signal = "bearish"  # Bearish momentum
        interpretation = "bearish_momentum"

    logger.info(f"RSI({period}): {rsi_value:.1f}, Signal: {signal}")

    return Indicator(
        name=f"RSI_{period}",
        value=round(rsi_value, 2),
        signal=signal,
        metadata={
            "period": period,
            "interpretation": interpretation,
            "avg_gain": round(avg_gain, 4),
            "avg_loss": round(avg_loss, 4),
        }
    )


def calculate_rsi_series(
    price_bars: List[PriceBar],
    period: int = 14,
) -> List[float]:
    """Calculate RSI series for all bars (for chart overlay).

    Unlike calculate_rsi() which returns just the latest value, this function
    returns the full RSI series for use in chart overlays and visual analysis.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: RSI period (default: 14)

    Returns:
        List of RSI values, one for each input bar (first `period` values may be NaN or 50).

    Raises:
        ValueError: If price_bars is empty or period is invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 100)
        >>> rsi_values = calculate_rsi_series(bars, period=14)
        >>> print(f"Latest RSI: {rsi_values[-1]:.1f}")
    """
    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if period < 2:
        raise ValueError("period must be >= 2")

    if len(price_bars) < period + 1:
        raise ValueError(f"Need at least {period + 1} bars for RSI({period})")

    # Extract closing prices
    closes = np.array([bar.close for bar in price_bars])

    # Calculate price changes
    deltas = np.diff(closes)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Calculate RSI series
    rsi_values = [50.0] * (period)  # Initialize first period with neutral RSI

    # Initial average gain/loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Calculate RSI for initial period
    if avg_loss == 0:
        rsi_values.append(100.0)
    else:
        rs = avg_gain / avg_loss
        rsi_values.append(100 - (100 / (1 + rs)))

    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            rsi_values.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsi_values.append(100 - (100 / (1 + rs)))

    # Round to 2 decimal places
    return [round(v, 2) for v in rsi_values]


def analyze_volume(
    price_bars: List[PriceBar],
    sma_period: int = 20,
) -> Indicator:
    """Analyze volume patterns and trends.

    Volume analysis is critical - "volume precedes price" is a fundamental market principle.
    This tool provides comprehensive volume metrics for confirming price moves.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        sma_period: Period for volume moving average (default: 20)

    Returns:
        Indicator object with volume analysis and trading signal

    Raises:
        ValueError: If price_bars is empty or period is invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> volume = analyze_volume(bars, sma_period=20)
        >>> print(f"Relative Volume: {volume.metadata['relative_volume']:.2f}x")
        >>> print(f"Signal: {volume.signal}")
    """
    logger.info(f"Analyzing volume for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < sma_period:
        raise ValueError(f"Need at least {sma_period} bars for volume analysis")

    # Extract volume data
    volumes = np.array([bar.volume for bar in price_bars])
    closes = np.array([bar.close for bar in price_bars])

    # Calculate volume moving average
    volume_sma = pd.Series(volumes).rolling(window=sma_period).mean()
    current_volume = volumes[-1]
    avg_volume = float(volume_sma.iloc[-1])

    # Relative volume (current vs average)
    relative_volume = current_volume / avg_volume if avg_volume > 0 else 1.0

    # On-Balance Volume (OBV) - cumulative volume based on price direction
    obv = np.zeros(len(closes))
    obv[0] = volumes[0]
    for i in range(1, len(closes)):
        if closes[i] > closes[i - 1]:
            obv[i] = obv[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv[i] = obv[i - 1] - volumes[i]
        else:
            obv[i] = obv[i - 1]

    # OBV trend (last 5 vs last 10 bars)
    if len(obv) >= 10:
        obv_recent = np.mean(obv[-5:])
        obv_older = np.mean(obv[-10:-5])
        obv_trend = "rising" if obv_recent > obv_older else "falling"
    else:
        obv_trend = "neutral"

    # Volume spike detection (>2x average)
    volume_spike = relative_volume > 2.0

    # Recent price change for volume confirmation
    if len(closes) >= 2:
        price_change = (closes[-1] - closes[-2]) / closes[-2]
    else:
        price_change = 0

    # Determine signal
    if relative_volume > 1.5 and price_change > 0 and obv_trend == "rising":
        signal = "bullish"  # High volume + price up + accumulation
        interpretation = "accumulation"
    elif relative_volume > 1.5 and price_change < 0 and obv_trend == "falling":
        signal = "bearish"  # High volume + price down + distribution
        interpretation = "distribution"
    elif relative_volume < 0.5:
        signal = "neutral"  # Low volume = unreliable move
        interpretation = "low_volume"
    else:
        signal = "neutral"
        interpretation = "mixed"

    logger.info(
        f"Volume: {current_volume:,.0f} (Relative: {relative_volume:.2f}x), "
        f"OBV Trend: {obv_trend}, Signal: {signal}"
    )

    return Indicator(
        name="Volume",
        value=round(float(current_volume), 0),
        signal=signal,
        metadata={
            "avg_volume": round(avg_volume, 0),
            "relative_volume": round(relative_volume, 2),
            "volume_spike": volume_spike,
            "obv": round(float(obv[-1]), 0),
            "obv_trend": obv_trend,
            "interpretation": interpretation,
            "sma_period": sma_period,
        }
    )


def calculate_macd(
    price_bars: List[PriceBar],
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9,
) -> Indicator:
    """Calculate MACD (Moving Average Convergence Divergence).

    MACD is one of the most popular indicators used by 90% of traders. It shows
    momentum shifts and trend changes through crossovers.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        fast_period: Fast EMA period (default: 12)
        slow_period: Slow EMA period (default: 26)
        signal_period: Signal line period (default: 9)

    Returns:
        Indicator object with MACD values and trading signal

    Raises:
        ValueError: If price_bars is empty or periods are invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> macd = calculate_macd(bars)
        >>> print(f"MACD: {macd.value:.2f}, Signal: {macd.signal}")
        >>> print(f"Histogram: {macd.metadata['histogram']:.2f}")
    """
    logger.info(
        f"Calculating MACD({fast_period}/{slow_period}/{signal_period}) "
        f"for {len(price_bars)} bars"
    )

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < slow_period + signal_period:
        raise ValueError(f"Need at least {slow_period + signal_period} bars for MACD")

    # Extract closing prices
    closes = pd.Series([bar.close for bar in price_bars])

    # Calculate EMAs
    ema_fast = closes.ewm(span=fast_period, adjust=False).mean()
    ema_slow = closes.ewm(span=slow_period, adjust=False).mean()

    # MACD line = fast EMA - slow EMA
    macd_line = ema_fast - ema_slow

    # Signal line = 9 EMA of MACD line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Histogram = MACD line - Signal line
    histogram = macd_line - signal_line

    # Current values
    current_macd = float(macd_line.iloc[-1])
    current_signal = float(signal_line.iloc[-1])
    current_histogram = float(histogram.iloc[-1])

    # Previous values for crossover detection
    if len(macd_line) >= 2:
        prev_macd = float(macd_line.iloc[-2])
        prev_signal = float(signal_line.iloc[-2])
        prev_histogram = float(histogram.iloc[-2])
    else:
        prev_macd = current_macd
        prev_signal = current_signal
        prev_histogram = current_histogram

    # Detect crossovers
    bullish_crossover = prev_macd <= prev_signal and current_macd > current_signal
    bearish_crossover = prev_macd >= prev_signal and current_macd < current_signal

    # Determine signal
    if bullish_crossover:
        signal = "bullish"
        interpretation = "bullish_crossover"
    elif bearish_crossover:
        signal = "bearish"
        interpretation = "bearish_crossover"
    elif current_macd > current_signal and current_histogram > 0:
        signal = "bullish"
        interpretation = "bullish_momentum"
    elif current_macd < current_signal and current_histogram < 0:
        signal = "bearish"
        interpretation = "bearish_momentum"
    else:
        signal = "neutral"
        interpretation = "neutral"

    # Histogram trend (strengthening or weakening)
    if abs(current_histogram) > abs(prev_histogram):
        histogram_trend = "strengthening"
    elif abs(current_histogram) < abs(prev_histogram):
        histogram_trend = "weakening"
    else:
        histogram_trend = "stable"

    logger.info(
        f"MACD: {current_macd:.2f}, Signal Line: {current_signal:.2f}, "
        f"Histogram: {current_histogram:.2f}, Signal: {signal}"
    )

    return Indicator(
        name="MACD",
        value=round(current_macd, 2),
        signal=signal,
        metadata={
            "macd_line": round(current_macd, 2),
            "signal_line": round(current_signal, 2),
            "histogram": round(current_histogram, 2),
            "bullish_crossover": bullish_crossover,
            "bearish_crossover": bearish_crossover,
            "histogram_trend": histogram_trend,
            "interpretation": interpretation,
            "fast_period": fast_period,
            "slow_period": slow_period,
            "signal_period": signal_period,
        }
    )


def calculate_atr(
    price_bars: List[PriceBar],
    period: int = 14,
) -> Indicator:
    """Calculate Average True Range (ATR).

    ATR measures volatility and is essential for setting volatility-based stop losses
    and position sizing. Professional traders use ATR instead of fixed percentage stops.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: ATR period (default: 14)

    Returns:
        Indicator object with ATR value and volatility assessment

    Raises:
        ValueError: If price_bars is empty or period is invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 30)
        >>> atr = calculate_atr(bars)
        >>> print(f"ATR: ${atr.value:.2f}")
        >>> print(f"Suggested stop distance: ${atr.metadata['stop_distance_2x']:.2f}")
    """
    logger.info(f"Calculating ATR({period}) for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < period + 1:
        raise ValueError(f"Need at least {period + 1} bars for ATR({period})")

    # Calculate True Range for each bar
    true_ranges = []
    for i in range(1, len(price_bars)):
        bar = price_bars[i]
        prev_bar = price_bars[i - 1]

        # True Range = max of:
        # 1. High - Low
        # 2. abs(High - Previous Close)
        # 3. abs(Low - Previous Close)
        tr = max(
            bar.high - bar.low,
            abs(bar.high - prev_bar.close),
            abs(bar.low - prev_bar.close),
        )
        true_ranges.append(tr)

    # Calculate ATR using Wilder's smoothing (industry standard)
    # This is more responsive to recent volatility than simple moving average
    atr_values = np.zeros(len(true_ranges))

    # First ATR is simple mean of first 'period' true ranges
    if len(true_ranges) >= period:
        atr_values[period - 1] = np.mean(true_ranges[:period])

        # Subsequent ATRs use Wilder's smoothing
        for i in range(period, len(true_ranges)):
            atr_values[i] = (atr_values[i - 1] * (period - 1) + true_ranges[i]) / period

    atr_series = pd.Series(atr_values)
    current_atr = float(atr_series.iloc[-1]) if atr_series.iloc[-1] > 0 else float(np.mean(true_ranges[-period:]))

    # ATR as percentage of price (for comparison across stocks)
    current_price = price_bars[-1].close
    atr_percentage = (current_atr / current_price) * 100

    # Calculate ATR percentile (current vs historical range)
    if len(atr_series) >= 50:
        atr_percentile = (
            (atr_series.iloc[-1] > atr_series.iloc[-50:]).sum() / 50 * 100
        )
    else:
        atr_percentile = 50.0

    # Volatility assessment
    # Note: ATR is non-directional - high volatility can occur in both uptrends and downtrends.
    # Signal is neutral as ATR alone doesn't indicate direction, only magnitude of moves.
    if atr_percentage > 3.0:
        volatility = "high"
        signal = "neutral"  # High volatility = larger moves (direction agnostic)
    elif atr_percentage > 2.0:
        volatility = "moderate"
        signal = "neutral"
    else:
        volatility = "low"
        signal = "neutral"  # Low volatility = smaller moves

    # Suggested stop loss distances (common ATR multiples)
    stop_1x = current_atr
    stop_2x = current_atr * 2
    stop_3x = current_atr * 3

    logger.info(
        f"ATR({period}): ${current_atr:.2f} ({atr_percentage:.2f}%), "
        f"Volatility: {volatility}"
    )

    return Indicator(
        name=f"ATR_{period}",
        value=round(current_atr, 2),
        signal=signal,
        metadata={
            "period": period,
            "atr_percentage": round(atr_percentage, 2),
            "volatility": volatility,
            "atr_percentile": round(atr_percentile, 1),
            "stop_distance_1x": round(stop_1x, 2),
            "stop_distance_2x": round(stop_2x, 2),
            "stop_distance_3x": round(stop_3x, 2),
            "current_price": round(current_price, 2),
        }
    )


def calculate_bollinger_bands(
    price_bars: List[PriceBar],
    period: int = 20,
    std_dev: float = 2.0,
) -> Indicator:
    """Calculate Bollinger Bands.

    Bollinger Bands show volatility and overbought/oversold conditions. Band squeezes
    signal potential breakouts, and price touching bands indicates extremes.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: Moving average period (default: 20)
        std_dev: Number of standard deviations (default: 2.0)

    Returns:
        Indicator object with Bollinger Bands values and trading signal

    Raises:
        ValueError: If price_bars is empty or parameters are invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> bb = calculate_bollinger_bands(bars)
        >>> print(f"Upper: ${bb.metadata['upper_band']:.2f}")
        >>> print(f"Middle: ${bb.metadata['middle_band']:.2f}")
        >>> print(f"Lower: ${bb.metadata['lower_band']:.2f}")
        >>> print(f"%B Position: {bb.metadata['percent_b']:.2f}")
    """
    logger.info(f"Calculating Bollinger Bands({period}, {std_dev}) for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < period:
        raise ValueError(f"Need at least {period} bars for Bollinger Bands")

    # Extract closing prices
    closes = pd.Series([bar.close for bar in price_bars])

    # Calculate SMA (middle band)
    sma = closes.rolling(window=period).mean()
    middle_band = float(sma.iloc[-1])

    # Calculate standard deviation
    std = closes.rolling(window=period).std()
    current_std = float(std.iloc[-1])

    # Calculate bands
    upper_band = middle_band + (std_dev * current_std)
    lower_band = middle_band - (std_dev * current_std)

    current_price = float(closes.iloc[-1])

    # Calculate %B (where price is within bands)
    # %B = 0 means at lower band, 1 means at upper band
    if upper_band != lower_band:
        percent_b = (current_price - lower_band) / (upper_band - lower_band)
    else:
        percent_b = 0.5

    # Calculate band width (volatility measure)
    band_width = ((upper_band - lower_band) / middle_band) * 100

    # Historical band width for squeeze detection
    if len(sma) >= 50:
        historical_widths = []
        for i in range(len(sma) - 50, len(sma)):
            if not pd.isna(sma.iloc[i]) and not pd.isna(std.iloc[i]):
                hist_upper = sma.iloc[i] + (std_dev * std.iloc[i])
                hist_lower = sma.iloc[i] - (std_dev * std.iloc[i])
                hist_width = ((hist_upper - hist_lower) / sma.iloc[i]) * 100
                historical_widths.append(hist_width)

        if historical_widths:
            avg_width = np.mean(historical_widths)
            # Squeeze: current width < 75% of average
            squeeze = band_width < (avg_width * 0.75)
        else:
            squeeze = False
    else:
        squeeze = False

    # Determine signal
    if percent_b > 1.0:
        signal = "bearish"  # Above upper band (overbought)
        interpretation = "overbought"
    elif percent_b < 0.0:
        signal = "bullish"  # Below lower band (oversold)
        interpretation = "oversold"
    elif percent_b > 0.8:
        signal = "bearish"  # Near upper band
        interpretation = "approaching_overbought"
    elif percent_b < 0.2:
        signal = "bullish"  # Near lower band
        interpretation = "approaching_oversold"
    elif squeeze:
        signal = "neutral"
        interpretation = "squeeze"  # Low volatility, potential breakout
    else:
        signal = "neutral"
        interpretation = "normal"

    logger.info(
        f"Bollinger Bands: Upper={upper_band:.2f}, Middle={middle_band:.2f}, "
        f"Lower={lower_band:.2f}, %B={percent_b:.2f}, Signal: {signal}"
    )

    return Indicator(
        name="BollingerBands",
        value=round(middle_band, 2),
        signal=signal,
        metadata={
            "upper_band": round(upper_band, 2),
            "middle_band": round(middle_band, 2),
            "lower_band": round(lower_band, 2),
            "percent_b": round(percent_b, 2),
            "band_width": round(band_width, 2),
            "squeeze": squeeze,
            "interpretation": interpretation,
            "period": period,
            "std_dev": std_dev,
            "current_price": round(current_price, 2),
        }
    )


def detect_divergences(
    price_bars: List[PriceBar],
    indicator_type: str = "rsi",
    lookback_period: int = 14,
    swing_detection_window: int = 5,
) -> Indicator:
    """Detect bullish and bearish divergences between price and indicators.

    Divergence is a powerful leading indicator that signals potential reversals:
    - Regular Bullish: Price makes lower lows, but indicator makes higher lows (reversal up)
    - Regular Bearish: Price makes higher highs, but indicator makes lower highs (reversal down)
    - Hidden Bullish: Price makes higher lows, but indicator makes lower lows (continuation up)
    - Hidden Bearish: Price makes lower highs, but indicator makes higher highs (continuation down)

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        indicator_type: "rsi" or "macd" (default: "rsi")
        lookback_period: Period for indicator calculation (default: 14)
        swing_detection_window: Window for finding swing highs/lows (default: 5)

    Returns:
        Indicator object with divergence detection results

    Raises:
        ValueError: If price_bars is empty or parameters are invalid

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 100)
        >>> divergence = detect_divergences(bars, indicator_type="rsi")
        >>> if divergence.metadata.get("regular_bullish"):
        >>>     print("Bullish divergence detected - potential reversal up!")
        >>> print(f"Divergences found: {divergence.metadata['divergence_types']}")
    """
    logger.info(
        f"Detecting {indicator_type.upper()} divergences for {len(price_bars)} bars "
        f"(lookback={lookback_period}, window={swing_detection_window})"
    )

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < lookback_period + swing_detection_window * 2:
        raise ValueError(
            f"Need at least {lookback_period + swing_detection_window * 2} bars "
            f"for divergence detection"
        )

    if indicator_type not in ["rsi", "macd"]:
        raise ValueError("indicator_type must be 'rsi' or 'macd'")

    # Calculate indicator values
    if indicator_type == "rsi":
        indicator = calculate_rsi(price_bars, period=lookback_period)
        # Get full RSI series for swing detection
        closes = np.array([bar.close for bar in price_bars])
        deltas = np.diff(closes)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.mean(gains[:lookback_period])
        avg_loss = np.mean(losses[:lookback_period])

        rsi_values = []
        for i in range(lookback_period, len(gains)):
            avg_gain = (avg_gain * (lookback_period - 1) + gains[i]) / lookback_period
            avg_loss = (avg_loss * (lookback_period - 1) + losses[i]) / lookback_period

            if avg_loss == 0:
                rsi_val = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_val = 100 - (100 / (1 + rs))
            rsi_values.append(rsi_val)

        # Pad with NaN for alignment
        indicator_values = [np.nan] * (lookback_period + 1) + rsi_values
    else:  # macd
        macd_indicator = calculate_macd(price_bars)
        closes_series = pd.Series([bar.close for bar in price_bars])
        ema_fast = closes_series.ewm(span=12, adjust=False).mean()
        ema_slow = closes_series.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        indicator_values = macd_line.tolist()

    # Extract price data
    closes = np.array([bar.close for bar in price_bars])

    # Find swing highs and lows
    def find_swing_points(data, window):
        """Find local maxima (swing highs) and minima (swing lows)."""
        swing_highs = []
        swing_lows = []

        for i in range(window, len(data) - window):
            # Check if current point is a swing high
            is_high = all(data[i] >= data[i - j] for j in range(1, window + 1))
            is_high = is_high and all(data[i] >= data[i + j] for j in range(1, window + 1))

            # Check if current point is a swing low
            is_low = all(data[i] <= data[i - j] for j in range(1, window + 1))
            is_low = is_low and all(data[i] <= data[i + j] for j in range(1, window + 1))

            if is_high:
                swing_highs.append((i, data[i]))
            if is_low:
                swing_lows.append((i, data[i]))

        return swing_highs, swing_lows

    # Find swing points for price and indicator
    price_highs, price_lows = find_swing_points(closes, swing_detection_window)

    # Filter out NaN values from indicator
    valid_indicator = np.array([
        val if not (isinstance(val, float) and np.isnan(val)) else None
        for val in indicator_values
    ])

    # Forward-fill invalid indicator values to avoid false swing points
    # Using 0 as placeholder creates artificial swings that trigger false divergences
    indicator_data = []
    first_valid = next((v for v in valid_indicator if v is not None), 50)  # Default to 50 (neutral RSI)
    last_valid = first_valid
    for val in valid_indicator:
        if val is not None:
            indicator_data.append(val)
            last_valid = val
        else:
            indicator_data.append(last_valid)  # Forward-fill with last valid value

    indicator_highs, indicator_lows = find_swing_points(
        np.array(indicator_data),
        swing_detection_window
    )

    # Detect divergences
    divergences = {
        "regular_bullish": False,
        "regular_bearish": False,
        "hidden_bullish": False,
        "hidden_bearish": False,
    }

    divergence_details = []

    # Check for regular bullish divergence (price: lower lows, indicator: higher lows)
    if len(price_lows) >= 2 and len(indicator_lows) >= 2:
        # Get last two swing lows
        p_low_1_idx, p_low_1_val = price_lows[-2]
        p_low_2_idx, p_low_2_val = price_lows[-1]

        # Find corresponding indicator lows (within +/- 3 bars)
        i_low_1 = None
        i_low_2 = None

        for idx, val in indicator_lows:
            if abs(idx - p_low_1_idx) <= 3 and i_low_1 is None:
                i_low_1 = (idx, val)
            if abs(idx - p_low_2_idx) <= 3:
                i_low_2 = (idx, val)

        if i_low_1 and i_low_2:
            # Regular bullish: price lower low, indicator higher low
            if p_low_2_val < p_low_1_val and i_low_2[1] > i_low_1[1]:
                divergences["regular_bullish"] = True
                divergence_details.append(
                    f"Regular Bullish: Price {p_low_1_val:.2f}->{p_low_2_val:.2f}, "
                    f"{indicator_type.upper()} {i_low_1[1]:.2f}->{i_low_2[1]:.2f}"
                )

            # Hidden bullish: price higher low, indicator lower low
            if p_low_2_val > p_low_1_val and i_low_2[1] < i_low_1[1]:
                divergences["hidden_bullish"] = True
                divergence_details.append(
                    f"Hidden Bullish: Price {p_low_1_val:.2f}->{p_low_2_val:.2f}, "
                    f"{indicator_type.upper()} {i_low_1[1]:.2f}->{i_low_2[1]:.2f}"
                )

    # Check for regular bearish divergence (price: higher highs, indicator: lower highs)
    if len(price_highs) >= 2 and len(indicator_highs) >= 2:
        # Get last two swing highs
        p_high_1_idx, p_high_1_val = price_highs[-2]
        p_high_2_idx, p_high_2_val = price_highs[-1]

        # Find corresponding indicator highs (within +/- 3 bars)
        i_high_1 = None
        i_high_2 = None

        for idx, val in indicator_highs:
            if abs(idx - p_high_1_idx) <= 3 and i_high_1 is None:
                i_high_1 = (idx, val)
            if abs(idx - p_high_2_idx) <= 3:
                i_high_2 = (idx, val)

        if i_high_1 and i_high_2:
            # Regular bearish: price higher high, indicator lower high
            if p_high_2_val > p_high_1_val and i_high_2[1] < i_high_1[1]:
                divergences["regular_bearish"] = True
                divergence_details.append(
                    f"Regular Bearish: Price {p_high_1_val:.2f}->{p_high_2_val:.2f}, "
                    f"{indicator_type.upper()} {i_high_1[1]:.2f}->{i_high_2[1]:.2f}"
                )

            # Hidden bearish: price lower high, indicator higher high
            if p_high_2_val < p_high_1_val and i_high_2[1] > i_high_1[1]:
                divergences["hidden_bearish"] = True
                divergence_details.append(
                    f"Hidden Bearish: Price {p_high_1_val:.2f}->{p_high_2_val:.2f}, "
                    f"{indicator_type.upper()} {i_high_1[1]:.2f}->{i_high_2[1]:.2f}"
                )

    # Determine overall signal
    divergence_types = [k for k, v in divergences.items() if v]

    if divergences["regular_bullish"]:
        signal = "bullish"
        interpretation = "regular_bullish_divergence"
    elif divergences["regular_bearish"]:
        signal = "bearish"
        interpretation = "regular_bearish_divergence"
    elif divergences["hidden_bullish"]:
        signal = "bullish"
        interpretation = "hidden_bullish_divergence"
    elif divergences["hidden_bearish"]:
        signal = "bearish"
        interpretation = "hidden_bearish_divergence"
    else:
        signal = "neutral"
        interpretation = "no_divergence"

    # Calculate divergence strength
    divergence_count = sum(divergences.values())
    if divergence_count >= 2:
        strength = "strong"
    elif divergence_count == 1:
        strength = "moderate"
    else:
        strength = "none"

    logger.info(
        f"Divergence Detection ({indicator_type.upper()}): "
        f"Found {divergence_count} divergence(s) - {', '.join(divergence_types) if divergence_types else 'none'}"
    )

    return Indicator(
        name=f"Divergence_{indicator_type.upper()}",
        value=divergence_count,
        signal=signal,
        metadata={
            "indicator_type": indicator_type,
            "regular_bullish": divergences["regular_bullish"],
            "regular_bearish": divergences["regular_bearish"],
            "hidden_bullish": divergences["hidden_bullish"],
            "hidden_bearish": divergences["hidden_bearish"],
            "divergence_types": divergence_types,
            "divergence_count": divergence_count,
            "strength": strength,
            "interpretation": interpretation,
            "details": divergence_details,
            "lookback_period": lookback_period,
            "swing_detection_window": swing_detection_window,
            "price_swing_highs_count": len(price_highs),
            "price_swing_lows_count": len(price_lows),
        }
    )


def calculate_adx(
    price_bars: List[PriceBar],
    period: int = 14,
) -> Indicator:
    """Calculate Average Directional Index (ADX) for trend strength.

    ADX measures trend strength regardless of direction:
    - ADX < 20: Weak trend or ranging market (avoid trend-following strategies)
    - ADX 20-25: Trend emerging
    - ADX 25-50: Strong trend (ideal for trend-following)
    - ADX > 50: Very strong trend (may be extended)

    Also calculates +DI and -DI for directional bias.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        period: ADX period (default: 14)

    Returns:
        Indicator object with ADX value and trend strength assessment

    Raises:
        ValueError: If price_bars is empty or insufficient

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> adx = calculate_adx(bars)
        >>> if adx.value > 25 and adx.metadata['plus_di'] > adx.metadata['minus_di']:
        >>>     print("Strong uptrend - trend-following strategies work well")
    """
    logger.info(f"Calculating ADX({period}) for {len(price_bars)} bars")

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    if len(price_bars) < period * 2:
        raise ValueError(f"Need at least {period * 2} bars for ADX({period})")

    highs = np.array([bar.high for bar in price_bars])
    lows = np.array([bar.low for bar in price_bars])
    closes = np.array([bar.close for bar in price_bars])

    # Calculate +DM and -DM
    plus_dm = np.zeros(len(highs))
    minus_dm = np.zeros(len(highs))

    for i in range(1, len(highs)):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Calculate True Range
    tr = np.zeros(len(highs))
    for i in range(1, len(highs)):
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )

    # Smooth with Wilder's smoothing (similar to EMA)
    def wilders_smooth(data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's smoothing method."""
        smoothed = np.zeros(len(data))
        smoothed[period] = np.sum(data[1 : period + 1])
        for i in range(period + 1, len(data)):
            smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + data[i]
        return smoothed

    smoothed_tr = wilders_smooth(tr, period)
    smoothed_plus_dm = wilders_smooth(plus_dm, period)
    smoothed_minus_dm = wilders_smooth(minus_dm, period)

    # Calculate +DI and -DI
    plus_di = np.zeros(len(highs))
    minus_di = np.zeros(len(highs))

    for i in range(period, len(highs)):
        if smoothed_tr[i] > 0:
            plus_di[i] = 100 * smoothed_plus_dm[i] / smoothed_tr[i]
            minus_di[i] = 100 * smoothed_minus_dm[i] / smoothed_tr[i]

    # Calculate DX
    dx = np.zeros(len(highs))
    for i in range(period, len(highs)):
        di_sum = plus_di[i] + minus_di[i]
        if di_sum > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / di_sum

    # Calculate ADX (smoothed DX)
    adx = wilders_smooth(dx, period)

    current_adx = float(adx[-1])
    current_plus_di = float(plus_di[-1])
    current_minus_di = float(minus_di[-1])

    # Determine trend strength
    if current_adx < 20:
        trend_strength = "weak"
        signal = "neutral"
    elif current_adx < 25:
        trend_strength = "emerging"
        signal = "bullish" if current_plus_di > current_minus_di else "bearish"
    elif current_adx < 50:
        trend_strength = "strong"
        signal = "bullish" if current_plus_di > current_minus_di else "bearish"
    else:
        trend_strength = "very_strong"
        signal = "neutral"  # May be extended, be cautious

    # Determine trend direction
    if current_plus_di > current_minus_di:
        trend_direction = "bullish"
    elif current_minus_di > current_plus_di:
        trend_direction = "bearish"
    else:
        trend_direction = "neutral"

    logger.info(
        f"ADX({period}): {current_adx:.1f} ({trend_strength}), "
        f"+DI: {current_plus_di:.1f}, -DI: {current_minus_di:.1f}"
    )

    return Indicator(
        name=f"ADX_{period}",
        value=round(current_adx, 2),
        signal=signal,
        metadata={
            "period": period,
            "plus_di": round(current_plus_di, 2),
            "minus_di": round(current_minus_di, 2),
            "trend_strength": trend_strength,
            "trend_direction": trend_direction,
            "is_trending": current_adx >= 25,
        },
    )


def calculate_stochastic(
    price_bars: List[PriceBar],
    k_period: int = 14,
    d_period: int = 3,
    smooth_k: int = 3,
) -> Indicator:
    """Calculate Stochastic Oscillator (%K and %D).

    Stochastic measures where the close is relative to the high-low range:
    - Overbought: %K > 80 (potential reversal down)
    - Oversold: %K < 20 (potential reversal up)
    - Bullish crossover: %K crosses above %D in oversold zone
    - Bearish crossover: %K crosses below %D in overbought zone

    Best used in ranging markets or for timing entries in trending markets.

    Args:
        price_bars: List of PriceBar objects (OHLCV data)
        k_period: Lookback period for %K (default: 14)
        d_period: Smoothing period for %D (default: 3)
        smooth_k: Smoothing for slow stochastic (default: 3)

    Returns:
        Indicator object with stochastic values and trading signal

    Raises:
        ValueError: If price_bars is empty or insufficient

    Example:
        >>> bars = fetch_price_bars("AAPL", "1d", 50)
        >>> stoch = calculate_stochastic(bars)
        >>> if stoch.metadata['bullish_crossover'] and stoch.value < 30:
        >>>     print("Bullish reversal signal - oversold with crossover")
    """
    logger.info(
        f"Calculating Stochastic({k_period}, {d_period}, {smooth_k}) "
        f"for {len(price_bars)} bars"
    )

    if not price_bars:
        raise ValueError("price_bars cannot be empty")

    min_bars = k_period + d_period + smooth_k
    if len(price_bars) < min_bars:
        raise ValueError(f"Need at least {min_bars} bars for Stochastic")

    highs = np.array([bar.high for bar in price_bars])
    lows = np.array([bar.low for bar in price_bars])
    closes = np.array([bar.close for bar in price_bars])

    # Calculate raw %K
    raw_k = np.zeros(len(closes))
    for i in range(k_period - 1, len(closes)):
        period_high = np.max(highs[i - k_period + 1 : i + 1])
        period_low = np.min(lows[i - k_period + 1 : i + 1])

        if period_high != period_low:
            raw_k[i] = 100 * (closes[i] - period_low) / (period_high - period_low)
        else:
            raw_k[i] = 50  # Neutral if no range

    # Smooth %K for "slow stochastic"
    slow_k = pd.Series(raw_k).rolling(window=smooth_k).mean().values

    # Calculate %D (signal line)
    slow_d = pd.Series(slow_k).rolling(window=d_period).mean().values

    current_k = float(slow_k[-1]) if not np.isnan(slow_k[-1]) else 50
    current_d = float(slow_d[-1]) if not np.isnan(slow_d[-1]) else 50
    prev_k = (
        float(slow_k[-2])
        if len(slow_k) > 1 and not np.isnan(slow_k[-2])
        else current_k
    )
    prev_d = (
        float(slow_d[-2])
        if len(slow_d) > 1 and not np.isnan(slow_d[-2])
        else current_d
    )

    # Detect crossovers
    bullish_crossover = prev_k <= prev_d and current_k > current_d
    bearish_crossover = prev_k >= prev_d and current_k < current_d

    # Determine signal
    if current_k < 20:
        if bullish_crossover:
            signal = "bullish"
            interpretation = "oversold_crossover"
        else:
            signal = "bullish"  # Oversold is generally bullish
            interpretation = "oversold"
    elif current_k > 80:
        if bearish_crossover:
            signal = "bearish"
            interpretation = "overbought_crossover"
        else:
            signal = "bearish"  # Overbought is generally bearish
            interpretation = "overbought"
    elif bullish_crossover:
        signal = "bullish"
        interpretation = "bullish_crossover"
    elif bearish_crossover:
        signal = "bearish"
        interpretation = "bearish_crossover"
    else:
        signal = "neutral"
        interpretation = "neutral"

    logger.info(
        f"Stochastic: %K={current_k:.1f}, %D={current_d:.1f}, "
        f"Signal: {signal}, "
        f"Crossover: {'bullish' if bullish_crossover else 'bearish' if bearish_crossover else 'none'}"
    )

    return Indicator(
        name="Stochastic",
        value=round(current_k, 2),
        signal=signal,
        metadata={
            "k_period": k_period,
            "d_period": d_period,
            "smooth_k": smooth_k,
            "percent_k": round(current_k, 2),
            "percent_d": round(current_d, 2),
            "bullish_crossover": bullish_crossover,
            "bearish_crossover": bearish_crossover,
            "interpretation": interpretation,
            "is_oversold": current_k < 20,
            "is_overbought": current_k > 80,
        },
    )
