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

    # Calculate ATR using smoothed moving average
    tr_series = pd.Series(true_ranges)
    atr_series = tr_series.rolling(window=period).mean()
    current_atr = float(atr_series.iloc[-1])

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
    if atr_percentage > 3.0:
        volatility = "high"
        signal = "bearish"  # High volatility = risky
    elif atr_percentage > 2.0:
        volatility = "moderate"
        signal = "neutral"
    else:
        volatility = "low"
        signal = "bullish"  # Low volatility = stable

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
