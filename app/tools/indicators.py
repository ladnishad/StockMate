"""Technical indicator calculation tools.

All functions are designed to be used as LLM agent tools with clear
signatures, comprehensive docstrings, and structured input/output.
"""

from typing import List
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
