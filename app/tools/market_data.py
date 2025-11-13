"""Market data fetching tools using Alpaca API.

All functions are designed to be used as LLM agent tools with clear
signatures, comprehensive docstrings, and structured input/output.
"""

from typing import List, Literal, Optional
from datetime import datetime, timedelta
import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

from app.config import get_settings
from app.models.data import PriceBar, Fundamentals, Sentiment

logger = logging.getLogger(__name__)

# Alpaca Data Feed Options
# - "iex": Free tier, 5 years of history (Investors Exchange)
# - "sip": Paid tier, 7 years of history (Securities Information Processor)
# - None: Defaults to best available feed based on subscription
ALPACA_DATA_FEED = None  # Will use best available feed for user's subscription


def _get_alpaca_client() -> StockHistoricalDataClient:
    """Get initialized Alpaca historical data client."""
    settings = get_settings()
    return StockHistoricalDataClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )


def _get_trading_client() -> TradingClient:
    """Get initialized Alpaca trading client."""
    settings = get_settings()
    return TradingClient(
        api_key=settings.alpaca_api_key,
        secret_key=settings.alpaca_secret_key,
    )


def fetch_price_bars(
    symbol: str,
    timeframe: Literal["1d", "1h", "15m", "5m"] = "1d",
    days_back: int = 100,
    feed: Optional[str] = None,
) -> List[PriceBar]:
    """Fetch historical price bars (OHLCV data) for a stock symbol.

    This tool retrieves historical price data from Alpaca Markets API using
    the official alpaca-py SDK. Works with both free (IEX) and paid (SIP) tiers.

    Data Feed Information:
    - Free tier: Uses IEX feed (5 years of history, Investors Exchange)
    - Paid tier: Can use SIP feed (7 years of history, all US exchanges)
    - Default: Automatically uses best available feed for your subscription

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        timeframe: Bar timeframe - '1d' (daily), '1h' (hourly), '15m' (15-minute), '5m' (5-minute)
        days_back: Number of days of historical data to fetch
        feed: Data feed to use ('iex', 'sip', or None for auto). Default: None

    Returns:
        List of PriceBar objects containing OHLCV data, sorted by timestamp ascending

    Raises:
        ValueError: If symbol is invalid or no data is available
        APIError: If Alpaca API returns an error (e.g., invalid credentials, rate limit)
        Exception: If API request fails

    Example:
        >>> bars = fetch_price_bars("AAPL", timeframe="1d", days_back=30)
        >>> print(f"Fetched {len(bars)} daily bars for AAPL")
        >>> print(f"Latest close: ${bars[-1].close:.2f}")
    """
    logger.info(f"Fetching {timeframe} bars for {symbol} (last {days_back} days)")

    # Map timeframe strings to Alpaca TimeFrame objects
    timeframe_map = {
        "1d": TimeFrame.Day,
        "1h": TimeFrame.Hour,
        "15m": TimeFrame.Minute,  # Will use 15 multiplier
        "5m": TimeFrame.Minute,   # Will use 5 multiplier
    }

    # Calculate start and end dates
    end = datetime.utcnow()
    start = end - timedelta(days=days_back)

    # Adjust timeframe for minutes
    if timeframe == "15m":
        tf = TimeFrame(15, TimeFrame.Minute)
    elif timeframe == "5m":
        tf = TimeFrame(5, TimeFrame.Minute)
    else:
        tf = timeframe_map[timeframe]

    try:
        client = _get_alpaca_client()

        # Use provided feed or default
        data_feed = feed or ALPACA_DATA_FEED

        # Build request parameters with feed specification
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=data_feed,  # Specify data feed (iex, sip, or None for auto)
        )

        bars_data = client.get_stock_bars(request_params)

        if symbol not in bars_data:
            raise ValueError(f"No data available for symbol: {symbol}")

        bars = bars_data[symbol]

        # Convert to PriceBar objects
        price_bars = [
            PriceBar(
                timestamp=bar.timestamp,
                open=float(bar.open),
                high=float(bar.high),
                low=float(bar.low),
                close=float(bar.close),
                volume=int(bar.volume),
            )
            for bar in bars
        ]

        feed_info = data_feed or "auto"
        logger.info(f"Successfully fetched {len(price_bars)} bars for {symbol} (feed: {feed_info})")
        return price_bars

    except APIError as e:
        # Alpaca API specific errors (auth, rate limits, subscription issues)
        logger.error(f"Alpaca API error fetching price bars for {symbol}: {str(e)}")
        raise ValueError(f"Alpaca API error: {str(e)}")
    except ValueError:
        # Re-raise ValueError as-is
        raise
    except Exception as e:
        logger.error(f"Error fetching price bars for {symbol}: {str(e)}")
        raise


def fetch_fundamentals(symbol: str) -> Fundamentals:
    """Fetch fundamental data for a stock symbol.

    This tool retrieves fundamental financial metrics. Since Alpaca's free tier
    has limited fundamental data, this function provides a best-effort approach
    using available data and sensible defaults.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Fundamentals object containing financial metrics

    Raises:
        ValueError: If symbol is invalid
        Exception: If API request fails

    Example:
        >>> fundamentals = fetch_fundamentals("AAPL")
        >>> print(f"Market Cap: ${fundamentals.market_cap:,.0f}")
        >>> print(f"P/E Ratio: {fundamentals.pe_ratio}")
    """
    logger.info(f"Fetching fundamentals for {symbol}")

    try:
        # For this implementation, we'll use Alpaca's asset info
        # and derive some metrics from recent price action
        client = _get_alpaca_client()
        trading_client = _get_trading_client()

        # Fetch recent daily bars to calculate 52-week high/low
        price_bars = fetch_price_bars(symbol, timeframe="1d", days_back=365)

        if not price_bars:
            raise ValueError(f"No price data available for {symbol}")

        # Calculate 52-week high and low
        highs = [bar.high for bar in price_bars]
        lows = [bar.low for bar in price_bars]

        fifty_two_week_high = max(highs) if highs else None
        fifty_two_week_low = min(lows) if lows else None

        # Note: Full fundamental data would require a dedicated financial data API
        # For production, integrate with services like Financial Modeling Prep,
        # Alpha Vantage, or IEX Cloud
        fundamentals = Fundamentals(
            market_cap=None,  # Would need external API
            pe_ratio=None,    # Would need external API
            eps=None,         # Would need external API
            dividend_yield=None,  # Would need external API
            beta=None,        # Would need external API
            fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low,
        )

        logger.info(f"Successfully fetched fundamentals for {symbol}")
        return fundamentals

    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {str(e)}")
        raise


def fetch_sentiment(symbol: str) -> Sentiment:
    """Fetch sentiment analysis data for a stock symbol.

    This tool analyzes market sentiment based on recent price action and volume.
    For production use, integrate with news sentiment APIs like Alpaca News API,
    MarketPsych, or social sentiment providers.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Sentiment object containing sentiment score, label, and news count

    Raises:
        ValueError: If symbol is invalid or insufficient data
        Exception: If API request fails

    Example:
        >>> sentiment = fetch_sentiment("AAPL")
        >>> print(f"Sentiment: {sentiment.label} (score: {sentiment.score:.2f})")
        >>> print(f"Based on {sentiment.news_count} data points")
    """
    logger.info(f"Fetching sentiment for {symbol}")

    try:
        # Fetch recent price action to derive sentiment
        price_bars = fetch_price_bars(symbol, timeframe="1d", days_back=20)

        if len(price_bars) < 10:
            raise ValueError(f"Insufficient data for sentiment analysis: {symbol}")

        # Simple sentiment calculation based on price momentum and volume
        recent_bars = price_bars[-10:]
        earlier_bars = price_bars[-20:-10] if len(price_bars) >= 20 else price_bars[:10]

        recent_close = recent_bars[-1].close
        earlier_close = earlier_bars[-1].close if earlier_bars else recent_bars[0].close

        # Calculate price change percentage
        price_change_pct = ((recent_close - earlier_close) / earlier_close) * 100

        # Calculate volume trend
        recent_avg_volume = sum(bar.volume for bar in recent_bars) / len(recent_bars)
        earlier_avg_volume = sum(bar.volume for bar in earlier_bars) / len(earlier_bars) if earlier_bars else recent_avg_volume
        volume_ratio = recent_avg_volume / earlier_avg_volume if earlier_avg_volume > 0 else 1.0

        # Derive sentiment score (-1 to 1)
        # Positive price momentum + increasing volume = bullish
        # Negative price momentum + increasing volume = bearish
        base_score = max(-1, min(1, price_change_pct / 10))  # Normalize to -1 to 1

        # Adjust score based on volume
        if volume_ratio > 1.2:  # 20% higher volume
            base_score *= 1.2  # Amplify the signal
        elif volume_ratio < 0.8:  # 20% lower volume
            base_score *= 0.8  # Dampen the signal

        sentiment_score = max(-1, min(1, base_score))

        # Determine label
        if sentiment_score > 0.3:
            label = "bullish"
        elif sentiment_score < -0.3:
            label = "bearish"
        else:
            label = "neutral"

        sentiment = Sentiment(
            score=round(sentiment_score, 2),
            label=label,
            news_count=len(price_bars),  # Using bar count as proxy
        )

        logger.info(f"Sentiment for {symbol}: {label} ({sentiment_score:.2f})")
        return sentiment

    except Exception as e:
        logger.error(f"Error calculating sentiment for {symbol}: {str(e)}")
        raise
