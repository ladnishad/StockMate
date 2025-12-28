"""Market data fetching tools using Alpaca API.

All functions are designed to be used as LLM agent tools with clear
signatures, comprehensive docstrings, and structured input/output.
"""

from typing import List, Literal, Optional
from datetime import datetime, timedelta
import logging

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import (
    StockBarsRequest,
    StockLatestQuoteRequest,
    StockLatestTradeRequest,
    StockSnapshotRequest,
    NewsRequest,
)
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

from app.config import get_settings
from app.models.data import PriceBar, Fundamentals, Sentiment

logger = logging.getLogger(__name__)


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
    timeframe: Literal["1d", "1h", "15m", "5m", "1m", "1w"] = "1d",
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
        timeframe: Bar timeframe - '1d' (daily), '1h' (hourly), '15m' (15-minute), '5m' (5-minute), '1w' (weekly)
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
        "1w": TimeFrame.Week,
        "1m": TimeFrame.Minute,   # 1 minute
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
    elif timeframe == "1m":
        tf = TimeFrame.Minute
    else:
        tf = timeframe_map[timeframe]

    try:
        client = _get_alpaca_client()
        settings = get_settings()

        # Use provided feed or default from config (sip for AlgoTrader Plus)
        data_feed = feed or settings.alpaca_data_feed

        # Build request parameters with feed specification
        request_params = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
            feed=data_feed,  # Specify data feed (iex, sip, or None for auto)
        )

        bars_data = client.get_stock_bars(request_params)

        # BarSet object requires checking .data dict for 'in' operator
        if symbol not in bars_data.data:
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
        base_score = price_change_pct / 10  # Normalize to approximately -1 to 1

        # Adjust score based on volume (amplify or dampen before clamping)
        if volume_ratio > 1.2:  # 20% higher volume
            base_score *= 1.2  # Amplify the signal
        elif volume_ratio < 0.8:  # 20% lower volume
            base_score *= 0.8  # Dampen the signal

        # Clamp final score to [-1, 1] range
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


def fetch_news_sentiment(
    symbol: str,
    days_back: int = 7,
    limit: int = 50,
) -> dict:
    """Fetch real news sentiment from Alpaca News API.

    This tool retrieves actual news articles and sentiment scores from Alpaca's News API,
    providing comprehensive news-based sentiment analysis including:
    - Sentiment scores from news articles
    - Article headlines and summaries
    - News volume and recency
    - Source credibility

    Note: Alpaca News API is available on paid plans. Free tier users will get
    a graceful fallback to the basic sentiment analysis.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        days_back: Number of days of news to fetch (default: 7)
        limit: Maximum number of articles to retrieve (default: 50)

    Returns:
        Dictionary containing:
        - sentiment_score: Aggregated sentiment (-1 to 1)
        - sentiment_label: "bullish", "bearish", or "neutral"
        - article_count: Number of articles analyzed
        - average_sentiment: Average sentiment from articles
        - recent_headlines: List of recent headlines with sentiment
        - news_volume_trend: "increasing", "stable", or "decreasing"

    Raises:
        ValueError: If symbol is invalid
        Exception: If API request fails

    Example:
        >>> news = fetch_news_sentiment("AAPL", days_back=7)
        >>> print(f"News Sentiment: {news['sentiment_label']} ({news['sentiment_score']:.2f})")
        >>> print(f"Analyzed {news['article_count']} articles")
        >>> for headline in news['recent_headlines'][:5]:
        >>>     print(f"  - {headline['title']} (sentiment: {headline['sentiment']})")
    """
    logger.info(f"Fetching news sentiment for {symbol} (last {days_back} days)")

    try:
        client = _get_alpaca_client()

        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days_back)

        # Create news request
        news_request = NewsRequest(
            symbols=symbol,
            start=start_date,
            end=end_date,
            limit=limit,
            sort="desc",  # Most recent first
        )

        # Fetch news articles
        try:
            news_data = client.get_news(news_request)
            articles = list(news_data) if news_data else []
        except AttributeError:
            # News API might not be available in all alpaca-py versions or subscription tiers
            logger.warning(
                f"Alpaca News API not available. "
                f"This may require a paid subscription or newer alpaca-py version. "
                f"Falling back to basic sentiment analysis."
            )
            # Fallback to basic sentiment
            basic_sentiment = fetch_sentiment(symbol)
            return {
                "sentiment_score": basic_sentiment.score,
                "sentiment_label": basic_sentiment.label,
                "article_count": 0,
                "average_sentiment": basic_sentiment.score,
                "recent_headlines": [],
                "news_volume_trend": "unavailable",
                "source": "price_based_fallback",
            }
        except APIError as e:
            logger.warning(f"Alpaca News API error: {str(e)}. Falling back to basic sentiment.")
            basic_sentiment = fetch_sentiment(symbol)
            return {
                "sentiment_score": basic_sentiment.score,
                "sentiment_label": basic_sentiment.label,
                "article_count": 0,
                "average_sentiment": basic_sentiment.score,
                "recent_headlines": [],
                "news_volume_trend": "unavailable",
                "source": "price_based_fallback",
            }

        if not articles:
            logger.warning(f"No news articles found for {symbol}. Using basic sentiment.")
            basic_sentiment = fetch_sentiment(symbol)
            return {
                "sentiment_score": basic_sentiment.score,
                "sentiment_label": basic_sentiment.label,
                "article_count": 0,
                "average_sentiment": basic_sentiment.score,
                "recent_headlines": [],
                "news_volume_trend": "no_news",
                "source": "price_based_fallback",
            }

        # Extract headlines and sentiment scores
        headlines_with_sentiment = []
        sentiment_scores = []

        for article in articles:
            # Alpaca news articles may have sentiment attached
            # If not available, we'll analyze based on headline tone
            article_sentiment = 0.0

            # Try to get sentiment from article metadata
            # Note: Exact field names may vary by Alpaca API version
            if hasattr(article, 'sentiment'):
                article_sentiment = float(article.sentiment)
            elif hasattr(article, 'sentiment_score'):
                article_sentiment = float(article.sentiment_score)
            else:
                # Fallback: Simple keyword-based sentiment
                headline = article.headline.lower() if hasattr(article, 'headline') else ""
                summary = article.summary.lower() if hasattr(article, 'summary') else ""
                text = headline + " " + summary

                # Bullish keywords
                bullish_keywords = [
                    'surge', 'gain', 'rise', 'jump', 'rally', 'beat', 'exceed',
                    'growth', 'profit', 'strong', 'up', 'high', 'record', 'upgrade',
                    'buy', 'outperform', 'positive', 'breakthrough', 'success'
                ]

                # Bearish keywords
                bearish_keywords = [
                    'fall', 'drop', 'decline', 'loss', 'miss', 'weak', 'down',
                    'low', 'concern', 'worry', 'risk', 'cut', 'downgrade', 'sell',
                    'underperform', 'negative', 'struggle', 'fail', 'lawsuit'
                ]

                bullish_count = sum(1 for word in bullish_keywords if word in text)
                bearish_count = sum(1 for word in bearish_keywords if word in text)

                # Calculate sentiment score (-1 to 1)
                if bullish_count + bearish_count > 0:
                    article_sentiment = (bullish_count - bearish_count) / (bullish_count + bearish_count)
                else:
                    article_sentiment = 0.0

            sentiment_scores.append(article_sentiment)

            # Store headline info
            headlines_with_sentiment.append({
                "title": article.headline if hasattr(article, 'headline') else "Unknown",
                "sentiment": round(article_sentiment, 2),
                "created_at": article.created_at if hasattr(article, 'created_at') else None,
                "url": article.url if hasattr(article, 'url') else None,
                "source": article.source if hasattr(article, 'source') else "Unknown",
            })

        # Calculate aggregated sentiment
        if sentiment_scores:
            average_sentiment = sum(sentiment_scores) / len(sentiment_scores)

            # Weight recent news more heavily (exponential decay)
            weighted_scores = []
            for i, score in enumerate(sentiment_scores):
                # More recent articles get higher weight (decay factor)
                recency_weight = 1.0 / (1.0 + (i * 0.1))
                weighted_scores.append(score * recency_weight)

            weighted_average = sum(weighted_scores) / sum(
                1.0 / (1.0 + (i * 0.1)) for i in range(len(weighted_scores))
            )

            sentiment_score = round(weighted_average, 2)
        else:
            average_sentiment = 0.0
            sentiment_score = 0.0

        # Determine sentiment label
        if sentiment_score > 0.2:
            sentiment_label = "bullish"
        elif sentiment_score < -0.2:
            sentiment_label = "bearish"
        else:
            sentiment_label = "neutral"

        # Calculate news volume trend
        if len(articles) > 10:
            recent_count = len([a for a in articles[:len(articles)//2]])
            older_count = len([a for a in articles[len(articles)//2:]])

            if recent_count > older_count * 1.5:
                news_volume_trend = "increasing"
            elif recent_count < older_count * 0.67:
                news_volume_trend = "decreasing"
            else:
                news_volume_trend = "stable"
        else:
            news_volume_trend = "low_volume"

        logger.info(
            f"News Sentiment for {symbol}: {sentiment_label} ({sentiment_score:.2f}) "
            f"based on {len(articles)} articles"
        )

        return {
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "article_count": len(articles),
            "average_sentiment": round(average_sentiment, 2),
            "recent_headlines": headlines_with_sentiment[:10],  # Top 10 most recent
            "news_volume_trend": news_volume_trend,
            "source": "alpaca_news_api",
            "days_analyzed": days_back,
        }

    except Exception as e:
        logger.error(f"Error fetching news sentiment for {symbol}: {str(e)}")
        # Graceful fallback
        logger.info("Falling back to price-based sentiment analysis")
        basic_sentiment = fetch_sentiment(symbol)
        return {
            "sentiment_score": basic_sentiment.score,
            "sentiment_label": basic_sentiment.label,
            "article_count": 0,
            "average_sentiment": basic_sentiment.score,
            "recent_headlines": [],
            "news_volume_trend": "unavailable",
            "source": "price_based_fallback",
            "error": str(e),
        }


def fetch_latest_quote(symbol: str) -> dict:
    """Fetch real-time bid/ask quote for a symbol.

    This tool retrieves the most recent quote (bid/ask prices and sizes) from
    Alpaca Markets API. With AlgoTrader Plus subscription, this provides real-time
    data from all US exchanges via the SIP feed.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Dictionary containing:
        - symbol: The stock symbol
        - bid_price: Current best bid price
        - ask_price: Current best ask price
        - bid_size: Size at best bid
        - ask_size: Size at best ask
        - spread: Absolute spread (ask - bid)
        - spread_pct: Spread as percentage of mid price
        - mid_price: Midpoint between bid and ask
        - timestamp: Quote timestamp

    Raises:
        ValueError: If symbol is invalid or no quote available
        APIError: If Alpaca API returns an error

    Example:
        >>> quote = fetch_latest_quote("AAPL")
        >>> print(f"AAPL Bid: ${quote['bid_price']:.2f} Ask: ${quote['ask_price']:.2f}")
        >>> print(f"Spread: {quote['spread_pct']:.3f}%")
    """
    logger.info(f"Fetching latest quote for {symbol}")

    try:
        client = _get_alpaca_client()
        settings = get_settings()

        request = StockLatestQuoteRequest(
            symbol_or_symbols=symbol,
            feed=settings.alpaca_data_feed,
        )

        quotes = client.get_stock_latest_quote(request)

        if symbol not in quotes:
            raise ValueError(f"No quote available for symbol: {symbol}")

        quote = quotes[symbol]

        bid_price = float(quote.bid_price)
        ask_price = float(quote.ask_price)
        spread = ask_price - bid_price
        mid_price = (bid_price + ask_price) / 2
        spread_pct = (spread / mid_price * 100) if mid_price > 0 else 0

        result = {
            "symbol": symbol,
            "bid_price": bid_price,
            "ask_price": ask_price,
            "bid_size": int(quote.bid_size),
            "ask_size": int(quote.ask_size),
            "spread": round(spread, 4),
            "spread_pct": round(spread_pct, 4),
            "mid_price": round(mid_price, 4),
            "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
        }

        logger.info(f"Quote for {symbol}: bid=${bid_price:.2f} ask=${ask_price:.2f} spread={spread_pct:.3f}%")
        return result

    except APIError as e:
        logger.error(f"Alpaca API error fetching quote for {symbol}: {str(e)}")
        raise ValueError(f"Alpaca API error: {str(e)}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {str(e)}")
        raise


def fetch_latest_trade(symbol: str) -> dict:
    """Fetch the most recent trade execution for a symbol.

    This tool retrieves the latest trade (price, size, exchange) from
    Alpaca Markets API. With AlgoTrader Plus subscription, this provides
    real-time data from all US exchanges via the SIP feed.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Dictionary containing:
        - symbol: The stock symbol
        - price: Trade execution price
        - size: Number of shares traded
        - exchange: Exchange where trade occurred
        - timestamp: Trade timestamp
        - conditions: Trade conditions (if available)

    Raises:
        ValueError: If symbol is invalid or no trade available
        APIError: If Alpaca API returns an error

    Example:
        >>> trade = fetch_latest_trade("AAPL")
        >>> print(f"AAPL Last: ${trade['price']:.2f} x {trade['size']} on {trade['exchange']}")
    """
    logger.info(f"Fetching latest trade for {symbol}")

    try:
        client = _get_alpaca_client()
        settings = get_settings()

        request = StockLatestTradeRequest(
            symbol_or_symbols=symbol,
            feed=settings.alpaca_data_feed,
        )

        trades = client.get_stock_latest_trade(request)

        if symbol not in trades:
            raise ValueError(f"No trade available for symbol: {symbol}")

        trade = trades[symbol]

        result = {
            "symbol": symbol,
            "price": float(trade.price),
            "size": int(trade.size),
            "exchange": trade.exchange if hasattr(trade, 'exchange') else None,
            "timestamp": trade.timestamp.isoformat() if trade.timestamp else None,
            "conditions": list(trade.conditions) if hasattr(trade, 'conditions') and trade.conditions else [],
        }

        logger.info(f"Latest trade for {symbol}: ${result['price']:.2f} x {result['size']}")
        return result

    except APIError as e:
        logger.error(f"Alpaca API error fetching trade for {symbol}: {str(e)}")
        raise ValueError(f"Alpaca API error: {str(e)}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error fetching trade for {symbol}: {str(e)}")
        raise


def fetch_snapshots(symbols: List[str]) -> dict:
    """Fetch real-time snapshots for multiple symbols in a single API call.

    This tool retrieves comprehensive snapshots including latest quote, latest trade,
    current daily bar, and previous daily bar for multiple symbols efficiently.
    This is much faster than fetching data for each symbol individually.

    With AlgoTrader Plus subscription, this provides real-time data from all
    US exchanges via the SIP feed.

    Args:
        symbols: List of stock ticker symbols (e.g., ['AAPL', 'TSLA', 'MSFT'])

    Returns:
        Dictionary mapping symbol to snapshot data:
        {
            "AAPL": {
                "symbol": "AAPL",
                "latest_quote": { bid_price, ask_price, spread, ... },
                "latest_trade": { price, size, exchange, ... },
                "daily_bar": { open, high, low, close, volume, ... },
                "prev_daily_bar": { open, high, low, close, volume, ... },
            },
            ...
        }

    Raises:
        ValueError: If symbols list is empty or invalid
        APIError: If Alpaca API returns an error

    Example:
        >>> snapshots = fetch_snapshots(["AAPL", "MSFT", "GOOGL"])
        >>> for symbol, data in snapshots.items():
        >>>     print(f"{symbol}: ${data['latest_trade']['price']:.2f}")
    """
    if not symbols:
        raise ValueError("Symbols list cannot be empty")

    logger.info(f"Fetching snapshots for {len(symbols)} symbols: {symbols[:5]}{'...' if len(symbols) > 5 else ''}")

    try:
        client = _get_alpaca_client()
        settings = get_settings()

        request = StockSnapshotRequest(
            symbol_or_symbols=symbols,
            feed=settings.alpaca_data_feed,
        )

        snapshots = client.get_stock_snapshot(request)

        result = {}
        for symbol in symbols:
            if symbol not in snapshots:
                logger.warning(f"No snapshot available for {symbol}")
                result[symbol] = None
                continue

            snapshot = snapshots[symbol]

            # Process latest quote
            latest_quote = None
            if snapshot.latest_quote:
                q = snapshot.latest_quote
                bid = float(q.bid_price)
                ask = float(q.ask_price)
                mid = (bid + ask) / 2
                spread = ask - bid
                latest_quote = {
                    "bid_price": bid,
                    "ask_price": ask,
                    "bid_size": int(q.bid_size),
                    "ask_size": int(q.ask_size),
                    "spread": round(spread, 4),
                    "spread_pct": round(spread / mid * 100, 4) if mid > 0 else 0,
                    "mid_price": round(mid, 4),
                    "timestamp": q.timestamp.isoformat() if q.timestamp else None,
                }

            # Process latest trade
            latest_trade = None
            if snapshot.latest_trade:
                t = snapshot.latest_trade
                latest_trade = {
                    "price": float(t.price),
                    "size": int(t.size),
                    "exchange": t.exchange if hasattr(t, 'exchange') else None,
                    "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                }

            # Process daily bar
            daily_bar = None
            if snapshot.daily_bar:
                b = snapshot.daily_bar
                daily_bar = {
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": int(b.volume),
                    "timestamp": b.timestamp.isoformat() if b.timestamp else None,
                    "vwap": float(b.vwap) if hasattr(b, 'vwap') and b.vwap else None,
                }

            # Process previous daily bar
            prev_daily_bar = None
            if snapshot.previous_daily_bar:
                pb = snapshot.previous_daily_bar
                prev_daily_bar = {
                    "open": float(pb.open),
                    "high": float(pb.high),
                    "low": float(pb.low),
                    "close": float(pb.close),
                    "volume": int(pb.volume),
                    "timestamp": pb.timestamp.isoformat() if pb.timestamp else None,
                    "vwap": float(pb.vwap) if hasattr(pb, 'vwap') and pb.vwap else None,
                }

            result[symbol] = {
                "symbol": symbol,
                "latest_quote": latest_quote,
                "latest_trade": latest_trade,
                "daily_bar": daily_bar,
                "prev_daily_bar": prev_daily_bar,
            }

        logger.info(f"Successfully fetched snapshots for {len(result)} symbols")
        return result

    except APIError as e:
        logger.error(f"Alpaca API error fetching snapshots: {str(e)}")
        raise ValueError(f"Alpaca API error: {str(e)}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error fetching snapshots: {str(e)}")
        raise
