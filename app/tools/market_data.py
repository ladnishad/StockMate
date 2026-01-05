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
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from alpaca.trading.client import TradingClient
from alpaca.common.exceptions import APIError

from app.config import get_settings
from app.models.data import PriceBar, Fundamentals, Sentiment, NewsArticle, NewsContext
from app.tools.finnhub_client import fetch_company_news as finnhub_fetch_news

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
        tf = TimeFrame(15, TimeFrameUnit.Minute)
    elif timeframe == "5m":
        tf = TimeFrame(5, TimeFrameUnit.Minute)
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


async def fetch_fundamentals(symbol: str) -> Fundamentals:
    """Fetch comprehensive fundamental data from Finnhub API.

    This tool retrieves fundamental financial metrics including:
    - Valuation (P/E, P/B, P/S, PEG)
    - Growth (EPS growth, revenue growth)
    - Profitability (margins, ROE, ROA)
    - Financial health (debt ratios, liquidity)
    - Earnings calendar (upcoming earnings risk!)
    - Earnings history (beat rate, surprises)

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')

    Returns:
        Fundamentals object with comprehensive financial data

    Example:
        >>> fundamentals = await fetch_fundamentals("AAPL")
        >>> print(f"P/E: {fundamentals.pe_ratio}, Valuation: {fundamentals.get_valuation_assessment()}")
        >>> if fundamentals.has_earnings_risk():
        ...     print(f"WARNING: Earnings in {fundamentals.next_earnings.days_until} days!")
    """
    import asyncio
    from app.tools.finnhub_client import (
        fetch_basic_financials,
        fetch_earnings_calendar,
        fetch_earnings_history,
    )
    from app.models.data import EarningsEvent, EarningsSurprise

    logger.info(f"Fetching Finnhub fundamentals for {symbol}")

    try:
        # Fetch all Finnhub data in parallel
        basic_task = fetch_basic_financials(symbol)
        calendar_task = fetch_earnings_calendar(symbol)
        history_task = fetch_earnings_history(symbol)

        basic, calendar, history = await asyncio.gather(
            basic_task, calendar_task, history_task
        )

        # Also get 52-week high/low from price bars as backup
        fallback_52w_high = None
        fallback_52w_low = None
        range_position = None
        try:
            price_bars = fetch_price_bars(symbol, timeframe="1d", days_back=365)
            if price_bars:
                highs = [bar.high for bar in price_bars]
                lows = [bar.low for bar in price_bars]
                current_price = price_bars[-1].close

                fallback_52w_high = max(highs) if highs else None
                fallback_52w_low = min(lows) if lows else None

                # Calculate position in 52w range
                if fallback_52w_high and fallback_52w_low and current_price:
                    range_size = fallback_52w_high - fallback_52w_low
                    if range_size > 0:
                        range_position = (
                            (current_price - fallback_52w_low) / range_size
                        ) * 100
        except Exception as price_err:
            logger.warning(f"Could not fetch price data for 52w range: {price_err}")

        # Build EarningsEvent if available
        next_earnings = None
        if calendar.get("next_earnings"):
            ne = calendar["next_earnings"]
            next_earnings = EarningsEvent(
                date=ne.get("date"),
                days_until=ne.get("days_until"),
                eps_estimate=ne.get("eps_estimate"),
                revenue_estimate=ne.get("revenue_estimate"),
                hour=ne.get("hour"),
            )

        # Build last earnings surprise if available
        last_earnings = None
        if history.get("last_earnings"):
            le = history["last_earnings"]
            last_earnings = EarningsSurprise(
                period=le.get("period", "unknown"),
                actual=le.get("actual"),
                estimate=le.get("estimate"),
                surprise_percent=le.get("surprise_percent"),
            )

        # Build Fundamentals object
        fundamentals = Fundamentals(
            # Valuation
            market_cap=(
                basic.get("market_cap") * 1_000_000
                if basic.get("market_cap")
                else None
            ),
            pe_ratio=basic.get("pe_ratio"),
            pe_forward=basic.get("pe_forward"),
            pb_ratio=basic.get("pb_ratio"),
            ps_ratio=basic.get("ps_ratio"),
            peg_ratio=basic.get("peg_ratio"),
            # Per-share
            eps=basic.get("eps"),
            eps_growth_qoq=basic.get("eps_growth_qoq"),
            eps_growth_yoy=basic.get("eps_growth_yoy"),
            revenue_per_share=basic.get("revenue_per_share"),
            book_value_per_share=basic.get("book_value_per_share"),
            # Growth
            revenue_growth_qoq=basic.get("revenue_growth_qoq"),
            revenue_growth_yoy=basic.get("revenue_growth_yoy"),
            revenue_growth_3y=basic.get("revenue_growth_3y"),
            # Profitability
            gross_margin=basic.get("gross_margin"),
            operating_margin=basic.get("operating_margin"),
            net_margin=basic.get("net_margin"),
            roe=basic.get("roe"),
            roa=basic.get("roa"),
            roic=basic.get("roic"),
            # Health
            debt_to_equity=basic.get("debt_to_equity"),
            current_ratio=basic.get("current_ratio"),
            quick_ratio=basic.get("quick_ratio"),
            interest_coverage=basic.get("interest_coverage"),
            # Dividends
            dividend_yield=basic.get("dividend_yield"),
            payout_ratio=basic.get("payout_ratio"),
            # Risk
            beta=basic.get("beta"),
            fifty_two_week_high=basic.get("52_week_high") or fallback_52w_high,
            fifty_two_week_low=basic.get("52_week_low") or fallback_52w_low,
            fifty_two_week_range_position=range_position,
            # Earnings
            next_earnings=next_earnings,
            last_earnings=last_earnings,
            earnings_beat_rate=history.get("earnings_beat_rate"),
            avg_earnings_surprise=history.get("avg_earnings_surprise"),
            # Meta
            data_timestamp=datetime.utcnow().isoformat(),
            data_source="finnhub",
        )

        logger.info(
            f"Successfully fetched fundamentals for {symbol}: "
            f"P/E={fundamentals.pe_ratio}, Health={fundamentals.get_financial_health_score()}"
        )
        return fundamentals

    except Exception as e:
        logger.error(f"Error fetching fundamentals for {symbol}: {e}")
        # Return empty fundamentals on error
        return Fundamentals(
            data_source="error",
            data_timestamp=datetime.utcnow().isoformat(),
        )


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


async def fetch_news_sentiment(
    symbol: str,
    days_back: int = 7,
    limit: int = 20,
) -> NewsContext:
    """Fetch news sentiment from Finnhub News API.

    This tool retrieves news articles and calculates sentiment scores from Finnhub's
    free News API, providing comprehensive news-based sentiment analysis including:
    - Sentiment scores from news article content
    - Article headlines and summaries
    - Key themes from article categories
    - Breaking news detection

    Finnhub free tier includes company news (60 API calls/minute).

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        days_back: Number of days of news to fetch (default: 7)
        limit: Maximum number of articles to retrieve (default: 20)

    Returns:
        NewsContext object containing:
        - articles: List of NewsArticle objects
        - overall_sentiment: "bullish", "bearish", or "neutral"
        - sentiment_score: Aggregated sentiment (-1 to 1)
        - article_count: Number of articles analyzed
        - key_themes: Common topics from article categories
        - has_breaking_news: Whether recent news exists

    Example:
        >>> news = await fetch_news_sentiment("AAPL", days_back=7)
        >>> print(f"News Sentiment: {news.overall_sentiment} ({news.sentiment_score:.2f})")
        >>> print(f"Analyzed {news.article_count} articles")
        >>> for article in news.articles[:5]:
        >>>     print(f"  - {article.title} ({article.sentiment_label})")
    """

    logger.info(f"Fetching news sentiment for {symbol} from Finnhub (last {days_back} days)")

    try:
        # Fetch from Finnhub
        news_data = await finnhub_fetch_news(symbol, days_back=days_back, limit=limit)

        # Check for error
        if news_data.get("error"):
            logger.warning(f"Finnhub API error: {news_data['error']}. Using fallback.")
            basic_sentiment = fetch_sentiment(symbol)
            return NewsContext(
                articles=[],
                overall_sentiment=basic_sentiment.label,
                sentiment_score=basic_sentiment.score,
                article_count=0,
                key_themes=[],
                has_breaking_news=False,
                data_source="price_based_fallback",
            )

        # Convert to NewsArticle objects
        articles = [
            NewsArticle(
                title=a.get("title", ""),
                description=a.get("description", ""),
                url=a.get("url", ""),
                source=a.get("source", ""),
                published_date=a.get("published_date", ""),
                tickers=a.get("tickers", []),
                tags=a.get("tags", []),
                sentiment_score=a.get("sentiment_score", 0.0),
                sentiment_label=a.get("sentiment_label", "neutral"),
            )
            for a in news_data.get("articles", [])
        ]

        news_context = NewsContext(
            articles=articles,
            overall_sentiment=news_data.get("overall_sentiment", "neutral"),
            sentiment_score=news_data.get("sentiment_score", 0.0),
            article_count=news_data.get("article_count", 0),
            key_themes=news_data.get("key_themes", []),
            has_breaking_news=news_data.get("has_breaking_news", False),
            data_source="finnhub",
        )

        logger.info(
            f"News Sentiment for {symbol}: {news_context.overall_sentiment} "
            f"({news_context.sentiment_score:.2f}) based on {news_context.article_count} articles"
        )

        return news_context

    except Exception as e:
        logger.error(f"Error fetching news sentiment for {symbol}: {str(e)}")
        # Graceful fallback to price-based sentiment
        logger.info("Falling back to price-based sentiment analysis")
        basic_sentiment = fetch_sentiment(symbol)
        return NewsContext(
            articles=[],
            overall_sentiment=basic_sentiment.label,
            sentiment_score=basic_sentiment.score,
            article_count=0,
            key_themes=[],
            has_breaking_news=False,
            data_source="price_based_fallback",
        )


async def fetch_news_for_trade_style(
    symbol: str,
    trade_style: str,
) -> NewsContext:
    """Fetch news optimized for a specific trade style.

    Different trade styles need different news time windows:
    - Day trades: Last 2 days, focus on breaking news
    - Swing trades: Last 7 days, balance of recency and context
    - Position trades: Last 30 days, comprehensive view

    Args:
        symbol: Stock ticker symbol
        trade_style: One of 'day', 'swing', 'position'

    Returns:
        NewsContext with trade-style appropriate data
    """
    from app.tools.finnhub_client import fetch_news_for_trade_style as finnhub_fetch_for_style

    logger.info(f"Fetching {trade_style} trade news for {symbol}")

    try:
        news_data = await finnhub_fetch_for_style(symbol, trade_style)

        # Convert to NewsContext
        articles = [
            NewsArticle(
                title=a.get("title", ""),
                description=a.get("description", ""),
                url=a.get("url", ""),
                source=a.get("source", ""),
                published_date=a.get("published_date", ""),
                tickers=a.get("tickers", []),
                tags=a.get("tags", []),
                sentiment_score=a.get("sentiment_score", 0.0),
                sentiment_label=a.get("sentiment_label", "neutral"),
            )
            for a in news_data.get("articles", [])
        ]

        return NewsContext(
            articles=articles,
            overall_sentiment=news_data.get("overall_sentiment", "neutral"),
            sentiment_score=news_data.get("sentiment_score", 0.0),
            article_count=news_data.get("article_count", 0),
            key_themes=news_data.get("key_themes", []),
            has_breaking_news=news_data.get("has_breaking_news", False),
            data_source="finnhub",
        )

    except Exception as e:
        logger.error(f"Error fetching trade-style news for {symbol}: {str(e)}")
        return NewsContext(
            articles=[],
            overall_sentiment="neutral",
            sentiment_score=0.0,
            article_count=0,
            key_themes=[],
            has_breaking_news=False,
            data_source="error",
        )


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
