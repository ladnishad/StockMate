"""Data models for internal use."""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PriceBar(BaseModel):
    """OHLCV price bar data.

    Attributes:
        timestamp: Bar timestamp
        open: Opening price
        high: High price
        low: Low price
        close: Closing price
        volume: Trading volume
    """

    timestamp: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(ge=0)


class EarningsEvent(BaseModel):
    """Upcoming or recent earnings event data.

    Attributes:
        date: Earnings announcement date (YYYY-MM-DD)
        days_until: Days until earnings (negative if past)
        eps_estimate: Consensus EPS estimate
        revenue_estimate: Consensus revenue estimate in dollars
        hour: Timing - bmo (before market), amc (after close), dmh (during hours)
    """

    date: Optional[str] = Field(None, description="Earnings date YYYY-MM-DD")
    days_until: Optional[int] = Field(None, description="Days until earnings")
    eps_estimate: Optional[float] = Field(None, description="Consensus EPS estimate")
    revenue_estimate: Optional[float] = Field(None, description="Consensus revenue estimate")
    hour: Optional[str] = Field(None, description="bmo, amc, or dmh")


class EarningsSurprise(BaseModel):
    """Historical earnings surprise data.

    Attributes:
        period: Earnings period date
        actual: Actual EPS reported
        estimate: Estimated EPS
        surprise_percent: Surprise as percentage
    """

    period: str = Field(description="Earnings period YYYY-MM-DD")
    actual: Optional[float] = Field(None, description="Actual EPS")
    estimate: Optional[float] = Field(None, description="Estimated EPS")
    surprise_percent: Optional[float] = Field(None, description="Surprise %")


class Fundamentals(BaseModel):
    """Comprehensive fundamental data for a stock.

    Data sourced from Finnhub API including valuation metrics, growth metrics,
    profitability, financial health, and earnings calendar.

    Attributes:
        Valuation: market_cap, pe_ratio, pe_forward, pb_ratio, ps_ratio, peg_ratio
        Per-Share: eps, eps_growth_qoq, eps_growth_yoy, revenue_per_share, book_value_per_share
        Growth: revenue_growth_qoq, revenue_growth_yoy, revenue_growth_3y
        Profitability: gross_margin, operating_margin, net_margin, roe, roa, roic
        Health: debt_to_equity, current_ratio, quick_ratio, interest_coverage
        Dividends: dividend_yield, payout_ratio
        Risk: beta, fifty_two_week_high, fifty_two_week_low, fifty_two_week_range_position
        Earnings: next_earnings, last_earnings, earnings_beat_rate, avg_earnings_surprise
    """

    # Valuation Metrics
    market_cap: Optional[float] = Field(None, ge=0, description="Market cap in dollars")
    pe_ratio: Optional[float] = Field(None, description="Trailing P/E ratio")
    pe_forward: Optional[float] = Field(None, description="Forward P/E ratio")
    pb_ratio: Optional[float] = Field(None, description="Price to book ratio")
    ps_ratio: Optional[float] = Field(None, description="Price to sales ratio")
    peg_ratio: Optional[float] = Field(None, description="PEG ratio")

    # Per-Share Metrics
    eps: Optional[float] = Field(None, description="Earnings per share TTM")
    eps_growth_qoq: Optional[float] = Field(None, description="EPS growth QoQ %")
    eps_growth_yoy: Optional[float] = Field(None, description="EPS growth YoY %")
    revenue_per_share: Optional[float] = Field(None, description="Revenue per share")
    book_value_per_share: Optional[float] = Field(None, description="Book value per share")

    # Growth Metrics
    revenue_growth_qoq: Optional[float] = Field(None, description="Revenue growth QoQ %")
    revenue_growth_yoy: Optional[float] = Field(None, description="Revenue growth YoY %")
    revenue_growth_3y: Optional[float] = Field(None, description="3-year revenue CAGR %")

    # Profitability Metrics
    gross_margin: Optional[float] = Field(None, description="Gross margin %")
    operating_margin: Optional[float] = Field(None, description="Operating margin %")
    net_margin: Optional[float] = Field(None, description="Net profit margin %")
    roe: Optional[float] = Field(None, description="Return on equity %")
    roa: Optional[float] = Field(None, description="Return on assets %")
    roic: Optional[float] = Field(None, description="Return on invested capital %")

    # Financial Health Metrics
    debt_to_equity: Optional[float] = Field(None, ge=0, description="Debt to equity ratio")
    current_ratio: Optional[float] = Field(None, ge=0, description="Current ratio")
    quick_ratio: Optional[float] = Field(None, ge=0, description="Quick ratio")
    interest_coverage: Optional[float] = Field(None, description="Interest coverage ratio")

    # Dividend Metrics
    dividend_yield: Optional[float] = Field(None, ge=0, description="Dividend yield %")
    payout_ratio: Optional[float] = Field(None, ge=0, description="Payout ratio %")

    # Volatility & Risk
    beta: Optional[float] = Field(None, description="Beta vs market")
    fifty_two_week_high: Optional[float] = Field(None, gt=0, description="52-week high")
    fifty_two_week_low: Optional[float] = Field(None, gt=0, description="52-week low")
    fifty_two_week_range_position: Optional[float] = Field(
        None, description="Position in 52w range 0-100%"
    )

    # Earnings Events (CRITICAL for risk warnings)
    next_earnings: Optional[EarningsEvent] = Field(None, description="Next earnings")
    last_earnings: Optional[EarningsSurprise] = Field(None, description="Last earnings")
    earnings_beat_rate: Optional[float] = Field(None, description="Beat rate last 4Q %")
    avg_earnings_surprise: Optional[float] = Field(None, description="Avg surprise %")

    # Meta
    data_timestamp: Optional[str] = Field(None, description="When data was fetched")
    data_source: str = Field(default="finnhub", description="Data provider")

    def get_valuation_assessment(self) -> str:
        """Get a simple valuation assessment based on P/E ratio."""
        if self.pe_ratio is None:
            return "unknown"
        if self.pe_ratio < 0:
            return "unprofitable"
        if self.pe_ratio < 15:
            return "undervalued"
        if self.pe_ratio < 25:
            return "fairly_valued"
        if self.pe_ratio < 40:
            return "growth_premium"
        return "highly_valued"

    def get_financial_health_score(self) -> str:
        """Get financial health assessment based on key metrics."""
        score = 0
        if self.current_ratio is not None and self.current_ratio > 1.5:
            score += 1
        if self.debt_to_equity is not None and self.debt_to_equity < 1.0:
            score += 1
        if self.interest_coverage is not None and self.interest_coverage > 5:
            score += 1
        if self.net_margin is not None and self.net_margin > 10:
            score += 1

        if score >= 3:
            return "strong"
        if score >= 2:
            return "moderate"
        return "weak"

    def has_earnings_risk(self, days_threshold: int = 7) -> bool:
        """Check if earnings are within the risk window.

        Args:
            days_threshold: Number of days to consider as risk window

        Returns:
            True if earnings are within threshold days
        """
        if self.next_earnings and self.next_earnings.days_until is not None:
            return 0 <= self.next_earnings.days_until <= days_threshold
        return False


class Sentiment(BaseModel):
    """Sentiment analysis data.

    Attributes:
        score: Sentiment score (-1 to 1, negative to positive)
        label: Sentiment label (bearish, neutral, bullish)
        news_count: Number of news articles analyzed
    """

    score: float = Field(..., ge=-1, le=1)
    label: str = Field(..., pattern="^(bearish|neutral|bullish)$")
    news_count: int = Field(ge=0)


class NewsArticle(BaseModel):
    """Individual news article from Tiingo.

    Attributes:
        title: Article headline
        description: Article summary/description (truncated to 500 chars)
        url: Link to full article
        source: News source name
        published_date: Publication timestamp (ISO format)
        tickers: Tickers mentioned in article
        tags: Topic tags from Tiingo
        sentiment_score: Calculated sentiment (-1 to 1)
        sentiment_label: Sentiment classification
    """

    title: str = Field(default="", description="Article headline")
    description: str = Field(default="", description="Article summary")
    url: str = Field(default="", description="Article URL")
    source: str = Field(default="", description="News source")
    published_date: str = Field(default="", description="Publication date ISO")
    tickers: List[str] = Field(default_factory=list, description="Mentioned tickers")
    tags: List[str] = Field(default_factory=list, description="Topic tags")
    sentiment_score: float = Field(default=0.0, ge=-1, le=1, description="Sentiment")
    sentiment_label: str = Field(default="neutral", description="Sentiment label")


class NewsContext(BaseModel):
    """Aggregated news context for a ticker.

    Contains processed news data from Tiingo including sentiment analysis,
    key themes, and breaking news detection.

    Attributes:
        articles: List of recent news articles
        overall_sentiment: Aggregated sentiment label (bullish/neutral/bearish)
        sentiment_score: Numeric sentiment score (-1 to 1)
        article_count: Total number of articles found
        key_themes: Most common topic tags
        has_breaking_news: Whether any articles are < 2 hours old
        data_source: Data provider name
    """

    articles: List[NewsArticle] = Field(default_factory=list, description="News articles")
    overall_sentiment: str = Field(default="neutral", description="Aggregated sentiment")
    sentiment_score: float = Field(default=0.0, ge=-1, le=1, description="Sentiment score")
    article_count: int = Field(default=0, ge=0, description="Article count")
    key_themes: List[str] = Field(default_factory=list, description="Top topic tags")
    has_breaking_news: bool = Field(default=False, description="Has recent news")
    data_source: str = Field(default="tiingo", description="Data provider")

    def get_sentiment_label(self) -> str:
        """Get human-readable sentiment label."""
        if self.sentiment_score > 0.2:
            return "bullish"
        elif self.sentiment_score < -0.2:
            return "bearish"
        return "neutral"

    def has_catalyst_risk(self) -> bool:
        """Check if there's breaking news that could cause volatility.

        Breaking news within 2 hours can cause unexpected price moves,
        especially important for day trading.
        """
        return self.has_breaking_news

    def sentiment_aligns_with_bias(self, bias: str) -> bool:
        """Check if news sentiment aligns with trade bias.

        Args:
            bias: Trade bias ('long', 'short', or 'neutral')

        Returns:
            True if sentiment supports the bias, False if conflicts
        """
        if bias == "long":
            return self.sentiment_score > 0.1  # Bullish or neutral-bullish
        elif bias == "short":
            return self.sentiment_score < -0.1  # Bearish or neutral-bearish
        return True  # Neutral bias always aligns

    def get_top_articles(self, count: int = 5) -> List[NewsArticle]:
        """Get the most recent articles.

        Args:
            count: Number of articles to return

        Returns:
            Top N most recent articles
        """
        return self.articles[:count]


class Indicator(BaseModel):
    """Technical indicator data.

    Attributes:
        name: Indicator name
        value: Current indicator value
        signal: Trading signal (bullish, bearish, neutral)
        metadata: Additional metadata
    """

    name: str
    value: float
    signal: str = Field(pattern="^(bullish|bearish|neutral)$")
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StructuralPivot(BaseModel):
    """Structural pivot point (support/resistance level).

    Attributes:
        price: Pivot price level
        type: Pivot type (support or resistance)
        strength: Strength score (0-100)
        touches: Number of times price touched this level
    """

    price: float = Field(gt=0)
    type: str = Field(pattern="^(support|resistance)$")
    strength: float = Field(ge=0, le=100)
    touches: int = Field(ge=1)


class MarketSnapshot(BaseModel):
    """Complete market snapshot for analysis.

    Attributes:
        symbol: Stock ticker symbol
        current_price: Current stock price
        price_bars_1d: Daily price bars
        price_bars_1h: Hourly price bars (optional)
        price_bars_15m: 15-minute price bars (optional)
        fundamentals: Fundamental data
        sentiment: Sentiment data
        indicators: List of calculated indicators
        pivots: List of structural pivots
        timestamp: Snapshot timestamp
    """

    symbol: str
    current_price: float = Field(gt=0)
    price_bars_1d: List[PriceBar]
    price_bars_1h: Optional[List[PriceBar]] = None
    price_bars_15m: Optional[List[PriceBar]] = None
    fundamentals: Fundamentals
    sentiment: Sentiment
    indicators: List[Indicator] = Field(default_factory=list)
    pivots: List[StructuralPivot] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============================================================================
# AlgoTrader Plus Real-Time Data Models
# ============================================================================


class Quote(BaseModel):
    """Real-time bid/ask quote data.

    Represents the current best bid and ask prices from the market.
    With AlgoTrader Plus, this data comes from the SIP feed (all US exchanges).

    Attributes:
        symbol: Stock ticker symbol
        bid_price: Current best bid price
        ask_price: Current best ask price
        bid_size: Number of shares at best bid
        ask_size: Number of shares at best ask
        spread: Absolute spread (ask - bid)
        spread_pct: Spread as percentage of mid price
        mid_price: Midpoint between bid and ask
        timestamp: Quote timestamp
    """

    symbol: str
    bid_price: float = Field(ge=0)
    ask_price: float = Field(ge=0)
    bid_size: int = Field(ge=0)
    ask_size: int = Field(ge=0)
    spread: float = Field(ge=0)
    spread_pct: float = Field(ge=0)
    mid_price: float = Field(ge=0)
    timestamp: Optional[datetime] = None


class Trade(BaseModel):
    """Real-time trade execution data.

    Represents the most recent trade execution from the market.
    With AlgoTrader Plus, this data comes from the SIP feed (all US exchanges).

    Attributes:
        symbol: Stock ticker symbol
        price: Trade execution price
        size: Number of shares traded
        exchange: Exchange where trade occurred
        timestamp: Trade timestamp
        conditions: Trade conditions/flags (if available)
    """

    symbol: str
    price: float = Field(gt=0)
    size: int = Field(ge=0)
    exchange: Optional[str] = None
    timestamp: Optional[datetime] = None
    conditions: List[str] = Field(default_factory=list)


class RealTimeSnapshot(BaseModel):
    """Complete real-time snapshot from AlgoTrader Plus.

    Contains latest quote, latest trade, current daily bar, and previous
    daily bar for a symbol. This is more efficient than fetching each
    piece of data separately.

    Attributes:
        symbol: Stock ticker symbol
        latest_quote: Most recent bid/ask quote
        latest_trade: Most recent trade execution
        daily_bar: Current day's OHLCV bar
        prev_daily_bar: Previous day's OHLCV bar
    """

    symbol: str
    latest_quote: Optional[Quote] = None
    latest_trade: Optional[Trade] = None
    daily_bar: Optional[PriceBar] = None
    prev_daily_bar: Optional[PriceBar] = None
