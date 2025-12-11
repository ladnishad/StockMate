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


class Fundamentals(BaseModel):
    """Fundamental data for a stock.

    Attributes:
        market_cap: Market capitalization
        pe_ratio: Price to earnings ratio
        eps: Earnings per share
        dividend_yield: Dividend yield percentage
        beta: Stock beta (volatility measure)
        fifty_two_week_high: 52-week high price
        fifty_two_week_low: 52-week low price
    """

    market_cap: Optional[float] = Field(None, ge=0)
    pe_ratio: Optional[float] = None
    eps: Optional[float] = None
    dividend_yield: Optional[float] = Field(None, ge=0)
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = Field(None, gt=0)
    fifty_two_week_low: Optional[float] = Field(None, gt=0)


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
