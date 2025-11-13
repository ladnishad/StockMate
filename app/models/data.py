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
