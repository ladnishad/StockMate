"""Watchlist data models for user-managed stock tracking."""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class WatchlistItem(BaseModel):
    """A single item in user's watchlist."""

    symbol: str = Field(..., description="Stock ticker symbol")
    added_at: datetime = Field(default_factory=datetime.utcnow, description="When added to watchlist")
    notes: Optional[str] = Field(None, description="User notes about this stock")
    alerts_enabled: bool = Field(False, description="Whether alerts are enabled for this stock")

    # Scanner metadata (populated when added from scanner)
    scanner_source: Optional[str] = Field(
        None, description="Scanner that flagged this stock (e.g., 'Day Trade Scanner')"
    )
    scanner_reason: Optional[str] = Field(
        None, description="Why scanner flagged this stock (e.g., 'Breakout Setup')"
    )

    # Live data (populated on fetch)
    current_price: Optional[float] = Field(None, description="Current stock price")
    change: Optional[float] = Field(None, description="Price change")
    change_pct: Optional[float] = Field(None, description="Percentage change")
    score: Optional[float] = Field(None, description="Analysis score (0-100)")
    recommendation: Optional[str] = Field(None, description="BUY or NO_BUY")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "added_at": "2025-01-10T10:30:00Z",
                "notes": "Strong fundamentals",
                "alerts_enabled": True,
                "scanner_source": "Day Trade Scanner",
                "scanner_reason": "Breakout Setup",
                "current_price": 175.50,
                "change": 2.35,
                "change_pct": 1.36,
                "score": 78.5,
                "recommendation": "BUY",
            }
        }


class WatchlistResponse(BaseModel):
    """Response for user watchlist endpoint."""

    user_id: str = Field(..., description="User identifier")
    items: List[WatchlistItem] = Field(default_factory=list, description="Watchlist items")
    count: int = Field(..., description="Number of items in watchlist")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "default",
                "items": [],
                "count": 0,
                "last_updated": "2025-01-10T10:30:00Z",
            }
        }


class SearchResult(BaseModel):
    """Ticker search result."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    exchange: str = Field(..., description="Exchange (e.g., NASDAQ, NYSE)")
    asset_type: str = Field(..., description="Asset type (stock, etf)")
    source: Optional[str] = Field(None, description="Data source: 'alpaca' or 'fallback'")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NASDAQ",
                "asset_type": "stock",
                "source": "alpaca",
            }
        }


class PriceBarResponse(BaseModel):
    """Single price bar for charting."""

    timestamp: str = Field(..., description="Bar timestamp")
    open: float = Field(..., description="Open price")
    high: float = Field(..., description="High price")
    low: float = Field(..., description="Low price")
    close: float = Field(..., description="Close price")
    volume: int = Field(..., description="Volume")


class StockDetailResponse(BaseModel):
    """Comprehensive stock detail for detail page."""

    symbol: str = Field(..., description="Stock ticker symbol")
    name: str = Field(..., description="Company name")
    current_price: float = Field(..., description="Current price")
    change: float = Field(..., description="Price change")
    change_pct: float = Field(..., description="Percentage change")

    # Key statistics
    open_price: Optional[float] = Field(None, description="Today's open price")
    high_price: Optional[float] = Field(None, description="Today's high price")
    low_price: Optional[float] = Field(None, description="Today's low price")
    volume: Optional[int] = Field(None, description="Today's volume")
    fifty_two_week_high: Optional[float] = Field(None, description="52-week high")
    fifty_two_week_low: Optional[float] = Field(None, description="52-week low")
    avg_volume: Optional[int] = Field(None, description="30-day average volume")

    # Analysis data
    score: float = Field(..., description="Analysis score (0-100)")
    recommendation: str = Field(..., description="BUY or NO_BUY")
    reasoning: str = Field(..., description="Analysis reasoning summary")
    reasons: List[str] = Field(default_factory=list, description="Key reasons for recommendation")

    # Trade plan (if BUY)
    trade_plan: Optional[dict] = Field(None, description="Trade plan with entry/stop/targets")

    # Multi-timeframe chart data
    bars_1d: List[PriceBarResponse] = Field(default_factory=list, description="Daily bars")
    bars_1h: List[PriceBarResponse] = Field(default_factory=list, description="Hourly bars")
    bars_15m: List[PriceBarResponse] = Field(default_factory=list, description="15-minute bars")

    # Key levels for chart overlays
    support_levels: List[float] = Field(default_factory=list, description="Support levels")
    resistance_levels: List[float] = Field(default_factory=list, description="Resistance levels")

    # Indicator data for overlays
    ema_9: List[float] = Field(default_factory=list, description="9-period EMA values")
    ema_21: List[float] = Field(default_factory=list, description="21-period EMA values")
    vwap: Optional[float] = Field(None, description="Current VWAP")

    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Response timestamp")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "name": "Apple Inc.",
                "current_price": 175.50,
                "change": 2.35,
                "change_pct": 1.36,
                "open_price": 173.50,
                "high_price": 176.25,
                "low_price": 172.80,
                "volume": 45000000,
                "fifty_two_week_high": 199.62,
                "fifty_two_week_low": 164.08,
                "avg_volume": 52000000,
                "score": 78.5,
                "recommendation": "BUY",
                "reasoning": "Strong bullish momentum with volume confirmation",
                "reasons": ["Price above key EMAs", "MACD bullish crossover", "RSI healthy"],
                "trade_plan": {
                    "entry_price": 175.50,
                    "stop_loss": 172.00,
                    "target_1": 180.00,
                    "target_2": 185.00,
                    "target_3": 190.00,
                },
                "bars_1d": [],
                "bars_1h": [],
                "bars_15m": [],
                "support_levels": [170.00, 165.00],
                "resistance_levels": [180.00, 185.00],
                "ema_9": [],
                "ema_21": [],
                "vwap": 174.25,
                "timestamp": "2025-01-10T10:30:00Z",
            }
        }
