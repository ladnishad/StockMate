"""Scanner data models for market scanner feature."""

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Literal, Any
from datetime import datetime
from enum import Enum


class TradingStyle(str, Enum):
    """Trading style categories for scanner."""

    DAY = "day"
    SWING = "swing"
    POSITION = "position"


class PatternType(str, Enum):
    """Pattern types detected by scanner."""

    # Gap patterns
    GAP_UP = "gap_up"
    GAP_DOWN = "gap_down"
    GAP_FILL = "gap_fill"

    # Breakout patterns
    RESISTANCE_BREAKOUT = "resistance_breakout"
    VWAP_RECLAIM = "vwap_reclaim"
    RANGE_BREAKOUT = "range_breakout"
    CHANNEL_BREAKOUT = "channel_breakout"
    BASE_BREAKOUT = "base_breakout"

    # Reversal patterns
    OVERSOLD_BOUNCE = "oversold_bounce"
    PANIC_DIP_BUY = "panic_dip_buy"
    FAILED_BREAKDOWN = "failed_breakdown"
    SUPPORT_BOUNCE = "support_bounce"

    # Continuation patterns
    BULL_FLAG = "bull_flag"
    BEAR_FLAG = "bear_flag"
    PULLBACK_TO_SUPPORT = "pullback_to_support"
    FIB_RETRACEMENT = "fib_retracement"
    HIGHER_LOW = "higher_low"
    LOWER_HIGH = "lower_high"
    TREND_CONTINUATION = "trend_continuation"

    # Momentum patterns
    NEW_HOD = "new_hod"  # New high of day
    NEW_LOD = "new_lod"  # New low of day
    MOMENTUM_SURGE = "momentum_surge"

    # Long-term patterns
    GOLDEN_CROSS = "golden_cross"
    DEATH_CROSS = "death_cross"
    WEEKLY_BREAKOUT = "weekly_breakout"
    MAJOR_SUPPORT_TEST = "major_support_test"
    TRENDLINE_BOUNCE = "trendline_bounce"


class ConfidenceGrade(str, Enum):
    """Letter grade for confidence scores."""

    A_PLUS = "A+"
    A = "A"
    B_PLUS = "B+"
    B = "B"
    C = "C"


def score_to_grade(score: float) -> ConfidenceGrade:
    """Convert numeric score (0-100) to letter grade.

    Grade Mapping:
        85-100: A+
        75-84: A
        65-74: B+
        55-64: B
        45-54: C
        Below 45: Not shown (returns C as minimum)
    """
    if score >= 85:
        return ConfidenceGrade.A_PLUS
    elif score >= 75:
        return ConfidenceGrade.A
    elif score >= 65:
        return ConfidenceGrade.B_PLUS
    elif score >= 55:
        return ConfidenceGrade.B
    else:
        return ConfidenceGrade.C


class ScannerResult(BaseModel):
    """A single scanner result for a detected setup.

    Attributes:
        symbol: Stock ticker symbol
        style: Trading style (day, swing, position)
        confidence_grade: Letter grade (A+, A, B+, B, C)
        confidence_score: Internal 0-100 score for ranking
        current_price: Current stock price
        description: Template-generated description of the setup
        pattern_type: Type of pattern detected
        key_levels: Key price levels for the setup (support, resistance, etc.)
        detected_at: When the setup was first detected
        is_new: Whether this appeared in the latest scan cycle
        is_watching: Whether the stock is already in user's watchlist
        warnings: List of warnings (earnings, halted, etc.)
        volume_multiple: Volume compared to average (e.g., 2.1x)
        gap_pct: Gap percentage if applicable
        fib_level: Fibonacci level if applicable
    """

    symbol: str = Field(..., description="Stock ticker symbol")
    style: TradingStyle = Field(..., description="Trading style")
    confidence_grade: ConfidenceGrade = Field(..., description="Letter grade")
    confidence_score: float = Field(..., ge=0, le=100, description="Internal score for ranking")
    current_price: float = Field(..., gt=0, description="Current price")
    description: str = Field(..., description="Human-readable setup description")
    pattern_type: PatternType = Field(..., description="Pattern detected")
    key_levels: Dict[str, float] = Field(
        default_factory=dict,
        description="Key levels: support, resistance, entry, stop, target, etc."
    )
    detected_at: datetime = Field(default_factory=datetime.utcnow)
    is_new: bool = Field(default=True, description="Appeared in latest scan")
    is_watching: bool = Field(default=False, description="Already in watchlist")
    warnings: List[str] = Field(default_factory=list, description="Warning messages")

    # Additional context
    volume_multiple: Optional[float] = Field(None, description="Volume vs average")
    gap_pct: Optional[float] = Field(None, description="Gap percentage")
    fib_level: Optional[float] = Field(None, description="Fibonacci level (38.2, 50, 61.8)")
    rsi_value: Optional[float] = Field(None, description="RSI value")
    vwap: Optional[float] = Field(None, description="VWAP price")

    # Expiration tracking
    expires_at: Optional[datetime] = Field(None, description="When setup expires")

    class Config:
        json_schema_extra = {
            "example": {
                "symbol": "AAPL",
                "style": "day",
                "confidence_grade": "A+",
                "confidence_score": 87.5,
                "current_price": 185.50,
                "description": "Bull flag breakout: Testing $186 resistance with 2.1x volume",
                "pattern_type": "resistance_breakout",
                "key_levels": {
                    "support": 183.00,
                    "resistance": 186.00,
                    "entry": 185.50,
                    "stop": 183.00,
                    "target_1": 188.00,
                },
                "detected_at": "2025-01-10T10:30:00Z",
                "is_new": True,
                "is_watching": False,
                "warnings": [],
                "volume_multiple": 2.1,
            }
        }


class ScannerResponse(BaseModel):
    """Response for scanner endpoint containing results for a style.

    Attributes:
        style: Trading style for these results
        results: List of scanner results (max 10)
        scan_time: When the scan was performed
        next_scheduled_scan: When the next scheduled scan will run
        total_stocks_scanned: How many stocks were evaluated
    """

    style: TradingStyle = Field(..., description="Trading style")
    results: List[ScannerResult] = Field(default_factory=list, max_length=10)
    scan_time: datetime = Field(default_factory=datetime.utcnow)
    next_scheduled_scan: Optional[datetime] = Field(None)
    total_stocks_scanned: int = Field(default=0, ge=0)

    class Config:
        json_schema_extra = {
            "example": {
                "style": "day",
                "results": [],
                "scan_time": "2025-01-10T10:30:00Z",
                "next_scheduled_scan": "2025-01-10T12:00:00Z",
                "total_stocks_scanned": 1500,
            }
        }


class AllScannersResponse(BaseModel):
    """Response containing results from all three scanner styles.

    Attributes:
        day: Day trading scan results
        swing: Swing trading scan results
        position: Position trading scan results
        scan_time: When the scan was performed
        next_scheduled_scan: When the next scheduled scan will run
    """

    day: ScannerResponse = Field(..., description="Day trading results")
    swing: ScannerResponse = Field(..., description="Swing trading results")
    position: ScannerResponse = Field(..., description="Position trading results")
    scan_time: datetime = Field(default_factory=datetime.utcnow)
    next_scheduled_scan: Optional[datetime] = Field(None)


class ScannerStatusResponse(BaseModel):
    """Response for scanner status endpoint.

    Attributes:
        last_scan_time: When the last scan was performed
        next_scheduled_scan: When the next scheduled scan will run
        current_scan_name: Name of the current/last scan (pre_market, market_open, etc.)
        is_scanning: Whether a scan is currently running
        total_results: Number of results per style
    """

    last_scan_time: Optional[datetime] = Field(None)
    next_scheduled_scan: Optional[datetime] = Field(None)
    current_scan_name: Optional[str] = Field(None)
    is_scanning: bool = Field(default=False)
    total_results: Dict[str, int] = Field(
        default_factory=lambda: {"day": 0, "swing": 0, "position": 0}
    )

    class Config:
        json_schema_extra = {
            "example": {
                "last_scan_time": "2025-01-10T09:30:00Z",
                "next_scheduled_scan": "2025-01-10T12:00:00Z",
                "current_scan_name": "market_open",
                "is_scanning": False,
                "total_results": {"day": 10, "swing": 8, "position": 5},
            }
        }


class AddFromScannerRequest(BaseModel):
    """Request to add a scanned stock to watchlist.

    Attributes:
        scanner_source: Which scanner the stock came from
        scanner_reason: Why the scanner flagged this stock
    """

    scanner_source: str = Field(..., description="e.g., 'Day Trade Scanner'")
    scanner_reason: str = Field(..., description="e.g., 'Breakout Setup'")

    class Config:
        json_schema_extra = {
            "example": {
                "scanner_source": "Day Trade Scanner",
                "scanner_reason": "Breakout Setup",
            }
        }


# ============================================================================
# Internal models for scoring and analysis
# ============================================================================


class ConfidenceScoreComponents(BaseModel):
    """Breakdown of confidence score components.

    Used internally to calculate and debug confidence scores.

    Components and weights:
        - pattern_clarity: 25% - How clean/textbook the pattern is
        - volume_confirmation: 25% - Multiple of average volume
        - indicator_alignment: 20% - RSI, MACD, EMA agreement
        - risk_reward_ratio: 15% - Calculated R:R from key levels
        - fib_level_proximity: 15% - Bonus if price at key Fib level
    """

    pattern_clarity: float = Field(default=0, ge=0, le=100)
    volume_confirmation: float = Field(default=0, ge=0, le=100)
    indicator_alignment: float = Field(default=0, ge=0, le=100)
    risk_reward_ratio: float = Field(default=0, ge=0, le=100)
    fib_level_proximity: float = Field(default=0, ge=0, le=100)

    def calculate_total(self) -> float:
        """Calculate weighted total score."""
        return (
            self.pattern_clarity * 0.25 +
            self.volume_confirmation * 0.25 +
            self.indicator_alignment * 0.20 +
            self.risk_reward_ratio * 0.15 +
            self.fib_level_proximity * 0.15
        )


class ScanCandidate(BaseModel):
    """Internal model for a potential scan candidate before scoring.

    Used during the scan process before final filtering and ranking.
    """

    symbol: str
    current_price: float
    style: TradingStyle
    patterns_detected: List[PatternType] = Field(default_factory=list)
    score_components: Optional[ConfidenceScoreComponents] = None
    raw_data: Dict[str, Any] = Field(default_factory=dict)
    avg_volume: Optional[float] = None
    current_volume: Optional[float] = None
    earnings_days_until: Optional[int] = None
    is_halted: bool = False
    exchange: Optional[str] = None


class ScanSession(BaseModel):
    """Metadata for a scan session.

    Tracks information about a single scan run.
    """

    session_id: str = Field(..., description="Unique session identifier")
    scan_name: str = Field(..., description="e.g., 'pre_market', 'market_open'")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    stocks_scanned: int = Field(default=0)
    results_found: Dict[str, int] = Field(
        default_factory=lambda: {"day": 0, "swing": 0, "position": 0}
    )
    errors: List[str] = Field(default_factory=list)

    def mark_complete(self):
        """Mark the session as complete."""
        self.completed_at = datetime.utcnow()


# ============================================================================
# Volume thresholds per trading style
# ============================================================================

VOLUME_THRESHOLDS = {
    TradingStyle.DAY: 500_000,      # 500K minimum daily volume
    TradingStyle.SWING: 200_000,    # 200K minimum daily volume
    TradingStyle.POSITION: 100_000, # 100K minimum daily volume
}

# High confidence volume multiples
VOLUME_CONFIDENCE_MULTIPLIER = {
    TradingStyle.DAY: 2.0,      # 2x+ for high confidence
    TradingStyle.SWING: 1.5,    # 1.5x+ for high confidence
    TradingStyle.POSITION: 1.0, # Above average for high confidence
}

# Minimum price filter
MIN_PRICE = 1.00

# Setup expiration times
EXPIRATION_HOURS = {
    TradingStyle.DAY: 8,        # Same day (market hours)
    TradingStyle.SWING: 72,     # 3 days
    TradingStyle.POSITION: 168, # 7 days
}
