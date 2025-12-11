"""Trader profile definitions and configurations.

This module defines the data models for trader profiles, which customize
the analysis behavior based on different trading styles:
- Day Trader: Intraday focus, tight stops, VWAP/volume emphasis
- Swing Trader: Multi-day holds, Fibonacci levels, structure-based stops
- Position Trader: Multi-week holds, trend-following, momentum focus
- Long-Term Investor: Multi-month holds, fundamentals, wide stops
"""

from enum import Enum
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TraderProfileType(str, Enum):
    """Enumeration of available trader profiles."""

    DAY_TRADER = "day_trader"
    SWING_TRADER = "swing_trader"
    POSITION_TRADER = "position_trader"
    LONG_TERM_INVESTOR = "long_term_investor"


class TimeframeConfig(BaseModel):
    """Timeframe configuration for analysis.

    Attributes:
        primary: Primary timeframe for analysis (1d, 1h, 15m)
        confirmation: Timeframe for trend confirmation
        entry: Timeframe for precise entry timing
    """

    primary: str = Field(..., description="Primary timeframe for analysis (1d, 1h, 15m)")
    confirmation: Optional[str] = Field(None, description="Confirmation timeframe")
    entry: Optional[str] = Field(None, description="Entry timeframe for precise entry")


class RiskConfig(BaseModel):
    """Risk management configuration.

    Attributes:
        risk_percentage: Percentage of account to risk per trade (0.1-5.0)
        stop_method: Method for calculating stop loss (atr, structure, percentage)
        atr_multiplier: Multiplier for ATR-based stops (1.0-5.0)
        max_position_percent: Maximum position size as % of account (5-50)
    """

    risk_percentage: float = Field(
        1.0, ge=0.1, le=5.0, description="% of account to risk per trade"
    )
    stop_method: Literal["atr", "structure", "percentage"] = Field(
        "atr", description="Method for calculating stop loss"
    )
    atr_multiplier: float = Field(
        2.0, ge=1.0, le=5.0, description="ATR multiplier for stops"
    )
    max_position_percent: float = Field(
        20.0, ge=5.0, le=50.0, description="Max % of account per position"
    )


class TargetConfig(BaseModel):
    """Target calculation configuration.

    Attributes:
        method: Method for calculating price targets
        rr_ratios: Risk/reward ratios for targets when using rr_ratio method
        use_fibonacci_extensions: Whether to use Fibonacci extensions for targets
        validate_against_resistance: Whether to validate targets against resistance
    """

    method: Literal["rr_ratio", "structure", "fibonacci"] = Field(
        "rr_ratio", description="Method for calculating targets"
    )
    rr_ratios: List[float] = Field(
        default=[1.5, 2.5, 3.5], description="R:R ratios for targets"
    )
    use_fibonacci_extensions: bool = Field(
        False, description="Use Fibonacci extensions for targets"
    )
    validate_against_resistance: bool = Field(
        True, description="Validate targets against resistance levels"
    )


class ScoringWeights(BaseModel):
    """Scoring weight configuration for each factor.

    All weights should sum to approximately 100 for proper scoring.
    Each weight represents the importance of that factor in the final score.
    """

    # Existing factors
    sentiment: float = Field(10.0, ge=0, le=30, description="Sentiment weight")
    ema_trend: float = Field(12.0, ge=0, le=30, description="EMA trend weight")
    rsi: float = Field(8.0, ge=0, le=20, description="RSI weight")
    vwap: float = Field(8.0, ge=0, le=25, description="VWAP weight")
    volume: float = Field(10.0, ge=0, le=25, description="Volume weight")
    macd: float = Field(10.0, ge=0, le=25, description="MACD weight")
    bollinger: float = Field(8.0, ge=0, le=20, description="Bollinger Bands weight")
    multi_tf: float = Field(8.0, ge=0, le=20, description="Multi-timeframe weight")
    support_resistance: float = Field(
        5.0, ge=0, le=15, description="Support/Resistance weight"
    )
    divergence: float = Field(8.0, ge=0, le=20, description="Divergence weight")
    volume_profile: float = Field(5.0, ge=0, le=15, description="Volume profile weight")
    chart_patterns: float = Field(8.0, ge=0, le=20, description="Chart patterns weight")

    # New indicators for enhanced swing trading
    fibonacci: float = Field(0.0, ge=0, le=15, description="Fibonacci levels weight")
    adx: float = Field(0.0, ge=0, le=15, description="ADX trend strength weight")
    stochastic: float = Field(0.0, ge=0, le=15, description="Stochastic weight")


class TraderProfile(BaseModel):
    """Complete trader profile configuration.

    A trader profile customizes the entire analysis pipeline:
    - Which timeframes to analyze
    - How to weight different technical factors
    - Risk management parameters
    - Target calculation methodology
    - Confidence thresholds

    Attributes:
        name: Human-readable profile name
        profile_type: Profile type enum value
        description: Description of the trading style
        timeframes: Timeframe configuration
        min_holding_period: Minimum expected holding period
        max_holding_period: Maximum expected holding period
        allowed_trade_types: Which trade types this profile generates
        risk: Risk management configuration
        targets: Target calculation configuration
        weights: Scoring weights for each factor
        buy_confidence_threshold: Minimum confidence for BUY recommendation
        rsi_overbought: RSI overbought threshold
        rsi_oversold: RSI oversold threshold
        adx_trend_threshold: ADX threshold for trending market
    """

    name: str = Field(..., description="Human-readable profile name")
    profile_type: TraderProfileType = Field(..., description="Profile type")
    description: str = Field(..., description="Description of the trading style")

    # Timeframe settings
    timeframes: TimeframeConfig = Field(..., description="Timeframe configuration")
    min_holding_period: str = Field(
        ..., description="Minimum holding period (e.g., '1h', '2d', '2w')"
    )
    max_holding_period: str = Field(
        ..., description="Maximum holding period (e.g., '1d', '2w', '3m')"
    )

    # Allowed trade types output
    allowed_trade_types: List[Literal["day", "swing", "long"]] = Field(
        ..., description="Trade types this profile can generate"
    )

    # Risk settings
    risk: RiskConfig = Field(..., description="Risk management configuration")

    # Target settings
    targets: TargetConfig = Field(..., description="Target calculation configuration")

    # Scoring weights
    weights: ScoringWeights = Field(..., description="Scoring weights for factors")

    # Thresholds
    buy_confidence_threshold: float = Field(
        65.0, ge=50, le=90, description="Minimum confidence for BUY"
    )

    # Indicator thresholds (profile-specific)
    rsi_overbought: float = Field(70.0, ge=60, le=90, description="RSI overbought level")
    rsi_oversold: float = Field(30.0, ge=10, le=40, description="RSI oversold level")
    adx_trend_threshold: float = Field(
        25.0, ge=15, le=40, description="ADX threshold for trending"
    )
