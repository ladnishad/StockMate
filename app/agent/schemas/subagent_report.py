"""SubAgentReport schema - structured output from each trade-style sub-agent.

Each sub-agent (Day/Swing/Position) returns this standardized report
after analyzing a stock through their specialized lens.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# Type aliases for clarity
TradeStyleLiteral = Literal["day", "swing", "position"]
BiasLiteral = Literal["bullish", "bearish", "neutral"]
TrendQualityLiteral = Literal["clean", "moderate", "choppy"]


class VisionAnalysisResult(BaseModel):
    """Result from Claude Vision chart analysis.

    Each sub-agent analyzes its own chart (5-min/daily/weekly)
    and returns visual pattern recognition results.
    """

    trend_quality: TrendQualityLiteral = Field(
        description="How clean is the trend structure? Clean trends have higher probability setups."
    )
    visual_patterns: List[str] = Field(
        default_factory=list,
        description="Patterns detected visually: 'bull flag', 'ascending triangle', 'double bottom', etc."
    )
    candlestick_patterns: List[str] = Field(
        default_factory=list,
        description="Candlestick patterns: 'engulfing', 'hammer', 'doji', 'three white soldiers', etc."
    )
    ema_structure: str = Field(
        default="",
        description="EMA alignment description: 'price above all EMAs', 'EMAs stacked bullish', etc."
    )
    volume_confirmation: str = Field(
        default="",
        description="Volume analysis: 'increasing on breakout', 'declining on pullback', etc."
    )
    warning_signs: List[str] = Field(
        default_factory=list,
        description="Potential issues: 'divergence forming', 'exhaustion candle', 'failed breakout', etc."
    )
    confidence_modifier: int = Field(
        default=0,
        ge=-20,
        le=20,
        description="Modifier to base confidence based on visual clarity. -20 to +20."
    )
    summary: str = Field(
        default="",
        description="One-sentence summary of visual analysis findings."
    )


class PriceTargetWithReasoning(BaseModel):
    """A price target with explanation of why this level matters."""

    price: float = Field(gt=0, description="Target price level")
    reasoning: str = Field(
        description="Why this target: 'previous resistance', 'measured move', 'Fibonacci 1.618', etc."
    )


class SubAgentReport(BaseModel):
    """Structured report from a trade-style sub-agent.

    Each sub-agent (day-trade-analyzer, swing-trade-analyzer, position-trade-analyzer)
    returns this report after analyzing a stock through their specialized lens.
    """

    # Metadata
    trade_style: TradeStyleLiteral = Field(
        description="The trade style this agent specializes in."
    )
    symbol: str = Field(description="Stock ticker symbol analyzed.")
    analysis_timestamp: str = Field(description="ISO timestamp when analysis was performed.")

    # Core Decision
    suitable: bool = Field(
        description="Is this a good setup for THIS trade style? True = viable setup exists."
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score 0-100 in this setup. Higher = more conviction."
    )

    # Trade Direction
    bias: BiasLiteral = Field(
        description="Directional bias: bullish (long), bearish (short), or neutral (no clear direction)."
    )
    thesis: str = Field(
        max_length=500,
        description="2-3 sentence explanation of why this trade works for THIS style."
    )

    # Vision Analysis (REQUIRED - each agent analyzes its own chart)
    vision_analysis: VisionAnalysisResult = Field(
        description="Results from Claude Vision analysis of the timeframe-specific chart."
    )

    # Entry Zone
    entry_zone_low: Optional[float] = Field(
        default=None,
        description="Lower bound of entry zone. None if no setup."
    )
    entry_zone_high: Optional[float] = Field(
        default=None,
        description="Upper bound of entry zone. None if no setup."
    )
    entry_reasoning: str = Field(
        default="",
        description="Why this entry zone: 'support retest', 'breakout above resistance', etc."
    )

    # Risk Management
    stop_loss: Optional[float] = Field(
        default=None,
        description="Stop loss price. Below entry for longs, above for shorts."
    )
    stop_reasoning: str = Field(
        default="",
        description="Why this stop: 'below swing low', 'below support', '1.5 ATR', etc."
    )

    # Targets
    targets: List[PriceTargetWithReasoning] = Field(
        default_factory=list,
        description="List of price targets with reasoning. Usually 1-3 targets."
    )

    # Risk/Reward Metrics
    risk_reward: Optional[float] = Field(
        default=None,
        description="Risk/reward ratio to first target. 2.0 = 2:1 R:R."
    )
    position_size_pct: Optional[float] = Field(
        default=None,
        ge=0,
        le=5,
        description="Suggested position size as % of account. Usually 1-3%."
    )

    # Holding Period
    holding_period: str = Field(
        default="",
        description="Expected holding period: '1-3 hours', '3-5 days', '2-4 weeks', etc."
    )

    # Key Levels
    key_supports: List[float] = Field(
        default_factory=list,
        description="Key support levels to watch."
    )
    key_resistances: List[float] = Field(
        default_factory=list,
        description="Key resistance levels to watch."
    )

    # Invalidation
    invalidation_criteria: str = Field(
        default="",
        description="What would invalidate this trade: 'close below $150', 'break of trend', etc."
    )

    # Position Awareness
    position_aligned: bool = Field(
        default=True,
        description="Does this recommendation align with user's existing position (if any)?"
    )
    position_recommendation: Optional[str] = Field(
        default=None,
        description="If user has position: 'hold', 'add', 'trim', 'exit', or None if no position."
    )

    # Educational Content
    setup_explanation: str = Field(
        default="",
        description="Plain English explanation of the setup for beginners."
    )
    what_to_watch: List[str] = Field(
        default_factory=list,
        description="Specific triggers to watch: 'green close above $155', 'volume spike', etc."
    )
    risk_warnings: List[str] = Field(
        default_factory=list,
        description="Honest assessment of risks: 'earnings in 3 days', 'low volume', etc."
    )

    # Technical Context
    atr_percent: Optional[float] = Field(
        default=None,
        description="ATR as percentage of price. Day: >3%, Swing: 1-3%, Position: <1.5%."
    )
    technical_summary: str = Field(
        default="",
        description="Brief summary of key technicals: RSI, MACD, EMA alignment, volume."
    )

    def has_valid_setup(self) -> bool:
        """Check if this report has a valid tradeable setup."""
        return (
            self.suitable
            and self.entry_zone_low is not None
            and self.entry_zone_high is not None
            and self.stop_loss is not None
            and len(self.targets) > 0
            and self.confidence >= 50
        )

    def get_risk_per_share(self) -> Optional[float]:
        """Calculate risk per share from entry to stop."""
        if self.entry_zone_low and self.stop_loss:
            # Use midpoint of entry zone
            entry_mid = (self.entry_zone_low + self.entry_zone_high) / 2 if self.entry_zone_high else self.entry_zone_low
            return abs(entry_mid - self.stop_loss)
        return None
