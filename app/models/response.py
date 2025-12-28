"""Response models for API endpoints."""

from typing import Optional, Literal, List, Dict, Tuple
from pydantic import BaseModel, Field


class TradePlan(BaseModel):
    """Trade plan details when recommendation is BUY.

    Attributes:
        trade_type: Type of trade (day, swing, or long)
        entry_price: Recommended entry price
        stop_loss: Stop loss price
        target_1: First price target
        target_2: Second price target (optional)
        target_3: Third price target (optional)
        position_size: Number of shares to buy
        risk_amount: Dollar amount at risk
        risk_percentage: Percentage of account at risk
    """

    trade_type: Literal["day", "swing", "long"] = Field(
        ...,
        description="Type of trade: day (intraday), swing (days to weeks), long (months+)"
    )
    entry_price: float = Field(
        ...,
        description="Recommended entry price",
        gt=0
    )
    stop_loss: float = Field(
        ...,
        description="Stop loss price",
        gt=0
    )
    target_1: float = Field(
        ...,
        description="First price target",
        gt=0
    )
    target_2: Optional[float] = Field(
        None,
        description="Second price target",
        gt=0
    )
    target_3: Optional[float] = Field(
        None,
        description="Third price target",
        gt=0
    )
    position_size: int = Field(
        ...,
        description="Number of shares to buy",
        ge=1
    )
    risk_amount: float = Field(
        ...,
        description="Dollar amount at risk (account_size * risk_percentage)",
        ge=0
    )
    risk_percentage: float = Field(
        ...,
        description="Percentage of account at risk (typically 1-2%)",
        ge=0,
        le=100
    )


class AnalysisResponse(BaseModel):
    """Response model for stock analysis endpoint.

    Attributes:
        symbol: Stock ticker symbol analyzed
        recommendation: Trading recommendation (BUY or NO_BUY)
        confidence: Confidence score (0-100)
        trade_plan: Trade plan details if recommendation is BUY
        reasoning: Brief explanation of the recommendation
        timestamp: ISO timestamp of analysis
    """

    symbol: str = Field(
        ...,
        description="Stock ticker symbol analyzed"
    )
    recommendation: Literal["BUY", "NO_BUY"] = Field(
        ...,
        description="Trading recommendation"
    )
    confidence: float = Field(
        ...,
        description="Confidence score from 0 to 100",
        ge=0,
        le=100
    )
    current_price: Optional[float] = Field(
        None,
        description="Current stock price at time of analysis"
    )
    trade_plan: Optional[TradePlan] = Field(
        None,
        description="Trade plan details (only present if recommendation is BUY)"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the recommendation"
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp of when analysis was performed"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "AAPL",
                    "recommendation": "BUY",
                    "confidence": 78.5,
                    "trade_plan": {
                        "trade_type": "swing",
                        "entry_price": 175.50,
                        "stop_loss": 172.00,
                        "target_1": 180.00,
                        "target_2": 185.00,
                        "target_3": 190.00,
                        "position_size": 28,
                        "risk_amount": 100.00,
                        "risk_percentage": 1.0
                    },
                    "reasoning": "Strong bullish momentum with RSI above 50, price above key EMAs, and positive sentiment.",
                    "timestamp": "2025-11-13T10:30:00Z"
                }
            ]
        }
    }


# ============================================================================
# Enhanced Smart Analysis Models
# ============================================================================


class TradeStyleRecommendation(BaseModel):
    """Agent-determined optimal trade style based on setup analysis."""

    recommended_style: Literal["day", "swing", "position"] = Field(
        ...,
        description="Optimal trade style: day (intraday), swing (days to weeks), position (weeks to months)"
    )
    reasoning: str = Field(
        ...,
        description="Why this trade style fits the current setup"
    )
    holding_period: str = Field(
        ...,
        description="Expected holding period, e.g., '2-5 days', '1-3 weeks'"
    )


class ScenarioPath(BaseModel):
    """One possible price path scenario with probability assessment."""

    scenario: Literal["bullish", "bearish", "sideways"] = Field(
        ...,
        description="Type of scenario"
    )
    probability: int = Field(
        ...,
        description="Probability percentage (0-100)",
        ge=0,
        le=100
    )
    description: str = Field(
        ...,
        description="Detailed description of this scenario"
    )
    price_target: Optional[float] = Field(
        None,
        description="Expected price target for this scenario"
    )
    key_trigger: str = Field(
        ...,
        description="What price action or event would confirm this scenario"
    )


class ChartAnnotation(BaseModel):
    """Annotation for chart visualization."""

    type: Literal["level", "zone", "pattern", "trend_line", "arrow"] = Field(
        ...,
        description="Type of chart annotation"
    )
    price: Optional[float] = Field(
        None,
        description="Price level for single-line annotations"
    )
    price_high: Optional[float] = Field(
        None,
        description="Upper price for zone annotations"
    )
    price_low: Optional[float] = Field(
        None,
        description="Lower price for zone annotations"
    )
    label: str = Field(
        ...,
        description="Short label to display on chart"
    )
    color: Literal["green", "red", "blue", "yellow", "orange", "purple", "gray", "teal", "cyan", "pink", "white", "black"] = Field(
        ...,
        description="Color for the annotation"
    )
    description: str = Field(
        ...,
        description="Detailed explanation shown when annotation is tapped"
    )


class PriceTarget(BaseModel):
    """Price target with reasoning."""

    price: float = Field(
        ...,
        description="Target price level",
        gt=0
    )
    reasoning: str = Field(
        ...,
        description="Why this level is a target (e.g., 'Previous resistance', '1.618 Fib extension')"
    )


class EducationalContent(BaseModel):
    """Comprehensive educational content for hand-holding user guidance."""

    setup_explanation: str = Field(
        ...,
        description="Plain English explanation of what technical setup/pattern is forming"
    )
    level_explanations: Dict[str, str] = Field(
        default_factory=dict,
        description="Map of price level to explanation of why it matters"
    )
    what_to_watch: List[str] = Field(
        default_factory=list,
        description="List of specific price action cues to monitor"
    )
    scenarios: List[ScenarioPath] = Field(
        default_factory=list,
        description="Possible future price paths with probabilities"
    )
    risk_warnings: List[str] = Field(
        default_factory=list,
        description="Important risks and warnings the trader should be aware of"
    )
    chart_annotations: List[ChartAnnotation] = Field(
        default_factory=list,
        description="Annotations to display on the chart for visualization"
    )


class EnhancedTradePlan(BaseModel):
    """Complete trade plan with educational hand-holding content."""

    # Trade style determination
    trade_style: TradeStyleRecommendation = Field(
        ...,
        description="Agent-determined optimal trade style"
    )

    # Core analysis
    bias: Literal["bullish", "bearish", "neutral"] = Field(
        ...,
        description="Overall directional bias"
    )
    thesis: str = Field(
        ...,
        description="1-2 sentence thesis explaining the trade opportunity"
    )
    confidence: int = Field(
        ...,
        description="Confidence score (0-100)",
        ge=0,
        le=100
    )

    # Entry
    entry_zone_low: Optional[float] = Field(
        None,
        description="Lower bound of entry zone",
        gt=0
    )
    entry_zone_high: Optional[float] = Field(
        None,
        description="Upper bound of entry zone",
        gt=0
    )

    # Risk management
    stop_loss: Optional[float] = Field(
        None,
        description="Stop loss price",
        gt=0
    )
    stop_reasoning: str = Field(
        "",
        description="Why this stop level was chosen"
    )

    # Targets
    targets: List[PriceTarget] = Field(
        default_factory=list,
        description="Price targets with reasoning"
    )

    # Position sizing
    risk_reward: Optional[float] = Field(
        None,
        description="Risk/reward ratio"
    )
    position_size_pct: Optional[float] = Field(
        None,
        description="Recommended position size as percentage of account (1-5)",
        ge=0,
        le=100
    )

    # Key levels
    key_supports: List[float] = Field(
        default_factory=list,
        description="Key support levels"
    )
    key_resistances: List[float] = Field(
        default_factory=list,
        description="Key resistance levels"
    )

    # Invalidation
    invalidation_criteria: str = Field(
        "",
        description="What would invalidate this trade setup"
    )

    # Educational content (expandable section)
    educational: EducationalContent = Field(
        default_factory=EducationalContent,
        description="Educational content for the expandable 'Learn More' section"
    )


class SmartAnalysisResponse(BaseModel):
    """Response model for smart (profile-less) stock analysis."""

    symbol: str = Field(
        ...,
        description="Stock ticker symbol analyzed"
    )
    current_price: float = Field(
        ...,
        description="Current stock price at time of analysis",
        gt=0
    )
    recommendation: Literal["BUY", "NO_BUY"] = Field(
        ...,
        description="Trading recommendation"
    )
    trade_plan: EnhancedTradePlan = Field(
        ...,
        description="Complete trade plan with educational content"
    )
    timestamp: str = Field(
        ...,
        description="ISO timestamp of when analysis was performed"
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "AAPL",
                    "current_price": 178.50,
                    "recommendation": "BUY",
                    "trade_plan": {
                        "trade_style": {
                            "recommended_style": "swing",
                            "reasoning": "Bull flag pattern on daily chart with 3-5 day resolution expected",
                            "holding_period": "3-5 days"
                        },
                        "bias": "bullish",
                        "thesis": "AAPL forming a bull flag after strong earnings breakout, pullback to support offers low-risk entry.",
                        "confidence": 72,
                        "entry_zone_low": 176.00,
                        "entry_zone_high": 178.50,
                        "stop_loss": 173.50,
                        "stop_reasoning": "Below flag low and 21 EMA",
                        "targets": [
                            {"price": 185.00, "reasoning": "Previous high / measured move"},
                            {"price": 190.00, "reasoning": "1.618 Fibonacci extension"}
                        ],
                        "risk_reward": 2.6,
                        "position_size_pct": 3,
                        "key_supports": [176.00, 173.50, 170.00],
                        "key_resistances": [180.00, 185.00, 190.00],
                        "invalidation_criteria": "Close below $173.50 would break the flag structure",
                        "educational": {
                            "setup_explanation": "This stock is forming a classic bull flag pattern...",
                            "level_explanations": {
                                "176.00": "Strong support - tested 3 times this week with buying volume each time",
                                "185.00": "Major resistance - previous all-time high from October"
                            },
                            "what_to_watch": [
                                "Watch for a green candle close above $180 with above-average volume",
                                "RSI should stay above 50 to confirm bullish momentum"
                            ],
                            "scenarios": [
                                {
                                    "scenario": "bullish",
                                    "probability": 60,
                                    "description": "Price breaks above the flag and runs to $185-190",
                                    "price_target": 187.50,
                                    "key_trigger": "Daily close above $180 with volume > 1.5x average"
                                }
                            ],
                            "risk_warnings": [
                                "Overall market showing some weakness - watch SPY for confirmation",
                                "Tech sector rotation possible ahead of Fed meeting"
                            ],
                            "chart_annotations": []
                        }
                    },
                    "timestamp": "2025-12-14T10:30:00Z"
                }
            ]
        }
    }
