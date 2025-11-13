"""Response models for API endpoints."""

from typing import Optional, Literal
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
