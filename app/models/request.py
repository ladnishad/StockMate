"""Request models for API endpoints."""

from typing import Optional, Literal
from pydantic import BaseModel, Field, field_validator


class AnalysisRequest(BaseModel):
    """Request model for stock analysis endpoint.

    Attributes:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        account_size: Total account size in dollars
        use_ai: Whether to use AI-enhanced analysis (future use)
        trader_profile: Optional trader profile for customized analysis
    """

    symbol: str = Field(
        ...,
        description="Stock ticker symbol",
        min_length=1,
        max_length=10,
        examples=["AAPL", "TSLA", "GOOGL"]
    )
    account_size: float = Field(
        ...,
        description="Total account size in dollars",
        gt=0,
        examples=[10000.0, 50000.0]
    )
    use_ai: bool = Field(
        default=False,
        description="Whether to use AI-enhanced analysis"
    )
    trader_profile: Optional[Literal[
        "day_trader", "swing_trader", "position_trader", "long_term_investor"
    ]] = Field(
        default=None,
        description="Trader profile for customized analysis. Options: day_trader, swing_trader, position_trader, long_term_investor"
    )

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, v: str) -> str:
        """Validate and normalize symbol."""
        return v.strip().upper()

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "symbol": "AAPL",
                    "account_size": 10000.0,
                    "use_ai": False,
                    "trader_profile": None
                },
                {
                    "symbol": "TSLA",
                    "account_size": 25000.0,
                    "use_ai": False,
                    "trader_profile": "swing_trader"
                }
            ]
        }
    }
