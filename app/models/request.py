"""Request models for API endpoints."""

from pydantic import BaseModel, Field, field_validator


class AnalysisRequest(BaseModel):
    """Request model for stock analysis endpoint.

    The analysis is fully automated - the agent determines the optimal
    trade style (day/swing/position) based on technical analysis.

    Attributes:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        account_size: Total account size in dollars
        use_ai: Whether to use AI-enhanced analysis (future use)
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
                    "use_ai": False
                },
                {
                    "symbol": "TSLA",
                    "account_size": 25000.0,
                    "use_ai": False
                }
            ]
        }
    }
