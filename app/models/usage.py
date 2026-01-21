"""Usage tracking models and pricing configuration.

Tracks AI API usage per user with cost calculations for billing and analytics.
"""

from datetime import datetime
from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field


class ModelProvider(str, Enum):
    """Supported AI model providers."""
    CLAUDE = "claude"
    GROK = "grok"


class OperationType(str, Enum):
    """Types of AI operations for tracking."""
    PLAN_GENERATION = "plan_generation"
    PLAN_EVALUATION = "plan_evaluation"
    CHAT = "chat"
    IMAGE_ANALYSIS = "image_analysis"
    SEARCH = "search"
    ORCHESTRATOR = "orchestrator"
    SUBAGENT = "subagent"


# =============================================================================
# PRICING CONFIGURATION (per 1M tokens)
# =============================================================================

# Claude Pricing (Anthropic) - as of Jan 2025
# https://platform.claude.com/docs/en/about-claude/pricing
CLAUDE_PRICING = {
    # Claude 4.5 Series (Latest)
    "claude-opus-4-5-20251101": {"input": 5.00, "output": 25.00},
    "claude-sonnet-4-5-20251101": {"input": 3.00, "output": 15.00},
    "claude-haiku-4-5-20251101": {"input": 1.00, "output": 5.00},
    # Claude 4 Series (Sonnet 4)
    "claude-sonnet-4-20250514": {"input": 3.00, "output": 15.00},
    # Claude 3.5 Series
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 1.00, "output": 5.00},
    # Claude 3 Series (Budget)
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
}

# Grok Pricing (xAI) - as of Jan 2025
# https://docs.x.ai/docs/models
# Note: Exact per-token pricing not publicly disclosed in docs
# These are estimated based on competitive analysis and API behavior
GROK_PRICING = {
    # Grok 4 Series
    "grok-4": {"input": 3.00, "output": 15.00},  # Flagship model
    "grok-4-1-fast": {"input": 1.00, "output": 5.00},  # Fast variant
    # Grok 3 Series (Legacy)
    "grok-3": {"input": 2.00, "output": 10.00},
    "grok-3-mini": {"input": 0.50, "output": 2.50},
}

# Tool call pricing (Grok specific)
# $5 per 1,000 calls for Web Search, X Search, Code Execution
GROK_TOOL_PRICING = {
    "web_search": 0.005,  # $5/1000 = $0.005 per call
    "x_search": 0.005,
    "news_search": 0.005,
    "code_execution": 0.005,
    "document_search": 0.005,
    "collections_search": 0.0025,  # $2.50/1000
}


# =============================================================================
# USAGE MODELS
# =============================================================================

class UsageRecord(BaseModel):
    """Single API usage record."""

    id: str = Field(..., description="Unique record ID")
    user_id: str = Field(..., description="User who made the request")
    provider: ModelProvider = Field(..., description="AI provider used")
    model: str = Field(..., description="Model ID used")
    operation_type: OperationType = Field(..., description="Type of operation")

    # Token counts
    input_tokens: int = Field(0, description="Input/prompt tokens")
    output_tokens: int = Field(0, description="Output/completion tokens")
    total_tokens: int = Field(0, description="Total tokens used")

    # Cost calculation
    estimated_cost: float = Field(0.0, description="Estimated cost in USD")

    # Tool usage (for Grok)
    tool_calls: int = Field(0, description="Number of tool calls (web search, etc.)")
    tool_cost: float = Field(0.0, description="Cost of tool calls")

    # Context
    symbol: Optional[str] = Field(None, description="Stock symbol if applicable")
    endpoint: Optional[str] = Field(None, description="API endpoint called")

    # Timestamps
    created_at: str = Field(..., description="When the request was made")

    class Config:
        from_attributes = True


class UsageSummary(BaseModel):
    """Aggregated usage summary for a user or time period."""

    user_id: Optional[str] = Field(None, description="User ID (None for all users)")
    period_start: str = Field(..., description="Start of the summary period")
    period_end: str = Field(..., description="End of the summary period")

    # Totals by provider
    claude_requests: int = Field(0)
    claude_input_tokens: int = Field(0)
    claude_output_tokens: int = Field(0)
    claude_cost: float = Field(0.0)

    grok_requests: int = Field(0)
    grok_input_tokens: int = Field(0)
    grok_output_tokens: int = Field(0)
    grok_cost: float = Field(0.0)
    grok_tool_calls: int = Field(0)
    grok_tool_cost: float = Field(0.0)

    # Grand totals
    total_requests: int = Field(0)
    total_tokens: int = Field(0)
    total_cost: float = Field(0.0)


class UserUsageSummary(BaseModel):
    """Usage summary for a single user (for admin dashboard)."""

    user_id: str
    email: Optional[str] = None
    total_requests: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # Breakdown by provider
    claude_cost: float = 0.0
    grok_cost: float = 0.0

    # Most recent activity
    last_request_at: Optional[str] = None

    # By operation type - counts
    plan_generations: int = 0
    chat_requests: int = 0
    evaluations: int = 0
    orchestrator_calls: int = 0
    subagent_calls: int = 0
    image_analyses: int = 0

    # By operation type - costs
    plan_generation_cost: float = 0.0
    chat_cost: float = 0.0
    evaluation_cost: float = 0.0
    orchestrator_cost: float = 0.0
    subagent_cost: float = 0.0
    image_analysis_cost: float = 0.0


class OperationTypeBreakdown(BaseModel):
    """Breakdown of usage by operation type."""

    operation_type: str
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_cost_per_request: float = 0.0


class UsageByOperationResponse(BaseModel):
    """Response model for usage by operation type."""

    user_id: Optional[str] = None
    period_start: str
    period_end: str
    breakdowns: List[OperationTypeBreakdown] = Field(default_factory=list)
    total_cost: float = 0.0


class UsageResponse(BaseModel):
    """Response model for usage API endpoints."""

    records: List[UsageRecord] = Field(default_factory=list)
    total_count: int = Field(0)
    page: int = Field(1)
    page_size: int = Field(50)
    has_more: bool = Field(False)


class UsageCostResponse(BaseModel):
    """Response model for cost breakdown endpoint."""

    summary: UsageSummary
    users: List[UserUsageSummary] = Field(default_factory=list)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def calculate_cost(
    provider: ModelProvider,
    model: str,
    input_tokens: int,
    output_tokens: int,
    tool_calls: int = 0,
) -> tuple[float, float]:
    """Calculate the estimated cost for an API call.

    Args:
        provider: AI provider (claude or grok)
        model: Model ID used
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        tool_calls: Number of tool calls (for Grok)

    Returns:
        Tuple of (token_cost, tool_cost)
    """
    # Get pricing table
    if provider == ModelProvider.CLAUDE:
        pricing = CLAUDE_PRICING
    else:
        pricing = GROK_PRICING

    # Get model pricing (fallback to default if model not found)
    model_pricing = pricing.get(model)
    if not model_pricing:
        # Fallback to sensible defaults
        if provider == ModelProvider.CLAUDE:
            model_pricing = {"input": 3.00, "output": 15.00}  # Sonnet-level
        else:
            model_pricing = {"input": 3.00, "output": 15.00}  # Grok-4 level

    # Calculate token cost (pricing is per 1M tokens)
    input_cost = (input_tokens / 1_000_000) * model_pricing["input"]
    output_cost = (output_tokens / 1_000_000) * model_pricing["output"]
    token_cost = input_cost + output_cost

    # Calculate tool cost (only for Grok)
    tool_cost = 0.0
    if provider == ModelProvider.GROK and tool_calls > 0:
        # Assume web/X search - could be more granular if we track tool types
        tool_cost = tool_calls * GROK_TOOL_PRICING.get("web_search", 0.005)

    return round(token_cost, 6), round(tool_cost, 6)
