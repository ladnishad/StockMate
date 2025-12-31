"""FinalPlanResponse schema - what the orchestrator returns after synthesis.

The orchestrator runs all 3 sub-agents in parallel, collects their reports,
and synthesizes a final response with the best plan selected.
"""

from typing import List, Optional, Literal
from pydantic import BaseModel, Field

from app.agent.schemas.subagent_report import (
    SubAgentReport,
    TradeStyleLiteral,
    BiasLiteral,
)


class AlternativePlan(BaseModel):
    """Summary of an alternative plan that wasn't selected.

    These are stored so user can view other options on request.
    """

    trade_style: TradeStyleLiteral = Field(
        description="The trade style of this alternative."
    )
    bias: BiasLiteral = Field(
        description="Directional bias of this alternative."
    )
    suitable: bool = Field(
        description="Whether this style had a valid setup."
    )
    confidence: int = Field(
        ge=0,
        le=100,
        description="Confidence score for this alternative."
    )
    holding_period: str = Field(
        default="",
        description="Expected holding period for this alternative."
    )
    brief_thesis: str = Field(
        max_length=200,
        description="One-sentence summary of this alternative's thesis."
    )
    why_not_selected: str = Field(
        max_length=200,
        description="Why this wasn't selected: 'lower confidence', 'worse R:R', 'no clear setup', etc."
    )
    risk_reward: Optional[float] = Field(
        default=None,
        description="Risk/reward ratio of this alternative."
    )
    position_recommendation: Optional[str] = Field(
        default=None,
        description="Position management recommendation: 'hold', 'add', 'trim', 'reduce', 'exit'."
    )
    risk_warnings: List[str] = Field(
        default_factory=list,
        description="Key risk warnings for this alternative."
    )

    @classmethod
    def from_report(cls, report: SubAgentReport, why_not_selected: str) -> "AlternativePlan":
        """Create an AlternativePlan summary from a full SubAgentReport."""
        return cls(
            trade_style=report.trade_style,
            bias=report.bias,
            suitable=report.suitable,
            confidence=report.confidence,
            holding_period=report.holding_period,
            brief_thesis=report.thesis[:200] if len(report.thesis) > 200 else report.thesis,
            why_not_selected=why_not_selected,
            risk_reward=report.risk_reward,
            position_recommendation=report.position_recommendation,
            risk_warnings=report.risk_warnings[:3] if report.risk_warnings else [],
        )


class FinalPlanResponse(BaseModel):
    """Final response from the orchestrator after running all sub-agents.

    Contains the selected best plan, alternatives, and metadata.
    """

    # Selected Plan (full detail)
    selected_plan: SubAgentReport = Field(
        description="The best plan selected by the orchestrator."
    )
    selected_style: TradeStyleLiteral = Field(
        description="The trade style that was selected as best."
    )
    selection_reasoning: str = Field(
        max_length=500,
        description="Why this plan was selected: 'highest confidence', 'best R:R', 'aligns with position', etc."
    )

    # Alternatives (available on request)
    alternatives: List[AlternativePlan] = Field(
        default_factory=list,
        description="Summary of other plans that weren't selected. User can request details."
    )

    # Metadata
    symbol: str = Field(description="Stock ticker symbol analyzed.")
    analysis_timestamp: str = Field(description="ISO timestamp when analysis completed.")
    total_analysis_time_ms: int = Field(
        default=0,
        description="Total time in milliseconds for all sub-agents to complete."
    )

    # Position Status
    has_existing_position: bool = Field(
        default=False,
        description="Whether the user has an existing position in this stock."
    )
    position_direction: Optional[Literal["long", "short"]] = Field(
        default=None,
        description="Direction of existing position, if any."
    )
    position_entry: Optional[float] = Field(
        default=None,
        description="Entry price of existing position, if any."
    )

    # Market Context
    market_direction: Optional[str] = Field(
        default=None,
        description="Overall market direction: 'bullish', 'bearish', 'mixed'."
    )
    current_price: Optional[float] = Field(
        default=None,
        description="Current stock price at time of analysis."
    )

    # Sub-agent Status
    day_trade_analyzed: bool = Field(
        default=True,
        description="Whether day trade analysis completed successfully."
    )
    swing_trade_analyzed: bool = Field(
        default=True,
        description="Whether swing trade analysis completed successfully."
    )
    position_trade_analyzed: bool = Field(
        default=True,
        description="Whether position trade analysis completed successfully."
    )

    def get_alternative_by_style(self, style: TradeStyleLiteral) -> Optional[AlternativePlan]:
        """Get alternative plan by trade style."""
        for alt in self.alternatives:
            if alt.trade_style == style:
                return alt
        return None

    def has_any_valid_setup(self) -> bool:
        """Check if any trade style found a valid setup."""
        if self.selected_plan.suitable:
            return True
        return any(alt.suitable for alt in self.alternatives)

    def get_styles_with_setups(self) -> List[TradeStyleLiteral]:
        """Get list of trade styles that found valid setups."""
        styles = []
        if self.selected_plan.suitable:
            styles.append(self.selected_plan.trade_style)
        for alt in self.alternatives:
            if alt.suitable:
                styles.append(alt.trade_style)
        return styles


class DataContext(BaseModel):
    """Common data context gathered by orchestrator before dispatching sub-agents.

    This contains data that's shared across all sub-agents (position, market context).
    Timeframe-specific data is gathered by each sub-agent individually.
    """

    symbol: str = Field(description="Stock ticker symbol.")
    user_id: str = Field(description="User ID for position lookup.")

    # Current Price (shared)
    current_price: float = Field(description="Current stock price.")
    bid: Optional[float] = Field(default=None, description="Current bid price.")
    ask: Optional[float] = Field(default=None, description="Current ask price.")

    # Position Status (shared - affects all sub-agents)
    has_position: bool = Field(default=False, description="Whether user has a position.")
    position_direction: Optional[Literal["long", "short"]] = Field(default=None)
    position_entry: Optional[float] = Field(default=None)
    position_size: Optional[int] = Field(default=None)
    position_pnl_pct: Optional[float] = Field(default=None)

    # Market Context (shared)
    market_direction: str = Field(
        default="mixed",
        description="Overall market direction: 'bullish', 'bearish', 'mixed'."
    )
    bullish_indices: int = Field(
        default=0,
        description="Number of major indices (SPY, QQQ, DIA, IWM) that are up."
    )

    # Timestamp
    timestamp: str = Field(description="ISO timestamp when context was gathered.")

    # News/Sentiment (shared)
    news_sentiment: Optional[str] = Field(
        default=None,
        description="Overall news sentiment: 'bullish', 'bearish', 'neutral'."
    )
    news_summary: Optional[str] = Field(
        default=None,
        description="Brief summary of recent news and catalysts."
    )
    recent_headlines: Optional[List[str]] = Field(
        default=None,
        description="Recent news headlines about the stock."
    )

    def to_prompt_context(self) -> str:
        """Format context for inclusion in sub-agent prompts."""
        lines = [
            f"Symbol: {self.symbol}",
            f"Current Price: ${self.current_price:.2f}",
            f"Market Direction: {self.market_direction} ({self.bullish_indices}/4 indices bullish)",
        ]

        if self.has_position:
            direction = self.position_direction.upper() if self.position_direction else "UNKNOWN"
            lines.append(f"\n** USER HAS {direction} POSITION **")
            lines.append(f"Entry: ${self.position_entry:.2f}" if self.position_entry else "Entry: Unknown")
            lines.append(f"Size: {self.position_size} shares" if self.position_size else "")
            if self.position_pnl_pct is not None:
                pnl_sign = "+" if self.position_pnl_pct >= 0 else ""
                lines.append(f"Unrealized P&L: {pnl_sign}{self.position_pnl_pct:.2f}%")
            lines.append("DO NOT SUGGEST OPPOSITE DIRECTION TRADES.")
        else:
            lines.append("\nNo existing position.")

        # News context
        if self.news_sentiment or self.news_summary:
            lines.append(f"\nNews Sentiment: {self.news_sentiment or 'Unknown'}")
            if self.news_summary:
                lines.append(f"News Summary: {self.news_summary}")
            if self.recent_headlines:
                lines.append("Recent Headlines:")
                for headline in self.recent_headlines[:3]:
                    lines.append(f"  - {headline}")

        return "\n".join(lines)
