"""FinalPlanResponse schema - what the orchestrator returns after synthesis.

The orchestrator runs all 3 sub-agents in parallel, collects their reports,
and synthesizes a final response with the best plan selected.
"""

from typing import List, Optional, Literal, Any
from pydantic import BaseModel, Field

from app.agent.schemas.subagent_report import (
    SubAgentReport,
    TradeStyleLiteral,
    BiasLiteral,
)

# Import Fundamentals for runtime use with Pydantic
from app.models.data import Fundamentals


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
        max_length=1500,
        description="Why this plan was selected: 'highest confidence', 'best R:R', 'aligns with position', etc."
    )

    # Alternatives (full SubAgentReports for other trade styles)
    alternatives: List[SubAgentReport] = Field(
        default_factory=list,
        description="Full reports from other trade styles that weren't selected."
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

    # X/Social Sentiment Citations (aggregated from all sub-agents)
    all_citations: List[str] = Field(
        default_factory=list,
        description="All unique URLs to X/Twitter posts and sources used across all analyses."
    )

    def get_alternative_by_style(self, style: TradeStyleLiteral) -> Optional[SubAgentReport]:
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
    news_score: Optional[float] = Field(
        default=None,
        description="Numeric news sentiment score from -1 (bearish) to 1 (bullish)."
    )
    news_article_count: int = Field(
        default=0,
        description="Number of news articles analyzed."
    )
    news_has_breaking: bool = Field(
        default=False,
        description="Whether there's breaking news (< 2 hours old)."
    )
    news_key_themes: Optional[List[str]] = Field(
        default=None,
        description="Key topics/themes from news articles."
    )

    # X/Social Sentiment (gathered once, shared with all sub-agents)
    x_sentiment: Optional[str] = Field(
        default=None,
        description="X/Twitter sentiment: 'bullish', 'bearish', 'neutral', 'mixed'."
    )
    x_sentiment_summary: Optional[str] = Field(
        default=None,
        description="Summary of X/social discussion about the stock."
    )
    x_citations: Optional[List[str]] = Field(
        default=None,
        description="URLs of X posts used in sentiment analysis."
    )

    # Fundamentals (gathered once, shared with all sub-agents)
    # Using Any to avoid circular import, actual type is Fundamentals
    fundamentals: Optional[Fundamentals] = Field(
        default=None,
        description="Fundamental financial data for the stock."
    )
    has_earnings_risk: bool = Field(
        default=False,
        description="True if earnings announcement is within 7 days."
    )
    days_until_earnings: Optional[int] = Field(
        default=None,
        description="Days until next earnings announcement."
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
        if self.news_sentiment or self.news_score is not None:
            lines.append(f"\n## NEWS CONTEXT:")
            lines.append(f"Sentiment: {self.news_sentiment or 'Unknown'}")
            if self.news_score is not None:
                lines.append(f"Sentiment Score: {self.news_score:+.2f} (-1 to +1 scale)")
            if self.news_article_count > 0:
                lines.append(f"Articles Analyzed: {self.news_article_count}")
            if self.news_has_breaking:
                lines.append("** BREAKING NEWS DETECTED - increased volatility risk **")
            if self.news_key_themes:
                lines.append(f"Key Themes: {', '.join(self.news_key_themes[:5])}")
            if self.recent_headlines:
                lines.append("Recent Headlines:")
                for headline in self.recent_headlines[:5]:
                    lines.append(f"  - {headline}")

        # X/Social sentiment context
        if self.x_sentiment or self.x_sentiment_summary:
            lines.append(f"\nX/Social Sentiment: {self.x_sentiment or 'Unknown'}")
            if self.x_sentiment_summary:
                lines.append(f"X Discussion: {self.x_sentiment_summary}")

        # Fundamentals context
        if self.fundamentals:
            f = self.fundamentals
            lines.append("\n## FUNDAMENTALS:")

            # Valuation
            if f.pe_ratio is not None:
                lines.append(f"P/E Ratio: {f.pe_ratio:.1f} ({f.get_valuation_assessment()})")
            if f.pb_ratio is not None:
                lines.append(f"P/B Ratio: {f.pb_ratio:.2f}")
            if f.ps_ratio is not None:
                lines.append(f"P/S Ratio: {f.ps_ratio:.2f}")

            # Growth
            if f.eps_growth_yoy is not None:
                lines.append(f"EPS Growth (YoY): {f.eps_growth_yoy:+.1f}%")
            if f.revenue_growth_yoy is not None:
                lines.append(f"Revenue Growth (YoY): {f.revenue_growth_yoy:+.1f}%")

            # Health
            lines.append(f"Financial Health: {f.get_financial_health_score()}")
            if f.debt_to_equity is not None:
                lines.append(f"Debt/Equity: {f.debt_to_equity:.2f}")

            # Earnings Risk - CRITICAL
            if self.has_earnings_risk and self.days_until_earnings is not None:
                lines.append(f"\n** EARNINGS WARNING: {self.days_until_earnings} DAYS UNTIL EARNINGS **")
                lines.append("HIGH RISK for swing/position trades - earnings can cause 10-30% gaps!")
                if f.earnings_beat_rate is not None:
                    lines.append(f"Beat rate (last 4Q): {f.earnings_beat_rate:.0f}%")

        return "\n".join(lines)
