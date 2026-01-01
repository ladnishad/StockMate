"""Sub-agent definitions for the Claude Agent SDK.

This module defines the three specialized trade analysis sub-agents:
- Day Trade Analyzer: Intraday setups (5-min charts, ATR > 3%)
- Swing Trade Analyzer: Multi-day setups (daily charts, ATR 1-3%)
- Position Trade Analyzer: Major trend setups (weekly charts, ATR < 1.5%)

Each agent has access to data-gathering tools and will use timeframe-specific
parameters as instructed in their system prompts.
"""

from typing import Dict, Union

# Import the SDK's AgentDefinition
try:
    from claude_agent_sdk import AgentDefinition
except ImportError:
    # Fallback for when SDK is not installed
    from dataclasses import dataclass
    from typing import Optional, List, Literal

    @dataclass
    class AgentDefinition:
        """Fallback AgentDefinition when SDK not installed."""
        description: str
        prompt: str
        tools: Optional[List[str]] = None
        model: Optional[Literal["sonnet", "opus", "haiku", "inherit"]] = None

from app.agent.prompts import (
    build_day_trade_prompt,
    build_swing_trade_prompt,
    build_position_trade_prompt,
)


# Tool names that each sub-agent can use
SUBAGENT_TOOLS = [
    "get_price_bars",
    "get_technical_indicators",
    "get_support_resistance",
    "get_volume_profile",
    "generate_chart",
    "analyze_chart_vision",
    "get_position_status",
    "get_market_context",
]


def create_day_trade_agent(
    position_context: str,
    news_context: str = "No recent news available.",
) -> AgentDefinition:
    """Create the Day Trade Analyzer agent definition.

    This agent specializes in intraday setups with:
    - 5-minute and 15-minute charts
    - EMAs: 5, 9, 20
    - VWAP, opening range, intraday pivots
    - Holding period: minutes to hours

    Args:
        position_context: Formatted position context string
        news_context: News and sentiment context string

    Returns:
        AgentDefinition for day trading analysis
    """
    return AgentDefinition(
        description="Day trading specialist for intraday setups. Analyzes 5-min charts, VWAP levels, and opening range breakouts. Best for stocks with ATR > 3%.",
        prompt=build_day_trade_prompt("", position_context, news_context),  # Symbol injected at runtime
        tools=SUBAGENT_TOOLS,
        model="sonnet",
    )


def create_swing_trade_agent(
    position_context: str,
    news_context: str = "No recent news available.",
) -> AgentDefinition:
    """Create the Swing Trade Analyzer agent definition.

    This agent specializes in multi-day setups with:
    - Daily candlestick charts (50-100 days)
    - EMAs: 9, 21, 50
    - Daily pivots, swing highs/lows
    - Holding period: 2-10 trading days

    Args:
        position_context: Formatted position context string
        news_context: News and sentiment context string

    Returns:
        AgentDefinition for swing trading analysis
    """
    return AgentDefinition(
        description="Swing trading specialist for multi-day patterns. Analyzes daily charts, bull flags, triangles, and base breakouts. Best for stocks with ATR 1-3%.",
        prompt=build_swing_trade_prompt("", position_context, news_context),  # Symbol injected at runtime
        tools=SUBAGENT_TOOLS,
        model="sonnet",
    )


def create_position_trade_agent(
    position_context: str,
    news_context: str = "No recent news available.",
) -> AgentDefinition:
    """Create the Position Trade Analyzer agent definition.

    This agent specializes in major trend setups with:
    - Weekly candlestick charts (52+ weeks)
    - EMAs: 21, 50, 200
    - Weekly/monthly support/resistance
    - Holding period: weeks to months

    Args:
        position_context: Formatted position context string
        news_context: News and sentiment context string

    Returns:
        AgentDefinition for position trading analysis
    """
    return AgentDefinition(
        description="Position trading specialist for major trend plays. Analyzes weekly charts, major breakouts, and long-term trend continuation. Best for stocks with ATR < 1.5%.",
        prompt=build_position_trade_prompt("", position_context, news_context),  # Symbol injected at runtime
        tools=SUBAGENT_TOOLS,
        model="sonnet",
    )


# Pre-built agent definitions with default position context
# These are used when no position context is needed or for documentation
DAY_TRADE_AGENT = create_day_trade_agent("No existing position.")
SWING_TRADE_AGENT = create_swing_trade_agent("No existing position.")
POSITION_TRADE_AGENT = create_position_trade_agent("No existing position.")


def get_all_subagent_definitions(
    position_context: str = "No existing position.",
    news_context: str = "No recent news available.",
) -> Dict[str, AgentDefinition]:
    """Get all sub-agent definitions with the given position and news context.

    This function creates fresh agent definitions with the proper position
    and news context injected into their system prompts. Used by the orchestrator
    to spawn sub-agents with position and market awareness.

    Args:
        position_context: Formatted position context string describing
            the user's current position (if any)
        news_context: News and sentiment context string

    Returns:
        Dictionary mapping agent names to their definitions:
        - "day-trade-analyzer": Day trading specialist
        - "swing-trade-analyzer": Swing trading specialist
        - "position-trade-analyzer": Position trading specialist
    """
    return {
        "day-trade-analyzer": create_day_trade_agent(position_context, news_context),
        "swing-trade-analyzer": create_swing_trade_agent(position_context, news_context),
        "position-trade-analyzer": create_position_trade_agent(position_context, news_context),
    }


def get_agent_by_style(
    trade_style: str,
    position_context: str = "No existing position.",
    news_context: str = "No recent news available.",
) -> AgentDefinition:
    """Get a single agent definition by trade style.

    Args:
        trade_style: One of "day", "swing", or "position"
        position_context: Formatted position context string
        news_context: News and sentiment context string

    Returns:
        AgentDefinition for the specified trade style

    Raises:
        ValueError: If trade_style is not recognized
    """
    style_map = {
        "day": create_day_trade_agent,
        "swing": create_swing_trade_agent,
        "position": create_position_trade_agent,
    }

    if trade_style not in style_map:
        raise ValueError(
            f"Unknown trade style: {trade_style}. "
            f"Must be one of: {', '.join(style_map.keys())}"
        )

    return style_map[trade_style](position_context, news_context)
