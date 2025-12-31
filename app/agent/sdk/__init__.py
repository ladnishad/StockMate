"""Claude Agent SDK integration for StockMate.

This module provides the SDK-based trading plan generation with:
- TradePlanOrchestrator: Main orchestrator that coordinates sub-agents
- Sub-agent definitions for Day/Swing/Position trade analysis
- SDK MCP tools for data gathering and chart analysis
- Streaming event generation for iOS consumption
"""

from app.agent.sdk.orchestrator import TradePlanOrchestrator
from app.agent.sdk.subagent_definitions import (
    DAY_TRADE_AGENT,
    SWING_TRADE_AGENT,
    POSITION_TRADE_AGENT,
    get_all_subagent_definitions,
)

__all__ = [
    "TradePlanOrchestrator",
    "DAY_TRADE_AGENT",
    "SWING_TRADE_AGENT",
    "POSITION_TRADE_AGENT",
    "get_all_subagent_definitions",
]
