"""Agent module for AI-powered stock monitoring.

This module provides the Claude Agent SDK integration for
autonomous stock monitoring and alert generation.
"""

from app.agent.tools import get_agent_tools
from app.agent.prompts import MASTER_AGENT_PROMPT, STOCK_AGENT_PROMPT_TEMPLATE
from app.agent.stock_agent import StockAgent
from app.agent.master_agent import MasterAgent, get_master_agent

__all__ = [
    "get_agent_tools",
    "MASTER_AGENT_PROMPT",
    "STOCK_AGENT_PROMPT_TEMPLATE",
    "StockAgent",
    "MasterAgent",
    "get_master_agent",
]
