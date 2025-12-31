"""Specialized prompts for the Claude Agent SDK orchestrator and sub-agents.

Each sub-agent has its own system prompt tailored to its trade style:
- Day Trade: Intraday focus, VWAP, opening range, quick resolution
- Swing Trade: Multi-day patterns, daily S/R, 2-10 day holding
- Position Trade: Weekly trends, major levels, weeks-months holding

Also re-exports legacy prompts from the original prompts.py for backward compatibility.
"""

# New SDK prompts
from app.agent.prompts.orchestrator_prompt import (
    ORCHESTRATOR_SYSTEM_PROMPT,
    build_orchestrator_prompt,
)
from app.agent.prompts.day_trade_prompt import (
    DAY_TRADE_SYSTEM_PROMPT,
    build_day_trade_prompt,
)
from app.agent.prompts.swing_trade_prompt import (
    SWING_TRADE_SYSTEM_PROMPT,
    build_swing_trade_prompt,
)
from app.agent.prompts.position_trade_prompt import (
    POSITION_TRADE_SYSTEM_PROMPT,
    build_position_trade_prompt,
)

# Legacy prompts - import from the .py file using importlib to avoid name conflict
import importlib.util
import os

_prompts_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "agent", "prompts.py")
# The prompts.py file is at app/agent/prompts.py, we're in app/agent/prompts/__init__.py
_prompts_file = os.path.join(os.path.dirname(__file__), "..", "prompts.py")
_prompts_file = os.path.normpath(_prompts_file)

_spec = importlib.util.spec_from_file_location("_legacy_prompts", _prompts_file)
_legacy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_legacy)

# Re-export legacy prompts
MASTER_AGENT_PROMPT = _legacy.MASTER_AGENT_PROMPT
STOCK_AGENT_PROMPT_TEMPLATE = _legacy.STOCK_AGENT_PROMPT_TEMPLATE
SMART_PLANNING_SYSTEM_PROMPT = _legacy.SMART_PLANNING_SYSTEM_PROMPT
SMART_PLAN_GENERATION_PROMPT = _legacy.SMART_PLAN_GENERATION_PROMPT
SMART_PLAN_EVALUATION_PROMPT = _legacy.SMART_PLAN_EVALUATION_PROMPT
VISUAL_ANALYSIS_PROMPT = _legacy.VISUAL_ANALYSIS_PROMPT
get_stock_agent_prompt = _legacy.get_stock_agent_prompt

__all__ = [
    # Orchestrator (new)
    "ORCHESTRATOR_SYSTEM_PROMPT",
    "build_orchestrator_prompt",
    # Day Trade (new)
    "DAY_TRADE_SYSTEM_PROMPT",
    "build_day_trade_prompt",
    # Swing Trade (new)
    "SWING_TRADE_SYSTEM_PROMPT",
    "build_swing_trade_prompt",
    # Position Trade (new)
    "POSITION_TRADE_SYSTEM_PROMPT",
    "build_position_trade_prompt",
    # Legacy prompts
    "MASTER_AGENT_PROMPT",
    "STOCK_AGENT_PROMPT_TEMPLATE",
    "SMART_PLANNING_SYSTEM_PROMPT",
    "SMART_PLAN_GENERATION_PROMPT",
    "SMART_PLAN_EVALUATION_PROMPT",
    "VISUAL_ANALYSIS_PROMPT",
    "get_stock_agent_prompt",
]
