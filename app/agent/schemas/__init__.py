"""Pydantic schemas for the Claude Agent SDK integration.

This module contains structured output schemas for:
- SubAgentReport: What each trade-style sub-agent returns
- FinalPlanResponse: What the orchestrator returns after synthesis
- StreamEvent: SSE event models for iOS streaming UI
"""

from app.agent.schemas.subagent_report import (
    SubAgentReport,
    VisionAnalysisResult,
    PriceTargetWithReasoning,
    TradeStyleLiteral,
    BiasLiteral,
)
from app.agent.schemas.final_response import (
    FinalPlanResponse,
    AlternativePlan,
)
from app.agent.schemas.streaming import (
    StreamEvent,
    SubAgentProgress,
    SubAgentStatus,
    OrchestratorStepType,
)

__all__ = [
    # SubAgent Report
    "SubAgentReport",
    "VisionAnalysisResult",
    "PriceTargetWithReasoning",
    "TradeStyleLiteral",
    "BiasLiteral",
    # Final Response
    "FinalPlanResponse",
    "AlternativePlan",
    # Streaming
    "StreamEvent",
    "SubAgentProgress",
    "SubAgentStatus",
    "OrchestratorStepType",
]
