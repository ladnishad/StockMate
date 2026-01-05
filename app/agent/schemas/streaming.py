"""Streaming event schemas for iOS consumption via SSE.

These models define the structure of Server-Sent Events that the
iOS app consumes to show Claude Code-style progress for each sub-agent.
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SubAgentStatus(str, Enum):
    """Status of a sub-agent during execution."""
    PENDING = "pending"
    RUNNING = "running"
    GATHERING_DATA = "gathering_data"
    CALCULATING_TECHNICALS = "calculating_technicals"
    GENERATING_CHART = "generating_chart"
    ANALYZING_CHART = "analyzing_chart"
    GENERATING_PLAN = "generating_plan"
    COMPLETED = "completed"
    FAILED = "failed"


class OrchestratorStepType(str, Enum):
    """Steps performed by the main orchestrator."""
    GATHERING_COMMON_DATA = "gathering_common_data"
    CHECKING_POSITION = "checking_position"
    SPAWNING_SUBAGENTS = "spawning_subagents"
    WAITING_FOR_SUBAGENTS = "waiting_for_subagents"
    SELECTING_BEST = "selecting_best"
    COMPLETE = "complete"
    # New agentic mode steps
    AGENTIC_ANALYSIS = "agentic_analysis"


class StreamEventType(str, Enum):
    """Types of streaming events for iOS consumption."""
    # Legacy multi-agent events
    ORCHESTRATOR_STEP = "orchestrator_step"
    SUBAGENT_PROGRESS = "subagent_progress"
    SUBAGENT_COMPLETE = "subagent_complete"
    FINAL_RESULT = "final_result"
    ERROR = "error"
    # New agentic mode events
    AGENT_THINKING = "agent_thinking"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"


class SubAgentProgress(BaseModel):
    """Progress update for a single sub-agent.

    The iOS app displays these in parallel columns showing
    what each trade-style agent is currently doing.
    """

    agent_name: str = Field(
        description="Agent identifier: 'day-trade-analyzer', 'swing-trade-analyzer', 'position-trade-analyzer'"
    )
    display_name: str = Field(
        default="",
        description="Human-readable name: 'Day Trade', 'Swing Trade', 'Position Trade'"
    )
    status: SubAgentStatus = Field(
        description="Current status of this sub-agent."
    )
    current_step: Optional[str] = Field(
        default=None,
        description="Human-readable description of current step: 'Gathering 5-min bars', 'Vision analysis', etc."
    )
    steps_completed: List[str] = Field(
        default_factory=list,
        description="List of completed step names for progress indicator."
    )
    findings: List[str] = Field(
        default_factory=list,
        description="Key findings so far: 'RSI: 62 (bullish)', 'Bull flag detected', etc."
    )
    elapsed_ms: int = Field(
        default=0,
        description="Time elapsed for this sub-agent in milliseconds."
    )
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if status is FAILED."
    )

    @classmethod
    def create_pending(cls, agent_name: str) -> "SubAgentProgress":
        """Create a pending sub-agent progress."""
        display_names = {
            "day-trade-analyzer": "Day Trade",
            "swing-trade-analyzer": "Swing Trade",
            "position-trade-analyzer": "Position Trade",
        }
        return cls(
            agent_name=agent_name,
            display_name=display_names.get(agent_name, agent_name),
            status=SubAgentStatus.PENDING,
        )


class StreamEvent(BaseModel):
    """SSE event for iOS consumption.

    The iOS app receives these events and updates the UI accordingly.
    Events flow: orchestrator_step -> subagent_progress -> subagent_complete -> final_result
    """

    type: str = Field(
        description="Event type: 'orchestrator_step', 'subagent_progress', 'subagent_complete', 'final_result', 'error'"
    )
    timestamp: float = Field(
        description="Unix timestamp when event was emitted."
    )

    # Orchestrator-level progress
    step_type: Optional[str] = Field(
        default=None,
        description="For orchestrator_step events: which step is being performed."
    )
    step_status: Optional[str] = Field(
        default=None,
        description="Status of orchestrator step: 'active', 'completed'."
    )
    step_findings: List[str] = Field(
        default_factory=list,
        description="Findings from orchestrator step."
    )

    # Sub-agent progress (for parallel display)
    subagents: Optional[Dict[str, SubAgentProgress]] = Field(
        default=None,
        description="Progress for all sub-agents. Keys are agent names."
    )

    # Single sub-agent completion
    agent_name: Optional[str] = Field(
        default=None,
        description="For subagent_complete: which agent completed."
    )
    agent_findings: List[str] = Field(
        default_factory=list,
        description="For subagent_complete: final findings from that agent."
    )

    # Final result
    analysis_id: Optional[str] = Field(
        default=None,
        description="For final_result: unique ID of the saved analysis."
    )
    plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For final_result: the selected plan as dict."
    )
    alternatives: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="For final_result: alternative plans as list of dicts."
    )
    selected_style: Optional[str] = Field(
        default=None,
        description="For final_result: which trade style was selected."
    )
    selection_reasoning: Optional[str] = Field(
        default=None,
        description="For final_result: why this plan was selected."
    )
    all_citations: Optional[List[str]] = Field(
        default=None,
        description="For final_result: X/social citations used across all analyses."
    )

    # Error
    error_message: Optional[str] = Field(
        default=None,
        description="For error events: the error message."
    )

    # Agentic mode fields
    thinking: Optional[str] = Field(
        default=None,
        description="For agent_thinking: AI's reasoning/thinking text."
    )
    tool_name: Optional[str] = Field(
        default=None,
        description="For tool_call/tool_result: name of the tool."
    )
    tool_arguments: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For tool_call: arguments passed to the tool."
    )
    tool_result: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For tool_result: result returned by the tool."
    )
    iteration: Optional[int] = Field(
        default=None,
        description="For agentic events: which iteration this occurred in."
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Generic data field for flexible event payloads."
    )
    agentic_plan: Optional[Dict[str, Any]] = Field(
        default=None,
        description="For agentic final_result: the full plan with day/swing/position_trade_plan structure."
    )

    @classmethod
    def orchestrator_step(
        cls,
        step_type: OrchestratorStepType,
        status: str,
        findings: List[str] = None,
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create an orchestrator step event."""
        import time
        return cls(
            type="orchestrator_step",
            timestamp=timestamp or time.time(),
            step_type=step_type.value,
            step_status=status,
            step_findings=findings or [],
        )

    @classmethod
    def subagent_progress(
        cls,
        subagents: Dict[str, SubAgentProgress],
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create a sub-agent progress event."""
        import time
        return cls(
            type="subagent_progress",
            timestamp=timestamp or time.time(),
            subagents=subagents,
        )

    @classmethod
    def subagent_complete(
        cls,
        agent_name: str,
        findings: List[str],
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create a sub-agent completion event."""
        import time
        return cls(
            type="subagent_complete",
            timestamp=timestamp or time.time(),
            agent_name=agent_name,
            agent_findings=findings,
        )

    @classmethod
    def final_result(
        cls,
        plan: Dict[str, Any],
        alternatives: List[Dict[str, Any]],
        selected_style: str,
        selection_reasoning: str,
        analysis_id: str = None,
        all_citations: List[str] = None,
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create a final result event."""
        import time
        return cls(
            type="final_result",
            timestamp=timestamp or time.time(),
            analysis_id=analysis_id,
            plan=plan,
            alternatives=alternatives,
            selected_style=selected_style,
            selection_reasoning=selection_reasoning,
            all_citations=all_citations or [],
        )

    @classmethod
    def error(cls, message: str, timestamp: float = None) -> "StreamEvent":
        """Create an error event."""
        import time
        return cls(
            type="error",
            timestamp=timestamp or time.time(),
            error_message=message,
        )

    @classmethod
    def agent_thinking(
        cls,
        thinking: str,
        iteration: int = None,
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create an agent thinking event (AI's reasoning)."""
        import time
        return cls(
            type=StreamEventType.AGENT_THINKING.value,
            timestamp=timestamp or time.time(),
            thinking=thinking,
            iteration=iteration,
        )

    @classmethod
    def tool_call(
        cls,
        tool_name: str,
        arguments: Dict[str, Any],
        iteration: int = None,
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create a tool call event."""
        import time
        return cls(
            type=StreamEventType.TOOL_CALL.value,
            timestamp=timestamp or time.time(),
            tool_name=tool_name,
            tool_arguments=arguments,
            iteration=iteration,
        )

    @classmethod
    def tool_result_event(
        cls,
        tool_name: str,
        result: Dict[str, Any],
        iteration: int = None,
        timestamp: float = None,
    ) -> "StreamEvent":
        """Create a tool result event."""
        import time
        return cls(
            type=StreamEventType.TOOL_RESULT.value,
            timestamp=timestamp or time.time(),
            tool_name=tool_name,
            tool_result=result,
            iteration=iteration,
        )

    def to_sse(self) -> str:
        """Format as Server-Sent Event string."""
        import json
        data = self.model_dump(exclude_none=True)
        return f"data: {json.dumps(data)}\n\n"
