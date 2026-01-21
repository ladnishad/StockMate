"""TradePlanOrchestrator - Main orchestrator that coordinates sub-agents.

This orchestrator:
1. Gathers common context (price, position, market)
2. Spawns 3 sub-agents in parallel (day/swing/position trade analyzers)
3. Collects their reports and synthesizes the best recommendation
4. Streams progress events to the iOS app
"""

import asyncio
import logging
import time
from typing import Dict, Any, AsyncGenerator, Optional, List
from datetime import datetime

from app.agent.schemas.final_response import FinalPlanResponse, DataContext
from app.agent.schemas.subagent_report import SubAgentReport
from app.agent.schemas.streaming import (
    StreamEvent,
    SubAgentProgress,
    SubAgentStatus,
    OrchestratorStepType,
)
from app.agent.sdk.tools import (
    get_current_price,
    get_position_status,
    get_market_context,
    get_news_sentiment,
)
from app.agent.providers.factory import get_user_provider
from app.agent.providers import AIMessage, AIProvider, SearchParameters, ModelProvider
from app.agent.providers.grok_provider import get_x_search_parameters
from app.services.usage_tracker import get_usage_tracker
from app.models.usage import OperationType, ModelProvider as UsageModelProvider

logger = logging.getLogger(__name__)


# =============================================================================
# Retry Helper for Resilient Tool Calls
# =============================================================================


async def retry_async(
    func,
    *args,
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    **kwargs
):
    """Retry an async function with exponential backoff.

    Args:
        func: Async function to call
        *args: Arguments to pass to func
        max_attempts: Maximum retry attempts (default 3)
        delay: Initial delay between retries in seconds (default 1.0)
        backoff: Multiplier for delay after each retry (default 2.0)
        **kwargs: Keyword arguments to pass to func

    Returns:
        Result from func

    Raises:
        Last exception if all retries fail
    """
    last_exception = None
    current_delay = delay

    for attempt in range(max_attempts):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            error_msg = str(e) or f"{type(e).__name__}"
            if attempt < max_attempts - 1:
                logger.warning(
                    f"Retry {attempt + 1}/{max_attempts} for {func.__name__} "
                    f"after error: {error_msg}. Waiting {current_delay:.1f}s..."
                )
                await asyncio.sleep(current_delay)
                current_delay *= backoff
            else:
                logger.error(f"All {max_attempts} attempts failed for {func.__name__}: {error_msg}")

    raise last_exception


# =============================================================================
# Helper Functions for Formatting Comprehensive Data
# =============================================================================


def format_institutional_indicators(indicators: Dict[str, Any]) -> str:
    """Format institutional indicators for the AI prompt."""
    parts = []

    # Ichimoku
    ichimoku = indicators.get("ichimoku", {})
    if ichimoku.get("available"):
        parts.append(f"- Ichimoku: {ichimoku.get('signal', 'N/A')}, "
                    f"Cloud: {ichimoku.get('price_vs_cloud', 'N/A')}, "
                    f"TK Cross: {ichimoku.get('tk_cross', 'N/A')}")

    # Williams %R
    williams = indicators.get("williams_r", {})
    if williams.get("available"):
        parts.append(f"- Williams %R: {williams.get('value', 'N/A')} ({williams.get('signal', 'N/A')})")

    # Parabolic SAR
    psar = indicators.get("parabolic_sar", {})
    if psar.get("available"):
        parts.append(f"- Parabolic SAR: {psar.get('trend_direction', 'N/A')} ({psar.get('signal', 'N/A')})")

    # CMF
    cmf = indicators.get("cmf", {})
    if cmf.get("available"):
        parts.append(f"- CMF: {cmf.get('value', 'N/A')} ({cmf.get('signal', 'N/A')}) - {cmf.get('interpretation', '')}")

    # ADL
    adl = indicators.get("adl", {})
    if adl.get("available"):
        divergence_str = " [DIVERGENCE]" if adl.get("divergence") else ""
        parts.append(f"- ADL: {adl.get('signal', 'N/A')}, Trend: {adl.get('trend', 'N/A')}{divergence_str}")

    return "\n".join(parts) if parts else "- Institutional indicators: Not available"


def format_levels_with_metrics(sr_levels: Dict[str, Any]) -> str:
    """Format S/R levels with institutional-grade metrics."""
    lines = []

    for level_type in ["support", "resistance"]:
        levels = sr_levels.get(level_type, [])
        if levels and isinstance(levels, list):
            lines.append(f"{level_type.capitalize()}:")
            for level in levels[:3]:
                # Defensive checks - ensure level is a dict with numeric price
                if not isinstance(level, dict):
                    continue
                price = level.get("price", 0)
                if not isinstance(price, (int, float)) or price is None:
                    price = 0
                reliability = str(level.get("reliability", "unknown")).upper()
                reclaimed = "[RECLAIMED]" if level.get("reclaimed") else ""
                touches = level.get("touches", 0) or 0
                hv_touches = level.get("high_volume_touches", 0) or 0
                bounce = level.get("bounce_quality", 0) or 0
                distance = level.get("distance_pct", 0) or 0
                if not isinstance(distance, (int, float)):
                    distance = 0

                lines.append(
                    f"  - ${price:.2f} [{reliability}] {reclaimed} - "
                    f"{touches} touches, {hv_touches} high-vol, "
                    f"bounce: {bounce}, distance: {distance:+.1f}%"
                )

    return "\n".join(lines) if lines else "- No levels identified"


def format_volume_profile(volume_profile: Dict[str, Any]) -> str:
    """Format volume profile data."""
    if not volume_profile or volume_profile.get("error"):
        return "- Volume profile: Not available"

    lines = []
    vpoc = volume_profile.get("vpoc")
    vah = volume_profile.get("value_area_high")
    val = volume_profile.get("value_area_low")
    hvn = volume_profile.get("high_volume_nodes", []) or []
    lvn = volume_profile.get("low_volume_nodes", []) or []

    # Defensive checks for numeric values
    if vpoc and isinstance(vpoc, (int, float)):
        lines.append(f"- VPOC (Point of Control): ${vpoc:.2f}")
    if vah and val and isinstance(vah, (int, float)) and isinstance(val, (int, float)):
        lines.append(f"- Value Area: ${val:.2f} - ${vah:.2f}")
    if hvn and isinstance(hvn, list):
        hvn_nums = [n for n in hvn[:3] if isinstance(n, (int, float))]
        if hvn_nums:
            hvn_str = ", ".join([f"${n:.2f}" for n in hvn_nums])
            lines.append(f"- High Volume Nodes: {hvn_str}")
    if lvn and isinstance(lvn, list):
        lvn_nums = [n for n in lvn[:3] if isinstance(n, (int, float))]
        if lvn_nums:
            lvn_str = ", ".join([f"${n:.2f}" for n in lvn_nums])
            lines.append(f"- Low Volume Nodes (fast moves): {lvn_str}")

    return "\n".join(lines) if lines else "- Volume profile: No data"


def format_chart_patterns(patterns_data: Dict[str, Any]) -> str:
    """Format chart patterns with success rates."""
    if not patterns_data or patterns_data.get("error"):
        return "- Chart patterns: Not available"

    patterns = patterns_data.get("patterns", []) or []
    if not patterns or not isinstance(patterns, list):
        return "- No chart patterns detected"

    lines = []
    for p in patterns[:3]:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "Unknown")
        ptype = p.get("type", "unknown")
        success = p.get("success_rate")
        target = p.get("target_price")
        confidence = p.get("confidence", 0) or 0

        success_str = f"{success}% success" if success else "N/A success"
        target_str = f", target: ${target:.2f}" if target and isinstance(target, (int, float)) else ""
        lines.append(f"- {name} ({ptype}) - {success_str}, confidence: {confidence}{target_str}")

    strongest = patterns_data.get("strongest_pattern")
    if strongest:
        lines.append(f"- Strongest pattern: {strongest}")

    return "\n".join(lines)


def format_divergences(divergence_data: Dict[str, Any]) -> str:
    """Format divergence detection results for AI analysis."""
    if not divergence_data or divergence_data.get("error"):
        return "- Divergences: Not available"

    lines = []

    rsi_div = divergence_data.get("rsi_divergence", {})
    if rsi_div.get("detected"):
        lines.append(f"- RSI Divergence: {rsi_div.get('type', 'unknown').upper()} - {rsi_div.get('interpretation', '')}")
    else:
        lines.append("- RSI Divergence: None detected")

    macd_div = divergence_data.get("macd_divergence", {})
    if macd_div.get("detected"):
        lines.append(f"- MACD Divergence: {macd_div.get('type', 'unknown').upper()} - {macd_div.get('interpretation', '')}")
    else:
        lines.append("- MACD Divergence: None detected")

    # Overall signal
    overall = divergence_data.get("overall_signal", "neutral")
    if divergence_data.get("has_divergence"):
        lines.append(f"- ⚠️ DIVERGENCE DETECTED: Overall {overall.upper()} signal")

    return "\n".join(lines)


def format_enhanced_technicals(indicators: Dict[str, Any], bars: Dict[str, Any]) -> str:
    """Format enhanced technical data with actual values."""
    lines = []

    # ATR with dollar value
    atr_data = indicators.get("atr", {})
    atr_value = atr_data.get("value")
    if atr_value is not None:
        lines.append(f"- ATR: ${atr_value:.2f} ({atr_data.get('pct', 0):.1f}% of price) - {atr_data.get('volatility_regime', 'unknown')} volatility")

    # RSI
    rsi_data = indicators.get("rsi", {})
    rsi_value = rsi_data.get("value")
    if rsi_value is not None:
        lines.append(f"- RSI(14): {rsi_value:.1f} ({rsi_data.get('signal', 'neutral')})")

    # Actual EMA values
    emas = indicators.get("emas", {})
    if emas:
        ema_str_parts = []
        for key, value in sorted(emas.items()):
            if value is not None and isinstance(value, (int, float)):
                period = key.replace("ema_", "")
                ema_str_parts.append(f"{period}-day: ${value:.2f}")
        if ema_str_parts:
            lines.append(f"- EMA Values: {', '.join(ema_str_parts)}")

    # EMA trend
    ema_trend = indicators.get("ema_trend", "unknown")
    lines.append(f"- EMA Trend: {ema_trend}")

    # VWAP
    vwap_data = indicators.get("vwap", {})
    vwap_value = vwap_data.get("value")
    if vwap_value is not None:
        position = vwap_data.get("price_vs_vwap", "unknown")
        distance = vwap_data.get("distance_pct", 0)
        lines.append(f"- VWAP: ${vwap_value:.2f} (price {position} VWAP by {abs(distance or 0):.1f}%)")

    # ADX
    adx_data = indicators.get("adx", {})
    adx_value = adx_data.get("value")
    if adx_value is not None:
        lines.append(f"- ADX: {adx_value:.1f} ({adx_data.get('strength', 'unknown')} trend) - {adx_data.get('interpretation', '')}")

    # MACD Histogram
    macd_data = indicators.get("macd", {})
    histogram = macd_data.get("histogram")
    if histogram is not None and isinstance(histogram, (int, float)):
        lines.append(f"- MACD Histogram: {histogram:.4f}")

    # Bollinger Position
    bollinger_data = indicators.get("bollinger", {})
    position = bollinger_data.get("position")
    if position:
        lines.append(f"- Bollinger Position: {position}")

    return "\n".join(lines)


def format_fibonacci_levels(fib_data: Dict[str, Any]) -> str:
    """Format Fibonacci levels for AI consumption."""
    if not fib_data or fib_data.get("error"):
        return "- Fibonacci: Not available"

    lines = []

    # Swing points and trend
    swing_high = fib_data.get("swing_high")
    swing_low = fib_data.get("swing_low")
    trend = fib_data.get("trend", "unknown")
    if swing_high and swing_low:
        lines.append(f"- Swing High: ${swing_high:.2f}, Swing Low: ${swing_low:.2f} ({trend})")

    # Signal
    signal = fib_data.get("signal", "neutral")
    at_entry = fib_data.get("at_entry_level", False)
    near_fib = fib_data.get("near_fib_level", False)
    lines.append(f"- Fibonacci Signal: {signal.upper()}" + (" - AT KEY ENTRY LEVEL" if at_entry else " - near level" if near_fib else ""))

    # Nearest level
    nearest_level = fib_data.get("nearest_level")
    nearest_price = fib_data.get("nearest_price")
    distance_pct = fib_data.get("distance_pct", 0)
    if nearest_level and nearest_price:
        lines.append(f"- Nearest Fib Level: {nearest_level} (${nearest_price:.2f}) - {abs(distance_pct):.1f}% away")

    # Retracement levels (key entries)
    retracements = fib_data.get("retracement_levels", {})
    if retracements:
        key_levels = ["0.382", "0.500", "0.618", "0.786"]
        ret_parts = []
        for level in key_levels:
            price = retracements.get(level)
            if price:
                ret_parts.append(f"{level}: ${price:.2f}")
        if ret_parts:
            lines.append(f"- Key Retracements: {', '.join(ret_parts)}")

    # Extension levels (targets)
    extensions = fib_data.get("extension_levels", {})
    if extensions:
        ext_parts = []
        for level in ["1.272", "1.618", "2.000", "2.618"]:
            price = extensions.get(level)
            if price:
                ext_parts.append(f"{level}: ${price:.2f}")
        if ext_parts:
            lines.append(f"- Extension Targets: {', '.join(ext_parts)}")

    # Suggested zones
    entry_zone = fib_data.get("suggested_entry_zone", {})
    stop_zone = fib_data.get("suggested_stop_zone", {})
    if entry_zone.get("low") and entry_zone.get("high"):
        lines.append(f"- Fib Entry Zone: ${entry_zone['low']:.2f} - ${entry_zone['high']:.2f}")
    if stop_zone.get("low") and stop_zone.get("high"):
        lines.append(f"- Fib Stop Zone: ${stop_zone['low']:.2f} - ${stop_zone['high']:.2f}")

    return "\n".join(lines) if lines else "- Fibonacci: No data"


class TradePlanOrchestrator:
    """Orchestrates parallel sub-agents for trading plan generation.

    This is the main entry point for the SDK-based plan generation.
    It coordinates three specialized sub-agents:
    - day-trade-analyzer
    - swing-trade-analyzer
    - position-trade-analyzer
    """

    def __init__(self, user_id: Optional[str] = None):
        """Initialize the orchestrator.

        Args:
            user_id: User ID for provider lookup. If not provided,
                     must be passed to generate_plan_stream.
        """
        self.context: Optional[DataContext] = None
        self.subagent_reports: Dict[str, SubAgentReport] = {}
        self.subagent_progress: Dict[str, SubAgentProgress] = {}
        self._start_time: float = 0
        self._user_id: Optional[str] = user_id
        self._provider: Optional[AIProvider] = None

    async def _get_provider(self, user_id: Optional[str] = None) -> AIProvider:
        """Get the AI provider for the current user.

        Args:
            user_id: Optional user ID override

        Returns:
            AIProvider instance for the user
        """
        if self._provider is None:
            uid = user_id or self._user_id
            if not uid:
                raise ValueError("user_id is required to get provider")
            self._provider = await get_user_provider(uid)
        return self._provider

    def _calculate_risk_reward(
        self,
        ai_plan: Dict[str, Any],
        entry: float,
        stop: float,
        targets: List,
        bias: str,
    ) -> float:
        """Calculate risk/reward ratio from AI response or fallback to calculation.

        Args:
            ai_plan: AI response dict (may contain risk_reward)
            entry: Entry price
            stop: Stop loss price
            targets: List of target prices
            bias: Trade bias (bullish/bearish)

        Returns:
            Risk/reward ratio
        """
        # Use AI-provided value if available
        ai_rr = ai_plan.get("risk_reward")
        if ai_rr:
            try:
                return float(ai_rr)
            except (ValueError, TypeError):
                pass

        # Calculate from entry/stop/targets
        try:
            if entry and stop and targets:
                risk = abs(entry - stop)
                if risk > 0:
                    # For bullish: target is above entry
                    # For bearish: target is below entry
                    first_target = targets[0].price if hasattr(targets[0], 'price') else targets[0]
                    reward = abs(first_target - entry)
                    return round(reward / risk, 2)
        except (TypeError, AttributeError, IndexError):
            pass

        # Default fallback
        return 2.0

    async def gather_common_context(
        self,
        symbol: str,
        user_id: str,
    ) -> DataContext:
        """Gather common data that all sub-agents need.

        This is called once at the start, then passed to all sub-agents.

        Args:
            symbol: Stock ticker symbol
            user_id: User ID for position lookup

        Returns:
            DataContext with shared data
        """
        # Gather in parallel
        price_task = asyncio.create_task(get_current_price(symbol))
        position_task = asyncio.create_task(get_position_status(symbol, user_id))
        market_task = asyncio.create_task(get_market_context())
        news_task = asyncio.create_task(get_news_sentiment(symbol))

        price_data, position_data, market_data, news_data = await asyncio.gather(
            price_task, position_task, market_task, news_task
        )

        # Build context
        context = DataContext(
            symbol=symbol.upper(),
            user_id=user_id,
            current_price=price_data.get("price", 0),
            bid=price_data.get("bid"),
            ask=price_data.get("ask"),
            has_position=position_data.get("has_position", False),
            position_direction=position_data.get("direction"),
            position_entry=position_data.get("entry_price"),
            position_size=position_data.get("current_size"),
            position_pnl_pct=position_data.get("unrealized_pnl_pct"),
            market_direction=market_data.get("market_direction", "mixed"),
            bullish_indices=market_data.get("bullish_indices", 0),
            timestamp=datetime.utcnow().isoformat(),
            # News data
            news_sentiment=news_data.get("sentiment"),
            news_summary=news_data.get("summary"),
            recent_headlines=news_data.get("headlines", []),
        )

        self.context = context
        return context

    def _initialize_subagent_progress(self) -> Dict[str, SubAgentProgress]:
        """Initialize progress tracking for all sub-agents."""
        agents = ["day-trade-analyzer", "swing-trade-analyzer", "position-trade-analyzer"]
        progress = {}
        for agent_name in agents:
            progress[agent_name] = SubAgentProgress.create_pending(agent_name)
        self.subagent_progress = progress
        return progress

    async def generate_plan_stream(
        self,
        symbol: str,
        user_id: str,
        force_new: bool = True,
        agentic_mode: bool = True,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate trading plan with streaming progress.

        This is the main entry point that yields StreamEvents for iOS consumption.

        Args:
            symbol: Stock ticker symbol
            user_id: User ID
            force_new: Force new plan generation
            agentic_mode: When True, uses iterative AI tool-calling where the AI
                decides what to investigate. When False, uses parallel sub-agents
                (legacy V2 mode). Currently both modes use the same implementation.

        Yields:
            StreamEvent objects for each progress update
        """
        self._start_time = time.time()
        self._user_id = user_id  # Store for provider lookup
        logger.info(f"[Orchestrator] Starting plan generation for {symbol}")

        try:
            # Step 1: Gather common data
            logger.info(f"[Orchestrator] Step 1/4: Gathering common data for {symbol}")
            yield StreamEvent.orchestrator_step(
                OrchestratorStepType.GATHERING_COMMON_DATA,
                "active",
            )

            context = await self.gather_common_context(symbol, user_id)

            findings = [
                f"Price: ${context.current_price:.2f}",
                f"Market: {context.market_direction.capitalize()} ({context.bullish_indices}/4 indices up)",
            ]
            if context.has_position and context.position_direction:
                entry_str = f"@ ${context.position_entry:.2f}" if context.position_entry else ""
                findings.append(
                    f"Position: {context.position_direction.upper()} {entry_str}".strip()
                )
            else:
                findings.append("Position: None")

            # Add news sentiment
            if context.news_sentiment:
                findings.append(f"News: {context.news_sentiment.capitalize()}")

            yield StreamEvent.orchestrator_step(
                OrchestratorStepType.GATHERING_COMMON_DATA,
                "completed",
                findings,
            )
            logger.info(f"[Orchestrator] Common data gathered: {', '.join(findings)}")

            # Step 2: Spawn sub-agents
            logger.info(f"[Orchestrator] Step 2/4: Spawning parallel sub-agents for {symbol}")
            yield StreamEvent.orchestrator_step(
                OrchestratorStepType.SPAWNING_SUBAGENTS,
                "active",
            )

            # Initialize progress tracking
            progress = self._initialize_subagent_progress()
            yield StreamEvent.subagent_progress(progress)

            # Run sub-agents in parallel
            async for event in self._run_subagents_parallel(symbol, context):
                yield event

            # Step 3: Synthesize results
            logger.info(f"[Orchestrator] Step 3/4: Selecting best plan for {symbol}")
            yield StreamEvent.orchestrator_step(
                OrchestratorStepType.SELECTING_BEST,
                "active",
            )

            final_response = await self._synthesize_results(symbol, context)

            # Final result
            elapsed_ms = int((time.time() - self._start_time) * 1000)
            logger.info(f"[Orchestrator] Selected: {final_response.selected_style.upper()} trade ({final_response.selected_plan.confidence}% confidence)")
            final_response.total_analysis_time_ms = elapsed_ms

            # Transform plan to include flat target fields for iOS compatibility
            plan_dict = final_response.selected_plan.model_dump()
            targets = plan_dict.get("targets", [])
            plan_dict["target_1"] = targets[0]["price"] if len(targets) > 0 and targets[0].get("price") else None
            plan_dict["target_2"] = targets[1]["price"] if len(targets) > 1 and targets[1].get("price") else None
            plan_dict["target_3"] = targets[2]["price"] if len(targets) > 2 and targets[2].get("price") else None
            plan_dict["target_reasoning"] = "; ".join([t.get("reasoning", "") for t in targets[:3] if t.get("reasoning")])

            # Transform alternatives with flat target fields and iOS compatibility
            alternative_dicts = []
            for alt in final_response.alternatives:
                alt_dict = alt.model_dump()
                alt_targets = alt_dict.get("targets", [])
                alt_dict["target_1"] = alt_targets[0]["price"] if len(alt_targets) > 0 and alt_targets[0].get("price") else None
                alt_dict["target_2"] = alt_targets[1]["price"] if len(alt_targets) > 1 and alt_targets[1].get("price") else None
                alt_dict["target_3"] = alt_targets[2]["price"] if len(alt_targets) > 2 and alt_targets[2].get("price") else None
                alt_dict["target_reasoning"] = "; ".join([t.get("reasoning", "") for t in alt_targets[:3] if t.get("reasoning")])
                # iOS compatibility: add brief_thesis as alias for thesis
                alt_dict["brief_thesis"] = alt_dict.get("thesis", "")
                alternative_dicts.append(alt_dict)

            yield StreamEvent.final_result(
                plan=plan_dict,
                alternatives=alternative_dicts,
                selected_style=final_response.selected_style,
                selection_reasoning=final_response.selection_reasoning,
                all_citations=final_response.all_citations,
            )

            yield StreamEvent.orchestrator_step(
                OrchestratorStepType.COMPLETE,
                "completed",
            )
            logger.info(f"[Orchestrator] Step 4/4: Plan generation complete for {symbol} ({elapsed_ms}ms)")

        except Exception as e:
            logger.error(f"[Orchestrator] Error for {symbol}: {e}")
            yield StreamEvent.error(str(e))

    async def _run_subagents_parallel(
        self,
        symbol: str,
        context: DataContext,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run all three sub-agents in parallel and stream progress.

        NOTE: The Claude Agent SDK is designed to work within Claude Code CLI,
        not as a standalone library. For our backend API, we use direct API calls
        to run sub-agents in parallel.

        Args:
            symbol: Stock ticker symbol
            context: Shared data context

        Yields:
            StreamEvent for progress updates
        """
        import os

        # Check if we should use simulation mode (for testing only)
        use_simulation = os.getenv("USE_SIMULATED_SUBAGENTS", "false").lower() == "true"

        if use_simulation:
            # Simulation mode - for testing/demo only
            logger.info(f"Running sub-agents in simulation mode for {symbol}")
            async for event in self._run_subagents_simulation(symbol, context):
                yield event
            return

        # Real implementation using the AI provider abstraction
        # This runs all 3 sub-agents in parallel using asyncio.gather
        logger.info(f"Running real sub-agents for {symbol}")
        async for event in self._run_subagents_real(symbol, context):
            yield event

    async def _run_subagents_real(
        self,
        symbol: str,
        context: DataContext,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run real sub-agents using the AI provider abstraction.

        This runs all 3 trade-style analyses in parallel using asyncio.gather.
        Each agent gathers its own timeframe-specific data and analyzes charts.
        Uses the provider abstraction to work with both Claude and Grok.
        """
        from app.agent.sdk.subagent_definitions import get_all_subagent_definitions
        from app.agent.sdk import tools as sdk_tools
        import json

        # Get the provider for this user
        provider = await self._get_provider()

        position_context = context.to_prompt_context() if context else "No context available."

        # Build news context string from gathered data
        news_context = "No recent news available."
        if context:
            news_parts = []
            if context.news_sentiment:
                news_parts.append(f"Sentiment: {context.news_sentiment}")
            if context.news_summary:
                news_parts.append(f"Summary: {context.news_summary}")
            if context.recent_headlines:
                headlines_str = ", ".join(context.recent_headlines[:3])
                news_parts.append(f"Headlines: {headlines_str}")
            if news_parts:
                news_context = " | ".join(news_parts)

        agents_def = get_all_subagent_definitions(position_context, news_context)

        # Update status to running for all agents
        for agent_name in self.subagent_progress:
            self.subagent_progress[agent_name].status = SubAgentStatus.RUNNING
            self.subagent_progress[agent_name].current_step = "Starting analysis"

        yield StreamEvent.subagent_progress(self.subagent_progress.copy())

        # Create queue for real-time progress updates
        progress_queue: asyncio.Queue = asyncio.Queue()

        async def emit_progress():
            """Helper to put current progress on queue for real-time streaming."""
            await progress_queue.put(self.subagent_progress.copy())

        async def run_single_agent(agent_name: str, agent_def) -> SubAgentReport:
            """Run a single sub-agent with real API calls and tools."""
            from app.agent.schemas.subagent_report import (
                SubAgentReport,
                VisionAnalysisResult,
                PriceTargetWithReasoning,
            )

            trade_style = agent_name.replace("-trade-analyzer", "")
            logger.info(f"[{agent_name}] Starting analysis")

            # Update progress
            self.subagent_progress[agent_name].status = SubAgentStatus.GATHERING_DATA
            self.subagent_progress[agent_name].current_step = "Gathering price data"
            await emit_progress()

            try:
                # Step 1: Gather timeframe-specific data
                if trade_style == "day":
                    bars = await sdk_tools.get_price_bars(symbol, "5m", 3)
                    indicators = await sdk_tools.get_technical_indicators(symbol, [5, 9, 20], 14, timeframe="5m")
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "intraday")
                    volume_profile = await sdk_tools.get_volume_profile(symbol, 5)  # 5 days
                    chart_patterns = await sdk_tools.get_chart_patterns(symbol, 20)  # 20 days
                elif trade_style == "swing":
                    bars = await sdk_tools.get_price_bars(symbol, "1d", 100)
                    indicators = await sdk_tools.get_technical_indicators(symbol, [9, 21, 50], 14, timeframe="1d")
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "daily")
                    volume_profile = await sdk_tools.get_volume_profile(symbol, 50)  # 50 days
                    chart_patterns = await sdk_tools.get_chart_patterns(symbol, 100)  # 100 days
                else:  # position
                    bars = await sdk_tools.get_price_bars(symbol, "1w", 365)
                    indicators = await sdk_tools.get_technical_indicators(symbol, [21, 50, 200], 14, timeframe="1w")
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "weekly")
                    volume_profile = await sdk_tools.get_volume_profile(symbol, 200)  # 200 days
                    chart_patterns = await sdk_tools.get_chart_patterns(symbol, 200)  # 200 days

                # Get divergences (same lookback for all trade styles - daily chart divergences)
                divergences = await sdk_tools.get_divergences(symbol, lookback=50)

                # Get Fibonacci levels for the trade style
                fib_levels = await sdk_tools.get_fibonacci_levels(
                    symbol,
                    bars.get("bars", []),
                    trade_style
                )

                self.subagent_progress[agent_name].current_step = "Generating chart"
                self.subagent_progress[agent_name].status = SubAgentStatus.GENERATING_CHART
                await emit_progress()

                # Step 2: Generate chart
                timeframe_map = {"day": "5m", "swing": "1d", "position": "1w"}
                days_map = {"day": 3, "swing": 100, "position": 365}
                chart_result = await sdk_tools.generate_chart(
                    symbol,
                    timeframe_map[trade_style],
                    days_map[trade_style]
                )

                self.subagent_progress[agent_name].current_step = "Analyzing chart with Vision"
                self.subagent_progress[agent_name].status = SubAgentStatus.ANALYZING_CHART
                await emit_progress()

                # Step 3: Vision analysis with retry (uses user's selected provider - Claude or Grok)
                vision_result = {}
                if chart_result.get("chart_image_base64"):
                    try:
                        vision_result = await retry_async(
                            sdk_tools.analyze_chart_vision,
                            symbol,
                            chart_result["chart_image_base64"],
                            trade_style,
                            provider,  # Pass user's selected provider for vision analysis
                            context.user_id,  # Pass user_id for usage tracking
                            max_attempts=2,  # Quick retry for vision
                            delay=0.5,
                        )
                    except Exception as ve:
                        logger.warning(f"[{agent_name}] Vision analysis failed after retries: {ve}")
                        vision_result = {}  # Continue without vision

                self.subagent_progress[agent_name].current_step = "Generating trade plan"
                self.subagent_progress[agent_name].status = SubAgentStatus.GENERATING_PLAN
                await emit_progress()

                # Step 4: Build report from gathered data
                # Handle None values explicitly - .get() returns None if key exists with None value
                current_price = bars.get("current_price") or context.current_price or 0
                atr_pct = bars.get("atr_pct") or 2.0

                # Determine suitability based on ATR (aligned with prompt thresholds)
                suitable = False
                if trade_style == "day" and atr_pct > 3.0:  # Day trades need high volatility (>3%)
                    suitable = True
                elif trade_style == "swing" and 1.0 <= atr_pct <= 3.0:  # Swing trades need moderate volatility (1-3%)
                    suitable = True
                elif trade_style == "position" and atr_pct < 1.5:  # Position trades need low volatility (<1.5%)
                    # Position trades also require EMA alignment for trending conditions
                    ema_trend = indicators.get("ema_trend") or "unknown"
                    suitable = ema_trend in ["bullish_aligned", "bearish_aligned"]

                # Build confidence from indicators
                base_confidence = 50
                ema_trend = indicators.get("ema_trend") or "unknown"
                if ema_trend == "bullish_aligned":
                    base_confidence += 15
                elif ema_trend == "bearish_aligned":
                    base_confidence += 10

                # Get RSI value, handling None explicitly
                rsi_data = indicators.get("rsi") or {}
                rsi = rsi_data.get("value") if isinstance(rsi_data, dict) else None
                rsi = rsi if rsi is not None else 50  # Default to 50 if None
                if 40 <= rsi <= 60:
                    base_confidence += 5

                # Add vision modifier (range matches vision prompt: -20 to +20)
                # Validate that the modifier is numeric to prevent crashes
                try:
                    raw_vision_modifier = vision_result.get("confidence_modifier", 0)
                    vision_modifier = float(raw_vision_modifier) if raw_vision_modifier is not None else 0
                    vision_modifier = max(-20, min(20, vision_modifier))  # Clamp to valid range
                except (ValueError, TypeError):
                    logger.warning(f"[{agent_name}] Invalid vision confidence_modifier: {raw_vision_modifier}, using 0")
                    vision_modifier = 0
                confidence = min(100, max(0, base_confidence + vision_modifier))

                # Determine bias
                if ema_trend == "bullish_aligned":
                    bias = "bullish"
                elif ema_trend == "bearish_aligned":
                    bias = "bearish"
                else:
                    bias = "neutral"

                # Build support/resistance - defensive extraction of prices
                raw_supports = sr_levels.get("support", []) or []
                raw_resistances = sr_levels.get("resistance", []) or []
                supports = [
                    s.get("price") for s in raw_supports[:3]
                    if isinstance(s, dict) and isinstance(s.get("price"), (int, float))
                ]
                resistances = [
                    r.get("price") for r in raw_resistances[:3]
                    if isinstance(r, dict) and isinstance(r.get("price"), (int, float))
                ]

                # Calculate entry/stop/targets based on bias
                if bias == "bullish" and supports:
                    # Bullish setup: entry near support, stop below, targets above
                    entry_low = supports[0] if supports else current_price * 0.98
                    entry_high = current_price * 0.995
                    stop_loss = entry_low * 0.97
                    targets = [
                        PriceTargetWithReasoning(price=round(current_price * 1.03, 2), reasoning="First resistance"),
                        PriceTargetWithReasoning(price=round(current_price * 1.06, 2), reasoning="Second resistance"),
                    ]
                elif bias == "bearish" and resistances:
                    # Bearish/short setup: entry near resistance, stop above, targets below
                    entry_low = current_price * 1.005
                    entry_high = resistances[0] if resistances else current_price * 1.02
                    stop_loss = entry_high * 1.03  # Stop above resistance for shorts
                    targets = [
                        PriceTargetWithReasoning(price=round(current_price * 0.97, 2), reasoning="First support target"),
                        PriceTargetWithReasoning(price=round(current_price * 0.94, 2), reasoning="Second support target"),
                    ]
                else:
                    # Neutral or missing levels - use conservative defaults
                    entry_low = current_price * 0.98
                    entry_high = current_price * 0.995
                    stop_loss = current_price * 0.95
                    targets = [
                        PriceTargetWithReasoning(price=round(current_price * 1.05, 2), reasoning="Target 1"),
                    ]

                holding_periods = {"day": "1-4 hours", "swing": "3-7 days", "position": "2-6 weeks"}

                # ============================================================
                # Position Management Logic
                # ============================================================
                position_recommendation = None
                position_aligned = True
                risk_warnings = []
                what_to_watch = ["Volume confirmation", "Break of key levels"]

                if context.has_position and context.position_entry:
                    entry_price = context.position_entry
                    pnl_pct = ((current_price - entry_price) / entry_price) * 100
                    is_long = context.position_direction == "long"

                    # Determine position recommendation based on P&L and technicals
                    if is_long:
                        # LONG position management
                        if pnl_pct >= 50:
                            # Large profit - consider taking some off
                            position_recommendation = "trim"
                            risk_warnings.append(f"Position up {pnl_pct:.1f}% - consider taking partial profits")
                            what_to_watch.append(f"Set trailing stop below ${supports[0]:.2f}" if supports else "Set trailing stop")
                        elif pnl_pct >= 20:
                            # Good profit - hold but tighten stops
                            if ema_trend == "bearish_aligned":
                                position_recommendation = "trim"
                                risk_warnings.append("Technicals turning bearish - protect profits")
                            else:
                                position_recommendation = "hold"
                                what_to_watch.append("Tighten stop to breakeven or better")
                        elif pnl_pct >= 0:
                            # Small profit - hold if trend intact
                            if ema_trend == "bearish_aligned":
                                position_recommendation = "reduce"
                                risk_warnings.append("Trend weakening - consider reducing exposure")
                            else:
                                position_recommendation = "hold"
                        elif pnl_pct >= -10:
                            # Small loss - evaluate trend
                            if ema_trend == "bullish_aligned":
                                position_recommendation = "hold"
                                what_to_watch.append("Watch for bounce at support")
                            else:
                                position_recommendation = "reduce"
                                risk_warnings.append("In drawdown with weak technicals")
                        else:
                            # Significant loss
                            if ema_trend == "bearish_aligned":
                                position_recommendation = "exit"
                                risk_warnings.append(f"Position down {abs(pnl_pct):.1f}% with bearish technicals")
                            else:
                                position_recommendation = "hold"
                                risk_warnings.append("Deep drawdown - reassess thesis")

                        # Check if price is near key levels
                        if supports and current_price <= supports[0] * 1.02:
                            what_to_watch.append(f"Testing support at ${supports[0]:.2f}")
                        if resistances and current_price >= resistances[0] * 0.98:
                            what_to_watch.append(f"Approaching resistance at ${resistances[0]:.2f}")

                        # Bias alignment check
                        if bias == "bearish":
                            position_aligned = False
                            risk_warnings.append("Analysis bias conflicts with long position")

                    else:
                        # SHORT position management (inverse logic)
                        pnl_pct = -pnl_pct  # Invert for shorts
                        if pnl_pct >= 20:
                            position_recommendation = "trim" if ema_trend == "bullish_aligned" else "hold"
                        elif pnl_pct >= 0:
                            position_recommendation = "hold"
                        else:
                            position_recommendation = "exit" if ema_trend == "bullish_aligned" else "hold"

                        if bias == "bullish":
                            position_aligned = False
                            risk_warnings.append("Analysis bias conflicts with short position")

                    # Add position context to thesis
                    position_context_str = f" You have a {context.position_direction.upper()} position @ ${entry_price:.2f} ({pnl_pct:+.1f}%)."
                else:
                    position_context_str = ""

                # ============================================================
                # CALL CLAUDE FOR REAL ANALYSIS
                # ============================================================
                self.subagent_progress[agent_name].current_step = "AI generating analysis"

                # Build context for Claude with comprehensive data
                analysis_context = f"""
## Stock: {symbol}
## Trade Style: {trade_style.upper()}
## Current Price: ${current_price:.2f}

## Technical Data (Enhanced):
{format_enhanced_technicals(indicators, bars)}

## Institutional-Grade Indicators:
{format_institutional_indicators(indicators)}

## Divergence Analysis:
{format_divergences(divergences)}

## Key Levels (with Institutional Metrics):
{format_levels_with_metrics(sr_levels)}

## Fibonacci Analysis:
{format_fibonacci_levels(fib_levels)}

## Volume Profile:
{format_volume_profile(volume_profile)}

## Chart Patterns:
{format_chart_patterns(chart_patterns)}

## Vision Analysis:
{vision_result.get('summary', 'No chart analysis available')}
- Visual Patterns: {', '.join(vision_result.get('visual_patterns', [])) or 'None'}
- Trend Quality: {vision_result.get('trend_quality', 'unknown')}
- Warning Signs: {', '.join(vision_result.get('warning_signs', [])) or 'None'}

## Position Context:
{position_context_str if position_context_str else 'No existing position.'}

Based on this comprehensive data, provide your {trade_style} trade analysis.
Use the institutional metrics to assess level reliability for stop placement.
Reference chart pattern success rates when setting confidence.
Consider divergence signals - bullish divergence suggests potential reversal up, bearish suggests reversal down.
For day trades, pay attention to VWAP positioning. For position trades, use ADX to assess trend strength.
Use Fibonacci retracement levels (38.2%, 50%, 61.8%, 78.6%) for entry zones and extensions (1.272, 1.618) for targets.
"""

                # Get the trade-style specific prompt
                from app.agent.prompts import (
                    build_day_trade_prompt,
                    build_swing_trade_prompt,
                    build_position_trade_prompt,
                )

                if trade_style == "day":
                    system_prompt = build_day_trade_prompt(symbol, context.to_prompt_context(), news_context)
                elif trade_style == "swing":
                    system_prompt = build_swing_trade_prompt(symbol, context.to_prompt_context(), news_context)
                else:
                    system_prompt = build_position_trade_prompt(symbol, context.to_prompt_context(), news_context)

                # Build the user message for analysis
                user_message = f"""Analyze {symbol} for a {trade_style} trade setup.

{analysis_context}

Respond with a JSON object containing:
{{
    "suitable": true/false,
    "confidence": 0-100,
    "bias": "bullish" or "bearish" or "neutral",
    "thesis": "Your detailed 2-3 sentence analysis of this setup...",
    "entry_zone_low": price,
    "entry_zone_high": price,
    "entry_reasoning": "Why this entry zone...",
    "stop_loss": price,
    "stop_reasoning": "Why this stop level...",
    "targets": [
        {{"price": X, "reasoning": "Why this target..."}}
    ],
    "risk_reward": ratio,
    "holding_period": "e.g., 3-5 days",
    "invalidation_criteria": "What would invalidate this setup...",
    "setup_explanation": "The technical pattern or setup type...",
    "what_to_watch": ["Key thing 1", "Key thing 2"],
    "risk_warnings": ["Warning 1 if any"]
}}

Return ONLY the JSON object, no other text."""

                # Call the provider for real analysis with retry
                # Use search if provider supports X search (Grok), otherwise no search for sub-agents
                search_params = None
                if provider.supports_x_search:
                    search_params = get_x_search_parameters()

                plan_response = await retry_async(
                    provider.create_message,
                    messages=[AIMessage(role="user", content=user_message)],
                    system=system_prompt,
                    model_type="planning",
                    max_tokens=2000,
                    search_parameters=search_params,
                    max_attempts=2,  # Retry once on failure
                    delay=1.0,
                )

                # Track usage for sub-agent analysis
                try:
                    tracker = get_usage_tracker()
                    usage_provider = UsageModelProvider.GROK if provider.supports_x_search else UsageModelProvider.CLAUDE
                    await tracker.track_ai_response(
                        user_id=context.user_id,
                        provider=usage_provider,
                        model=provider.get_model("planning"),
                        operation_type=OperationType.SUBAGENT,
                        response=plan_response,
                        symbol=symbol,
                        endpoint=f"/create-plan-v2-stream/{agent_name}",
                    )
                except Exception as track_err:
                    logger.warning(f"[{agent_name}] Failed to track usage: {track_err}")

                # Capture X/social citations from Grok response
                x_citations = []
                logger.debug(f"[{agent_name}] Raw citations from response: {plan_response.citations}")
                logger.debug(f"[{agent_name}] Raw response keys: {list(plan_response.raw_response.keys()) if plan_response.raw_response else 'None'}")
                if plan_response.citations:
                    for citation in plan_response.citations:
                        if isinstance(citation, str):
                            x_citations.append(citation)
                        elif isinstance(citation, dict) and "url" in citation:
                            x_citations.append(citation["url"])
                    logger.info(f"[{agent_name}] Captured {len(x_citations)} X/social citations")
                else:
                    logger.warning(f"[{agent_name}] No citations returned from Grok X search")

                # Parse the AI response with robust JSON extraction
                ai_text = plan_response.content.strip()
                ai_plan = {}

                # Strategy 1: Try to find JSON in markdown code blocks
                import re
                json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', ai_text)
                if json_match:
                    try:
                        ai_plan = json.loads(json_match.group(1))
                    except json.JSONDecodeError:
                        pass

                # Strategy 2: If no code block, try to find JSON by brackets
                if not ai_plan:
                    # Find the outermost JSON object
                    first_brace = ai_text.find('{')
                    last_brace = ai_text.rfind('}')
                    if first_brace != -1 and last_brace > first_brace:
                        try:
                            ai_plan = json.loads(ai_text[first_brace:last_brace + 1])
                        except json.JSONDecodeError:
                            pass

                # Strategy 3: Try the whole response as JSON
                if not ai_plan:
                    try:
                        ai_plan = json.loads(ai_text)
                    except json.JSONDecodeError:
                        logger.warning(f"[{agent_name}] Failed to parse AI response, using fallback. Response preview: {ai_text[:200]}")

                # Use AI's analysis or fall back to calculated values
                # Note: Use `or` to handle both missing keys AND null values
                thesis = ai_plan.get("thesis") or f"{trade_style.capitalize()} trade analysis for {symbol}. ATR: {atr_pct:.1f}%, EMA trend: {ema_trend}.{position_context_str}"
                final_suitable = ai_plan.get("suitable") if ai_plan.get("suitable") is not None else suitable
                final_confidence = ai_plan.get("confidence") or confidence
                final_bias = ai_plan.get("bias") or bias
                final_entry_low = ai_plan.get("entry_zone_low") or entry_low
                final_entry_high = ai_plan.get("entry_zone_high") or entry_high
                final_stop = ai_plan.get("stop_loss") or stop_loss
                final_holding = ai_plan.get("holding_period") or holding_periods[trade_style]

                # Parse and validate targets from AI response
                ai_targets = ai_plan.get("targets") or []
                if ai_targets and isinstance(ai_targets, list):
                    parsed_targets = []
                    for t in ai_targets[:3]:
                        if isinstance(t, dict):
                            target_price = t.get("price")
                            if target_price is not None:
                                try:
                                    price_float = float(target_price)
                                    # Validate: positive and within reasonable range (0.5x to 3x current price)
                                    if price_float > 0 and current_price * 0.5 <= price_float <= current_price * 3:
                                        parsed_targets.append(
                                            PriceTargetWithReasoning(
                                                price=round(price_float, 2),
                                                reasoning=t.get("reasoning") or "Target level"
                                            )
                                        )
                                    else:
                                        logger.warning(f"[{agent_name}] Invalid target price {price_float} (current: {current_price}), skipping")
                                except (ValueError, TypeError):
                                    logger.warning(f"[{agent_name}] Non-numeric target price: {target_price}, skipping")
                    if parsed_targets:
                        targets = parsed_targets

                # Merge what_to_watch and risk_warnings
                ai_watch = ai_plan.get("what_to_watch", [])
                ai_warnings = ai_plan.get("risk_warnings", [])
                if ai_watch:
                    what_to_watch = ai_watch
                if ai_warnings:
                    risk_warnings = list(set(risk_warnings + ai_warnings))

                logger.info(f"[{agent_name}] Completed: {final_bias} bias, {final_confidence}% confidence")
                return SubAgentReport(
                    trade_style=trade_style,
                    symbol=symbol,
                    analysis_timestamp=datetime.utcnow().isoformat(),
                    suitable=final_suitable,
                    confidence=final_confidence,
                    bias=final_bias,
                    thesis=thesis,
                    vision_analysis=VisionAnalysisResult(
                        trend_quality=vision_result.get("trend_quality", "moderate"),
                        visual_patterns=vision_result.get("visual_patterns", []),
                        candlestick_patterns=vision_result.get("candlestick_patterns", []),
                        ema_structure=vision_result.get("ema_structure", indicators.get("ema_trend", "")),
                        volume_confirmation=vision_result.get("volume_confirmation", ""),
                        warning_signs=vision_result.get("warning_signs", []),
                        confidence_modifier=vision_modifier,
                        summary=vision_result.get("summary", ""),
                    ),
                    entry_zone_low=round(float(final_entry_low), 2) if final_entry_low else round(current_price * 0.98, 2),
                    entry_zone_high=round(float(final_entry_high), 2) if final_entry_high else round(current_price * 0.995, 2),
                    entry_reasoning=ai_plan.get("entry_reasoning") or (f"Near support at ${final_entry_low:.2f}" if final_entry_low else "Near current price"),
                    stop_loss=round(float(final_stop), 2) if final_stop else round(current_price * 0.95, 2),
                    stop_reasoning=ai_plan.get("stop_reasoning") or "Below recent support",
                    targets=targets,
                    risk_reward=self._calculate_risk_reward(ai_plan, final_entry_high, final_stop, targets, bias),
                    position_size_pct=2.0,
                    holding_period=final_holding,
                    key_supports=supports or [],
                    key_resistances=resistances or [],
                    invalidation_criteria=ai_plan.get("invalidation_criteria") or (f"Close below ${final_stop:.2f}" if final_stop else "Price closes below stop level"),
                    position_aligned=position_aligned,
                    position_recommendation=position_recommendation,
                    setup_explanation=ai_plan.get("setup_explanation") or f"This {trade_style} setup is based on {ema_trend} EMA alignment.",
                    what_to_watch=what_to_watch or ["Monitor price action", "Watch for volume confirmation"],
                    risk_warnings=risk_warnings or [],
                    atr_percent=atr_pct,
                    technical_summary=f"RSI: {rsi:.1f}, EMA trend: {ema_trend}",
                    x_citations=x_citations,
                )

            except Exception as e:
                error_msg = str(e) or f"{type(e).__name__}: {repr(e)}"
                logger.error(f"[{agent_name}] Error: {error_msg}")
                self.subagent_progress[agent_name].status = SubAgentStatus.FAILED
                self.subagent_progress[agent_name].error_message = error_msg
                self.subagent_progress[agent_name].findings = [
                    f"Error: {error_msg[:40]}",
                    "Using fallback data",
                ]
                await emit_progress()  # Emit error progress
                # Return mock report on error
                return self._create_mock_report(symbol, trade_style, context)

        # Run all agents in parallel while yielding progress events
        try:
            # Create tasks (don't await yet)
            tasks = [
                asyncio.create_task(run_single_agent(name, agent_def))
                for name, agent_def in agents_def.items()
            ]

            # While tasks are running, consume progress events from queue and yield them
            while not all(task.done() for task in tasks):
                try:
                    # Wait for progress update with short timeout
                    progress = await asyncio.wait_for(progress_queue.get(), timeout=0.2)
                    yield StreamEvent.subagent_progress(progress)
                except asyncio.TimeoutError:
                    pass  # No progress update, check if tasks are done

            # Drain any remaining progress events in queue
            while not progress_queue.empty():
                try:
                    progress = progress_queue.get_nowait()
                    yield StreamEvent.subagent_progress(progress)
                except asyncio.QueueEmpty:
                    break

            # Get results from completed tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for (agent_name, _), result in zip(agents_def.items(), results):
                if isinstance(result, Exception):
                    error_msg = str(result) or f"{type(result).__name__}: {repr(result)}"
                    self.subagent_progress[agent_name].status = SubAgentStatus.FAILED
                    self.subagent_progress[agent_name].error_message = error_msg
                    self.subagent_progress[agent_name].findings = [f"Error: {error_msg[:50]}"]
                else:
                    self.subagent_reports[agent_name] = result
                    self.subagent_progress[agent_name].status = SubAgentStatus.COMPLETED
                    self.subagent_progress[agent_name].findings = [
                        f"{result.bias.capitalize()}",
                        f"{result.confidence}% confidence",
                        f"Setup: {'Yes' if result.suitable else 'No'}",
                    ]

                yield StreamEvent.subagent_complete(
                    agent_name=agent_name,
                    findings=self.subagent_progress[agent_name].findings,
                )

        except Exception as e:
            logger.error(f"Error running parallel agents: {e}")
            # Fall back to simulation
            async for event in self._run_subagents_simulation(symbol, context):
                yield event

    async def _run_subagents_simulation(
        self,
        symbol: str,
        context: DataContext,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Simulate sub-agent execution for development/testing.

        This is used when the Claude Agent SDK is not available.
        It demonstrates the expected flow without actual API calls.
        """
        agents = ["day-trade-analyzer", "swing-trade-analyzer", "position-trade-analyzer"]
        steps = [
            ("gathering_data", "Gathering data"),
            ("calculating_technicals", "Calculating technicals"),
            ("generating_chart", "Generating chart"),
            ("analyzing_chart", "Analyzing chart"),
            ("generating_plan", "Generating plan"),
        ]

        # Simulate progress for each agent
        for step_idx, (status, step_name) in enumerate(steps):
            await asyncio.sleep(0.5)  # Simulate work

            # Update all agents to this step
            for agent_name in agents:
                self.subagent_progress[agent_name].status = SubAgentStatus(status)
                self.subagent_progress[agent_name].current_step = step_name
                self.subagent_progress[agent_name].steps_completed = [
                    s[1] for s in steps[:step_idx]
                ]

            yield StreamEvent.subagent_progress(self.subagent_progress.copy())

        # Simulate completion with mock reports
        for agent_name in agents:
            await asyncio.sleep(0.3)

            # Create mock report
            trade_style = agent_name.replace("-analyzer", "").replace("-trade", "")
            mock_report = self._create_mock_report(symbol, trade_style, context)
            self.subagent_reports[agent_name] = mock_report

            self.subagent_progress[agent_name].status = SubAgentStatus.COMPLETED
            self.subagent_progress[agent_name].findings = [
                f"{mock_report.bias.capitalize()}",
                f"{mock_report.confidence}% confidence",
                f"Setup: {'Yes' if mock_report.suitable else 'No'}",
            ]

            yield StreamEvent.subagent_complete(
                agent_name=agent_name,
                findings=self.subagent_progress[agent_name].findings,
            )

    def _create_mock_report(
        self,
        symbol: str,
        trade_style: str,
        context: DataContext,
    ) -> SubAgentReport:
        """Create a mock SubAgentReport for simulation mode."""
        from app.agent.schemas.subagent_report import (
            SubAgentReport,
            VisionAnalysisResult,
            PriceTargetWithReasoning,
        )

        current_price = context.current_price if context else 100.0

        # Different confidence based on style (for demo variety)
        confidence_map = {"day": 65, "swing": 78, "position": 45}
        suitable_map = {"day": True, "swing": True, "position": False}

        return SubAgentReport(
            trade_style=trade_style,
            symbol=symbol,
            analysis_timestamp=datetime.utcnow().isoformat(),
            suitable=suitable_map.get(trade_style, False),
            confidence=confidence_map.get(trade_style, 50),
            bias="bullish",
            thesis=f"[FALLBACK DATA] Analysis unavailable due to API timeout. This is simulated {trade_style} trade data for {symbol}.",
            vision_analysis=VisionAnalysisResult(
                trend_quality="moderate",
                visual_patterns=["bull flag"] if trade_style == "swing" else [],
                candlestick_patterns=[],
                ema_structure="Price above EMAs",
                volume_confirmation="Volume confirming",
                warning_signs=[],
                confidence_modifier=5,
                summary=f"Simulated vision analysis for {trade_style}",
            ),
            entry_zone_low=round(current_price * 0.98, 2),
            entry_zone_high=round(current_price * 0.99, 2),
            entry_reasoning="Simulated entry zone",
            stop_loss=round(current_price * 0.95, 2),
            stop_reasoning="Simulated stop",
            targets=[
                PriceTargetWithReasoning(
                    price=round(current_price * 1.05, 2),
                    reasoning="Target 1",
                ),
                PriceTargetWithReasoning(
                    price=round(current_price * 1.10, 2),
                    reasoning="Target 2",
                ),
            ],
            risk_reward=2.5,
            position_size_pct=2.0,
            holding_period="2-5 days" if trade_style == "swing" else "1-3 hours" if trade_style == "day" else "2-4 weeks",
            key_supports=[round(current_price * 0.95, 2)],
            key_resistances=[round(current_price * 1.05, 2)],
            invalidation_criteria="Simulated invalidation",
            position_aligned=True,
            setup_explanation=f"This is a simulated {trade_style} setup.",
            what_to_watch=["Watch for breakout", "Volume confirmation"],
            risk_warnings=["Simulated risk warning"],
            atr_percent=2.5,
            technical_summary="Simulated technicals",
        )

    def _process_sdk_message(self, message: Any) -> List[StreamEvent]:
        """Process a message from the Claude Agent SDK and extract progress events."""
        events = []

        # Check for Task tool invocations (sub-agent spawning)
        if hasattr(message, 'content') and message.content:
            for block in message.content:
                if hasattr(block, 'type') and block.type == 'tool_use':
                    if block.name == 'Task':
                        agent_type = block.input.get('subagent_type', 'unknown')
                        if agent_type in self.subagent_progress:
                            self.subagent_progress[agent_type].status = SubAgentStatus.RUNNING
                            self.subagent_progress[agent_type].current_step = "Starting analysis"
                            events.append(StreamEvent.subagent_progress(self.subagent_progress.copy()))

        # Check for parent_tool_use_id (message from within sub-agent)
        if hasattr(message, 'parent_tool_use_id') and message.parent_tool_use_id:
            # This is from a sub-agent - try to determine which one and update progress
            pass  # Would need to track tool_use_id -> agent_name mapping

        return events

    async def _synthesize_results(
        self,
        symbol: str,
        context: DataContext,
    ) -> FinalPlanResponse:
        """Synthesize results from all sub-agents and select the best plan.

        This method now calls Claude to create a comprehensive synthesis
        after selecting the best plan programmatically.

        Selection criteria (in order):
        1. Position alignment (never recommend opposite direction)
        2. Setup suitability (must have valid setup)
        3. Confidence score (higher is better)
        4. Risk/reward ratio (2:1 minimum for swing, 1.5:1 for day)

        Args:
            symbol: Stock ticker symbol
            context: Shared data context

        Returns:
            FinalPlanResponse with selected plan and alternatives
        """
        from app.agent.schemas.subagent_report import SubAgentReport, VisionAnalysisResult

        reports = list(self.subagent_reports.values())

        if not reports:
            # No reports - create empty response
            logger.warning(f"No sub-agent reports for {symbol}")

            no_setup_report = SubAgentReport(
                trade_style="swing",
                symbol=symbol,
                analysis_timestamp=datetime.utcnow().isoformat(),
                suitable=False,
                confidence=0,
                bias="neutral",
                thesis="No valid trading setup found across any trade style.",
                vision_analysis=VisionAnalysisResult(
                    trend_quality="choppy",
                    confidence_modifier=0,
                    summary="No analysis available",
                ),
            )

            return FinalPlanResponse(
                selected_plan=no_setup_report,
                selected_style="swing",
                selection_reasoning="No valid setups found.",
                alternatives=[],
                symbol=symbol,
                analysis_timestamp=datetime.utcnow().isoformat(),
                has_existing_position=context.has_position if context else False,
                position_direction=context.position_direction if context else None,
                position_entry=context.position_entry if context else None,
                market_direction=context.market_direction if context else None,
                current_price=context.current_price if context else None,
            )

        # Sort position-aligned reports first, but keep all reports
        # (We want to show conflicting analyses as alternatives with warnings)
        if context and context.has_position:
            # Sort so aligned reports come first
            reports = sorted(reports, key=lambda r: (not r.position_aligned, -r.confidence))

        # Sort by: suitable (True first) -> confidence (high first) -> risk_reward (high first)
        def sort_key(report: SubAgentReport):
            return (
                report.suitable,  # True (1) > False (0)
                report.confidence,
                report.risk_reward or 0,
            )

        sorted_reports = sorted(reports, key=sort_key, reverse=True)

        # Select best
        best_report = sorted_reports[0]

        # Keep full reports for alternatives (not truncated AlternativePlan)
        alternatives = []
        for report in sorted_reports[1:]:
            # Determine why not selected
            why_parts = []
            if not report.position_aligned:
                why_parts.append("Conflicts with existing position")
            if not report.suitable:
                why_parts.append("No valid setup for this trade style")
            elif report.confidence < best_report.confidence:
                why_parts.append(f"Lower confidence ({report.confidence}% vs {best_report.confidence}%)")
            elif (report.risk_reward or 0) < (best_report.risk_reward or 0):
                why_parts.append(f"Lower risk/reward ({report.risk_reward:.1f} vs {best_report.risk_reward:.1f})")

            why_not = ". ".join(why_parts) if why_parts else "Another style better fits current conditions"

            # Set why_not_selected on the full report and keep it
            report.why_not_selected = why_not
            alternatives.append(report)

        # Build selection reasoning
        reasoning_parts = []
        if best_report.suitable:
            reasoning_parts.append(f"{best_report.trade_style.capitalize()} trade has valid setup")
            reasoning_parts.append(f"{best_report.confidence}% confidence")
            if best_report.risk_reward:
                reasoning_parts.append(f"{best_report.risk_reward:.1f}:1 risk/reward")
        else:
            reasoning_parts.append("No valid setups found - passing on trade")

        # ============================================================
        # FINAL SYNTHESIS: Call AI provider to create comprehensive plan
        # ============================================================
        try:
            import json

            # Get the provider for this user
            provider = await self._get_provider()

            # Build summary of all analyses
            analyses_summary = []
            for report in sorted_reports:
                analyses_summary.append(f"""
## {report.trade_style.upper()} TRADE ANALYSIS:
- Suitable: {report.suitable}
- Confidence: {report.confidence}%
- Bias: {report.bias}
- Thesis: {report.thesis}
- Entry: ${report.entry_zone_low:.2f} - ${report.entry_zone_high:.2f}
- Stop: ${report.stop_loss:.2f}
- Targets: {', '.join([f'${t.price:.2f}' for t in (report.targets or [])])}
- R:R: {report.risk_reward:.1f}:1
- What to Watch: {', '.join(report.what_to_watch or [])}
- Risk Warnings: {', '.join(report.risk_warnings or [])}
""")

            position_context = ""
            if context.has_position and context.position_entry:
                pnl_pct = ((context.current_price - context.position_entry) / context.position_entry) * 100
                position_context = f"""
## EXISTING POSITION:
- Direction: {context.position_direction.upper()}
- Entry: ${context.position_entry:.2f}
- Current P&L: {pnl_pct:+.1f}%
"""

            # Build news context
            news_context = ""
            if context.news_sentiment or context.recent_headlines:
                news_context = f"""
## NEWS & SENTIMENT:
- Sentiment: {context.news_sentiment or 'Unknown'}
- Summary: {context.news_summary or 'No recent news'}
"""
                if context.recent_headlines:
                    news_context += "- Recent Headlines:\n"
                    for headline in context.recent_headlines[:3]:
                        news_context += f"  * {headline}\n"

            synthesis_prompt = f"""You are a professional trading advisor synthesizing analyses from 3 specialized agents.

## STOCK: {symbol}
## CURRENT PRICE: ${context.current_price:.2f}
## MARKET DIRECTION: {context.market_direction}
{position_context}
{news_context}

## REAL-TIME SENTIMENT (You have X/Twitter search - USE IT!)
You have access to real-time X (Twitter) search. **Search X for {symbol} right now** to find:
- Current trader sentiment and social buzz
- Breaking news or rumors being discussed
- Key influencer opinions

**Incorporate X sentiment into your synthesis**, especially in the thesis and news_impact fields.

## ALL ANALYSES:
{''.join(analyses_summary)}

## SELECTED: {best_report.trade_style.upper()} TRADE

Create a comprehensive trading plan synthesis. Respond with JSON:
{{
    "thesis": "A detailed 3-4 sentence thesis explaining the selected trade setup, why this timeframe was chosen over others, and what makes this setup compelling. Be SPECIFIC about the technical pattern, key price levels, and any news/catalysts. INCLUDE any relevant X/social sentiment you found. If there's an existing position, explicitly address whether to add, hold, trim, or exit.",
    "selection_reasoning": "2-3 sentences explaining why {best_report.trade_style} was selected over the alternatives, comparing confidence levels and setup quality.",
    "targets": [
        {{"price": <number>, "reasoning": "Why this is a valid target level based on resistance/technicals"}},
        {{"price": <number>, "reasoning": "Second target reasoning"}},
        {{"price": <number>, "reasoning": "Third target reasoning (if applicable)"}}
    ],
    "what_to_watch": ["5-7 specific, actionable items to monitor - include exact price levels like 'Break above $XXX triggers acceleration', volume thresholds, time-based triggers, and any relevant news catalysts or social sentiment shifts"],
    "risk_warnings": ["3-5 specific risks - technical invalidation levels, market risks, news/earnings risks, social sentiment risks, position-specific warnings"],
    "entry_reasoning": "Why this specific entry zone makes sense based on support levels and the technical setup",
    "stop_reasoning": "Why this stop level is appropriate - reference specific support/ATR/pattern invalidation",
    "news_impact": "Assessment of how recent news AND real-time X sentiment affects this trade thesis (2-3 sentences). Reference specific sentiment trends you found."
}}

Return ONLY valid JSON."""

            # Use search if provider supports X search (Grok) for real-time sentiment
            search_params = None
            if provider.supports_x_search:
                search_params = get_x_search_parameters()
            elif provider.supports_web_search:
                # Claude: use web search for synthesis
                search_params = SearchParameters(mode="on", sources=[{"type": "web"}], return_citations=True)

            synthesis_response = await retry_async(
                provider.create_message,
                messages=[AIMessage(role="user", content=synthesis_prompt)],
                system=None,
                model_type="planning",
                max_tokens=1500,
                search_parameters=search_params,
                max_attempts=2,  # Retry once on failure
                delay=1.0,
            )

            # Track usage for synthesis
            try:
                tracker = get_usage_tracker()
                usage_provider = UsageModelProvider.GROK if provider.supports_x_search else UsageModelProvider.CLAUDE
                await tracker.track_ai_response(
                    user_id=context.user_id if context else self._user_id,
                    provider=usage_provider,
                    model=provider.get_model("planning"),
                    operation_type=OperationType.ORCHESTRATOR,
                    response=synthesis_response,
                    symbol=symbol,
                    endpoint="/create-plan-v2-stream/synthesis",
                )
            except Exception as track_err:
                logger.warning(f"[Orchestrator] Failed to track synthesis usage: {track_err}")

            synthesis_text = synthesis_response.content.strip()
            synthesis = {}

            # Robust JSON parsing with multiple strategies
            import re

            # Strategy 1: Try to find JSON in markdown code blocks
            json_match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', synthesis_text)
            if json_match:
                try:
                    synthesis = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Strategy 2: If no code block, try to find JSON by brackets
            if not synthesis:
                first_brace = synthesis_text.find('{')
                last_brace = synthesis_text.rfind('}')
                if first_brace != -1 and last_brace > first_brace:
                    try:
                        synthesis = json.loads(synthesis_text[first_brace:last_brace + 1])
                    except json.JSONDecodeError:
                        pass

            # Strategy 3: Try the whole response as JSON
            if not synthesis:
                try:
                    synthesis = json.loads(synthesis_text)
                except json.JSONDecodeError:
                    logger.warning(f"[Orchestrator] Failed to parse synthesis response, using best agent output. Preview: {synthesis_text[:200]}")

            # Update best_report with synthesized content
            if synthesis.get("thesis"):
                best_report.thesis = synthesis["thesis"]
            if synthesis.get("what_to_watch"):
                best_report.what_to_watch = synthesis["what_to_watch"]
            if synthesis.get("risk_warnings"):
                best_report.risk_warnings = synthesis["risk_warnings"]
            if synthesis.get("entry_reasoning"):
                best_report.entry_reasoning = synthesis["entry_reasoning"]
            if synthesis.get("stop_reasoning"):
                best_report.stop_reasoning = synthesis["stop_reasoning"]
            if synthesis.get("selection_reasoning"):
                reasoning_parts = [synthesis["selection_reasoning"]]

            # Parse and validate synthesized targets
            from app.agent.schemas.subagent_report import PriceTargetWithReasoning
            synth_targets = synthesis.get("targets", [])
            current_price = context.current_price if context else 0
            if synth_targets and isinstance(synth_targets, list):
                parsed_targets = []
                for t in synth_targets[:3]:
                    if isinstance(t, dict) and t.get("price") is not None:
                        try:
                            price_float = float(t["price"])
                            # Validate: positive and within reasonable range (0.5x to 3x current price)
                            if price_float > 0 and (current_price == 0 or current_price * 0.5 <= price_float <= current_price * 3):
                                parsed_targets.append(
                                    PriceTargetWithReasoning(
                                        price=round(price_float, 2),
                                        reasoning=t.get("reasoning") or "Target level"
                                    )
                                )
                            else:
                                logger.warning(f"[Orchestrator] Invalid synthesis target {price_float} (current: {current_price}), skipping")
                        except (ValueError, TypeError) as te:
                            logger.warning(f"[Orchestrator] Failed to parse target {t}: {te}")
                if parsed_targets:
                    best_report.targets = parsed_targets
                    logger.info(f"[Orchestrator] Set {len(parsed_targets)} targets from synthesis")
                else:
                    logger.warning(f"[Orchestrator] Synthesis returned no valid targets")

            # Fallback: ensure we have targets even if synthesis failed
            if not best_report.targets or len(best_report.targets) == 0:
                logger.warning(f"[Orchestrator] No targets from synthesis, attempting fallback")
                # Try to generate targets from entry zone and risk/reward
                if best_report.entry_zone_high and best_report.stop_loss and context.current_price:
                    entry = best_report.entry_zone_high
                    stop = best_report.stop_loss
                    risk = abs(entry - stop)

                    if best_report.bias == "bullish":
                        # Bullish targets: above entry
                        t1 = round(entry + (risk * 1.5), 2)
                        t2 = round(entry + (risk * 2.5), 2)
                        t3 = round(entry + (risk * 4.0), 2)
                    else:
                        # Bearish targets: below entry
                        t1 = round(entry - (risk * 1.5), 2)
                        t2 = round(entry - (risk * 2.5), 2)
                        t3 = round(entry - (risk * 4.0), 2)

                    best_report.targets = [
                        PriceTargetWithReasoning(price=t1, reasoning="1.5R target based on risk/reward"),
                        PriceTargetWithReasoning(price=t2, reasoning="2.5R target for partial profits"),
                        PriceTargetWithReasoning(price=t3, reasoning="4R extended target"),
                    ]
                    logger.info(f"[Orchestrator] Generated fallback targets: {t1}, {t2}, {t3}")

            # Add news impact to thesis if provided
            if synthesis.get("news_impact"):
                # Append news context to thesis
                best_report.thesis = f"{best_report.thesis} {synthesis['news_impact']}"

            logger.info(f"[Orchestrator] Synthesis complete for {symbol}")

        except Exception as e:
            logger.warning(f"[Orchestrator] Synthesis call failed, using agent output: {e}")
            # Continue with the best_report as-is

        # Aggregate all X/social citations from all sub-agent reports
        all_citations = set()
        if best_report.x_citations:
            all_citations.update(best_report.x_citations)
        for alt in alternatives:
            if alt.x_citations:
                all_citations.update(alt.x_citations)
        # Prioritize actual X posts (x.com URLs)
        sorted_citations = sorted(
            all_citations,
            key=lambda url: (0 if "x.com" in url else 1, url)
        )
        logger.info(f"[Orchestrator] Aggregated {len(sorted_citations)} unique citations")

        return FinalPlanResponse(
            selected_plan=best_report,
            selected_style=best_report.trade_style,
            selection_reasoning=". ".join(reasoning_parts),
            alternatives=alternatives,
            symbol=symbol,
            analysis_timestamp=datetime.utcnow().isoformat(),
            has_existing_position=context.has_position if context else False,
            position_direction=context.position_direction if context else None,
            position_entry=context.position_entry if context else None,
            market_direction=context.market_direction if context else None,
            current_price=context.current_price if context else None,
            day_trade_analyzed="day-trade-analyzer" in self.subagent_reports,
            swing_trade_analyzed="swing-trade-analyzer" in self.subagent_reports,
            position_trade_analyzed="position-trade-analyzer" in self.subagent_reports,
            all_citations=sorted_citations,
        )
