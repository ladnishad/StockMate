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
from app.agent.providers import AIMessage, AIProvider, SearchParameters
from app.agent.providers.grok_provider import get_x_search_parameters

logger = logging.getLogger(__name__)


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
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate trading plan with streaming progress.

        This is the main entry point that yields StreamEvents for iOS consumption.

        Args:
            symbol: Stock ticker symbol
            user_id: User ID
            force_new: Force new plan generation

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

            try:
                # Step 1: Gather timeframe-specific data
                if trade_style == "day":
                    bars = await sdk_tools.get_price_bars(symbol, "5m", 3)
                    indicators = await sdk_tools.get_technical_indicators(symbol, [5, 9, 20], 14)
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "intraday")
                elif trade_style == "swing":
                    bars = await sdk_tools.get_price_bars(symbol, "1d", 100)
                    indicators = await sdk_tools.get_technical_indicators(symbol, [9, 21, 50], 14)
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "daily")
                else:  # position
                    bars = await sdk_tools.get_price_bars(symbol, "1w", 52)
                    indicators = await sdk_tools.get_technical_indicators(symbol, [21, 50, 200], 14)
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "weekly")

                self.subagent_progress[agent_name].current_step = "Generating chart"
                self.subagent_progress[agent_name].status = SubAgentStatus.GENERATING_CHART

                # Step 2: Generate chart
                timeframe_map = {"day": "5m", "swing": "1d", "position": "1w"}
                days_map = {"day": 3, "swing": 100, "position": 52}
                chart_result = await sdk_tools.generate_chart(
                    symbol,
                    timeframe_map[trade_style],
                    days_map[trade_style]
                )

                self.subagent_progress[agent_name].current_step = "Analyzing chart with Vision"
                self.subagent_progress[agent_name].status = SubAgentStatus.ANALYZING_CHART

                # Step 3: Vision analysis
                vision_result = {}
                if chart_result.get("chart_image_base64"):
                    vision_result = await sdk_tools.analyze_chart_vision(
                        symbol,
                        chart_result["chart_image_base64"],
                        trade_style
                    )

                self.subagent_progress[agent_name].current_step = "Generating trade plan"
                self.subagent_progress[agent_name].status = SubAgentStatus.GENERATING_PLAN

                # Step 4: Build report from gathered data
                current_price = bars.get("current_price", context.current_price)
                atr_pct = bars.get("atr_pct", 2.0)

                # Determine suitability based on ATR
                suitable = False
                if trade_style == "day" and atr_pct > 2.5:
                    suitable = True
                elif trade_style == "swing" and 1.0 <= atr_pct <= 3.5:
                    suitable = True
                elif trade_style == "position" and atr_pct < 2.0:
                    suitable = True

                # Build confidence from indicators
                base_confidence = 50
                ema_trend = indicators.get("ema_trend", "unknown")
                if ema_trend == "bullish_aligned":
                    base_confidence += 15
                elif ema_trend == "bearish_aligned":
                    base_confidence += 10

                rsi = indicators.get("rsi", {}).get("value", 50)
                if 40 <= rsi <= 60:
                    base_confidence += 5

                # Add vision modifier
                vision_modifier = vision_result.get("confidence_modifier", 0)
                confidence = min(100, max(0, base_confidence + vision_modifier))

                # Determine bias
                if ema_trend == "bullish_aligned":
                    bias = "bullish"
                elif ema_trend == "bearish_aligned":
                    bias = "bearish"
                else:
                    bias = "neutral"

                # Build support/resistance
                supports = [s["price"] for s in sr_levels.get("support", [])[:3]]
                resistances = [r["price"] for r in sr_levels.get("resistance", [])[:3]]

                # Calculate entry/stop/targets
                if bias == "bullish" and supports:
                    entry_low = supports[0] if supports else current_price * 0.98
                    entry_high = current_price * 0.995
                    stop_loss = entry_low * 0.97
                    targets = [
                        PriceTargetWithReasoning(price=round(current_price * 1.03, 2), reasoning="First resistance"),
                        PriceTargetWithReasoning(price=round(current_price * 1.06, 2), reasoning="Second resistance"),
                    ]
                else:
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

                # Build context for Claude
                analysis_context = f"""
## Stock: {symbol}
## Trade Style: {trade_style.upper()}
## Current Price: ${current_price:.2f}

## Technical Data:
- ATR%: {atr_pct:.1f}%
- RSI(14): {rsi:.1f}
- EMA Trend: {ema_trend}
- MACD: {indicators.get('macd', {}).get('histogram', 'N/A')}

## Key Levels:
- Support: {', '.join([f'${s:.2f}' for s in supports]) if supports else 'None identified'}
- Resistance: {', '.join([f'${r:.2f}' for r in resistances]) if resistances else 'None identified'}

## Vision Analysis:
{vision_result.get('summary', 'No chart analysis available')}
- Patterns: {', '.join(vision_result.get('visual_patterns', [])) or 'None'}
- Trend Quality: {vision_result.get('trend_quality', 'unknown')}
- Warning Signs: {', '.join(vision_result.get('warning_signs', [])) or 'None'}

## Position Context:
{position_context_str if position_context_str else 'No existing position.'}

Based on this data, provide your {trade_style} trade analysis.
"""

                # Get the trade-style specific prompt
                from app.agent.prompts import (
                    build_day_trade_prompt,
                    build_swing_trade_prompt,
                    build_position_trade_prompt,
                )

                if trade_style == "day":
                    system_prompt = build_day_trade_prompt(symbol, context.to_prompt_context())
                elif trade_style == "swing":
                    system_prompt = build_swing_trade_prompt(symbol, context.to_prompt_context())
                else:
                    system_prompt = build_position_trade_prompt(symbol, context.to_prompt_context())

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

                # Call the provider for real analysis
                # Use search if provider supports X search (Grok), otherwise no search for sub-agents
                search_params = None
                if provider.supports_x_search:
                    search_params = get_x_search_parameters()

                plan_response = await provider.create_message(
                    messages=[AIMessage(role="user", content=user_message)],
                    system=system_prompt,
                    model_type="planning",
                    max_tokens=2000,
                    search_parameters=search_params,
                )

                # Parse the AI response
                ai_text = plan_response.content.strip()
                # Remove markdown code blocks if present
                if ai_text.startswith("```"):
                    ai_text = ai_text.split("```")[1]
                    if ai_text.startswith("json"):
                        ai_text = ai_text[4:]
                    ai_text = ai_text.strip()

                try:
                    ai_plan = json.loads(ai_text)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse AI response for {agent_name}, using fallback")
                    ai_plan = {}

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

                # Parse targets from AI response
                ai_targets = ai_plan.get("targets") or []
                if ai_targets and isinstance(ai_targets, list):
                    parsed_targets = []
                    for t in ai_targets[:3]:
                        if isinstance(t, dict):
                            target_price = t.get("price")
                            if target_price is not None:
                                parsed_targets.append(
                                    PriceTargetWithReasoning(
                                        price=round(float(target_price), 2),
                                        reasoning=t.get("reasoning") or "Target level"
                                    )
                                )
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
                    entry_reasoning=ai_plan.get("entry_reasoning") or f"Near support at ${final_entry_low:.2f}" if final_entry_low else "Near current price",
                    stop_loss=round(float(final_stop), 2) if final_stop else round(current_price * 0.95, 2),
                    stop_reasoning=ai_plan.get("stop_reasoning") or "Below recent support",
                    targets=targets,
                    risk_reward=float(ai_plan.get("risk_reward") or 2.0),
                    position_size_pct=2.0,
                    holding_period=final_holding,
                    key_supports=supports or [],
                    key_resistances=resistances or [],
                    invalidation_criteria=ai_plan.get("invalidation_criteria") or f"Close below ${final_stop:.2f}" if final_stop else "Price closes below stop level",
                    position_aligned=position_aligned,
                    position_recommendation=position_recommendation,
                    setup_explanation=ai_plan.get("setup_explanation") or f"This {trade_style} setup is based on {ema_trend} EMA alignment.",
                    what_to_watch=what_to_watch or ["Monitor price action", "Watch for volume confirmation"],
                    risk_warnings=risk_warnings or [],
                    atr_percent=atr_pct,
                    technical_summary=f"RSI: {rsi:.1f}, EMA trend: {ema_trend}",
                )

            except Exception as e:
                logger.error(f"[{agent_name}] Error: {e}")
                self.subagent_progress[agent_name].status = SubAgentStatus.FAILED
                self.subagent_progress[agent_name].error_message = str(e)
                # Return mock report on error
                return self._create_mock_report(symbol, trade_style, context)

        # Run all agents in parallel
        try:
            tasks = [
                run_single_agent(name, agent_def)
                for name, agent_def in agents_def.items()
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Process results
            for (agent_name, _), result in zip(agents_def.items(), results):
                if isinstance(result, Exception):
                    self.subagent_progress[agent_name].status = SubAgentStatus.FAILED
                    self.subagent_progress[agent_name].error_message = str(result)
                    self.subagent_progress[agent_name].findings = ["Error occurred"]
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
            thesis=f"Simulated {trade_style} trade thesis for {symbol}",
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

## ALL ANALYSES:
{''.join(analyses_summary)}

## SELECTED: {best_report.trade_style.upper()} TRADE

Create a comprehensive trading plan synthesis. Respond with JSON:
{{
    "thesis": "A detailed 3-4 sentence thesis explaining the selected trade setup, why this timeframe was chosen over others, and what makes this setup compelling. Be SPECIFIC about the technical pattern, key price levels, and any news/catalysts. If there's an existing position, explicitly address whether to add, hold, trim, or exit.",
    "selection_reasoning": "2-3 sentences explaining why {best_report.trade_style} was selected over the alternatives, comparing confidence levels and setup quality.",
    "targets": [
        {{"price": <number>, "reasoning": "Why this is a valid target level based on resistance/technicals"}},
        {{"price": <number>, "reasoning": "Second target reasoning"}},
        {{"price": <number>, "reasoning": "Third target reasoning (if applicable)"}}
    ],
    "what_to_watch": ["5-7 specific, actionable items to monitor - include exact price levels like 'Break above $XXX triggers acceleration', volume thresholds, time-based triggers, and any relevant news catalysts"],
    "risk_warnings": ["3-5 specific risks - technical invalidation levels, market risks, news/earnings risks, position-specific warnings"],
    "entry_reasoning": "Why this specific entry zone makes sense based on support levels and the technical setup",
    "stop_reasoning": "Why this stop level is appropriate - reference specific support/ATR/pattern invalidation",
    "news_impact": "Brief assessment of how recent news/sentiment affects this trade thesis (1-2 sentences)"
}}

Return ONLY valid JSON."""

            # Use search if provider supports X search (Grok) for real-time sentiment
            search_params = None
            if provider.supports_x_search:
                search_params = get_x_search_parameters()
            elif provider.supports_web_search:
                # Claude: use web search for synthesis
                search_params = SearchParameters(mode="on", sources=[{"type": "web"}], return_citations=True)

            synthesis_response = await provider.create_message(
                messages=[AIMessage(role="user", content=synthesis_prompt)],
                system=None,
                model_type="planning",
                max_tokens=1500,
                search_parameters=search_params,
            )

            synthesis_text = synthesis_response.content.strip()
            # Remove markdown code blocks if present
            if synthesis_text.startswith("```"):
                synthesis_text = synthesis_text.split("```")[1]
                if synthesis_text.startswith("json"):
                    synthesis_text = synthesis_text[4:]
                synthesis_text = synthesis_text.strip()

            synthesis = json.loads(synthesis_text)

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

            # Parse synthesized targets
            from app.agent.schemas.subagent_report import PriceTargetWithReasoning
            synth_targets = synthesis.get("targets", [])
            if synth_targets and isinstance(synth_targets, list):
                parsed_targets = []
                for t in synth_targets[:3]:
                    if isinstance(t, dict) and t.get("price") is not None:
                        try:
                            parsed_targets.append(
                                PriceTargetWithReasoning(
                                    price=round(float(t["price"]), 2),
                                    reasoning=t.get("reasoning") or "Target level"
                                )
                            )
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
        )
