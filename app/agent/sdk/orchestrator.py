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

    async def _fetch_x_sentiment(self, symbol: str, user_id: str) -> Dict[str, Any]:
        """Fetch X/Twitter sentiment for the symbol using Grok.

        This is called once during common data gathering, and the results
        are shared with all sub-agents to avoid rate limiting.

        Args:
            symbol: Stock ticker symbol
            user_id: User ID for provider lookup

        Returns:
            Dict with x_sentiment, x_sentiment_summary, and x_citations
        """
        try:
            provider = await self._get_provider(user_id)

            # Only Grok supports X search
            if not provider.supports_x_search:
                logger.info(f"[Orchestrator] Provider doesn't support X search, skipping")
                return {}

            # Simple prompt to get X sentiment
            prompt = f"""Analyze the current X/Twitter sentiment for ${symbol} stock.

Search X for recent posts about ${symbol} and provide:
1. Overall sentiment: bullish, bearish, neutral, or mixed
2. A brief 2-3 sentence summary of what traders are saying
3. Key themes or catalysts being discussed

Respond in JSON format:
{{
    "sentiment": "bullish" or "bearish" or "neutral" or "mixed",
    "summary": "Brief summary of X discussion...",
    "key_themes": ["theme1", "theme2"]
}}

Return ONLY the JSON object."""

            search_params = get_x_search_parameters()

            response = await provider.create_message(
                messages=[AIMessage(role="user", content=prompt)],
                system="You are a financial sentiment analyst. Analyze X/Twitter posts about stocks and provide objective sentiment assessment.",
                model_type="quick",  # Use faster model for sentiment
                max_tokens=500,
                search_parameters=search_params,
            )

            # Extract citations
            x_citations = []
            if response.citations:
                for citation in response.citations:
                    if isinstance(citation, str):
                        x_citations.append(citation)
                    elif isinstance(citation, dict) and "url" in citation:
                        x_citations.append(citation["url"])

            # Parse response
            import json
            response_text = response.content.strip()
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
                response_text = response_text.strip()

            try:
                parsed = json.loads(response_text)
                return {
                    "x_sentiment": parsed.get("sentiment", "unknown"),
                    "x_sentiment_summary": parsed.get("summary", ""),
                    "x_citations": x_citations[:20],  # Limit to 20 citations
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, try to extract sentiment from text
                text_lower = response_text.lower()
                if "bullish" in text_lower:
                    sentiment = "bullish"
                elif "bearish" in text_lower:
                    sentiment = "bearish"
                elif "mixed" in text_lower:
                    sentiment = "mixed"
                else:
                    sentiment = "neutral"

                return {
                    "x_sentiment": sentiment,
                    "x_sentiment_summary": response_text[:500],
                    "x_citations": x_citations[:20],
                }

        except Exception as e:
            logger.warning(f"[Orchestrator] Error fetching X sentiment: {e}")
            return {}

    async def _fetch_fundamentals(self, symbol: str) -> Dict[str, Any]:
        """Fetch fundamental data for the symbol using Finnhub API.

        This is called once during common data gathering, and the results
        are shared with all sub-agents.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Dict with fundamentals object and derived flags
        """
        try:
            from app.tools.market_data import fetch_fundamentals

            fundamentals = await fetch_fundamentals(symbol)

            # Derive earnings risk flag
            has_earnings_risk = fundamentals.has_earnings_risk(days_threshold=7)
            days_until = None
            if fundamentals.next_earnings and fundamentals.next_earnings.days_until is not None:
                days_until = fundamentals.next_earnings.days_until

            logger.info(
                f"[Orchestrator] Fundamentals: P/E={fundamentals.pe_ratio}, "
                f"Health={fundamentals.get_financial_health_score()}, "
                f"EarningsRisk={has_earnings_risk}"
            )

            return {
                "fundamentals": fundamentals,
                "has_earnings_risk": has_earnings_risk,
                "days_until_earnings": days_until,
            }
        except Exception as e:
            logger.warning(f"[Orchestrator] Error fetching fundamentals: {e}")
            return {
                "fundamentals": None,
                "has_earnings_risk": False,
                "days_until_earnings": None,
            }

    async def gather_common_context(
        self,
        symbol: str,
        user_id: str,
    ) -> DataContext:
        """Gather common data that all sub-agents need.

        This is called once at the start, then passed to all sub-agents.
        Includes X/social sentiment fetched once and shared with all agents
        to avoid rate limiting from parallel X search requests.

        Args:
            symbol: Stock ticker symbol
            user_id: User ID for position lookup

        Returns:
            DataContext with shared data including X sentiment
        """
        # Gather market data in parallel (fast, local API calls)
        price_task = asyncio.create_task(get_current_price(symbol))
        position_task = asyncio.create_task(get_position_status(symbol, user_id))
        market_task = asyncio.create_task(get_market_context())
        news_task = asyncio.create_task(get_news_sentiment(symbol))
        fundamentals_task = asyncio.create_task(self._fetch_fundamentals(symbol))

        price_data, position_data, market_data, news_data, fundamentals_data = await asyncio.gather(
            price_task, position_task, market_task, news_task, fundamentals_task
        )

        # Fetch X sentiment separately (uses Grok API with X search)
        # This is done ONCE here instead of in each sub-agent to avoid rate limiting
        logger.info(f"[Orchestrator] Fetching X/social sentiment for {symbol}")
        x_data = await self._fetch_x_sentiment(symbol, user_id)
        if x_data:
            logger.info(f"[Orchestrator] X sentiment: {x_data.get('x_sentiment', 'unknown')}, {len(x_data.get('x_citations', []))} citations")

        # Log news results
        logger.info(
            f"[Orchestrator] News sentiment for {symbol}: {news_data.get('sentiment', 'neutral')} "
            f"(score: {news_data.get('sentiment_score', 0):.2f}, {news_data.get('article_count', 0)} articles)"
        )
        if news_data.get("has_breaking_news"):
            logger.info(f"[Orchestrator] Breaking news detected for {symbol}!")

        # Build context with all gathered data
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
            # News data (from Finnhub)
            news_sentiment=news_data.get("sentiment"),
            news_summary=news_data.get("summary"),
            recent_headlines=news_data.get("headlines", []),
            news_score=news_data.get("sentiment_score"),
            news_article_count=news_data.get("article_count", 0),
            news_has_breaking=news_data.get("has_breaking_news", False),
            news_key_themes=news_data.get("key_themes"),
            # X/Social sentiment (gathered once, shared with all sub-agents)
            x_sentiment=x_data.get("x_sentiment"),
            x_sentiment_summary=x_data.get("x_sentiment_summary"),
            x_citations=x_data.get("x_citations"),
            # Fundamentals (gathered once, shared with all sub-agents)
            fundamentals=fundamentals_data.get("fundamentals"),
            has_earnings_risk=fundamentals_data.get("has_earnings_risk", False),
            days_until_earnings=fundamentals_data.get("days_until_earnings"),
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

    async def _generate_plan_agentic(
        self,
        symbol: str,
        user_id: str,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate trading plan using the agentic tool-calling approach.

        This method uses the AgenticStockAnalyzer which:
        1. Gives the AI access to tools (price, chart, fundamentals, news, X search)
        2. Lets the AI decide what to investigate based on what it discovers
        3. Iteratively calls tools and reasons about results
        4. Provides transparent reasoning visible to the user in real-time

        Args:
            symbol: Stock ticker symbol
            user_id: User ID for provider preferences

        Yields:
            StreamEvent objects for real-time progress updates
        """
        from app.agent.sdk.agentic_analyzer import create_agentic_analyzer

        try:
            # Create the agentic analyzer with user's preferred providers
            analyzer = await create_agentic_analyzer(user_id)

            # Run the agentic analysis
            async for event in analyzer.analyze(symbol, user_id):
                yield event

        except Exception as e:
            logger.error(f"[Agentic] Error in agentic analysis: {e}", exc_info=True)
            yield StreamEvent.error(f"Agentic analysis failed: {str(e)}")

    async def generate_plan_stream(
        self,
        symbol: str,
        user_id: str,
        force_new: bool = True,
        agentic_mode: bool = False,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Generate trading plan with streaming progress.

        This is the main entry point that yields StreamEvents for iOS consumption.

        Args:
            symbol: Stock ticker symbol
            user_id: User ID
            force_new: Force new plan generation
            agentic_mode: If True, use new agentic tool-calling approach.
                         If False (default), use legacy multi-agent approach.

        Yields:
            StreamEvent objects for each progress update
        """
        self._start_time = time.time()
        self._user_id = user_id  # Store for provider lookup

        if agentic_mode:
            logger.info(f"[Orchestrator] Starting AGENTIC plan generation for {symbol}")
            async for event in self._generate_plan_agentic(symbol, user_id):
                yield event
            return

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

            # Add news sentiment with details
            if context.news_sentiment:
                news_str = f"News: {context.news_sentiment.capitalize()}"
                if context.news_article_count > 0:
                    news_str += f" ({context.news_article_count} articles)"
                if context.news_has_breaking:
                    news_str += " [BREAKING]"
                findings.append(news_str)

            # Add fundamentals summary
            if context.fundamentals:
                f = context.fundamentals
                if f.pe_ratio is not None:
                    findings.append(f"P/E: {f.pe_ratio:.1f} ({f.get_valuation_assessment()})")
                findings.append(f"Health: {f.get_financial_health_score().capitalize()}")
                if context.has_earnings_risk and context.days_until_earnings is not None:
                    findings.append(f"EARNINGS IN {context.days_until_earnings} DAYS!")

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
                    bars = await sdk_tools.get_price_bars(symbol, "1w", 364)  # 52 weeks = 364 days
                    indicators = await sdk_tools.get_technical_indicators(symbol, [21, 50, 200], 14)
                    sr_levels = await sdk_tools.get_support_resistance(symbol, "weekly")

                # Step 1.5: Calculate Fibonacci levels for this timeframe
                fib_result = await sdk_tools.get_fibonacci_levels(
                    symbol,
                    bars.get("bars", []),
                    trade_style
                )

                self.subagent_progress[agent_name].current_step = "Generating chart"
                self.subagent_progress[agent_name].status = SubAgentStatus.GENERATING_CHART

                # Step 2: Generate chart
                timeframe_map = {"day": "5m", "swing": "1d", "position": "1w"}
                days_map = {"day": 3, "swing": 100, "position": 364}  # 52 weeks = 364 days
                chart_result = await sdk_tools.generate_chart(
                    symbol,
                    timeframe_map[trade_style],
                    days_map[trade_style]
                )

                self.subagent_progress[agent_name].current_step = "Analyzing chart with Vision"
                self.subagent_progress[agent_name].status = SubAgentStatus.ANALYZING_CHART

                # Step 3: Vision analysis (uses user's selected provider - Claude or Grok)
                vision_result = {}
                if chart_result.get("chart_image_base64"):
                    vision_result = await sdk_tools.analyze_chart_vision(
                        symbol,
                        chart_result["chart_image_base64"],
                        trade_style,
                        provider,  # Pass user's selected provider for vision analysis
                    )

                self.subagent_progress[agent_name].current_step = "Generating trade plan"
                self.subagent_progress[agent_name].status = SubAgentStatus.GENERATING_PLAN

                # Step 4: Build report from gathered data
                current_price = bars.get("current_price", context.current_price)
                atr_pct = bars.get("atr_pct", 2.0)

                # Determine suitability based on ATR with TIMEFRAME-SPECIFIC thresholds
                # Each timeframe produces different ATR% values that can't be compared directly:
                # - Day (5m bars): Low ATR% is normal (0.2-0.5% typical)
                # - Swing (1d bars): Moderate ATR% expected (1-5% typical)
                # - Position (1w bars): Higher ATR% expected (5-25% typical for growth stocks)
                ATR_THRESHOLDS = {
                    "day":      {"min": 0.2, "max": 100},   # 5m bars - any reasonable intraday volatility
                    "swing":    {"min": 1.0, "max": 5.0},   # 1d bars - moderate daily volatility
                    "position": {"min": 0,   "max": 25.0},  # 1w bars - allow growth stock volatility
                }
                threshold = ATR_THRESHOLDS.get(trade_style, {"min": 0, "max": 100})
                suitable = threshold["min"] <= atr_pct <= threshold["max"]

                # Position trades also require EMA alignment for trending conditions
                if trade_style == "position" and suitable:
                    ema_trend = indicators.get("ema_trend", "unknown")
                    suitable = ema_trend in ["bullish_aligned", "bearish_aligned"]

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

                # Add vision modifier (range matches vision prompt: -20 to +20)
                raw_vision_modifier = vision_result.get("confidence_modifier", 0)
                vision_modifier = max(-20, min(20, raw_vision_modifier))
                confidence = min(100, max(0, base_confidence + vision_modifier))

                # ============================================================
                # FUNDAMENTAL INFLUENCE ON CONFIDENCE (Timeframe-Weighted)
                # ============================================================
                fundamental_modifier = 0
                if context.fundamentals:
                    f = context.fundamentals

                    # Weight by trade style: day=0.2, swing=0.5, position=1.0
                    weight_map = {"day": 0.2, "swing": 0.5, "position": 1.0}
                    weight = weight_map.get(trade_style, 0.5)

                    # Valuation factor (-10 to +5)
                    if f.pe_ratio is not None:
                        if f.pe_ratio < 0:
                            fundamental_modifier -= 5 * weight  # Unprofitable
                        elif f.pe_ratio < 15:
                            fundamental_modifier += 3 * weight  # Undervalued
                        elif f.pe_ratio > 50:
                            fundamental_modifier -= 5 * weight  # Very expensive

                    # Financial health factor (-5 to +5)
                    health = f.get_financial_health_score()
                    if health == "strong":
                        fundamental_modifier += 5 * weight
                    elif health == "weak":
                        fundamental_modifier -= 5 * weight

                    # Growth factor (-5 to +5)
                    if f.eps_growth_yoy is not None:
                        if f.eps_growth_yoy > 20:
                            fundamental_modifier += 5 * weight
                        elif f.eps_growth_yoy < -10:
                            fundamental_modifier -= 5 * weight

                    # Earnings risk penalty (CRITICAL - affects all styles)
                    if context.has_earnings_risk:
                        if trade_style == "day":
                            fundamental_modifier -= 10  # High risk even for day trades
                        elif trade_style == "swing":
                            fundamental_modifier -= 20  # Very high risk
                        else:  # position
                            fundamental_modifier -= 25  # Extreme risk

                    # Apply fundamental modifier
                    confidence = min(100, max(0, confidence + int(fundamental_modifier)))
                    logger.info(f"[{agent_name}] Fundamental modifier: {fundamental_modifier:.1f} (weight: {weight})")

                # Determine bias from EMA trend
                if ema_trend == "bullish_aligned":
                    bias = "bullish"
                elif ema_trend == "bearish_aligned":
                    bias = "bearish"
                else:
                    bias = "neutral"

                # ============================================================
                # VISION PATTERN INFLUENCE ON BIAS
                # If chart patterns conflict with EMA-based bias, adjust accordingly
                # ============================================================
                visual_patterns = [p.lower() for p in vision_result.get("visual_patterns", [])]
                pattern_string = " ".join(visual_patterns)

                bearish_patterns = ["descending channel", "bear flag", "head and shoulders", "double top", "breakdown", "rising wedge"]
                bullish_patterns = ["ascending channel", "bull flag", "inverse head and shoulders", "double bottom", "breakout", "falling wedge"]

                has_bearish_pattern = any(bp in pattern_string for bp in bearish_patterns)
                has_bullish_pattern = any(bp in pattern_string for bp in bullish_patterns)

                # Adjust bias when vision patterns conflict with EMA-based bias
                if has_bearish_pattern and not has_bullish_pattern:
                    if bias == "bullish":
                        # Conflicting signals: EMA bullish but chart bearish → downgrade to neutral
                        bias = "neutral"
                        confidence = max(0, confidence - 10)  # Reduce confidence for mixed signals
                    elif bias == "neutral":
                        # Neutral EMA + bearish patterns → lean bearish
                        bias = "bearish"
                elif has_bullish_pattern and not has_bearish_pattern:
                    if bias == "bearish":
                        # Conflicting signals: EMA bearish but chart bullish → downgrade to neutral
                        bias = "neutral"
                        confidence = max(0, confidence - 10)
                    elif bias == "neutral":
                        # Neutral EMA + bullish patterns → lean bullish
                        bias = "bullish"
                # If both patterns present or neither, keep EMA-based bias

                # ============================================================
                # NEWS INFLUENCE ON CONFIDENCE (Timeframe-Weighted)
                # News impacts trades differently based on holding period
                # Day trades: 20% weight (intraday momentum dominates)
                # Swing trades: 50% weight (catalysts affect multi-day holds)
                # Position trades: 100% weight (news critical for weeks/months)
                # ============================================================
                news_modifier = 0
                if context.news_score is not None and context.news_article_count > 0:
                    # Weight by trade style
                    news_weight_map = {"day": 0.2, "swing": 0.5, "position": 1.0}
                    news_weight = news_weight_map.get(trade_style, 0.5)

                    # Sentiment alignment factor (-10 to +5)
                    # If news aligns with bias: boost confidence
                    # If news conflicts with bias: reduce confidence
                    if bias == "bullish":
                        if context.news_score > 0.2:
                            news_modifier += 5 * news_weight  # Bullish news + bullish bias = aligned
                        elif context.news_score < -0.2:
                            news_modifier -= 10 * news_weight  # Bearish news + bullish bias = conflict
                    elif bias == "bearish":
                        if context.news_score < -0.2:
                            news_modifier += 5 * news_weight  # Bearish news + bearish bias = aligned
                        elif context.news_score > 0.2:
                            news_modifier -= 10 * news_weight  # Bullish news + bearish bias = conflict

                    # Breaking news risk penalty (especially for day trades)
                    if context.news_has_breaking:
                        if trade_style == "day":
                            news_modifier -= 10  # High volatility risk for intraday
                        elif trade_style == "swing":
                            news_modifier -= 5  # Moderate risk for swing
                        # Position trades can absorb breaking news better

                    confidence = min(100, max(0, confidence + int(news_modifier)))
                    logger.info(f"[{agent_name}] News modifier: {news_modifier:.1f} (weight: {news_weight}, score: {context.news_score:.2f})")

                # Build support/resistance
                supports = [s["price"] for s in sr_levels.get("support", [])[:3]]
                resistances = [r["price"] for r in sr_levels.get("resistance", [])[:3]]

                # ============================================================
                # HYBRID FIBONACCI + S/R LOGIC (Like a Real Trader)
                # Use Fibonacci when conditions are right, otherwise fall back to S/R
                # ============================================================

                # Extract Fibonacci data
                retracements = fib_result.get("retracement_levels", {})
                extensions = fib_result.get("extension_levels", {})
                fib_trend = fib_result.get("trend", "unknown")
                swing_range = fib_result.get("swing_range", 0) or 0

                # Calculate ATR for comparison
                atr_value = bars.get("atr", current_price * 0.02)  # fallback 2%

                # DECISION: Should we use Fibonacci for this setup?
                # A real trader asks: "Is there a clear trend with a meaningful swing?"
                use_fib_for_levels = (
                    # 1. Clear trend direction (EMA alignment matches)
                    fib_trend in ["uptrend", "downtrend"] and
                    ema_trend in ["bullish_aligned", "bearish_aligned"] and
                    # 2. Meaningful swing range (not choppy noise)
                    swing_range > atr_value * 2 and
                    # 3. Trend direction matches bias
                    ((fib_trend == "uptrend" and bias == "bullish") or
                     (fib_trend == "downtrend" and bias == "bearish"))
                )

                # Track which method was used
                levels_method = "fibonacci" if use_fib_for_levels else "support_resistance"

                # ============================================================
                # FIBONACCI-BASED LEVELS (When trend is clear)
                # ============================================================
                if use_fib_for_levels and bias == "bullish":
                    # Bullish setup in uptrend: Buy pullbacks to Fib support
                    entry_low = retracements.get("0.618", supports[0] if supports else current_price * 0.98)
                    entry_high = retracements.get("0.382", current_price * 0.995)
                    # Stop below 0.786 with 5% buffer (avoid stop hunts at exact Fib)
                    fib_786 = retracements.get("0.786", entry_low * 0.95)
                    stop_loss = fib_786 * 0.95
                    # Targets at Fib extensions beyond swing high
                    targets = [
                        PriceTargetWithReasoning(
                            price=round(extensions.get("1.272", current_price * 1.05), 2),
                            reasoning="Fibonacci 1.272 extension"
                        ),
                        PriceTargetWithReasoning(
                            price=round(extensions.get("1.618", current_price * 1.10), 2),
                            reasoning="Fibonacci 1.618 extension (golden ratio)"
                        ),
                    ]
                    logger.info(f"[{agent_name}] Using FIBONACCI levels (bullish uptrend)")

                elif use_fib_for_levels and bias == "bearish":
                    # Bearish setup in downtrend: Short rallies to Fib resistance
                    entry_low = current_price * 1.005
                    entry_high = retracements.get("0.382", resistances[0] if resistances else current_price * 1.02)
                    # Stop above 0.786 with buffer
                    fib_786 = retracements.get("0.786", entry_high * 1.05)
                    stop_loss = fib_786 * 1.05
                    # Downside Fib extension targets
                    targets = [
                        PriceTargetWithReasoning(
                            price=round(extensions.get("1.272", current_price * 0.95), 2),
                            reasoning="Fibonacci 1.272 extension (downside)"
                        ),
                        PriceTargetWithReasoning(
                            price=round(extensions.get("1.618", current_price * 0.90), 2),
                            reasoning="Fibonacci 1.618 extension (downside)"
                        ),
                    ]
                    logger.info(f"[{agent_name}] Using FIBONACCI levels (bearish downtrend)")

                # ============================================================
                # S/R + ATR-BASED LEVELS (When Fibonacci isn't appropriate)
                # Used for: choppy markets, neutral bias, day trades, small swings
                # ============================================================
                else:
                    if bias == "bullish" and supports:
                        entry_low = supports[0] if supports else current_price * 0.98
                        entry_high = current_price * 0.995
                        stop_loss = entry_low * 0.97  # ATR-based stop
                        targets = [
                            PriceTargetWithReasoning(
                                price=round(resistances[0] if resistances else current_price * 1.03, 2),
                                reasoning="First resistance level"
                            ),
                            PriceTargetWithReasoning(
                                price=round(resistances[1] if len(resistances) > 1 else current_price * 1.06, 2),
                                reasoning="Second resistance level"
                            ),
                        ]
                    elif bias == "bearish" and resistances:
                        entry_low = current_price * 1.005
                        entry_high = resistances[0] if resistances else current_price * 1.02
                        stop_loss = entry_high * 1.03
                        targets = [
                            PriceTargetWithReasoning(
                                price=round(supports[0] if supports else current_price * 0.97, 2),
                                reasoning="First support target"
                            ),
                        ]
                    else:
                        # Neutral - conservative defaults
                        entry_low = current_price * 0.98
                        entry_high = current_price * 0.995
                        stop_loss = current_price * 0.95
                        targets = [
                            PriceTargetWithReasoning(
                                price=round(current_price * 1.05, 2),
                                reasoning="Conservative target"
                            ),
                        ]
                    logger.info(f"[{agent_name}] Using S/R levels (fib_trend={fib_trend}, ema_trend={ema_trend})")

                holding_periods = {"day": "1-4 hours", "swing": "3-7 days", "position": "2-6 weeks"}

                # ============================================================
                # Position Management Logic
                # ============================================================
                position_recommendation = None
                position_aligned = True
                risk_warnings = []
                what_to_watch = ["Volume confirmation", "Break of key levels"]

                # Auto-add earnings risk warning if applicable (CRITICAL)
                if context.has_earnings_risk and context.days_until_earnings is not None:
                    earnings_warning = f"EARNINGS IN {context.days_until_earnings} DAYS - High gap risk!"
                    if context.fundamentals and context.fundamentals.next_earnings:
                        ne = context.fundamentals.next_earnings
                        if ne.hour:
                            timing = {
                                "bmo": "before market open",
                                "amc": "after close",
                                "dmh": "during hours"
                            }.get(ne.hour, ne.hour)
                            earnings_warning += f" ({timing})"
                    risk_warnings.insert(0, earnings_warning)  # Make it first warning
                    what_to_watch.insert(0, f"Earnings date: {context.fundamentals.next_earnings.date if context.fundamentals and context.fundamentals.next_earnings else 'upcoming'}")

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
{f'''
## Fibonacci Analysis:
- Trend: {fib_trend}
- Swing Range: ${fib_result.get("swing_low", 0):.2f} to ${fib_result.get("swing_high", 0):.2f}
- Key Retracements: {", ".join([f"{k}: ${v:.2f}" for k, v in sorted(retracements.items())[:4]]) if retracements else "None"}
- Key Extensions: {", ".join([f"{k}: ${v:.2f}" for k, v in sorted(extensions.items())[:3]]) if extensions else "None"}
- Using Fibonacci for levels: {levels_method == "fibonacci"}
''' if fib_trend in ["uptrend", "downtrend"] else ""}
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

                # Call the provider for real analysis
                # NOTE: X search is DISABLED for sub-agents to avoid rate limiting
                # X/social sentiment is fetched ONCE in gather_common_context and shared via DataContext
                # This prevents 3 parallel X search requests which caused timeouts
                plan_response = await provider.create_message(
                    messages=[AIMessage(role="user", content=user_message)],
                    system=system_prompt,
                    model_type="planning",
                    max_tokens=2000,
                    search_parameters=None,  # No X search - data already in context
                )

                # Use X citations from the shared context (fetched once in common step)
                x_citations = context.x_citations or []
                if x_citations:
                    logger.info(f"[{agent_name}] Using {len(x_citations)} pre-fetched X/social citations from context")

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
                    risk_reward=self._calculate_risk_reward(ai_plan, final_entry_high, final_stop, targets, bias),
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
                    technical_summary=f"RSI: {rsi:.1f}, EMA trend: {ema_trend}, Levels: {levels_method}" + (
                        f", Fib trend: {fib_trend}" if levels_method == "fibonacci" else ""
                    ),
                    x_citations=x_citations,
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
