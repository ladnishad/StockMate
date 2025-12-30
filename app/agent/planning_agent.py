"""Stock Planning Agent - Comprehensive analysis and trading plan generation.

This agent works like a professional trader:
1. Gathers ALL available data for a stock
2. Runs comprehensive technical analysis
3. Creates an AI-generated trading plan with thesis
4. Continuously evaluates the plan as price progresses
5. Supports conversation with memory about the plan
"""

import logging
import json
import uuid
import time
from datetime import datetime
from typing import Optional, Dict, Any, List

import anthropic
from anthropic._exceptions import OverloadedError, RateLimitError, APIStatusError

from app.config import get_settings
from app.agent.tools import (
    get_current_price,
    get_key_levels,
    get_technical_indicators,
    run_full_analysis,
    get_volume_profile,
    get_chart_patterns,
    get_position_status,
    get_market_context,
)
from app.storage.plan_store import TradingPlan, get_plan_store
from app.storage.conversation_store import get_conversation_store
from app.storage.position_store import get_position_store
from app.agent.prompts import (
    SMART_PLANNING_SYSTEM_PROMPT,
    SMART_PLAN_GENERATION_PROMPT,
    SMART_PLAN_EVALUATION_PROMPT,
)
from app.models.response import (
    EnhancedTradePlan,
    TradeStyleRecommendation,
    EducationalContent,
    ScenarioPath,
    ChartAnnotation,
    PriceTarget,
)

logger = logging.getLogger(__name__)

# System prompt for the planning agent
PLANNING_AGENT_SYSTEM = """You are an expert swing trader and technical analyst. Your job is to analyze stocks comprehensively and create actionable trading plans.

## Your Approach
1. **Search for context first** - Before analyzing technicals, search for recent news and Reddit sentiment about the stock
2. **Analyze the data thoroughly** - Look at all technical indicators, key levels, volume, and market context
3. **Form a thesis** - What's the story? Why would this trade work?
4. **Define precise levels** - Entry zones, stop loss, and profit targets based on actual support/resistance
5. **Consider risk** - Position sizing, risk/reward, and what would invalidate the trade
6. **Be honest** - If the setup isn't good, say so. Not every stock is a trade.

## Web Search Strategy
When web search is available, ALWAYS search for:
1. "{symbol} stock news" - Get recent news, earnings, analyst ratings
2. "site:reddit.com/r/wallstreetbets {symbol}" - Check retail sentiment and buzz

Focus on recent, relevant information. Ignore outdated articles or old Reddit posts.

## When Creating a Plan
- Entry should be at support or on a pullback, not chasing
- Stop loss should be below a meaningful level (support, swing low)
- Targets should be at resistance levels or Fibonacci extensions
- Risk/reward should be at least 2:1 for swing trades
- Consider the broader market direction
- Factor in any news catalysts or Reddit buzz into your thesis

## When Evaluating a Plan
- Has price moved toward entry? Away from it?
- Have key levels held or broken?
- Has the thesis changed?
- Should targets or stops be adjusted?
- Is the trade still valid?

## Communication Style
- Be direct and actionable
- Reference specific price levels and indicators
- Explain your reasoning briefly
- Write in plain text without markdown formatting (no bold, bullets, headers, or code blocks)
- Respond conversationally like a human trader texting a friend
- Use natural line breaks instead of bullet points
- If asked about buy/sell decisions, remind this is not financial advice
"""

PLAN_GENERATION_PROMPT = """Based on the comprehensive data below, create a detailed trading plan for {symbol}.

## Current Market Data
{market_data}

## Technical Analysis
{technical_data}

## Key Levels
{levels_data}

## Volume Analysis
{volume_data}

## Chart Patterns
{patterns_data}

## Existing Position (if any)
{position_data}

---

Analyze this stock like an expert trader. Follow this framework:

1. **Multi-Timeframe Analysis**: Assess the overall trend and key structure levels
2. **Pattern Recognition**: Identify any dominant chart patterns and their implications
3. **Momentum & Trend**: Evaluate EMA alignment, RSI, MACD signals
4. **Key Levels**: Find nearest support/resistance and confluence zones
5. **Trade Style Determination**: Based on ATR%, volatility, and setup type:
   - DAY TRADE: High ATR% (>3%), tight spread, quick resolution, intraday patterns
   - SWING TRADE: Moderate ATR (1-3%), multi-day patterns, clear S/R, 2-10 day hold
   - POSITION TRADE: Major trend alignment, wide levels, low volatility, weeks-months

IMPORTANT: If web search is available, search for news and Reddit sentiment FIRST before responding.

Respond in this exact JSON format:

{{
    "trade_style": "day" | "swing" | "position",
    "trade_style_reasoning": "1 sentence explaining why this style fits the current setup",
    "holding_period": "e.g., '1-3 days', '1-2 weeks', '2-4 weeks'",
    "confidence": <0-100, how confident in this setup>,
    "bias": "bullish" | "bearish" | "neutral",
    "thesis": "2-3 sentence explanation of why this trade makes sense or why you're passing. Include any relevant news/catalyst info.",
    "entry_zone_low": <price or null if no trade>,
    "entry_zone_high": <price or null>,
    "stop_loss": <price or null>,
    "stop_reasoning": "Why this stop level",
    "target_1": <price - conservative target>,
    "target_2": <price - moderate target>,
    "target_3": <price - aggressive target or null>,
    "target_reasoning": "Why these targets",
    "risk_reward": <ratio like 2.5>,
    "position_size_pct": <1-5, percentage of account>,
    "key_supports": [<price>, <price>],
    "key_resistances": [<price>, <price>],
    "invalidation_criteria": "What would invalidate this plan",
    "technical_summary": "Brief summary of key technical factors",
    "news_summary": "Brief summary of recent news/catalysts (from web search). Empty string if no news found or web search unavailable.",
    "reddit_sentiment": "bullish" | "bearish" | "neutral" | "mixed" | "none",
    "reddit_buzz": "Summary of Reddit discussion if found. Empty string if none."
}}

If this is NOT a good setup, still provide your analysis but set entry_zone, stop_loss, and targets to null and explain in the thesis why you're passing. Set confidence to how confident you are that there's no good trade here.
"""

PLAN_EVALUATION_PROMPT = """Evaluate the existing trading plan for {symbol} based on current market conditions.

## Existing Plan
{plan_data}

## Current Market Data
{market_data}

## Current Technical Analysis
{technical_data}

## User's Position
{position_data}

---

Evaluate if this plan is still valid. Consider:
1. Has price moved toward or away from entry?
2. Have key levels held or broken?
3. Has the market context changed?
4. Should we adjust stops or targets?
5. **If user has a position**: How is their trade performing? Should they adjust stops based on their entry? Are targets still realistic?

**IMPORTANT**: You must respond with a JSON object in this exact format:

```json
{{
    "status": "VALID" | "ADJUST" | "INVALIDATED",
    "evaluation": "2-3 sentence summary of current status",
    "action": "What the trader should do now",
    "adjustments": {{
        "stop_loss": null or new_price,
        "target_1": null or new_price,
        "target_2": null or new_price,
        "target_3": null or new_price,
        "entry_zone_low": null or new_price,
        "entry_zone_high": null or new_price,
        "stop_reasoning": null or "new reasoning if stop changed",
        "target_reasoning": null or "new reasoning if targets changed",
        "key_supports": null or [new_support_levels],
        "key_resistances": null or [new_resistance_levels]
    }},
    "adjustment_rationale": "Why these specific adjustments were made (if any)"
}}
```

Guidelines for adjustments:
- Set fields to null if no change needed
- Tighten stop loss if price moved favorably and new support formed
- Adjust targets if new resistance/support levels emerged
- Update key_supports/key_resistances if significant new levels have formed or old ones have broken
- Only adjust if there's a clear technical reason
- Keep risk/reward reasonable (ideally 2:1 or better)
"""


class StockPlanningAgent:
    """Comprehensive stock analysis and planning agent."""

    def __init__(self, symbol: str, user_id: str = "default"):
        self.symbol = symbol.upper()
        self.user_id = user_id
        self._client: Optional[anthropic.Anthropic] = None

        # Stores
        self._plan_store = get_plan_store()
        self._conversation_store = get_conversation_store()
        self._position_store = get_position_store()

        # Cached data (refreshed on analyze)
        self._market_data: Dict[str, Any] = {}
        self._technical_data: Dict[str, Any] = {}
        self._levels_data: Dict[str, Any] = {}
        self._volume_data: Dict[str, Any] = {}
        self._patterns_data: Dict[str, Any] = {}
        self._position_data: Dict[str, Any] = {}
        self._current_plan: Optional[TradingPlan] = None

    def _get_client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            settings = get_settings()
            if not settings.claude_api_key:
                raise ValueError("CLAUDE_API_KEY not configured")
            self._client = anthropic.Anthropic(api_key=settings.claude_api_key)
        return self._client

    async def gather_comprehensive_data(self) -> Dict[str, Any]:
        """Gather ALL available data for the stock."""
        logger.info(f"Gathering comprehensive data for {self.symbol}")

        # Gather all data in parallel-ish (async)
        try:
            self._market_data = {
                "price": await get_current_price(self.symbol),
                "market": await get_market_context(),
            }
        except Exception as e:
            logger.error(f"Error getting market data: {e}")
            self._market_data = {"error": str(e)}

        try:
            self._technical_data = await get_technical_indicators(self.symbol)
        except Exception as e:
            logger.error(f"Error getting technical indicators: {e}")
            self._technical_data = {"error": str(e)}

        try:
            self._levels_data = await get_key_levels(self.symbol)
        except Exception as e:
            logger.error(f"Error getting key levels: {e}")
            self._levels_data = {"error": str(e)}

        try:
            self._volume_data = await get_volume_profile(self.symbol)
        except Exception as e:
            logger.error(f"Error getting volume profile: {e}")
            self._volume_data = {"error": str(e)}

        try:
            self._patterns_data = await get_chart_patterns(self.symbol)
        except Exception as e:
            logger.error(f"Error getting chart patterns: {e}")
            self._patterns_data = {"error": str(e)}

        try:
            self._position_data = await get_position_status(self.symbol, self.user_id)
        except Exception as e:
            logger.error(f"Error getting position status: {e}")
            self._position_data = {"error": str(e)}

        # Load existing plan
        self._current_plan = await self._plan_store.get_plan(self.user_id, self.symbol)

        return {
            "market": self._market_data,
            "technical": self._technical_data,
            "levels": self._levels_data,
            "volume": self._volume_data,
            "patterns": self._patterns_data,
            "position": self._position_data,
            "has_plan": self._current_plan is not None,
        }

    def _format_data_for_prompt(self) -> Dict[str, str]:
        """Format gathered data for the AI prompt."""
        # Market data
        price_data = self._market_data.get("price", {})
        market_ctx = self._market_data.get("market", {})
        market_str = f"""Current Price: ${price_data.get('price', 'N/A')}
Daily Change: {price_data.get('change_pct', 'N/A')}%
Market Direction: {market_ctx.get('market_direction', 'unknown').upper()}
Bullish Indices: {market_ctx.get('bullish_indices', 0)}/{market_ctx.get('total_indices', 4)}
"""

        # Technical data
        tech = self._technical_data
        if tech.get("error"):
            tech_str = f"Error: {tech['error']}"
        else:
            rsi = tech.get("rsi", {})
            macd = tech.get("macd", {})
            emas = tech.get("emas", {})
            vol = tech.get("volume", {})
            bb = tech.get("bollinger", {})

            tech_str = f"""RSI: {rsi.get('value', 'N/A'):.1f} - {rsi.get('signal', 'N/A')}
MACD: {macd.get('signal', 'N/A')} (Histogram: {macd.get('histogram', 'N/A')})
EMAs: Price {'above' if emas.get('above_9') else 'below'} 9 EMA, {'above' if emas.get('above_21') else 'below'} 21 EMA, {'above' if emas.get('above_50') else 'below'} 50 EMA
EMA Trend: {emas.get('trend', 'N/A')}
Volume: {vol.get('relative', 'N/A')}x average ({vol.get('trend', 'N/A')})
Bollinger: {bb.get('position', 'N/A')} (Width: {bb.get('width', 'N/A')})
"""

        # Key levels
        levels = self._levels_data
        if levels.get("error"):
            levels_str = f"Error: {levels['error']}"
        else:
            supports = levels.get("support_levels", [])[:5]
            resistances = levels.get("resistance_levels", [])[:5]
            levels_str = f"""Support Levels: {', '.join([f'${s:.2f}' for s in supports]) if supports else 'None found'}
Resistance Levels: {', '.join([f'${r:.2f}' for r in resistances]) if resistances else 'None found'}
ATR (14): ${levels.get('atr', 'N/A')}
"""

        # Volume profile
        vol_profile = self._volume_data
        if vol_profile.get("error"):
            volume_str = f"Error: {vol_profile['error']}"
        else:
            volume_str = f"""Volume Profile POC: ${vol_profile.get('poc', 'N/A')}
Value Area High: ${vol_profile.get('value_area_high', 'N/A')}
Value Area Low: ${vol_profile.get('value_area_low', 'N/A')}
"""

        # Chart patterns
        patterns = self._patterns_data
        if patterns.get("error"):
            patterns_str = f"Error: {patterns['error']}"
        else:
            detected = patterns.get("patterns", [])
            if detected:
                patterns_str = "\n".join([
                    f"- {p.get('name', 'Unknown')}: {p.get('type', 'N/A')} (confidence: {p.get('confidence', 'N/A')}%, target: ${p.get('target', 'N/A')})"
                    for p in detected[:5]
                ])
            else:
                patterns_str = "No significant patterns detected"

        # Position
        pos = self._position_data
        if pos.get("has_position"):
            targets = pos.get("targets", {})
            position_str = f"""Status: {pos.get('status', 'unknown')}
Entry Price: ${pos.get('entry_price', 'N/A')}
Current Size: {pos.get('current_size', 0)} shares
Cost Basis: ${pos.get('cost_basis', 'N/A')}
Stop Loss: ${pos.get('stop_loss', 'N/A')}
Targets: ${targets.get('target_1', 'N/A')} / ${targets.get('target_2', 'N/A')} / ${targets.get('target_3', 'N/A')}
Current Price: ${pos.get('current_price', 'N/A')}
Unrealized P&L: ${pos.get('unrealized_pnl', 'N/A')} ({pos.get('unrealized_pnl_pct', 'N/A')}%)
R-Multiple: {pos.get('r_multiple', 'N/A')}
"""
        else:
            position_str = "No current position"

        return {
            "market_data": market_str,
            "technical_data": tech_str,
            "levels_data": levels_str,
            "volume_data": volume_str,
            "patterns_data": patterns_str,
            "position_data": position_str,
        }

    async def generate_plan(self, force_new: bool = False) -> TradingPlan:
        """Generate a new trading plan based on comprehensive analysis."""
        logger.info(f"Generating trading plan for {self.symbol}")

        # Check for existing plan
        if not force_new and self._current_plan and self._current_plan.status == "active":
            logger.info(f"Active plan exists for {self.symbol}, returning existing")
            return self._current_plan

        # Gather fresh data
        await self.gather_comprehensive_data()

        # Format data for prompt
        data = self._format_data_for_prompt()

        prompt = PLAN_GENERATION_PROMPT.format(
            symbol=self.symbol,
            **data
        )

        try:
            client = self._get_client()
            settings = get_settings()

            # Build tools list (web search if enabled)
            tools = None
            if settings.web_search_enabled:
                tools = [{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": settings.web_search_max_uses,
                }]
                logger.info(f"Web search enabled for {self.symbol} plan generation")

            response = client.messages.create(
                model=settings.claude_model_planning,
                max_tokens=4000,  # Increased for web search results
                system=PLANNING_AGENT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
            )

            # Extract text from response (may contain multiple content blocks with web search)
            response_text = self._extract_text_from_response(response)

            # Parse JSON response
            plan_data = self._parse_plan_response(response_text)

            # Create TradingPlan object
            price_data = self._market_data.get("price", {})
            rsi_data = self._technical_data.get("rsi", {})
            market_ctx = self._market_data.get("market", {})

            plan = TradingPlan(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                symbol=self.symbol,
                status="active",
                bias=plan_data.get("bias", "neutral"),
                thesis=plan_data.get("thesis", ""),
                entry_zone_low=plan_data.get("entry_zone_low"),
                entry_zone_high=plan_data.get("entry_zone_high"),
                stop_loss=plan_data.get("stop_loss"),
                stop_reasoning=plan_data.get("stop_reasoning", ""),
                target_1=plan_data.get("target_1"),
                target_2=plan_data.get("target_2"),
                target_3=plan_data.get("target_3"),
                target_reasoning=plan_data.get("target_reasoning", ""),
                risk_reward=plan_data.get("risk_reward"),
                position_size_pct=plan_data.get("position_size_pct"),
                key_supports=plan_data.get("key_supports", []),
                key_resistances=plan_data.get("key_resistances", []),
                invalidation_criteria=plan_data.get("invalidation_criteria", ""),
                # Trade style fields
                trade_style=plan_data.get("trade_style", "swing"),
                trade_style_reasoning=plan_data.get("trade_style_reasoning", ""),
                holding_period=plan_data.get("holding_period", ""),
                confidence=plan_data.get("confidence", 0),
                # Context at creation
                price_at_creation=price_data.get("price"),
                rsi_at_creation=rsi_data.get("value"),
                market_direction_at_creation=market_ctx.get("market_direction", ""),
                technical_summary=plan_data.get("technical_summary", ""),
                # External sentiment from web search
                news_summary=plan_data.get("news_summary", ""),
                reddit_sentiment=plan_data.get("reddit_sentiment", "none"),
                reddit_buzz=plan_data.get("reddit_buzz", ""),
            )

            # Save plan
            self._current_plan = await self._plan_store.save_plan(plan)
            logger.info(f"Created new trading plan for {self.symbol}: {plan.bias}")

            return self._current_plan

        except Exception as e:
            logger.error(f"Error generating plan for {self.symbol}: {e}")
            raise

    async def generate_plan_streaming(self, force_new: bool = False):
        """Generate a trading plan with streaming output.

        Yields chunks of the AI's response as it generates the plan,
        allowing real-time display of the analysis process.

        Yields:
            dict: Event objects with types:
                - {"type": "existing_plan", "plan": {...}} - If returning cached plan
                - {"type": "text", "content": "..."} - Text chunks from AI
                - {"type": "plan_complete", "plan": {...}} - Final parsed plan
                - {"type": "error", "message": "..."} - On error
        """
        logger.info(f"Generating streaming plan for {self.symbol}")

        # Check for existing plan
        if not force_new and self._current_plan and self._current_plan.status == "active":
            logger.info(f"Active plan exists for {self.symbol}, returning existing")
            yield {"type": "existing_plan", "plan": self._plan_to_dict(self._current_plan)}
            return

        # Data should already be gathered by the endpoint
        if not self._market_data:
            await self.gather_comprehensive_data()

        # Format data for prompt
        data = self._format_data_for_prompt()

        prompt = PLAN_GENERATION_PROMPT.format(
            symbol=self.symbol,
            **data
        )

        try:
            client = self._get_client()
            settings = get_settings()

            # Build tools list (web search if enabled)
            # Note: Streaming with tools requires handling tool_use blocks
            tools = None
            if settings.web_search_enabled:
                tools = [{
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "max_uses": settings.web_search_max_uses,
                }]
                logger.info(f"Web search enabled for {self.symbol} streaming plan")

            # Use streaming API
            full_text = ""
            with client.messages.stream(
                model=settings.claude_model_planning,
                max_tokens=4000,
                system=PLANNING_AGENT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
                tools=tools,
            ) as stream:
                for text in stream.text_stream:
                    full_text += text
                    yield {"type": "text", "content": text}

            # Parse the complete response and save plan
            plan_data = self._parse_plan_response(full_text)

            # Create TradingPlan object
            price_data = self._market_data.get("price", {})
            rsi_data = self._technical_data.get("rsi", {})
            market_ctx = self._market_data.get("market", {})

            plan = TradingPlan(
                id=str(uuid.uuid4()),
                user_id=self.user_id,
                symbol=self.symbol,
                status="active",
                bias=plan_data.get("bias", "neutral"),
                thesis=plan_data.get("thesis", ""),
                entry_zone_low=plan_data.get("entry_zone_low"),
                entry_zone_high=plan_data.get("entry_zone_high"),
                stop_loss=plan_data.get("stop_loss"),
                stop_reasoning=plan_data.get("stop_reasoning", ""),
                target_1=plan_data.get("target_1"),
                target_2=plan_data.get("target_2"),
                target_3=plan_data.get("target_3"),
                target_reasoning=plan_data.get("target_reasoning", ""),
                risk_reward=plan_data.get("risk_reward"),
                position_size_pct=plan_data.get("position_size_pct"),
                key_supports=plan_data.get("key_supports", []),
                key_resistances=plan_data.get("key_resistances", []),
                invalidation_criteria=plan_data.get("invalidation_criteria", ""),
                trade_style=plan_data.get("trade_style", "swing"),
                trade_style_reasoning=plan_data.get("trade_style_reasoning", ""),
                holding_period=plan_data.get("holding_period", ""),
                confidence=plan_data.get("confidence", 0),
                price_at_creation=price_data.get("price"),
                rsi_at_creation=rsi_data.get("value"),
                market_direction_at_creation=market_ctx.get("market_direction", ""),
                technical_summary=plan_data.get("technical_summary", ""),
                news_summary=plan_data.get("news_summary", ""),
                reddit_sentiment=plan_data.get("reddit_sentiment", "none"),
                reddit_buzz=plan_data.get("reddit_buzz", ""),
            )

            # Save plan
            self._current_plan = await self._plan_store.save_plan(plan)
            logger.info(f"Created streaming plan for {self.symbol}: {plan.bias}")

            # Yield the final parsed plan
            yield {"type": "plan_complete", "plan": self._plan_to_dict(self._current_plan)}

        except Exception as e:
            logger.error(f"Streaming error for {self.symbol}: {e}")
            yield {"type": "error", "message": str(e)}

    def _plan_to_dict(self, plan: TradingPlan) -> dict:
        """Convert TradingPlan to dictionary for JSON serialization."""
        return {
            "symbol": plan.symbol,
            "bias": plan.bias,
            "thesis": plan.thesis,
            "entry_zone_low": plan.entry_zone_low,
            "entry_zone_high": plan.entry_zone_high,
            "stop_loss": plan.stop_loss,
            "stop_reasoning": plan.stop_reasoning,
            "target_1": plan.target_1,
            "target_2": plan.target_2,
            "target_3": plan.target_3,
            "target_reasoning": plan.target_reasoning,
            "risk_reward": plan.risk_reward,
            "key_supports": plan.key_supports,
            "key_resistances": plan.key_resistances,
            "invalidation_criteria": plan.invalidation_criteria,
            "technical_summary": plan.technical_summary,
            "status": plan.status,
            "created_at": plan.created_at,
            "last_evaluation": plan.last_evaluation,
            "evaluation_notes": plan.evaluation_notes,
            "trade_style": plan.trade_style,
            "trade_style_reasoning": plan.trade_style_reasoning,
            "holding_period": plan.holding_period,
            "confidence": plan.confidence,
            "news_summary": plan.news_summary,
            "reddit_sentiment": plan.reddit_sentiment,
            "reddit_buzz": plan.reddit_buzz,
        }

    async def generate_smart_plan(self) -> EnhancedTradePlan:
        """Generate a smart trading plan with educational content.

        This method:
        1. Gathers all available market data
        2. Determines the optimal trade style (day/swing/position)
        3. Creates comprehensive educational content
        4. Returns an EnhancedTradePlan with hand-holding guidance
        """
        logger.info(f"Generating smart plan for {self.symbol}")

        # Gather fresh data
        await self.gather_comprehensive_data()

        # Format data for the enhanced prompt
        data = self._format_data_for_smart_prompt()

        prompt = SMART_PLAN_GENERATION_PROMPT.format(
            symbol=self.symbol,
            market_data=data["market_data"],
            technical_data=data["technical_data"],
            levels_data=data["levels_data"],
            volume_data=data["volume_data"],
            patterns_data=data["patterns_data"],
            market_context=data["market_context"],
            position_data=data["position_data"],
        )

        client = self._get_client()
        settings = get_settings()

        # Retry logic with exponential backoff for transient errors
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = client.messages.create(
                    model=settings.claude_model_planning,
                    max_tokens=4000,  # Larger for educational content
                    system=SMART_PLANNING_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": prompt}],
                )

                response_text = response.content[0].text
                logger.debug(f"Smart plan response: {response_text[:500]}...")

                # Parse the JSON response
                plan_data = self._parse_smart_plan_response(response_text)

                # Build the EnhancedTradePlan
                enhanced_plan = self._build_enhanced_plan(plan_data)

                logger.info(f"Generated smart plan for {self.symbol}: {enhanced_plan.bias} ({enhanced_plan.trade_style.recommended_style})")

                return enhanced_plan

            except (OverloadedError, RateLimitError) as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API overloaded for {self.symbol}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
                    logger.error(f"API overloaded after {max_retries} attempts for {self.symbol}: {e}")
                    raise ValueError(f"Claude API is currently overloaded. Please try again in a few moments.")

            except APIStatusError as e:
                logger.error(f"API error generating smart plan for {self.symbol}: {e}")
                raise ValueError(f"API error: {str(e)}")

            except Exception as e:
                logger.error(f"Error generating smart plan for {self.symbol}: {e}")
                raise

    def _format_data_for_smart_prompt(self) -> Dict[str, str]:
        """Format gathered data for the smart planning prompt."""
        # Market data with more detail
        price_data = self._market_data.get("price", {})
        market_ctx = self._market_data.get("market", {})

        current_price = price_data.get('price', 0)
        market_str = f"""Current Price: ${current_price:.2f}
Daily Change: {price_data.get('change_pct', 'N/A')}%
Bid: ${price_data.get('bid', 'N/A')} | Ask: ${price_data.get('ask', 'N/A')}
Spread: {price_data.get('spread_pct', 'N/A')}%
"""

        # Market context
        market_context_str = f"""Market Direction: {market_ctx.get('market_direction', 'unknown').upper()}
Bullish Indices: {market_ctx.get('bullish_indices', 0)}/{market_ctx.get('total_indices', 4)}
SPY: {market_ctx.get('spy_direction', 'N/A')}
QQQ: {market_ctx.get('qqq_direction', 'N/A')}
"""

        # Technical data with more indicators
        tech = self._technical_data
        if tech.get("error"):
            tech_str = f"Error: {tech['error']}"
        else:
            rsi = tech.get("rsi", {})
            macd = tech.get("macd", {})
            emas = tech.get("emas", {})
            vol = tech.get("volume", {})
            bb = tech.get("bollinger", {})
            atr = tech.get("atr", {})

            # Calculate ATR percentage for trade style determination
            atr_value = atr.get('value', 0)
            atr_pct = (atr_value / current_price * 100) if current_price > 0 else 0

            tech_str = f"""RSI (14): {rsi.get('value', 0):.1f} - {rsi.get('signal', 'N/A')}
MACD: {macd.get('signal', 'N/A')} | Histogram: {macd.get('histogram', 0):.3f}
Histogram Trend: {macd.get('histogram_trend', 'N/A')}

EMA Alignment:
- Price vs 9 EMA: {'Above' if emas.get('above_9') else 'Below'}
- Price vs 21 EMA: {'Above' if emas.get('above_21') else 'Below'}
- Price vs 50 EMA: {'Above' if emas.get('above_50') else 'Below'}
- EMA Trend: {emas.get('trend', 'N/A')}

Volume:
- Relative Volume: {vol.get('relative', 0):.2f}x average
- Volume Trend: {vol.get('trend', 'N/A')}

Bollinger Bands:
- Position: {bb.get('position', 'N/A')}
- Width: {bb.get('width', 'N/A')}
- %B: {bb.get('percent_b', 'N/A')}

Volatility:
- ATR (14): ${atr_value:.2f}
- ATR %: {atr_pct:.2f}% (of price)
"""

        # Key levels with more detail
        levels = self._levels_data
        if levels.get("error"):
            levels_str = f"Error: {levels['error']}"
        else:
            supports = levels.get("support_levels", [])[:5]
            resistances = levels.get("resistance_levels", [])[:5]

            support_details = []
            for s in supports:
                if isinstance(s, dict):
                    support_details.append(f"${s.get('price', s):.2f} (touches: {s.get('touches', 'N/A')})")
                else:
                    support_details.append(f"${s:.2f}")

            resistance_details = []
            for r in resistances:
                if isinstance(r, dict):
                    resistance_details.append(f"${r.get('price', r):.2f} (touches: {r.get('touches', 'N/A')})")
                else:
                    resistance_details.append(f"${r:.2f}")

            levels_str = f"""Support Levels:
{chr(10).join(['  - ' + s for s in support_details]) if support_details else '  None identified'}

Resistance Levels:
{chr(10).join(['  - ' + r for r in resistance_details]) if resistance_details else '  None identified'}

ATR (14): ${levels.get('atr', 0):.2f}
"""

        # Volume profile
        vol_profile = self._volume_data
        if vol_profile.get("error"):
            volume_str = f"Error: {vol_profile['error']}"
        else:
            volume_str = f"""Volume Profile:
- Point of Control (POC): ${vol_profile.get('poc', 'N/A')}
- Value Area High: ${vol_profile.get('value_area_high', 'N/A')}
- Value Area Low: ${vol_profile.get('value_area_low', 'N/A')}
- Current Price vs POC: {vol_profile.get('price_vs_poc', 'N/A')}
"""

        # Chart patterns with targets
        patterns = self._patterns_data
        if patterns.get("error"):
            patterns_str = f"Error: {patterns['error']}"
        else:
            detected = patterns.get("patterns", [])
            if detected:
                pattern_lines = []
                for p in detected[:5]:
                    pattern_lines.append(
                        f"- {p.get('name', 'Unknown')}: {p.get('type', 'N/A')}\n"
                        f"  Confidence: {p.get('confidence', 'N/A')}%\n"
                        f"  Target: ${p.get('target', 'N/A')}"
                    )
                patterns_str = "Detected Patterns:\n" + "\n".join(pattern_lines)
            else:
                patterns_str = "No significant chart patterns detected"

        # Position data
        pos = self._position_data
        if pos.get("has_position"):
            position_str = f"""Current Position:
- Status: {pos.get('status', 'unknown')}
- Entry: ${pos.get('entry_price', 'N/A')}
- Size: {pos.get('current_size', 0)} shares
- Stop Loss: ${pos.get('stop_loss', 'N/A')}
- Target 1: ${pos.get('target_1', 'N/A')}
- Target 2: ${pos.get('target_2', 'N/A')}
- Target 3: ${pos.get('target_3', 'N/A')}
- Targets Hit: {pos.get('targets_hit', [])}
"""
        else:
            position_str = "No current position"

        return {
            "market_data": market_str,
            "technical_data": tech_str,
            "levels_data": levels_str,
            "volume_data": volume_str,
            "patterns_data": patterns_str,
            "market_context": market_context_str,
            "position_data": position_str,
        }

    def _parse_smart_plan_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the smart plan JSON response."""
        try:
            # Look for JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse smart plan JSON: {e}")
            logger.error(f"Response was: {response_text[:1000]}")
            # Return a minimal valid structure
            return {
                "trade_style": {
                    "recommended_style": "swing",
                    "reasoning": "Unable to parse response",
                    "holding_period": "Unknown"
                },
                "bias": "neutral",
                "thesis": f"Analysis parsing failed: {str(e)[:100]}",
                "confidence": 0,
                "entry_zone_low": None,
                "entry_zone_high": None,
                "stop_loss": None,
                "stop_reasoning": "",
                "targets": [],
                "risk_reward": None,
                "position_size_pct": None,
                "key_supports": [],
                "key_resistances": [],
                "invalidation_criteria": "",
                "educational": {
                    "setup_explanation": "Analysis could not be completed",
                    "level_explanations": {},
                    "what_to_watch": [],
                    "scenarios": [],
                    "risk_warnings": ["Analysis parsing failed - please retry"],
                    "chart_annotations": []
                }
            }

    def _build_enhanced_plan(self, plan_data: Dict[str, Any]) -> EnhancedTradePlan:
        """Build an EnhancedTradePlan from parsed JSON data."""
        # Build trade style
        trade_style_data = plan_data.get("trade_style", {})
        trade_style = TradeStyleRecommendation(
            recommended_style=trade_style_data.get("recommended_style", "swing"),
            reasoning=trade_style_data.get("reasoning", ""),
            holding_period=trade_style_data.get("holding_period", "Unknown")
        )

        # Build targets
        targets = []
        for t in plan_data.get("targets", []):
            if isinstance(t, dict) and t.get("price"):
                targets.append(PriceTarget(
                    price=float(t["price"]),
                    reasoning=t.get("reasoning", "")
                ))

        # Build educational content
        edu_data = plan_data.get("educational", {})

        # Build scenarios
        scenarios = []
        for s in edu_data.get("scenarios", []):
            if isinstance(s, dict):
                scenarios.append(ScenarioPath(
                    scenario=s.get("scenario", "sideways"),
                    probability=int(s.get("probability", 33)),
                    description=s.get("description", ""),
                    price_target=s.get("price_target"),
                    key_trigger=s.get("key_trigger", "")
                ))

        # Build chart annotations
        annotations = []
        for a in edu_data.get("chart_annotations", []):
            if isinstance(a, dict):
                annotations.append(ChartAnnotation(
                    type=a.get("type", "level"),
                    price=a.get("price"),
                    price_high=a.get("price_high"),
                    price_low=a.get("price_low"),
                    label=a.get("label", ""),
                    color=a.get("color", "gray"),
                    description=a.get("description", "")
                ))

        educational = EducationalContent(
            setup_explanation=edu_data.get("setup_explanation", ""),
            level_explanations=edu_data.get("level_explanations", {}),
            what_to_watch=edu_data.get("what_to_watch", []),
            scenarios=scenarios,
            risk_warnings=edu_data.get("risk_warnings", []),
            chart_annotations=annotations
        )

        # Build the complete plan
        return EnhancedTradePlan(
            trade_style=trade_style,
            bias=plan_data.get("bias", "neutral"),
            thesis=plan_data.get("thesis", ""),
            confidence=int(plan_data.get("confidence", 0)),
            entry_zone_low=plan_data.get("entry_zone_low"),
            entry_zone_high=plan_data.get("entry_zone_high"),
            stop_loss=plan_data.get("stop_loss"),
            stop_reasoning=plan_data.get("stop_reasoning", ""),
            targets=targets,
            risk_reward=plan_data.get("risk_reward"),
            position_size_pct=plan_data.get("position_size_pct"),
            key_supports=plan_data.get("key_supports", []),
            key_resistances=plan_data.get("key_resistances", []),
            invalidation_criteria=plan_data.get("invalidation_criteria", ""),
            educational=educational
        )

    def _extract_text_from_response(self, response) -> str:
        """Extract text from Claude response that may contain web search blocks.

        When web search is enabled, the response contains multiple content blocks:
        - server_tool_use: Claude's search query
        - web_search_tool_result: Search results
        - text: Claude's analysis with citations

        We extract the final text block(s) for JSON parsing.
        """
        text_parts = []
        search_count = 0

        for block in response.content:
            if hasattr(block, 'type'):
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "server_tool_use":
                    # Log what Claude searched for
                    if hasattr(block, 'name') and block.name == "web_search":
                        search_count += 1
                        if hasattr(block, 'input') and isinstance(block.input, dict):
                            query = block.input.get('query', 'unknown')
                            logger.info(f"Web search #{search_count} for {self.symbol}: '{query}'")
                elif block.type == "web_search_tool_result":
                    # Log that results were received
                    logger.debug(f"Received web search results for {self.symbol}")

        if search_count > 0:
            logger.info(f"Web search completed: {search_count} searches for {self.symbol}")

        return "\n".join(text_parts)

    def _parse_plan_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI's JSON response into plan data."""
        # Try to extract JSON from the response
        try:
            # Look for JSON block
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse plan JSON: {e}")
            # Return a basic structure
            return {
                "bias": "neutral",
                "thesis": response_text[:500],
                "technical_summary": "Failed to parse structured response",
            }

    async def evaluate_plan(self) -> Dict[str, Any]:
        """Evaluate the current plan against latest market data."""
        if not self._current_plan:
            self._current_plan = await self._plan_store.get_plan(self.user_id, self.symbol)

        if not self._current_plan:
            return {"error": "No plan exists to evaluate"}

        # Gather fresh data
        await self.gather_comprehensive_data()

        # Format data
        data = self._format_data_for_prompt()

        # Format plan for prompt
        plan = self._current_plan
        plan_str = f"""Bias: {plan.bias}
Thesis: {plan.thesis}
Entry Zone: ${plan.entry_zone_low or 'N/A'} - ${plan.entry_zone_high or 'N/A'}
Stop Loss: ${plan.stop_loss or 'N/A'} ({plan.stop_reasoning})
Targets: ${plan.target_1 or 'N/A'} / ${plan.target_2 or 'N/A'} / ${plan.target_3 or 'N/A'}
Risk/Reward: {plan.risk_reward or 'N/A'}
Price at Creation: ${plan.price_at_creation or 'N/A'}
Created: {plan.created_at}
Invalidation: {plan.invalidation_criteria}
"""

        prompt = PLAN_EVALUATION_PROMPT.format(
            symbol=self.symbol,
            plan_data=plan_str,
            market_data=data["market_data"],
            technical_data=data["technical_data"],
            position_data=data["position_data"],
        )

        try:
            client = self._get_client()
            settings = get_settings()

            # Use the planning model for evaluation since it makes important decisions
            response = client.messages.create(
                model=settings.claude_model_planning,
                max_tokens=1500,
                system=PLANNING_AGENT_SYSTEM,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse JSON response
            eval_data = self._parse_evaluation_response(response_text)

            # Determine status
            status_str = eval_data.get("status", "VALID").upper()
            new_status = None
            if status_str == "INVALIDATED":
                new_status = "invalidated"
            elif status_str == "ADJUST":
                new_status = "active"  # Keep active but apply adjustments

            # Build evaluation notes for display
            evaluation_notes = eval_data.get("evaluation", "")
            if eval_data.get("action"):
                evaluation_notes += f"\n\n**Action**: {eval_data['action']}"
            if eval_data.get("adjustment_rationale"):
                evaluation_notes += f"\n\n**Adjustments**: {eval_data['adjustment_rationale']}"

            # Extract adjustments (filter out nulls)
            adjustments = {}
            if eval_data.get("adjustments"):
                for key, value in eval_data["adjustments"].items():
                    if value is not None:
                        adjustments[key] = value

            # Capture previous values before applying adjustments
            previous_values = {}
            new_values = {}
            if adjustments:
                for field in adjustments.keys():
                    # Get the previous value from the current plan
                    prev_value = getattr(plan, field, None)
                    if prev_value is not None:
                        previous_values[field] = prev_value
                    new_values[field] = adjustments[field]

            # Update plan with evaluation and any adjustments
            updated_plan = await self._plan_store.update_evaluation(
                self.user_id,
                self.symbol,
                evaluation_notes,
                new_status,
                adjustments if adjustments else None
            )

            # Update cached plan
            if updated_plan:
                self._current_plan = updated_plan

            return {
                "symbol": self.symbol,
                "evaluation": evaluation_notes,
                "plan_status": new_status or plan.status,
                "current_price": self._market_data.get("price", {}).get("price"),
                "price_at_creation": plan.price_at_creation,
                "adjustments_made": list(adjustments.keys()) if adjustments else [],
                "previous_values": previous_values,
                "new_values": new_values,
            }

        except Exception as e:
            logger.error(f"Error evaluating plan for {self.symbol}: {e}")
            raise

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse the AI's JSON evaluation response."""
        try:
            # Try to find JSON in the response
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                json_str = response_text[json_start:json_end].strip()
            elif "{" in response_text:
                json_start = response_text.find("{")
                json_end = response_text.rfind("}") + 1
                json_str = response_text[json_start:json_end]
            else:
                raise ValueError("No JSON found in response")

            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse evaluation JSON: {e}")
            # Fall back to basic parsing
            return {
                "status": "VALID" if "INVALIDATED" not in response_text.upper() else "INVALIDATED",
                "evaluation": response_text[:500],
                "adjustments": {}
            }

    async def chat(self, user_message: str) -> str:
        """Chat with the agent about the stock and plan.

        The agent has full context of:
        - All technical data
        - The trading plan
        - Previous conversation history
        """
        logger.info(f"Chat message for {self.symbol}: {user_message[:50]}...")

        # Load conversation history
        conversation = await self._conversation_store.get_conversation(
            self.user_id, self.symbol
        )

        # Gather fresh data if we haven't recently
        if not self._market_data:
            await self.gather_comprehensive_data()

        # Load plan if not loaded
        if not self._current_plan:
            self._current_plan = await self._plan_store.get_plan(self.user_id, self.symbol)

        # Build comprehensive context
        data = self._format_data_for_prompt()

        context_sections = [
            f"## Current Market Data for {self.symbol}",
            data["market_data"],
            "",
            "## Technical Analysis",
            data["technical_data"],
            "",
            "## Key Levels",
            data["levels_data"],
        ]

        if self._current_plan:
            plan = self._current_plan
            context_sections.extend([
                "",
                "## Current Trading Plan",
                f"Bias: {plan.bias}",
                f"Thesis: {plan.thesis}",
                f"Entry Zone: ${plan.entry_zone_low or 'N/A'} - ${plan.entry_zone_high or 'N/A'}",
                f"Stop Loss: ${plan.stop_loss or 'N/A'}",
                f"Targets: ${plan.target_1 or 'N/A'} / ${plan.target_2 or 'N/A'} / ${plan.target_3 or 'N/A'}",
                f"Risk/Reward: {plan.risk_reward or 'N/A'}",
                f"Status: {plan.status}",
                "",
                f"Last Evaluation: {plan.evaluation_notes or 'None yet'}",
            ])
        else:
            context_sections.extend([
                "",
                "## Trading Plan",
                "No trading plan created yet. User can ask to create one.",
            ])

        if self._position_data.get("has_position"):
            context_sections.extend([
                "",
                "## Current Position",
                data["position_data"],
            ])

        context_str = "\n".join(context_sections)

        system_prompt = f"""{PLANNING_AGENT_SYSTEM}

---

## Current Context for {self.symbol}

{context_str}

---

Answer the user's question based on this data. If they ask to create or update a plan, explain what you see and what you'd recommend. Be concise but thorough.
"""

        # Build messages with history
        messages = conversation.to_claude_messages(max_messages=10)
        messages.append({"role": "user", "content": user_message})

        try:
            client = self._get_client()
            settings = get_settings()

            response = client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=1500,
                system=system_prompt,
                messages=messages,
            )

            response_text = response.content[0].text

            # Save conversation
            conversation.add_message("user", user_message)
            conversation.add_message("assistant", response_text)
            await self._conversation_store.save_conversation(conversation)

            return response_text

        except Exception as e:
            logger.error(f"Chat error for {self.symbol}: {e}")
            return f"Sorry, I encountered an error: {e}"

    async def get_plan(self) -> Optional[TradingPlan]:
        """Get the current trading plan."""
        if not self._current_plan:
            self._current_plan = await self._plan_store.get_plan(self.user_id, self.symbol)
        return self._current_plan

    async def clear_conversation(self) -> bool:
        """Clear the conversation history."""
        return await self._conversation_store.clear_conversation(self.user_id, self.symbol)

    async def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the current state."""
        await self.gather_comprehensive_data()

        price_data = self._market_data.get("price", {})

        return {
            "symbol": self.symbol,
            "current_price": price_data.get("price"),
            "has_plan": self._current_plan is not None,
            "plan_status": self._current_plan.status if self._current_plan else None,
            "plan_bias": self._current_plan.bias if self._current_plan else None,
            "has_position": self._position_data.get("has_position", False),
            "market_direction": self._market_data.get("market", {}).get("market_direction"),
        }

    # =========================================================================
    # Interactive Plan Session Methods (Claude Code-style planning)
    # =========================================================================

    async def handle_user_question(
        self,
        draft_plan: Dict[str, Any],
        question: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle a user's question about the draft plan.

        Returns an explanation and optionally presents options.
        """
        settings = get_settings()

        # Build context from the plan
        plan_context = self._format_plan_for_prompt(draft_plan)

        # Format conversation history
        history_text = ""
        for msg in conversation_history[-5:]:  # Last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

        prompt = f"""The user has a draft trading plan and is asking a question about it.

## Draft Plan for {self.symbol}
{plan_context}

## Recent Conversation
{history_text}

## User's Question
{question}

## Instructions
Answer the user's question thoughtfully and thoroughly. Key principles:

1. EXPLAIN YOUR REASONING - Don't just state facts, explain WHY
2. BE HONEST - If there are tradeoffs or risks, mention them
3. REFERENCE SPECIFIC DATA - Use actual price levels, indicators, ATR values
4. IF THE QUESTION SUGGESTS A CHANGE - Explain the implications but don't automatically agree

For questions about trade style, explain based on:
- Pattern timeframe (intraday vs daily vs weekly)
- ATR and volatility characteristics
- Distance to targets vs typical daily range

For questions about levels (stop, entry, targets), explain based on:
- Technical significance of the level
- Risk/reward implications of changing it
- Probability of getting stopped out by normal volatility

If the user's question implies they want something that might not be optimal, respectfully explain the tradeoff and offer alternatives.

Respond conversationally in plain text. No markdown formatting."""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=1000,
                system="You are an expert trading advisor explaining a trading plan. Be thorough but conversational.",
                messages=[{"role": "user", "content": prompt}]
            )

            ai_response = response.content[0].text

            # Check if the response suggests options
            options = []
            if "option" in ai_response.lower() or "alternative" in ai_response.lower():
                # AI mentioned alternatives, but we let the UI parse them from the text
                pass

            return {
                "response": ai_response,
                "options": options
            }

        except Exception as e:
            logger.error(f"Error handling user question: {e}")
            return {
                "response": f"I apologize, but I encountered an error processing your question. Please try again.",
                "options": []
            }

    async def handle_adjustment_request(
        self,
        draft_plan: Dict[str, Any],
        request: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Handle a user's request to adjust the draft plan.

        Returns an explanation of tradeoffs and options, NOT blind agreement.
        """
        settings = get_settings()

        # Gather fresh market data for context
        await self.gather_comprehensive_data()

        # Build context
        plan_context = self._format_plan_for_prompt(draft_plan)
        market_context = self._format_market_data_for_prompt()

        # Format conversation history
        history_text = ""
        for msg in conversation_history[-5:]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            history_text += f"{role.upper()}: {content}\n"

        prompt = f"""The user wants to adjust their draft trading plan. Your job is to evaluate the request and provide thoughtful guidance - NOT to blindly agree.

## Draft Plan for {self.symbol}
{plan_context}

## Current Market Data
{market_context}

## Recent Conversation
{history_text}

## User's Adjustment Request
{request}

## Instructions
Evaluate the user's adjustment request carefully. DO NOT automatically agree with changes that might harm the trade's probability of success.

1. UNDERSTAND THE REQUEST - What specifically does the user want to change?

2. EVALUATE THE TRADEOFFS - For any adjustment:
   - How does it affect risk/reward ratio?
   - How does it affect probability of success?
   - Are there technical reasons why this might not work?

3. PRESENT OPTIONS - Always give 2-3 concrete options:
   - Option 1: Keep the original (explain why it's reasonable)
   - Option 2: The user's requested change (explain the tradeoffs)
   - Option 3: A compromise (if applicable)

4. MAKE A RECOMMENDATION - State which option you recommend and why

5. If the user's request is reasonable, you can apply it. If it's problematic, explain why and offer alternatives.

## Response Format
Respond with your analysis in plain conversational text.

At the end, if you recommend applying a change, include a JSON block with the updated plan values:

UPDATED_PLAN_JSON:
{{"stop_loss": 184.50, "risk_reward": 2.8}}

Only include fields that should be changed. If you don't recommend a change, don't include the JSON block."""

        try:
            client = self._get_client()
            response = client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=1500,
                system="You are an expert trading advisor. Your job is to protect the user from making bad trades while respecting their preferences. Never blindly agree - always explain tradeoffs.",
                messages=[{"role": "user", "content": prompt}]
            )

            ai_response = response.content[0].text

            # Parse options from the response (simplified - UI can do more sophisticated parsing)
            options = []

            # Check if AI provided updated plan values
            updated_plan = None
            if "UPDATED_PLAN_JSON:" in ai_response:
                try:
                    json_start = ai_response.index("UPDATED_PLAN_JSON:") + len("UPDATED_PLAN_JSON:")
                    json_text = ai_response[json_start:].strip()
                    # Find the JSON object
                    if json_text.startswith("{"):
                        # Find matching closing brace
                        brace_count = 0
                        end_idx = 0
                        for i, char in enumerate(json_text):
                            if char == "{":
                                brace_count += 1
                            elif char == "}":
                                brace_count -= 1
                                if brace_count == 0:
                                    end_idx = i + 1
                                    break
                        json_str = json_text[:end_idx]
                        adjustments = json.loads(json_str)

                        # Apply adjustments to draft plan
                        updated_plan = draft_plan.copy()
                        for key, value in adjustments.items():
                            if key in updated_plan:
                                updated_plan[key] = value

                        # Recalculate risk/reward if stop or targets changed
                        if "stop_loss" in adjustments or "target_1" in adjustments:
                            entry_mid = (updated_plan.get("entry_zone_low", 0) + updated_plan.get("entry_zone_high", 0)) / 2
                            if entry_mid and updated_plan.get("stop_loss") and updated_plan.get("target_1"):
                                risk = abs(entry_mid - updated_plan["stop_loss"])
                                reward = abs(updated_plan["target_1"] - entry_mid)
                                if risk > 0:
                                    updated_plan["risk_reward"] = round(reward / risk, 2)

                        # Remove the JSON from the response text
                        ai_response = ai_response[:ai_response.index("UPDATED_PLAN_JSON:")].strip()

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Could not parse updated plan JSON: {e}")

            return {
                "response": ai_response,
                "options": options,
                "updated_plan": updated_plan
            }

        except Exception as e:
            logger.error(f"Error handling adjustment request: {e}")
            return {
                "response": f"I apologize, but I encountered an error processing your adjustment request. Please try again.",
                "options": [],
                "updated_plan": None
            }

    def _format_plan_for_prompt(self, plan: Dict[str, Any]) -> str:
        """Format a plan dict into a readable prompt string."""
        lines = []

        if plan.get("trade_style"):
            lines.append(f"Trade Style: {plan['trade_style']} ({plan.get('holding_period', 'N/A')})")

        if plan.get("bias"):
            lines.append(f"Bias: {plan['bias'].upper()}")

        if plan.get("confidence"):
            lines.append(f"Confidence: {plan['confidence']}%")

        if plan.get("thesis"):
            lines.append(f"\nThesis: {plan['thesis']}")

        lines.append("\nLevels:")
        if plan.get("entry_zone_low") and plan.get("entry_zone_high"):
            lines.append(f"  Entry Zone: ${plan['entry_zone_low']:.2f} - ${plan['entry_zone_high']:.2f}")
        if plan.get("stop_loss"):
            lines.append(f"  Stop Loss: ${plan['stop_loss']:.2f}")
            if plan.get("stop_reasoning"):
                lines.append(f"    Reasoning: {plan['stop_reasoning']}")
        if plan.get("target_1"):
            lines.append(f"  Target 1: ${plan['target_1']:.2f}")
        if plan.get("target_2"):
            lines.append(f"  Target 2: ${plan['target_2']:.2f}")
        if plan.get("target_3"):
            lines.append(f"  Target 3: ${plan['target_3']:.2f}")
        if plan.get("risk_reward"):
            lines.append(f"  Risk/Reward: {plan['risk_reward']:.1f}:1")

        if plan.get("key_supports"):
            lines.append(f"\nKey Supports: {', '.join(f'${s:.2f}' for s in plan['key_supports'][:3])}")
        if plan.get("key_resistances"):
            lines.append(f"Key Resistances: {', '.join(f'${r:.2f}' for r in plan['key_resistances'][:3])}")

        if plan.get("invalidation_criteria"):
            lines.append(f"\nInvalidation: {plan['invalidation_criteria']}")

        return "\n".join(lines)

    def _format_market_data_for_prompt(self) -> str:
        """Format current market data for prompt context."""
        lines = []

        price_data = self._market_data.get("price", {})
        if price_data:
            lines.append(f"Current Price: ${price_data.get('price', 0):.2f}")
            if price_data.get("change_pct"):
                lines.append(f"Daily Change: {price_data['change_pct']:+.2f}%")

        tech_data = self._market_data.get("technicals", {})
        if tech_data:
            if tech_data.get("rsi"):
                lines.append(f"RSI(14): {tech_data['rsi']:.1f}")
            if tech_data.get("atr"):
                lines.append(f"ATR: ${tech_data['atr']:.2f}")

        levels_data = self._market_data.get("levels", {})
        if levels_data:
            if levels_data.get("supports"):
                lines.append(f"Nearby Supports: {', '.join(f'${s:.2f}' for s in levels_data['supports'][:2])}")
            if levels_data.get("resistances"):
                lines.append(f"Nearby Resistances: {', '.join(f'${r:.2f}' for r in levels_data['resistances'][:2])}")

        market_ctx = self._market_data.get("market", {})
        if market_ctx:
            lines.append(f"Market Direction: {market_ctx.get('market_direction', 'N/A')}")

        return "\n".join(lines) if lines else "No current market data available."
