"""Agentic stock analyzer using iterative tool calling.

This module implements an AI-driven analysis loop where the AI:
1. Has access to a set of tools (price, chart, fundamentals, news, X search)
2. Decides what to investigate based on what it discovers
3. Iteratively calls tools and reasons about results
4. Eventually provides a comprehensive trading analysis

This mimics how ChatGPT/Claude/Grok work with function calling.
"""

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from app.agent.providers import AIProvider, AIMessage
from app.agent.sdk.agentic_tools import AGENTIC_TOOLS
from app.agent.sdk.tool_executor import execute_tool
from app.agent.schemas.streaming import StreamEvent, StreamEventType, OrchestratorStepType

logger = logging.getLogger(__name__)


def _create_stream_event(event_type: str, **kwargs) -> StreamEvent:
    """Create a StreamEvent with proper timestamp."""
    return StreamEvent(
        type=event_type,
        timestamp=time.time(),
        **kwargs
    )


class AgenticStockAnalyzer:
    """AI-driven stock analyzer using iterative tool calls.

    The AI investigates a stock naturally, calling tools as needed,
    building understanding incrementally, and providing transparent reasoning.
    """

    def __init__(
        self,
        provider: AIProvider,
        x_provider: Optional[AIProvider] = None,
        max_iterations: int = 15,
    ):
        """Initialize the agentic analyzer.

        Args:
            provider: Primary AI provider (Claude or Grok)
            x_provider: Provider for X/Twitter search (preferably Grok)
            max_iterations: Maximum tool call iterations to prevent infinite loops
        """
        self.provider = provider
        self.x_provider = x_provider or provider
        self.max_iterations = max_iterations

    async def analyze(
        self,
        symbol: str,
        user_id: str,
        additional_context: Optional[str] = None,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Run agentic analysis with streaming progress.

        Args:
            symbol: Stock ticker symbol to analyze
            user_id: User ID for position lookups and preferences
            additional_context: Optional additional context from user

        Yields:
            StreamEvent objects for real-time progress updates
        """
        start_time = time.time()

        # Build execution context
        context = {
            "symbol": symbol.upper(),
            "user_id": user_id,
            "provider": self.provider,
            "x_provider": self.x_provider,
        }

        # Build system prompt
        system_prompt = self._build_system_prompt(symbol, additional_context)

        # Initialize conversation
        initial_message = self._build_initial_message(symbol, additional_context)
        messages: List[Dict[str, Any]] = [
            {"role": "user", "content": initial_message}
        ]

        # Stream start event
        yield StreamEvent.orchestrator_step(
            OrchestratorStepType.AGENTIC_ANALYSIS,
            "active",
            [f"Starting AI-driven analysis of {symbol}"],
        )

        iteration = 0
        tool_calls_made = []

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(f"[Agentic] Iteration {iteration}/{self.max_iterations}")

            try:
                # Call AI with tools
                response = await self.provider.create_message(
                    messages=[AIMessage(role=m["role"], content=m["content"]) for m in messages],
                    system=system_prompt,
                    tools=self._format_tools_for_provider(),
                    model_type="planning",
                    max_tokens=4000,
                )

                # Extract thinking/reasoning text
                thinking_text = self._extract_thinking(response)
                if thinking_text:
                    yield StreamEvent.agent_thinking(thinking_text, iteration)

                # Check for tool calls
                if response.tool_calls and len(response.tool_calls) > 0:
                    # Process each tool call
                    for tool_call in response.tool_calls:
                        tool_name = tool_call.name
                        tool_args = tool_call.arguments

                        logger.info(f"[Agentic] Tool call: {tool_name}({tool_args})")

                        # Stream tool call event
                        yield StreamEvent.tool_call(tool_name, tool_args, iteration)

                        # Execute the tool
                        tool_result = await execute_tool(tool_name, tool_args, context)
                        tool_calls_made.append({
                            "tool": tool_name,
                            "args": tool_args,
                            "result_summary": self._summarize_result(tool_result),
                        })

                        # Stream tool result event
                        yield StreamEvent.tool_result_event(tool_name, tool_result, iteration)

                        # Add assistant message and tool result to conversation
                        # Handle the case where response.content might be empty
                        assistant_content = response.content if response.content else f"Calling {tool_name}..."
                        messages.append({"role": "assistant", "content": assistant_content})
                        messages.append({
                            "role": "user",
                            "content": f"Tool '{tool_name}' result:\n```json\n{json.dumps(tool_result, indent=2, default=str)}\n```"
                        })

                else:
                    # No tool calls - AI is done, parse final response
                    logger.info(f"[Agentic] AI finished after {iteration} iterations")

                    final_plan = self._parse_final_response(response.content, symbol)

                    elapsed = time.time() - start_time

                    # Stream final result using class method
                    # Use agentic_plan (not plan) for the raw agentic format
                    yield _create_stream_event(
                        StreamEventType.FINAL_RESULT.value,
                        agentic_plan=final_plan,  # Agentic plan with day/swing/position_trade_plan
                        selected_style=final_plan.get("recommended_style") or "swing",
                        selection_reasoning=final_plan.get("recommendation_reasoning", ""),
                        data={
                            "symbol": symbol,
                            "iterations": iteration,
                            "tools_called": len(tool_calls_made),
                            "tool_history": tool_calls_made,
                            "elapsed_seconds": round(elapsed, 2),
                        }
                    )
                    return

            except Exception as e:
                logger.error(f"[Agentic] Error in iteration {iteration}: {e}", exc_info=True)
                yield StreamEvent.error(f"Error in iteration {iteration}: {str(e)}")
                # Try to continue if recoverable
                if iteration >= self.max_iterations:
                    raise

        # Max iterations reached without final response
        logger.warning(f"[Agentic] Max iterations ({self.max_iterations}) reached")
        yield StreamEvent.error(f"Maximum iterations ({self.max_iterations}) reached without completing analysis")

    def _build_system_prompt(self, symbol: str, additional_context: Optional[str] = None) -> str:
        """Build the system prompt for the agentic analyzer."""
        context_section = ""
        if additional_context:
            context_section = f"\n\n## ADDITIONAL CONTEXT FROM USER\n{additional_context}\n"

        return f"""You are an expert stock trading analyst. Your task is to analyze {symbol} for trading opportunities across day trade, swing trade, and position trade timeframes.

## YOUR INVESTIGATION APPROACH (MUST COMPLETE ALL STEPS)

1. **Start with context**: Check current price and market direction first
2. **Check user position**: See if user has an existing position (CRITICAL for recommendations)
3. **Check volatility**: Get ATR for all 3 timeframes (5m, 1d, 1w)
4. **Analyze ALL 3 charts** (REQUIRED - do not skip any):
   - 5m chart with trade_style="day" for day trade analysis
   - 1d chart with trade_style="swing" for swing trade analysis
   - 1w chart with trade_style="position" for position trade analysis
5. **Technical analysis**: Get indicators for each relevant timeframe
6. **Support/Resistance**: Get key levels for daily and weekly
7. **Fibonacci levels**: Get retracement/extension levels
8. **Fundamentals**: Get company fundamentals (REQUIRED)
9. **News sentiment**: Check recent news (REQUIRED)
10. **X/Twitter sentiment**: Search for real-time trader sentiment (REQUIRED)
11. **Form complete picture**: Only after gathering ALL data, provide final recommendation

## INVESTIGATION STYLE

- **Think out loud**: After each tool result, explain what you learned and what you need to investigate next
- **Be thorough**: You MUST analyze all 3 chart timeframes before making any recommendations
- **Never skip fundamentals or sentiment**: These are required for conviction assessment
- **Be deliberate**: Understand each piece before moving on
{context_section}
## CRITICAL RULES

1. **LONG ONLY - NO SHORTING**: This platform is for LONG trades only. NEVER recommend shorting or bearish trades.
   - If your analysis is bearish for a timeframe: mark it as "suitable": false with bias "neutral" and explain why it's not a good time to enter
   - If user has NO position and outlook is bearish: recommend staying out / waiting for better setup
   - If user HAS a position and outlook turns bearish: recommend "hold", "trim", or "exit" based on severity

2. **Position awareness**: If user has an existing position, provide hold/add/trim/exit recommendations. Never recommend actions against their position direction.

3. **All 3 timeframes**: Your final output MUST include analysis for day trade, swing trade, AND position trade - even if some are not suitable.

4. **Specific levels**: Always provide specific entry, stop loss, and target prices based on technical analysis. For unsuitable/bearish setups, still provide levels for reference but mark suitable=false.

5. **Conviction with reasoning**: For each timeframe, provide conviction (high/medium/low) WITH specific reasoning referencing the data you gathered.

## FINAL OUTPUT FORMAT

When you've gathered enough information, provide your final analysis as JSON:

```json
{{
  "recommended_style": "swing",
  "recommendation_reasoning": "Detailed explanation of why this style is best, referencing specific data points...",

  "day_trade_plan": {{
    "suitable": false,
    "conviction": "low",
    "conviction_reasoning": "ATR is only 1.8%, not enough intraday volatility for day trading...",
    "bias": "neutral",
    "thesis": "Not suitable for day trading due to low volatility. Wait for higher ATR environment.",
    "entry_zone": [148.50, 149.00],
    "stop_loss": 147.80,
    "targets": [150.00, 151.50],
    "holding_period": "1-3 hours"
  }},

  "swing_trade_plan": {{
    "suitable": true,
    "conviction": "high",
    "conviction_reasoning": "Clean bull flag on daily chart, 61.8% Fib holding, volume drying up in consolidation, strong fundamentals with 30% EPS growth...",
    "bias": "bullish",
    "thesis": "Detailed thesis integrating technicals, fundamentals, and sentiment...",
    "entry_zone": [147.00, 148.00],
    "stop_loss": 144.50,
    "targets": [155.00, 162.00, 170.00],
    "holding_period": "5-10 days"
  }},

  "position_trade_plan": {{
    "suitable": false,
    "conviction": "low",
    "conviction_reasoning": "Chart showing distribution pattern, breaking below key EMAs. Not a good time to enter long.",
    "bias": "neutral",
    "thesis": "Current price action is bearish. Stay out and wait for reversal confirmation before entering.",
    "entry_zone": null,
    "stop_loss": null,
    "targets": null,
    "holding_period": null
  }},

  "position_recommendation": null,
  "risk_warnings": ["Warning 1", "Warning 2"],
  "what_to_watch": ["Trigger 1 at $X", "Trigger 2"]
}}
```

If user has a position, include position_recommendation with one of these actions:
- "hold" - Keep position, outlook still favorable
- "add" - Consider adding to position on pullback
- "trim" - Reduce position size due to warning signs
- "exit" - Close position, outlook has turned negative

```json
"position_recommendation": {{
  "action": "trim",
  "reasoning": "Breaking below key support with increasing volume. Reducing risk exposure.",
  "adjustment": "Trim 50% at current levels, hold rest with stop at $X"
}}
```

Start your investigation now. Remember to think out loud about what you're learning at each step."""

    def _build_initial_message(self, symbol: str, additional_context: Optional[str] = None) -> str:
        """Build the initial user message."""
        msg = f"Analyze {symbol} for trading opportunities. Investigate thoroughly and provide complete analysis for day trade, swing trade, and position trade timeframes."

        if additional_context:
            msg += f"\n\nAdditional context: {additional_context}"

        return msg

    def _format_tools_for_provider(self) -> List[Dict[str, Any]]:
        """Format tools for the AI provider.

        Claude expects native format:
        {
            "name": "...",
            "description": "...",
            "input_schema": {...}
        }

        Grok/OpenAI expects:
        {
            "type": "function",
            "function": {
                "name": "...",
                "description": "...",
                "parameters": {...}
            }
        }
        """
        from app.agent.providers import ModelProvider

        # Check which provider we're using
        provider_type = self.provider.config.provider

        if provider_type == ModelProvider.CLAUDE:
            # Return tools in Claude's native format
            return AGENTIC_TOOLS
        else:
            # Convert to OpenAI/Grok format
            openai_tools = []
            for tool in AGENTIC_TOOLS:
                openai_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["input_schema"]
                    }
                }
                openai_tools.append(openai_tool)
            return openai_tools

    def _extract_thinking(self, response) -> Optional[str]:
        """Extract thinking/reasoning text from response."""
        if hasattr(response, "content") and response.content:
            # Return content that isn't just a tool call
            content = response.content.strip()
            if content and not content.startswith("{"):
                return content
        return None

    def _summarize_result(self, result: Dict) -> str:
        """Create a brief summary of a tool result for logging."""
        if "error" in result:
            return f"Error: {result['error']}"

        # Create brief summaries based on result type
        if "price" in result:
            return f"Price: ${result.get('price')}"
        elif "market_direction" in result:
            return f"Market: {result.get('market_direction')}"
        elif "has_position" in result:
            return f"Position: {'Yes' if result.get('has_position') else 'No'}"
        elif "chart_image" in result:
            return "Chart generated"
        elif "trend_quality" in result:
            return f"Vision: {result.get('trend_quality')}"
        elif "ema_trend" in result:
            return f"Technicals: {result.get('ema_trend')}"
        elif "support_levels" in result:
            return f"S/R levels found"
        elif "retracement_levels" in result:
            return f"Fibonacci: {result.get('trend')}"
        elif "valuation" in result:
            return f"Fundamentals fetched"
        elif "sentiment" in result:
            return f"News: {result.get('sentiment')}"
        elif "sentiment_analysis" in result:
            return f"X sentiment analyzed"
        elif "atr" in result:
            return f"ATR: {result.get('atr_pct'):.2f}%"

        return "Result received"

    def _parse_final_response(self, content: str, symbol: str) -> Dict[str, Any]:
        """Parse the final JSON response from the AI."""
        try:
            # Try to extract JSON from the response
            # The AI might wrap it in markdown code blocks
            json_str = content

            # Remove markdown code blocks if present
            if "```json" in json_str:
                start = json_str.find("```json") + 7
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()
            elif "```" in json_str:
                start = json_str.find("```") + 3
                end = json_str.find("```", start)
                json_str = json_str[start:end].strip()

            # Parse JSON
            plan = json.loads(json_str)

            # Ensure required fields exist
            plan["symbol"] = symbol
            if "recommended_style" not in plan:
                plan["recommended_style"] = "swing"  # Default
            if "recommendation_reasoning" not in plan:
                plan["recommendation_reasoning"] = content[:500]  # Use raw content as fallback

            return plan

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse final response as JSON: {e}")
            # Return a structured fallback
            return {
                "symbol": symbol,
                "parse_error": True,
                "raw_response": content,
                "recommended_style": "swing",
                "recommendation_reasoning": content[:500],
                "day_trade_plan": {"suitable": False, "conviction": "low", "conviction_reasoning": "Parse error"},
                "swing_trade_plan": {"suitable": False, "conviction": "low", "conviction_reasoning": "Parse error"},
                "position_trade_plan": {"suitable": False, "conviction": "low", "conviction_reasoning": "Parse error"},
            }


# Helper function to create analyzer with proper providers
async def create_agentic_analyzer(user_id: str) -> AgenticStockAnalyzer:
    """Create an agentic analyzer with the user's preferred providers.

    Args:
        user_id: User ID for provider preferences

    Returns:
        Configured AgenticStockAnalyzer
    """
    from app.agent.providers.factory import get_user_provider, get_provider, ModelProvider, is_provider_available

    # Get user's preferred provider
    primary_provider = await get_user_provider(user_id)

    # For X search, prefer Grok if available
    x_provider = primary_provider
    if is_provider_available(ModelProvider.GROK):
        try:
            x_provider = get_provider(ModelProvider.GROK)
        except ValueError:
            logger.warning("Grok not available for X search, using primary provider")

    return AgenticStockAnalyzer(
        provider=primary_provider,
        x_provider=x_provider,
        max_iterations=15,
    )
