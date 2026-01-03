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

import httpx
from anthropic._exceptions import OverloadedError, RateLimitError, APIStatusError

from app.config import get_settings
from app.agent.providers.factory import get_user_provider
from app.agent.providers import AIMessage, AIProvider, ModelProvider, SearchParameters
from app.agent.providers.grok_provider import get_x_search_parameters
from app.agent.tools import (
    get_current_price,
    get_key_levels,
    get_technical_indicators,
    run_full_analysis,
    get_volume_profile,
    get_chart_patterns,
    get_position_status,
    get_market_context,
    get_fibonacci_levels,
)
from app.storage.plan_store import TradingPlan, get_plan_store
from app.storage.conversation_store import get_conversation_store
from app.storage.position_store import get_position_store
from app.agent.prompts import (
    SMART_PLANNING_SYSTEM_PROMPT,
    SMART_PLAN_GENERATION_PROMPT,
    SMART_PLAN_EVALUATION_PROMPT,
    VISUAL_ANALYSIS_PROMPT,
)
from app.tools.chart_generator import generate_chart_image
from app.models.response import (
    EnhancedTradePlan,
    TradeStyleRecommendation,
    EducationalContent,
    ScenarioPath,
    ChartAnnotation,
    PriceTarget,
)

logger = logging.getLogger(__name__)


# Price fields that should be numeric (not strings with $ signs)
PRICE_FIELDS = {
    "entry_zone_low", "entry_zone_high", "stop_loss",
    "target_1", "target_2", "target_3", "risk_reward",
    "price_at_creation", "position_size_pct"
}

# List fields that contain prices
PRICE_LIST_FIELDS = {"key_supports", "key_resistances"}


def _sanitize_price_value(value: Any) -> Optional[float]:
    """Convert a price value to float, handling dollar signs and formatting.

    Handles cases like:
    - "$122.50" -> 122.50
    - "122.50" -> 122.50
    - 122.50 -> 122.50
    - "$1,234.56" -> 1234.56
    - None -> None
    """
    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        # Remove $ and commas
        cleaned = value.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            logger.warning(f"Could not parse price value: {value}")
            return None

    return None


def _sanitize_plan_data(plan_data: Dict[str, Any]) -> Dict[str, Any]:
    """Sanitize all price fields in plan data to ensure they are numeric."""
    sanitized = plan_data.copy()

    # Sanitize individual price fields
    for field in PRICE_FIELDS:
        if field in sanitized:
            sanitized[field] = _sanitize_price_value(sanitized[field])

    # Sanitize price lists
    for field in PRICE_LIST_FIELDS:
        if field in sanitized and isinstance(sanitized[field], list):
            sanitized[field] = [
                _sanitize_price_value(v) for v in sanitized[field]
                if _sanitize_price_value(v) is not None
            ]

    # Handle nested targets if present (for smart plan format)
    if "targets" in sanitized and isinstance(sanitized["targets"], list):
        for target in sanitized["targets"]:
            if isinstance(target, dict) and "price" in target:
                target["price"] = _sanitize_price_value(target["price"])

    return sanitized


def validate_plan_price_consistency(
    bias: str,
    entry_zone_low: Optional[float],
    entry_zone_high: Optional[float],
    stop_loss: Optional[float],
    targets: List[Optional[float]],
) -> List[str]:
    """Validate price levels are consistent with trade bias.

    For BULLISH (long) trades:
    - stop_loss should be BELOW entry
    - targets should be ABOVE entry

    For BEARISH (short) trades:
    - stop_loss should be ABOVE entry
    - targets should be BELOW entry

    Returns list of warning messages if inconsistent.
    """
    warnings = []

    if not entry_zone_low or not entry_zone_high:
        return warnings

    entry_mid = (entry_zone_low + entry_zone_high) / 2

    if bias == "bullish":
        # LONG: stop < entry, targets > entry
        if stop_loss and stop_loss >= entry_mid:
            warnings.append(
                f"BULLISH plan has stop_loss ${stop_loss:.2f} >= entry ${entry_mid:.2f}. "
                "Stop should be BELOW entry for long trades."
            )
        for i, target in enumerate(targets, 1):
            if target and target <= entry_mid:
                warnings.append(
                    f"BULLISH plan has target_{i} ${target:.2f} <= entry ${entry_mid:.2f}. "
                    "Targets should be ABOVE entry for long trades."
                )

    elif bias == "bearish":
        # SHORT: stop > entry, targets < entry
        if stop_loss and stop_loss <= entry_mid:
            warnings.append(
                f"BEARISH plan has stop_loss ${stop_loss:.2f} <= entry ${entry_mid:.2f}. "
                "Stop should be ABOVE entry for short trades."
            )
        for i, target in enumerate(targets, 1):
            if target and target >= entry_mid:
                warnings.append(
                    f"BEARISH plan has target_{i} ${target:.2f} >= entry ${entry_mid:.2f}. "
                    "Targets should be BELOW entry for short trades."
                )

    return warnings


# System prompt for the planning agent
PLANNING_AGENT_SYSTEM = """You are an expert swing trader and technical analyst. Your job is to analyze stocks comprehensively and create actionable trading plans.

## Your Approach
1. **Search for context first** - Before analyzing technicals, search for recent news and social sentiment about the stock
2. **Analyze the data thoroughly** - Look at all technical indicators, key levels, volume, and market context
3. **Form a thesis** - What's the story? Why would this trade work?
4. **Define precise levels** - Entry zones, stop loss, and profit targets based on actual support/resistance
5. **Consider risk** - Position sizing, risk/reward, and what would invalidate the trade
6. **Be honest** - If the setup isn't good, say so. Not every stock is a trade.

## Social Search Strategy
When search is available, ALWAYS search for:
1. "{symbol} stock news" - Get recent news, earnings, analyst ratings
2. Social sentiment - Check retail sentiment and buzz (X/Twitter posts, Reddit discussions, etc.)

Focus on recent, relevant information. Ignore outdated articles or old posts.

## Fibonacci Level Strategy
Fibonacci levels are critical for precision trading - institutional traders and algorithms watch these levels, creating self-fulfilling prophecies.

**ENTRIES - Look for entries at key retracement levels:**
- 38.2%: Shallow retracement in strong trends (highest probability but requires confirmation)
- 50%: The psychological midpoint, most commonly used retracement
- 61.8%: The "golden ratio", optimal risk/reward for swing trades
- 78.6%: Deep retracement, higher risk but often marks major reversals

**STOP LOSSES - Place stops beyond the next Fibonacci level with 5-10% buffer:**
- If entering at 61.8%, stop goes below 78.6% or the swing low (whichever is further)
- If entering at 50%, stop goes below 61.8%
- If entering at 38.2%, stop goes below 50%
- Never place stops exactly at Fibonacci levels - they get hunted. Use buffers.

**TARGETS - Use Fibonacci extension levels for profit targets:**
- Target 1: 1.272 extension (conservative, high probability)
- Target 2: 1.618 extension (moderate, the golden extension)
- Target 3: 2.618 extension (aggressive, trend extension plays)

**VALIDATION - Always validate Fib levels against structural support/resistance:**
- Fibonacci + volume node = highest probability zone
- Fibonacci + prior swing high/low = strong confluence
- Fibonacci + EMA convergence = institutional interest zone
- Isolated Fibonacci levels without confluence = lower reliability

## When Creating a Plan - Level Placement by Bias

### For BULLISH (Long) Plans:
- Entry should be at support, Fibonacci retracement, or on a pullback - not chasing
- Stop loss MUST be BELOW entry (below next Fib level, swing low, or support)
- Targets should be ABOVE entry (at Fib extensions, resistance levels, or measured moves)

### For BEARISH (Short) Plans:
- Entry should be at resistance, Fibonacci retracement, or on a bounce into overhead supply
- Stop loss MUST be ABOVE entry (above next Fib level, swing high, or resistance)
- Targets should be BELOW entry (at Fib extensions or support levels where you cover)

### For Both:
- Risk/reward should be at least 2:1 for swing trades
- Consider the broader market direction
- Factor in any news catalysts or social sentiment into your thesis
- When Fibonacci levels are available, USE THEM for precise level placement

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

## CRITICAL: Position-Aware Planning

When the user has an EXISTING POSITION, your plan MUST align with it:

### If User is LONG (owns shares):
- NEVER suggest opening a SHORT position or bearish trade
- If technicals turn bearish, recommend:
  - Tightening stop loss to protect gains
  - Scaling out at current levels to lock in profits
  - Moving stop to breakeven if in profit
  - Waiting for clearer direction
- If technicals are bullish, recommend:
  - Holding the position
  - Adding at key support levels (if appropriate)
  - Setting/adjusting profit targets

### If User is SHORT (has sold short):
- NEVER suggest going LONG while they're short
- If technicals turn bullish, recommend covering the short
- If technicals remain bearish, recommend holding with adjusted targets

### If User has NO POSITION:
- Generate a directional plan based purely on technicals
- Can suggest either LONG or SHORT based on the setup

The thesis should ALWAYS be framed around optimizing the user's ACTUAL situation.
"""

PLAN_GENERATION_PROMPT = """Based on the comprehensive data below, create a detailed trading plan for {symbol}.

## Current Market Data
{market_data}

## Technical Analysis
{technical_data}

## Key Levels
{levels_data}

## Fibonacci Levels
{fibonacci_data}

## Volume Analysis
{volume_data}

## Chart Patterns
{patterns_data}

## Existing Position (if any)
{position_data}

---

## IMPORTANT: Position-Aligned Planning Rules

IF THE USER HAS AN EXISTING POSITION:
1. Your bias and recommendations MUST optimize their existing position
2. If they are LONG and technicals are bearish:
   - Set bias to "bullish" or "neutral" (NEVER "bearish")
   - Recommend exit strategies, stop adjustments, or profit-taking - NOT short entries
   - Entry zones become "scale-out zones" or "do not add here"
   - Focus on protecting their gains or minimizing losses
3. If they are LONG and technicals are bullish:
   - Set bias to "bullish"
   - Recommend holding or adding to position
   - Set target levels for scaling out profits
4. The thesis should explain how to OPTIMIZE their existing position, not suggest a new directional trade

IF THE USER HAS NO POSITION:
- Generate a fresh directional plan based purely on technical analysis
- Can suggest LONG or SHORT based on the setup

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

IMPORTANT: If search is available, search for news and social sentiment FIRST before responding.

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
    "fib_entry_level": "38.2%" | "50%" | "61.8%" | "78.6%" | null (if entry aligns with a Fib retracement),
    "stop_loss": <price or null>,
    "stop_reasoning": "Why this stop level (mention Fibonacci level if applicable)",
    "fib_stop_level": "50%" | "61.8%" | "78.6%" | "100%" | null (Fib level below/above stop),
    "target_1": <price - conservative target>,
    "target_2": <price - moderate target>,
    "target_3": <price - aggressive target or null>,
    "target_reasoning": "Why these targets (mention Fibonacci extensions if applicable)",
    "fib_target_levels": ["1.272", "1.618", "2.618"] or [] (Fib extensions used for targets),
    "risk_reward": <ratio like 2.5>,
    "position_size_pct": <1-5, percentage of account>,
    "key_supports": [<price>, <price>],
    "key_resistances": [<price>, <price>],
    "invalidation_criteria": "What would invalidate this plan",
    "technical_summary": "Brief summary of key technical factors",
    "news_summary": "Brief summary of recent news/catalysts (from search). Empty string if no news found or search unavailable.",
    "social_sentiment": "bullish" | "bearish" | "neutral" | "mixed" | "none",
    "social_buzz": "Summary of social discussion (X/Twitter, Reddit, etc.) if found. Empty string if none."
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
5. Should the thesis be updated to reflect new developments?
6. **If user has a position**: How is their trade performing? Should they adjust stops based on their entry? Are targets still realistic? NEVER suggest flipping direction if user has an active position.

**IMPORTANT**: You must respond with a JSON object in this exact format:

```json
{{
    "status": "VALID" | "ADJUST" | "INVALIDATED",
    "evaluation": "2-3 sentence summary of current status",
    "action": "What the trader should do now",
    "adjustments": {{
        "thesis": null or "Updated thesis reflecting current market conditions",
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
- **Thesis**: Update if market conditions have materially changed (new catalysts, trend change, key level break). Keep same directional bias if user has position. Make it reflect current reality, not just the original reasoning.
- Tighten stop loss if price moved favorably and new support formed
- Adjust targets if new resistance/support levels emerged
- Update key_supports/key_resistances if significant new levels have formed or old ones have broken
- Only adjust if there's a clear technical reason
- Keep risk/reward reasonable (ideally 2:1 or better)
- **If user has active position**: Respect their position direction - never suggest opposite direction trades
"""


class StockPlanningAgent:
    """Comprehensive stock analysis and planning agent."""

    def __init__(self, symbol: str, user_id: str = "default"):
        self.symbol = symbol.upper()
        self.user_id = user_id
        self._provider: Optional[AIProvider] = None

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
        self._fibonacci_data: Dict[str, Any] = {}
        self._current_plan: Optional[TradingPlan] = None

    async def _get_provider(self) -> AIProvider:
        """Get or create AI provider for the user."""
        if self._provider is None:
            self._provider = await get_user_provider(self.user_id)
        return self._provider

    def _get_search_parameters(self) -> Optional[SearchParameters]:
        """Get search parameters based on the provider's capabilities.

        Uses X search for Grok (real-time sentiment), web search for Claude.
        """
        if self._provider is None:
            return None

        settings = get_settings()
        if not settings.web_search_enabled:
            return None

        # Use X search if available (Grok), otherwise web search only
        if self._provider.supports_x_search:
            return get_x_search_parameters()
        else:
            return SearchParameters(mode="on", sources=[{"type": "web"}], return_citations=True)

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

        # Get Fibonacci levels for precision trade planning
        try:
            self._fibonacci_data = await get_fibonacci_levels(self.symbol)
        except Exception as e:
            logger.error(f"Error getting Fibonacci levels: {e}")
            self._fibonacci_data = {"error": str(e)}

        # Load existing plan
        self._current_plan = await self._plan_store.get_plan(self.user_id, self.symbol)

        return {
            "market": self._market_data,
            "technical": self._technical_data,
            "levels": self._levels_data,
            "volume": self._volume_data,
            "patterns": self._patterns_data,
            "position": self._position_data,
            "fibonacci": self._fibonacci_data,
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

        # Fibonacci levels
        fib = self._fibonacci_data
        if fib.get("error"):
            fibonacci_str = f"Fibonacci levels unavailable: {fib.get('error')}"
        else:
            retracements = fib.get("retracement", {})
            extensions = fib.get("extension", {})
            current_price = price_data.get('price', 0)

            # Describe current price position vs Fibonacci
            nearest_level = fib.get("nearest_level", "")
            nearest_price = fib.get("nearest_price", 0)
            at_entry = fib.get("at_entry_level", False)

            position_desc = ""
            if at_entry and nearest_level in ["0.382", "0.500", "0.618", "0.786"]:
                position_desc = f"Currently AT {nearest_level} retracement (${nearest_price:.2f}) - POTENTIAL ENTRY ZONE"
            elif nearest_level:
                position_desc = f"Near {nearest_level} level (${nearest_price:.2f})"

            fibonacci_str = f"""Fibonacci Analysis:
Swing Range: ${fib.get('swing_low', 'N/A'):.2f} - ${fib.get('swing_high', 'N/A'):.2f} (trend: {fib.get('trend', 'unknown')})
Current Price: ${current_price:.2f} - {position_desc}

Key Retracements (potential entry zones):
  38.2%: ${retracements.get('0.382', 0):.2f}
  50.0%: ${retracements.get('0.500', 0):.2f}
  61.8%: ${retracements.get('0.618', 0):.2f} (golden ratio)
  78.6%: ${retracements.get('0.786', 0):.2f}

Extension Targets (potential profit targets):
  1.272: ${extensions.get('1.272', 0):.2f} (conservative)
  1.618: ${extensions.get('1.618', 0):.2f} (golden extension)
  2.618: ${extensions.get('2.618', 0):.2f} (aggressive)
"""

        # Position
        pos = self._position_data
        if pos.get("has_position"):
            targets = pos.get("targets", {})
            current_size = pos.get('current_size', 0)
            entry_price = pos.get('entry_price', 0)
            current_price = pos.get('current_price', 0)
            unrealized_pnl = pos.get('unrealized_pnl', 0)
            unrealized_pnl_pct = pos.get('unrealized_pnl_pct', 0)

            # Determine position direction (positive size = LONG, negative = SHORT)
            position_direction = "LONG" if current_size > 0 else "SHORT"
            pnl_status = "IN PROFIT" if unrealized_pnl >= 0 else "AT A LOSS"

            position_str = f"""
*******************************************
*** USER HAS ACTIVE {position_direction} POSITION ***
*******************************************

Position Direction: {position_direction}
Status: {pos.get('status', 'unknown')}
Entry Price: ${entry_price}
Current Size: {abs(current_size)} shares
Cost Basis: ${pos.get('cost_basis', 'N/A')}
Stop Loss: ${pos.get('stop_loss', 'N/A')}
Targets: ${targets.get('target_1', 'N/A')} / ${targets.get('target_2', 'N/A')} / ${targets.get('target_3', 'N/A')}
Current Price: ${current_price}
Unrealized P&L: ${unrealized_pnl} ({unrealized_pnl_pct}%) - {pnl_status}
R-Multiple: {pos.get('r_multiple', 'N/A')}

*******************************************
*** YOUR PLAN MUST HELP OPTIMIZE THIS {position_direction} POSITION ***
*** DO NOT SUGGEST AN OPPOSITE DIRECTION TRADE ***
*******************************************
"""
        else:
            position_str = "No current position - can suggest LONG or SHORT based on technicals"

        return {
            "market_data": market_str,
            "technical_data": tech_str,
            "levels_data": levels_str,
            "fibonacci_data": fibonacci_str,
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
            provider = await self._get_provider()
            settings = get_settings()

            # Get search parameters based on provider capabilities
            search_params = self._get_search_parameters()
            if search_params:
                logger.info(f"Search enabled for {self.symbol} plan generation (X search: {provider.supports_x_search})")

            response = await provider.create_message(
                messages=[AIMessage(role="user", content=prompt)],
                system=PLANNING_AGENT_SYSTEM,
                model_type="planning",
                max_tokens=4000,  # Increased for web search results
                search_parameters=search_params,
            )

            # Extract text from response
            response_text = response.content

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
                original_thesis=plan_data.get("thesis", ""),  # Preserve original thesis
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
                # External sentiment from web/social search
                news_summary=plan_data.get("news_summary", ""),
                social_sentiment=plan_data.get("social_sentiment", "none"),
                social_buzz=plan_data.get("social_buzz", ""),
                sentiment_source="x" if provider.supports_x_search else "reddit",
                # Validation warnings (if price levels don't match bias)
                validation_warnings=plan_data.get("_validation_warnings", []),
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
            provider = await self._get_provider()
            settings = get_settings()

            # Get search parameters based on provider capabilities
            search_params = self._get_search_parameters()
            if search_params:
                logger.info(f"Search enabled for {self.symbol} streaming plan (X search: {provider.supports_x_search})")

            # Use streaming API
            full_text = ""
            async for text in provider.create_message_stream(
                messages=[AIMessage(role="user", content=prompt)],
                system=PLANNING_AGENT_SYSTEM,
                model_type="planning",
                max_tokens=4000,
                search_parameters=search_params,
            ):
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
                original_thesis=plan_data.get("thesis", ""),  # Preserve original thesis
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
                social_sentiment=plan_data.get("social_sentiment", "none"),
                social_buzz=plan_data.get("social_buzz", ""),
                sentiment_source="x" if provider.supports_x_search else "reddit",
                # Validation warnings (if price levels don't match bias)
                validation_warnings=plan_data.get("_validation_warnings", []),
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
            "social_sentiment": plan.social_sentiment,
            "social_buzz": plan.social_buzz,
            "sentiment_source": plan.sentiment_source,
            "validation_warnings": plan.validation_warnings,
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
            fibonacci_data=data["fibonacci_data"],
            volume_data=data["volume_data"],
            patterns_data=data["patterns_data"],
            market_context=data["market_context"],
            position_data=data["position_data"],
        )

        provider = await self._get_provider()

        # Retry logic with exponential backoff for transient errors
        max_retries = 3
        base_delay = 2  # seconds

        for attempt in range(max_retries):
            try:
                response = await provider.create_message(
                    messages=[AIMessage(role="user", content=prompt)],
                    system=SMART_PLANNING_SYSTEM_PROMPT,
                    model_type="planning",
                    max_tokens=4000,  # Larger for educational content
                )

                response_text = response.content
                logger.debug(f"Smart plan response: {response_text[:500]}...")

                # Parse the JSON response
                plan_data = self._parse_smart_plan_response(response_text)

                # Perform visual chart analysis
                visual_analysis = await self._perform_visual_analysis()

                if visual_analysis:
                    # Apply visual confidence modifier to the plan
                    original_confidence = plan_data.get("confidence", 0)
                    visual_modifier = visual_analysis.get("visual_confidence_modifier", 0)

                    # Clamp modifier to reasonable range
                    visual_modifier = max(-20, min(20, visual_modifier))

                    # Apply modifier and clamp final confidence to 0-100
                    adjusted_confidence = max(0, min(100, original_confidence + visual_modifier))
                    plan_data["confidence"] = adjusted_confidence

                    # Store visual analysis data in the plan
                    plan_data["visual_analysis"] = {
                        "confidence_modifier": visual_modifier,
                        "original_confidence": original_confidence,
                        "trend_quality": visual_analysis.get("trend_quality", {}),
                        "patterns_identified": visual_analysis.get("visual_patterns_identified", []),
                        "warning_signs": visual_analysis.get("warning_signs", []),
                        "visual_summary": visual_analysis.get("visual_summary", ""),
                    }

                    logger.info(
                        f"Visual analysis applied for {self.symbol}: "
                        f"confidence {original_confidence} -> {adjusted_confidence} "
                        f"(modifier: {visual_modifier:+d})"
                    )

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
                    raise ValueError(f"AI API is currently overloaded. Please try again in a few moments.")

            except (APIStatusError, httpx.HTTPStatusError) as e:
                # Handle rate limits from Grok (httpx) similar to Claude
                status_code = getattr(e, 'status_code', None) or getattr(getattr(e, 'response', None), 'status_code', None)
                if status_code == 429 and attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"API rate limited for {self.symbol}, retrying in {delay}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                else:
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

        # Fibonacci levels (enhanced formatting for smart prompts)
        fib = self._fibonacci_data
        if fib.get("error"):
            fibonacci_str = f"Fibonacci levels unavailable: {fib.get('error')}"
        else:
            retracements = fib.get("retracement", {})
            extensions = fib.get("extension", {})

            # Describe current price position vs Fibonacci
            nearest_level = fib.get("nearest_level", "")
            nearest_price = fib.get("nearest_price", 0)
            at_entry = fib.get("at_entry_level", False)

            position_desc = ""
            if at_entry and nearest_level in ["0.382", "0.500", "0.618", "0.786"]:
                position_desc = f"**AT {nearest_level} RETRACEMENT - POTENTIAL ENTRY ZONE**"
            elif nearest_level:
                dist_pct = abs(current_price - nearest_price) / current_price * 100
                position_desc = f"Near {nearest_level} level ({dist_pct:.1f}% away)"

            fibonacci_str = f"""Fibonacci Analysis (Institutional Levels):
Swing Range: ${fib.get('swing_low', 0):.2f} - ${fib.get('swing_high', 0):.2f}
Trend Direction: {fib.get('trend', 'unknown').upper()}
Current Price: ${current_price:.2f} - {position_desc}

Retracement Levels (Entry Zones):
  - 38.2%: ${retracements.get('0.382', 0):.2f} (shallow retracement, strong trend)
  - 50.0%: ${retracements.get('0.500', 0):.2f} (psychological midpoint, most common)
  - 61.8%: ${retracements.get('0.618', 0):.2f} (golden ratio, optimal R/R)
  - 78.6%: ${retracements.get('0.786', 0):.2f} (deep retracement, reversal zone)

Extension Levels (Profit Targets):
  - 1.272: ${extensions.get('1.272', 0):.2f} (conservative target)
  - 1.618: ${extensions.get('1.618', 0):.2f} (golden extension, moderate target)
  - 2.618: ${extensions.get('2.618', 0):.2f} (aggressive target, trend extension)

**Use these levels for precise entry, stop, and target placement**
"""

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
            "fibonacci_data": fibonacci_str,
            "volume_data": volume_str,
            "patterns_data": patterns_str,
            "market_context": market_context_str,
            "position_data": position_str,
        }

    async def _perform_visual_analysis(self) -> Optional[Dict[str, Any]]:
        """Generate chart and perform visual analysis using Claude Vision.

        Returns the visual analysis results including confidence modifier,
        or None if visual analysis fails.
        """
        try:
            from app.tools.market_data import fetch_price_bars
            from app.tools.indicators import calculate_ema_series, calculate_rsi_series

            settings = get_settings()

            # Get price bars for charting (need enough for indicators + lookback)
            bars = fetch_price_bars(self.symbol, timeframe="1d", days_back=120)

            if not bars or len(bars) < 60:
                logger.warning(f"Insufficient bars for visual analysis: {len(bars) if bars else 0}")
                return None

            # Calculate indicator arrays using series functions (for chart overlay)
            ema_9 = calculate_ema_series(bars, 9)
            ema_21 = calculate_ema_series(bars, 21)
            ema_50 = calculate_ema_series(bars, 50)
            rsi_values = calculate_rsi_series(bars, 14)

            indicators = {
                "ema_9": ema_9,
                "ema_21": ema_21,
                "ema_50": ema_50,
                "rsi": rsi_values,
            }

            # Generate chart image
            logger.info(f"Generating chart for visual analysis: {self.symbol}")
            chart_image = generate_chart_image(
                symbol=self.symbol,
                bars=bars,
                indicators=indicators,
                lookback=60,
                show_volume=True,
                show_rsi=True,
            )

            # Send to AI provider for vision analysis
            logger.info(f"Sending chart to AI vision for {self.symbol}")
            provider = await self._get_provider()

            vision_response = await provider.analyze_image(
                image_base64=chart_image,
                prompt=VISUAL_ANALYSIS_PROMPT,
                model_type="planning",
            )

            response_text = vision_response.content

            # Parse the visual analysis JSON
            visual_analysis = self._parse_visual_analysis_response(response_text)

            logger.info(
                f"Visual analysis complete for {self.symbol}: "
                f"modifier={visual_analysis.get('visual_confidence_modifier', 0)}, "
                f"trend={visual_analysis.get('trend_quality', {}).get('assessment', 'unknown')}"
            )

            return visual_analysis

        except ImportError as e:
            logger.warning(f"Visual analysis dependencies not available: {e}")
            return None
        except Exception as e:
            logger.error(f"Visual analysis failed for {self.symbol}: {e}")
            return None

    def _parse_visual_analysis_response(self, response_text: str) -> Dict[str, Any]:
        """Parse Claude Vision's visual analysis response."""
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
                logger.warning("No JSON found in visual analysis response")
                return {"visual_confidence_modifier": 0, "visual_summary": response_text[:500]}

            return json.loads(json_str)

        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse visual analysis JSON: {e}")
            return {"visual_confidence_modifier": 0, "visual_summary": "Analysis parsing failed"}

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

            plan_data = json.loads(json_str)
            # Sanitize price fields to handle $ signs and ensure numeric types
            plan_data = _sanitize_plan_data(plan_data)

            # Validate price level consistency with bias
            # Smart plan format uses nested targets array
            bias = plan_data.get("bias", "neutral")
            targets_data = plan_data.get("targets", [])
            targets = [t.get("price") if isinstance(t, dict) else t for t in targets_data]
            warnings = validate_plan_price_consistency(
                bias=bias,
                entry_zone_low=plan_data.get("entry_zone_low"),
                entry_zone_high=plan_data.get("entry_zone_high"),
                stop_loss=plan_data.get("stop_loss"),
                targets=targets,
            )

            for warning in warnings:
                logger.warning(f"Smart plan validation for {self.symbol}: {warning}")

            # Flag the plan with warnings so iOS can display badge
            if warnings:
                plan_data["_validation_warnings"] = warnings

            return plan_data
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

            plan_data = json.loads(json_str)
            # Sanitize price fields to handle $ signs and ensure numeric types
            plan_data = _sanitize_plan_data(plan_data)

            # Validate price level consistency with bias
            bias = plan_data.get("bias", "neutral")
            targets = [plan_data.get("target_1"), plan_data.get("target_2"), plan_data.get("target_3")]
            warnings = validate_plan_price_consistency(
                bias=bias,
                entry_zone_low=plan_data.get("entry_zone_low"),
                entry_zone_high=plan_data.get("entry_zone_high"),
                stop_loss=plan_data.get("stop_loss"),
                targets=targets,
            )

            for warning in warnings:
                logger.warning(f"Plan validation for {self.symbol}: {warning}")

            # Flag the plan with warnings so iOS can display badge
            if warnings:
                plan_data["_validation_warnings"] = warnings

            return plan_data
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
            provider = await self._get_provider()

            # Use the planning model for evaluation since it makes important decisions
            response = await provider.create_message(
                messages=[AIMessage(role="user", content=prompt)],
                system=PLANNING_AGENT_SYSTEM,
                model_type="planning",
                max_tokens=1500,
            )

            response_text = response.content

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

            eval_data = json.loads(json_str)
            # Sanitize adjustments if present
            if eval_data.get("adjustments"):
                eval_data["adjustments"] = _sanitize_plan_data(eval_data["adjustments"])
            return eval_data
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
        history_messages = conversation.to_claude_messages(max_messages=10)
        ai_messages = [AIMessage(role=msg["role"], content=msg["content"]) for msg in history_messages]
        ai_messages.append(AIMessage(role="user", content=user_message))

        try:
            provider = await self._get_provider()

            response = await provider.create_message(
                messages=ai_messages,
                system=system_prompt,
                model_type="fast",  # Use fast model for chat
                max_tokens=1500,
            )

            response_text = response.content

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
            provider = await self._get_provider()
            response = await provider.create_message(
                messages=[AIMessage(role="user", content=prompt)],
                system="You are an expert trading advisor explaining a trading plan. Be thorough but conversational.",
                model_type="fast",
                max_tokens=1000,
            )

            ai_response = response.content

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
            provider = await self._get_provider()
            response = await provider.create_message(
                messages=[AIMessage(role="user", content=prompt)],
                system="You are an expert trading advisor. Your job is to protect the user from making bad trades while respecting their preferences. Never blindly agree - always explain tradeoffs.",
                model_type="fast",
                max_tokens=1500,
            )

            ai_response = response.content

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
                        # Sanitize price fields in adjustments
                        adjustments = _sanitize_plan_data(adjustments)

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
