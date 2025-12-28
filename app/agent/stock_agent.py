"""Stock-specific AI agent for monitoring and alert generation.

Each StockAgent instance monitors a single stock and generates
alerts based on price action and the user's trade plan.
"""

import logging
from typing import Dict, Any, Optional, Literal
from datetime import datetime

import anthropic

from app.config import get_settings
from app.agent.prompts import get_stock_agent_prompt, STOCK_AGENT_PROMPT_TEMPLATE
from app.agent.tools import (
    get_current_price,
    get_key_levels,
    get_technical_indicators,
    get_position_status,
    get_market_context,
)
from app.services.price_monitor import TriggerEvent
from app.storage.position_store import Position

logger = logging.getLogger(__name__)

AlertType = Literal["BUY", "STOP", "SELL", "NONE"]


class AlertResult:
    """Result of agent analysis."""

    def __init__(
        self,
        should_alert: bool,
        alert_type: AlertType = "NONE",
        message: str = "",
        context: Dict[str, Any] = None,
    ):
        self.should_alert = should_alert
        self.alert_type = alert_type
        self.message = message
        self.context = context or {}
        self.timestamp = datetime.utcnow().isoformat()


class StockAgent:
    """AI agent for monitoring a single stock.

    Uses Claude API to analyze price events and generate trading alerts.
    Maintains context about the stock's trade plan and position status.
    """

    def __init__(
        self,
        symbol: str,
        user_id: str = "default",
        trade_plan: Optional[Dict[str, Any]] = None,
    ):
        """Initialize stock agent.

        Args:
            symbol: Stock ticker symbol
            user_id: User identifier
            trade_plan: Trade plan with entry, stop, targets
        """
        self.symbol = symbol.upper()
        self.user_id = user_id
        self.trade_plan = trade_plan
        self.position: Optional[Position] = None
        self.key_levels: Optional[Dict[str, Any]] = None

        self._client: Optional[anthropic.Anthropic] = None
        self._analysis_count = 0
        self._last_alert_type: Optional[str] = None

    def _get_client(self) -> anthropic.Anthropic:
        """Get or create Anthropic client."""
        if self._client is None:
            settings = get_settings()
            if not settings.claude_api_key:
                raise ValueError("CLAUDE_API_KEY not configured")
            self._client = anthropic.Anthropic(api_key=settings.claude_api_key)
        return self._client

    async def update_context(
        self,
        trade_plan: Optional[Dict[str, Any]] = None,
        position: Optional[Position] = None,
    ) -> None:
        """Update agent context with new data.

        Args:
            trade_plan: New trade plan
            position: Current position status
        """
        if trade_plan:
            self.trade_plan = trade_plan

        if position is not None:
            self.position = position

        # Refresh key levels
        try:
            self.key_levels = await get_key_levels(self.symbol)
        except Exception as e:
            logger.warning(f"Failed to refresh levels for {self.symbol}: {e}")

    async def analyze_trigger(self, event: TriggerEvent) -> AlertResult:
        """Analyze a trigger event and determine if alert is needed.

        Args:
            event: Trigger event from price monitor

        Returns:
            AlertResult with decision and message
        """
        self._analysis_count += 1

        # Get current context
        try:
            position_data = await get_position_status(self.symbol, self.user_id)
            indicators = await get_technical_indicators(self.symbol)
            market = await get_market_context()
        except Exception as e:
            logger.error(f"Error gathering context for {self.symbol}: {e}")
            return AlertResult(should_alert=False, message=f"Context error: {e}")

        # Build the analysis prompt
        context_str = self._build_context_string(position_data, indicators, market)

        prompt = STOCK_AGENT_PROMPT_TEMPLATE.format(
            symbol=self.symbol,
            context=context_str,
            event_type=event.event_type,
            current_price=event.current_price,
            trigger_price=event.trigger_price,
            distance_pct=f"{event.distance_pct:.2f}",
        )

        # Call Claude API
        try:
            client = self._get_client()
            settings = get_settings()

            response = client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=500,
                system=get_stock_agent_prompt(
                    self.symbol,
                    self.trade_plan,
                    position_data,
                    self.key_levels,
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text

            # Parse the response
            return self._parse_response(response_text, event)

        except Exception as e:
            logger.error(f"Claude API error for {self.symbol}: {e}")
            return AlertResult(should_alert=False, message=f"API error: {e}")

    def _build_context_string(
        self,
        position: Dict[str, Any],
        indicators: Dict[str, Any],
        market: Dict[str, Any],
    ) -> str:
        """Build context string for the prompt.

        Args:
            position: Position status
            indicators: Technical indicators
            market: Market context

        Returns:
            Formatted context string
        """
        lines = []

        # Position status
        if position.get("has_position"):
            lines.append(f"Position: {position['status']} - {position.get('current_size', 0)} shares @ ${position.get('entry_price', 'N/A')}")
            lines.append(f"Stop: ${position.get('stop_loss', 'N/A')}")
        else:
            lines.append("Position: Not in position (watching)")

        # Trade plan
        if self.trade_plan:
            lines.append(f"Trade Plan - Entry: ${self.trade_plan.get('entry', 'N/A')}, Stop: ${self.trade_plan.get('stop_loss', 'N/A')}")

        # Indicators
        if not indicators.get("error"):
            rsi = indicators.get("rsi", {})
            lines.append(f"RSI: {rsi.get('value', 'N/A'):.1f} ({rsi.get('signal', 'N/A')})")

            emas = indicators.get("emas", {})
            above_emas = sum([
                emas.get("above_9", False),
                emas.get("above_21", False),
                emas.get("above_50", False),
            ])
            lines.append(f"EMAs: Price above {above_emas}/3 EMAs")

            vol = indicators.get("volume", {})
            lines.append(f"Volume: {vol.get('relative', 'N/A')}x average ({vol.get('trend', 'N/A')})")

        # Market context
        if not market.get("error"):
            lines.append(f"Market: {market.get('market_direction', 'unknown').upper()} ({market.get('bullish_indices', 0)}/4 indices up)")

        return "\n".join(lines)

    def _parse_response(self, response_text: str, event: TriggerEvent) -> AlertResult:
        """Parse Claude's response into an AlertResult.

        Args:
            response_text: Raw response from Claude
            event: Original trigger event

        Returns:
            AlertResult with parsed data
        """
        response_upper = response_text.upper()

        # Check for NO_ALERT
        if "NO_ALERT" in response_upper:
            return AlertResult(
                should_alert=False,
                alert_type="NONE",
                message=response_text,
            )

        # Determine alert type
        alert_type: AlertType = "NONE"
        if "BUY SIGNAL" in response_upper or "BUY:" in response_upper:
            alert_type = "BUY"
        elif "STOP ALERT" in response_upper or "STOP:" in response_upper:
            alert_type = "STOP"
        elif "TARGET" in response_upper and "HIT" in response_upper:
            alert_type = "SELL"
        elif "SELL" in response_upper:
            alert_type = "SELL"

        # If we detected an alert type, return it
        if alert_type != "NONE":
            self._last_alert_type = alert_type
            return AlertResult(
                should_alert=True,
                alert_type=alert_type,
                message=response_text.strip(),
                context={
                    "trigger_event": event.event_type,
                    "trigger_price": event.trigger_price,
                    "current_price": event.current_price,
                },
            )

        # Fallback - no clear alert detected
        return AlertResult(
            should_alert=False,
            alert_type="NONE",
            message=response_text,
        )

    async def quick_check(self, current_price: float) -> Optional[AlertResult]:
        """Quick rule-based check without Claude API call.

        Use this for obvious conditions to save API costs.

        Args:
            current_price: Current stock price

        Returns:
            AlertResult if obvious condition met, None otherwise
        """
        # Check stop loss proximity (no API needed)
        if self.position and self.position.status in ["entered", "partial"]:
            stop = self.position.stop_loss
            if stop and current_price > 0:
                distance_pct = (current_price - stop) / current_price

                # Very close to stop (0.5%)
                if 0 < distance_pct <= 0.005:
                    return AlertResult(
                        should_alert=True,
                        alert_type="STOP",
                        message=f"{self.symbol} CRITICAL STOP ALERT\nPrice: ${current_price:.2f} ({distance_pct*100:.1f}% above stop at ${stop:.2f})\nSTOP LOSS IMMINENT - Review position immediately",
                        context={"stop_loss": stop, "distance_pct": distance_pct},
                    )

        # Check target hits (no API needed)
        if self.position and self.position.status in ["entered", "partial"]:
            targets = [
                (1, self.position.target_1),
                (2, self.position.target_2),
                (3, self.position.target_3),
            ]

            for target_num, target_price in targets:
                if target_price and target_num not in self.position.targets_hit:
                    if current_price >= target_price:
                        return AlertResult(
                            should_alert=True,
                            alert_type="SELL",
                            message=f"{self.symbol} TARGET {target_num} HIT!\nPrice: ${current_price:.2f} (Target: ${target_price:.2f})\nConsider scaling out position",
                            context={"target_number": target_num, "target_price": target_price},
                        )

        return None

    async def chat(self, user_message: str) -> str:
        """Chat with the agent about this stock.

        Args:
            user_message: User's question about the stock

        Returns:
            Agent's response
        """
        # Get current context
        try:
            position_data = await get_position_status(self.symbol, self.user_id)
            indicators = await get_technical_indicators(self.symbol)
            price_data = await get_current_price(self.symbol)
            market = await get_market_context()
        except Exception as e:
            logger.error(f"Error gathering context for chat: {e}")
            return f"Sorry, I couldn't fetch the latest data for {self.symbol}. Error: {e}"

        # Build context for the chat
        context_lines = [
            f"Stock: {self.symbol}",
            f"Current Price: ${price_data.get('price', 'N/A')}",
        ]

        if position_data.get("has_position"):
            context_lines.append(f"Position: {position_data['status']} - {position_data.get('current_size', 0)} shares @ ${position_data.get('entry_price', 'N/A')}")
            context_lines.append(f"Stop Loss: ${position_data.get('stop_loss', 'N/A')}")
        else:
            context_lines.append("Position: Not in position")

        if self.trade_plan:
            context_lines.append(f"Trade Plan: Entry ${self.trade_plan.get('entry', 'N/A')}, Stop ${self.trade_plan.get('stop_loss', 'N/A')}, Targets ${self.trade_plan.get('target_1', 'N/A')}/${self.trade_plan.get('target_2', 'N/A')}/{self.trade_plan.get('target_3', 'N/A')}")

        if not indicators.get("error"):
            rsi = indicators.get("rsi", {})
            context_lines.append(f"RSI: {rsi.get('value', 'N/A'):.1f} ({rsi.get('signal', 'neutral')})")
            vol = indicators.get("volume", {})
            context_lines.append(f"Volume: {vol.get('relative', 'N/A')}x average")

        context_lines.append(f"Market: {market.get('market_direction', 'unknown').upper()}")

        context_str = "\n".join(context_lines)

        system_prompt = f"""You are a helpful trading assistant for {self.symbol}. Answer the user's questions about this stock based on the current data.

Current Data:
{context_str}

Guidelines:
- Be concise and actionable
- Reference specific numbers from the data
- If asked about buy/sell decisions, remind them this is not financial advice
- Focus on technical analysis and the trade plan if one exists
"""

        try:
            client = self._get_client()
            settings = get_settings()

            response = client.messages.create(
                model=settings.claude_model_fast,
                max_tokens=1000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Chat error for {self.symbol}: {e}")
            return f"Sorry, I encountered an error: {e}"

    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics.

        Returns:
            Dictionary with agent stats
        """
        return {
            "symbol": self.symbol,
            "user_id": self.user_id,
            "analysis_count": self._analysis_count,
            "last_alert_type": self._last_alert_type,
            "has_trade_plan": self.trade_plan is not None,
            "has_position": self.position is not None,
            "position_status": self.position.status if self.position else None,
        }
