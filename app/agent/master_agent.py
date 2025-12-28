"""Master agent that orchestrates stock monitoring and alert generation.

The MasterAgent:
- Manages multiple StockAgent instances (one per watchlist stock)
- Coordinates with PriceMonitor for trigger events
- Handles alert generation and delivery
- Tracks costs and performance
"""

import logging
import asyncio
from typing import Dict, List, Optional, Callable
from datetime import datetime

from app.agent.stock_agent import StockAgent, AlertResult
from app.agent.prompts import MASTER_AGENT_PROMPT
from app.agent.tools import get_market_context
from app.services.price_monitor import TriggerEvent, get_price_monitor
from app.services.scheduler import get_scheduler, is_market_open
from app.storage.position_store import get_position_store
from app.storage.alert_history import get_alert_history
from app.storage.watchlist_store import get_watchlist_store
from app.config import get_settings

logger = logging.getLogger(__name__)


class MasterAgent:
    """Master agent that coordinates all stock monitoring.

    Manages the lifecycle of stock-specific agents and handles
    the flow from trigger events to alert delivery.
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[str, str, str, str, float], None]] = None,
    ):
        """Initialize master agent.

        Args:
            alert_callback: Async function called with (user_id, symbol, alert_type, message, price)
        """
        self.alert_callback = alert_callback
        self._stock_agents: Dict[str, StockAgent] = {}  # symbol -> agent
        self._running = False
        self._total_analyses = 0
        self._total_alerts = 0
        self._start_time: Optional[str] = None

        # Get services
        self._price_monitor = get_price_monitor()
        self._scheduler = get_scheduler()
        self._position_store = get_position_store()
        self._alert_history = get_alert_history()
        self._watchlist_store = get_watchlist_store()

    async def _handle_trigger_event(self, event: TriggerEvent) -> None:
        """Handle a trigger event from the price monitor.

        Args:
            event: Trigger event with symbol and price data
        """
        symbol = event.symbol
        logger.info(f"Trigger event for {symbol}: {event.event_type} @ ${event.current_price:.2f}")

        # Get or create stock agent
        agent = self._stock_agents.get(symbol)
        if not agent:
            agent = await self._create_stock_agent(symbol, event.user_id)
            if not agent:
                return

        # Quick rule-based check first (saves API costs)
        quick_result = await agent.quick_check(event.current_price)
        if quick_result and quick_result.should_alert:
            await self._handle_alert(event.user_id, symbol, quick_result, event.current_price)
            return

        # Full AI analysis for complex situations
        try:
            result = await agent.analyze_trigger(event)
            self._total_analyses += 1

            if result.should_alert:
                await self._handle_alert(event.user_id, symbol, result, event.current_price)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")

    async def _handle_alert(
        self,
        user_id: str,
        symbol: str,
        result: AlertResult,
        current_price: float,
    ) -> None:
        """Handle an alert result - record and deliver.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            result: Alert result from agent
            current_price: Current price when alert generated
        """
        # Check cooldown
        can_send = await self._alert_history.can_send_alert(
            user_id, symbol, result.alert_type
        )

        if not can_send:
            logger.info(f"Alert cooldown active for {symbol} {result.alert_type}")
            return

        # Record alert
        await self._alert_history.record_alert(
            user_id=user_id,
            symbol=symbol,
            alert_type=result.alert_type,
            message=result.message,
            price_at_alert=current_price,
        )

        self._total_alerts += 1
        logger.info(f"Alert generated for {symbol}: {result.alert_type}")

        # Deliver via callback
        if self.alert_callback:
            try:
                if asyncio.iscoroutinefunction(self.alert_callback):
                    await self.alert_callback(
                        user_id, symbol, result.alert_type, result.message, current_price
                    )
                else:
                    self.alert_callback(
                        user_id, symbol, result.alert_type, result.message, current_price
                    )
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    async def _create_stock_agent(
        self,
        symbol: str,
        user_id: str = "default",
    ) -> Optional[StockAgent]:
        """Create a stock agent for a symbol.

        Args:
            symbol: Stock ticker symbol
            user_id: User identifier

        Returns:
            StockAgent instance or None if creation failed
        """
        try:
            # Get position and trade plan
            position = await self._position_store.get_position(user_id, symbol)

            trade_plan = None
            if position:
                trade_plan = {
                    "entry": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "target_1": position.target_1,
                    "target_2": position.target_2,
                    "target_3": position.target_3,
                    "trade_type": position.trade_type,
                }

            agent = StockAgent(symbol, user_id, trade_plan)
            await agent.update_context(trade_plan, position)

            self._stock_agents[symbol] = agent
            logger.info(f"Created stock agent for {symbol}")

            return agent

        except Exception as e:
            logger.error(f"Failed to create agent for {symbol}: {e}")
            return None

    async def _load_watchlist_agents(self) -> None:
        """Load agents for all watchlist stocks with alerts enabled."""
        try:
            items = self._watchlist_store.get_watchlist("default")

            for item in items:
                if item.get("alerts_enabled", False):
                    symbol = item["symbol"]
                    if symbol not in self._stock_agents:
                        await self._create_stock_agent(symbol, "default")

            logger.info(f"Loaded {len(self._stock_agents)} stock agents")

        except Exception as e:
            logger.error(f"Error loading watchlist agents: {e}")

    async def _on_market_open(self) -> None:
        """Called when market opens."""
        logger.info("Market opened - starting agent monitoring")

        # Load watchlist agents
        await self._load_watchlist_agents()

        # Get market context
        try:
            market = await get_market_context()
            logger.info(f"Market context: {market.get('market_direction', 'unknown')}")
        except Exception as e:
            logger.warning(f"Failed to get market context: {e}")

    async def _on_market_close(self) -> None:
        """Called when market closes."""
        logger.info("Market closed - stopping agent monitoring")

        # Log stats
        stats = self.get_stats()
        logger.info(f"Day stats: {stats['total_analyses']} analyses, {stats['total_alerts']} alerts")

    async def start(self) -> None:
        """Start the master agent and all monitoring."""
        if self._running:
            logger.warning("Master agent already running")
            return

        self._running = True
        self._start_time = datetime.utcnow().isoformat()

        settings = get_settings()
        interval = settings.agent_analysis_interval

        # Set up price monitor callback
        self._price_monitor.trigger_callback = self._handle_trigger_event

        # Register market hours callbacks
        self._scheduler.on_market_open(self._on_market_open)
        self._scheduler.on_market_close(self._on_market_close)

        # Start price monitor as interval task
        self._scheduler.add_interval_task(
            self._price_monitor._poll_prices,
            interval,
        )

        # Start scheduler
        await self._scheduler.start()

        # If market is already open, run open callback immediately
        if is_market_open():
            await self._on_market_open()
            await self._price_monitor.start(interval)

        logger.info("Master agent started")

    async def stop(self) -> None:
        """Stop the master agent and all monitoring."""
        if not self._running:
            return

        self._running = False

        # Stop services
        await self._price_monitor.stop()
        await self._scheduler.stop()

        # Clear agents
        self._stock_agents.clear()

        logger.info("Master agent stopped")

    async def add_symbol(self, symbol: str, user_id: str = "default") -> bool:
        """Add a symbol to monitoring.

        Args:
            symbol: Stock ticker symbol
            user_id: User identifier

        Returns:
            True if added successfully
        """
        symbol = symbol.upper()

        if symbol in self._stock_agents:
            return True

        agent = await self._create_stock_agent(symbol, user_id)
        if agent:
            # Also add to price monitor
            position = await self._position_store.get_position(user_id, symbol)
            await self._price_monitor.add_symbol(symbol, user_id, position)
            return True

        return False

    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from monitoring.

        Args:
            symbol: Stock ticker symbol
        """
        symbol = symbol.upper()

        if symbol in self._stock_agents:
            del self._stock_agents[symbol]

        await self._price_monitor.remove_symbol(symbol)

    async def update_position(self, symbol: str, user_id: str = "default") -> None:
        """Update position data for a symbol's agent.

        Args:
            symbol: Stock ticker symbol
            user_id: User identifier
        """
        symbol = symbol.upper()

        position = await self._position_store.get_position(user_id, symbol)

        # Update stock agent
        agent = self._stock_agents.get(symbol)
        if agent:
            trade_plan = None
            if position:
                trade_plan = {
                    "entry": position.entry_price,
                    "stop_loss": position.stop_loss,
                    "target_1": position.target_1,
                    "target_2": position.target_2,
                    "target_3": position.target_3,
                    "trade_type": position.trade_type,
                }
            await agent.update_context(trade_plan, position)

        # Update price monitor
        await self._price_monitor.update_position(symbol, position)

    def get_stats(self) -> Dict:
        """Get master agent statistics.

        Returns:
            Dictionary with agent statistics
        """
        return {
            "running": self._running,
            "start_time": self._start_time,
            "market_open": is_market_open(),
            "monitored_symbols": list(self._stock_agents.keys()),
            "agent_count": len(self._stock_agents),
            "total_analyses": self._total_analyses,
            "total_alerts": self._total_alerts,
            "price_monitor": self._price_monitor.get_status(),
            "scheduler": self._scheduler.get_status(),
            "agent_stats": {
                symbol: agent.get_stats()
                for symbol, agent in self._stock_agents.items()
            },
        }


# Singleton instance
_master_agent: Optional[MasterAgent] = None


def get_master_agent(
    alert_callback: Optional[Callable] = None,
) -> MasterAgent:
    """Get the singleton master agent instance.

    Args:
        alert_callback: Optional callback for alert delivery

    Returns:
        MasterAgent instance
    """
    global _master_agent
    if _master_agent is None:
        _master_agent = MasterAgent(alert_callback)
    elif alert_callback:
        _master_agent.alert_callback = alert_callback
    return _master_agent
