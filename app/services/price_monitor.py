"""Price monitoring service.

Monitors real-time price data for watchlist stocks and generates
trigger events when conditions warrant agent analysis.

Trigger events include:
- Price approaching support/resistance levels
- Breakouts and breakdowns
- Stop loss proximity
- Target price hits
- Volume spikes
- Large price moves
"""

import logging
import asyncio
from datetime import datetime
from typing import Callable, Dict, List, Optional, Literal, Set
from pydantic import BaseModel

from app.tools.streaming import get_streamer
from app.tools.market_data import fetch_snapshots
from app.tools.analysis import find_comprehensive_levels
from app.storage.position_store import get_position_store, Position
from app.storage.watchlist_store import get_watchlist_store

logger = logging.getLogger(__name__)


TriggerType = Literal[
    "approaching_support",
    "approaching_resistance",
    "breakout",
    "breakdown",
    "stop_proximity",
    "target_reached",
    "volume_spike",
    "large_move",
]


class TriggerEvent(BaseModel):
    """Event generated when a price condition warrants analysis."""

    event_type: TriggerType
    symbol: str
    current_price: float
    trigger_price: float
    distance_pct: float  # Percentage from trigger price
    context: dict  # Additional data specific to event type
    timestamp: str
    user_id: str = "default"


class StockContext(BaseModel):
    """Context for a monitored stock."""

    symbol: str
    user_id: str
    last_price: float
    last_volume: int
    support_levels: List[float]
    resistance_levels: List[float]
    position: Optional[Position] = None
    average_volume: int = 0
    last_check: str
    price_5min_ago: Optional[float] = None


class PriceMonitor:
    """Monitors prices and generates trigger events for agent analysis.

    Uses a combination of:
    1. Real-time WebSocket streaming for instant updates
    2. Periodic snapshot polling for reliability
    """

    def __init__(
        self,
        trigger_callback: Optional[Callable[[TriggerEvent], None]] = None,
        approach_threshold: float = 0.02,  # 2% from level
        stop_threshold: float = 0.01,  # 1% from stop
        volume_spike_multiplier: float = 2.0,  # 2x average
        large_move_threshold: float = 0.01,  # 1% in 5 minutes
    ):
        """Initialize price monitor.

        Args:
            trigger_callback: Async function to call with trigger events
            approach_threshold: % distance to trigger level approach event
            stop_threshold: % distance to trigger stop proximity event
            volume_spike_multiplier: Multiplier for volume spike detection
            large_move_threshold: % move in 5 minutes to trigger event
        """
        self.trigger_callback = trigger_callback
        self.approach_threshold = approach_threshold
        self.stop_threshold = stop_threshold
        self.volume_spike_multiplier = volume_spike_multiplier
        self.large_move_threshold = large_move_threshold

        self._running = False
        self._monitored: Dict[str, StockContext] = {}  # symbol -> context
        self._recent_triggers: Dict[str, Set[str]] = {}  # symbol -> set of recent trigger types
        self._streamer = get_streamer()
        self._poll_task: Optional[asyncio.Task] = None

    async def add_symbol(
        self,
        symbol: str,
        user_id: str = "default",
        position: Optional[Position] = None,
    ) -> None:
        """Add a symbol to monitor.

        Args:
            symbol: Stock ticker symbol
            user_id: User identifier
            position: Optional position data for stop/target monitoring
        """
        symbol = symbol.upper()

        if symbol in self._monitored:
            # Update position if provided
            if position:
                self._monitored[symbol].position = position
            return

        # Fetch initial data
        try:
            snapshots = fetch_snapshots([symbol])
            snapshot = snapshots.get(symbol, {})

            current_price = snapshot.get("latest_trade", {}).get("price", 0)
            current_volume = snapshot.get("daily_bar", {}).get("volume", 0)

            # Get support/resistance levels
            levels = find_comprehensive_levels(symbol)
            support_levels = [l["price"] for l in levels.get("support", [])]
            resistance_levels = [l["price"] for l in levels.get("resistance", [])]

            context = StockContext(
                symbol=symbol,
                user_id=user_id,
                last_price=current_price,
                last_volume=current_volume,
                support_levels=support_levels[:5],  # Top 5 nearest
                resistance_levels=resistance_levels[:5],
                position=position,
                average_volume=current_volume,  # Will be updated over time
                last_check=datetime.utcnow().isoformat(),
            )

            self._monitored[symbol] = context
            self._recent_triggers[symbol] = set()

            logger.info(f"Added {symbol} to price monitor")

        except Exception as e:
            logger.error(f"Error adding {symbol} to monitor: {e}")

    async def remove_symbol(self, symbol: str) -> None:
        """Remove a symbol from monitoring.

        Args:
            symbol: Stock ticker symbol
        """
        symbol = symbol.upper()

        if symbol in self._monitored:
            del self._monitored[symbol]
            self._recent_triggers.pop(symbol, None)
            logger.info(f"Removed {symbol} from price monitor")

    async def update_position(self, symbol: str, position: Optional[Position]) -> None:
        """Update position data for a symbol.

        Args:
            symbol: Stock ticker symbol
            position: New position data (or None to clear)
        """
        symbol = symbol.upper()

        if symbol in self._monitored:
            self._monitored[symbol].position = position

    def _check_trigger_conditions(
        self,
        symbol: str,
        current_price: float,
        current_volume: int,
    ) -> List[TriggerEvent]:
        """Check if current price/volume warrants trigger events.

        Args:
            symbol: Stock ticker symbol
            current_price: Current price
            current_volume: Current volume

        Returns:
            List of TriggerEvent objects
        """
        context = self._monitored.get(symbol)
        if not context:
            return []

        events = []
        now = datetime.utcnow().isoformat()

        # Skip if we recently triggered this type (prevent spam)
        def can_trigger(trigger_type: str) -> bool:
            return trigger_type not in self._recent_triggers.get(symbol, set())

        def mark_triggered(trigger_type: str) -> None:
            if symbol not in self._recent_triggers:
                self._recent_triggers[symbol] = set()
            self._recent_triggers[symbol].add(trigger_type)

        # Check support approach
        for support in context.support_levels:
            distance_pct = (current_price - support) / support if support > 0 else 0

            if 0 < distance_pct <= self.approach_threshold and can_trigger("approaching_support"):
                events.append(TriggerEvent(
                    event_type="approaching_support",
                    symbol=symbol,
                    current_price=current_price,
                    trigger_price=support,
                    distance_pct=distance_pct * 100,
                    context={"level_type": "support"},
                    timestamp=now,
                    user_id=context.user_id,
                ))
                mark_triggered("approaching_support")
                break

            # Breakdown below support
            if distance_pct < 0 and abs(distance_pct) <= 0.01 and can_trigger("breakdown"):
                events.append(TriggerEvent(
                    event_type="breakdown",
                    symbol=symbol,
                    current_price=current_price,
                    trigger_price=support,
                    distance_pct=distance_pct * 100,
                    context={"broken_level": support},
                    timestamp=now,
                    user_id=context.user_id,
                ))
                mark_triggered("breakdown")
                break

        # Check resistance approach
        for resistance in context.resistance_levels:
            distance_pct = (resistance - current_price) / current_price if current_price > 0 else 0

            if 0 < distance_pct <= self.approach_threshold and can_trigger("approaching_resistance"):
                events.append(TriggerEvent(
                    event_type="approaching_resistance",
                    symbol=symbol,
                    current_price=current_price,
                    trigger_price=resistance,
                    distance_pct=distance_pct * 100,
                    context={"level_type": "resistance"},
                    timestamp=now,
                    user_id=context.user_id,
                ))
                mark_triggered("approaching_resistance")
                break

            # Breakout above resistance
            if distance_pct < 0 and abs(distance_pct) <= 0.01 and can_trigger("breakout"):
                events.append(TriggerEvent(
                    event_type="breakout",
                    symbol=symbol,
                    current_price=current_price,
                    trigger_price=resistance,
                    distance_pct=distance_pct * 100,
                    context={"broken_level": resistance},
                    timestamp=now,
                    user_id=context.user_id,
                ))
                mark_triggered("breakout")
                break

        # Check position-related triggers
        if context.position and context.position.status in ["entered", "partial"]:
            position = context.position

            # Stop loss proximity
            if position.stop_loss:
                stop_distance = (current_price - position.stop_loss) / current_price
                if 0 < stop_distance <= self.stop_threshold and can_trigger("stop_proximity"):
                    events.append(TriggerEvent(
                        event_type="stop_proximity",
                        symbol=symbol,
                        current_price=current_price,
                        trigger_price=position.stop_loss,
                        distance_pct=stop_distance * 100,
                        context={
                            "stop_loss": position.stop_loss,
                            "entry_price": position.entry_price,
                            "position_size": position.current_size,
                        },
                        timestamp=now,
                        user_id=context.user_id,
                    ))
                    mark_triggered("stop_proximity")

            # Target reached
            targets = [
                (1, position.target_1),
                (2, position.target_2),
                (3, position.target_3),
            ]
            for target_num, target_price in targets:
                if target_price and target_num not in position.targets_hit:
                    if current_price >= target_price and can_trigger(f"target_{target_num}"):
                        events.append(TriggerEvent(
                            event_type="target_reached",
                            symbol=symbol,
                            current_price=current_price,
                            trigger_price=target_price,
                            distance_pct=0,
                            context={
                                "target_number": target_num,
                                "entry_price": position.entry_price,
                                "position_size": position.current_size,
                            },
                            timestamp=now,
                            user_id=context.user_id,
                        ))
                        mark_triggered(f"target_{target_num}")
                        break  # Only one target event at a time

        # Volume spike
        if context.average_volume > 0 and current_volume > 0:
            volume_ratio = current_volume / context.average_volume
            if volume_ratio >= self.volume_spike_multiplier and can_trigger("volume_spike"):
                events.append(TriggerEvent(
                    event_type="volume_spike",
                    symbol=symbol,
                    current_price=current_price,
                    trigger_price=current_price,
                    distance_pct=0,
                    context={
                        "volume_ratio": volume_ratio,
                        "current_volume": current_volume,
                        "average_volume": context.average_volume,
                    },
                    timestamp=now,
                    user_id=context.user_id,
                ))
                mark_triggered("volume_spike")

        # Large price move (5 minute)
        if context.price_5min_ago and context.price_5min_ago > 0:
            move_pct = (current_price - context.price_5min_ago) / context.price_5min_ago
            if abs(move_pct) >= self.large_move_threshold and can_trigger("large_move"):
                events.append(TriggerEvent(
                    event_type="large_move",
                    symbol=symbol,
                    current_price=current_price,
                    trigger_price=context.price_5min_ago,
                    distance_pct=move_pct * 100,
                    context={
                        "direction": "up" if move_pct > 0 else "down",
                        "price_5min_ago": context.price_5min_ago,
                    },
                    timestamp=now,
                    user_id=context.user_id,
                ))
                mark_triggered("large_move")

        return events

    async def _handle_price_update(self, symbol: str, price: float, volume: int) -> None:
        """Handle a price update and generate triggers if needed.

        Args:
            symbol: Stock ticker symbol
            price: Current price
            volume: Current volume
        """
        if symbol not in self._monitored:
            return

        # Check for trigger conditions
        events = self._check_trigger_conditions(symbol, price, volume)

        # Update context
        context = self._monitored[symbol]
        context.last_price = price
        context.last_volume = volume
        context.last_check = datetime.utcnow().isoformat()

        # Fire callback for each event
        if events and self.trigger_callback:
            for event in events:
                try:
                    if asyncio.iscoroutinefunction(self.trigger_callback):
                        await self.trigger_callback(event)
                    else:
                        self.trigger_callback(event)
                except Exception as e:
                    logger.error(f"Trigger callback error for {symbol}: {e}")

    async def _poll_prices(self) -> None:
        """Poll prices for all monitored symbols."""
        if not self._monitored:
            return

        symbols = list(self._monitored.keys())

        try:
            snapshots = fetch_snapshots(symbols)

            for symbol, snapshot in snapshots.items():
                price = snapshot.get("latest_trade", {}).get("price", 0)
                volume = snapshot.get("daily_bar", {}).get("volume", 0)

                if price > 0:
                    await self._handle_price_update(symbol, price, volume)

        except Exception as e:
            logger.error(f"Error polling prices: {e}")

    async def _update_5min_prices(self) -> None:
        """Update 5-minute-ago prices for large move detection."""
        for symbol, context in self._monitored.items():
            context.price_5min_ago = context.last_price

    async def _clear_recent_triggers(self) -> None:
        """Clear recent triggers to allow new alerts."""
        for symbol in self._monitored:
            self._recent_triggers[symbol] = set()

    async def _polling_loop(self, interval: int = 60) -> None:
        """Main polling loop.

        Args:
            interval: Seconds between polls
        """
        poll_count = 0

        while self._running:
            await self._poll_prices()

            poll_count += 1

            # Update 5-min prices every 5 polls
            if poll_count % 5 == 0:
                await self._update_5min_prices()

            # Clear recent triggers every 15 minutes (15 polls at 60s interval)
            if poll_count % 15 == 0:
                await self._clear_recent_triggers()

            await asyncio.sleep(interval)

    async def start(self, interval: int = 60) -> None:
        """Start price monitoring.

        Args:
            interval: Seconds between price polls
        """
        if self._running:
            logger.warning("Price monitor already running")
            return

        self._running = True

        # Load watchlist symbols
        await self._load_watchlist_symbols()

        # Start polling loop
        self._poll_task = asyncio.create_task(self._polling_loop(interval))

        logger.info(f"Price monitor started ({len(self._monitored)} symbols)")

    async def stop(self) -> None:
        """Stop price monitoring."""
        if not self._running:
            return

        self._running = False

        if self._poll_task:
            self._poll_task.cancel()
            try:
                await self._poll_task
            except asyncio.CancelledError:
                pass
            self._poll_task = None

        logger.info("Price monitor stopped")

    async def _load_watchlist_symbols(self) -> None:
        """Load symbols from watchlist with alerts enabled."""
        try:
            store = get_watchlist_store()
            position_store = get_position_store()

            # Get all watchlist items
            items = store.get_watchlist("default")

            for item in items:
                if item.get("alerts_enabled", False):
                    symbol = item["symbol"]

                    # Check for active position
                    position = await position_store.get_position("default", symbol)

                    await self.add_symbol(symbol, "default", position)

        except Exception as e:
            logger.error(f"Error loading watchlist: {e}")

    def get_status(self) -> dict:
        """Get monitor status.

        Returns:
            Dictionary with monitor status information
        """
        return {
            "running": self._running,
            "monitored_symbols": list(self._monitored.keys()),
            "symbol_count": len(self._monitored),
            "contexts": {
                symbol: {
                    "last_price": ctx.last_price,
                    "last_check": ctx.last_check,
                    "has_position": ctx.position is not None,
                    "support_levels": ctx.support_levels[:3],
                    "resistance_levels": ctx.resistance_levels[:3],
                }
                for symbol, ctx in self._monitored.items()
            },
        }


# Singleton instance
_monitor: Optional[PriceMonitor] = None


def get_price_monitor() -> PriceMonitor:
    """Get the singleton price monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PriceMonitor()
    return _monitor
