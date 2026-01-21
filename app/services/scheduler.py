"""Market hours scheduler.

Manages starting and stopping of monitoring during US market hours.
Uses Alpaca's /v2/clock endpoint for accurate market status with
local timezone-based fallback for resilience.
"""

import logging
from datetime import datetime, time, timedelta
from typing import Callable, Optional, List
from zoneinfo import ZoneInfo
import asyncio

from app.services.alpaca_clock import (
    get_market_clock,
    get_clock_service,
    MarketClockData,
)

logger = logging.getLogger(__name__)

# US Eastern timezone (handles DST automatically)
ET = ZoneInfo("America/New_York")


def is_market_open() -> bool:
    """Check if the US stock market is currently open.

    This function uses Alpaca's /v2/clock endpoint for accurate market status,
    with automatic fallback to local timezone calculations if the API is unavailable.

    Returns:
        True if market is open, False otherwise
    """
    try:
        clock = get_market_clock()
        return clock.is_open
    except Exception as e:
        logger.error(f"Error checking market status: {e}")
        # Emergency fallback - assume closed to be safe
        return False


def get_next_market_open() -> datetime:
    """Get the next market open time.

    This function uses Alpaca's /v2/clock endpoint for accurate timing,
    with automatic fallback to local timezone calculations if the API is unavailable.

    Returns:
        datetime of next market open in ET (timezone-aware)
    """
    try:
        clock = get_market_clock()

        # If market is currently open, return now
        if clock.is_open:
            return datetime.now(ET)

        return clock.next_open

    except Exception as e:
        logger.error(f"Error getting next market open: {e}")
        # Emergency fallback - return tomorrow's open
        now = datetime.now(ET)
        tomorrow = now.date() + timedelta(days=1)
        return datetime.combine(tomorrow, time(9, 30), tzinfo=ET)


def get_market_close_time() -> Optional[datetime]:
    """Get today's market close time if market is open.

    This function uses Alpaca's /v2/clock endpoint for accurate timing,
    with automatic fallback to local timezone calculations if the API is unavailable.

    Returns:
        datetime of today's close in ET, or None if market is closed
    """
    try:
        clock = get_market_clock()

        if not clock.is_open:
            return None

        return clock.next_close

    except Exception as e:
        logger.error(f"Error getting market close time: {e}")
        return None


def seconds_until_market_open() -> float:
    """Get seconds until next market open.

    This function uses Alpaca's /v2/clock endpoint for accurate timing.

    Returns:
        Seconds until market open (0 if currently open)
    """
    try:
        clock = get_market_clock()

        if clock.is_open:
            return 0

        now = datetime.now(ET)
        delta = clock.next_open - now
        return max(0, delta.total_seconds())

    except Exception as e:
        logger.error(f"Error calculating seconds until market open: {e}")
        return 0


def seconds_until_market_close() -> float:
    """Get seconds until market close.

    This function uses Alpaca's /v2/clock endpoint for accurate timing.

    Returns:
        Seconds until close (0 if market is closed)
    """
    try:
        clock = get_market_clock()

        if not clock.is_open:
            return 0

        now = datetime.now(ET)
        delta = clock.next_close - now
        return max(0, delta.total_seconds())

    except Exception as e:
        logger.error(f"Error calculating seconds until market close: {e}")
        return 0


def get_market_status() -> dict:
    """Get comprehensive market status information.

    This function returns detailed market status including data source
    information for debugging and monitoring.

    Returns:
        Dictionary with market status details including:
        - is_open: Current market open/closed status
        - timestamp: Current market time
        - next_open: Next market open datetime (ISO format)
        - next_close: Next market close datetime (ISO format)
        - seconds_until_open: Seconds until market opens
        - seconds_until_close: Seconds until market closes
        - data_source: "alpaca" or "fallback"
        - service_status: Clock service diagnostics
    """
    try:
        clock = get_market_clock()
        service = get_clock_service()

        return {
            "is_open": clock.is_open,
            "timestamp": clock.timestamp.isoformat(),
            "next_open": clock.next_open.isoformat(),
            "next_close": clock.next_close.isoformat(),
            "seconds_until_open": seconds_until_market_open(),
            "seconds_until_close": seconds_until_market_close(),
            "data_source": clock.source,
            "fetched_at": clock.fetched_at.isoformat(),
            "service_status": service.get_status(),
        }

    except Exception as e:
        logger.error(f"Error getting market status: {e}")
        return {
            "is_open": False,
            "error": str(e),
            "data_source": "error",
        }


class MarketHoursScheduler:
    """Manages scheduling of tasks during market hours.

    Automatically starts tasks when market opens and stops when it closes.
    Uses Alpaca's /v2/clock endpoint for accurate market status detection.
    """

    def __init__(self):
        """Initialize the scheduler."""
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._on_market_open_callbacks: List[Callable] = []
        self._on_market_close_callbacks: List[Callable] = []
        self._interval_callbacks: List[tuple[Callable, int]] = []  # (callback, interval_seconds)

    def on_market_open(self, callback: Callable) -> None:
        """Register a callback to run when market opens.

        Args:
            callback: Async function to call on market open
        """
        self._on_market_open_callbacks.append(callback)

    def on_market_close(self, callback: Callable) -> None:
        """Register a callback to run when market closes.

        Args:
            callback: Async function to call on market close
        """
        self._on_market_close_callbacks.append(callback)

    def add_interval_task(self, callback: Callable, interval_seconds: int) -> None:
        """Register a task to run at intervals during market hours.

        Args:
            callback: Async function to call at intervals
            interval_seconds: Seconds between calls
        """
        self._interval_callbacks.append((callback, interval_seconds))

    async def _run_callbacks(self, callbacks: List[Callable]) -> None:
        """Run a list of callbacks, handling errors."""
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback()
                else:
                    callback()
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def _run_interval_loop(self) -> None:
        """Main loop that runs during market hours."""
        logger.info("Starting market hours monitoring loop")

        # Wait for market to open if not already
        wait_seconds = seconds_until_market_open()
        if wait_seconds > 0:
            logger.info(f"Market closed. Waiting {wait_seconds/3600:.1f} hours until open")

            # Sleep in chunks to allow for cancellation and status updates
            while wait_seconds > 0 and self._running:
                sleep_time = min(wait_seconds, 60)  # Check every minute
                await asyncio.sleep(sleep_time)
                wait_seconds = seconds_until_market_open()

        if not self._running:
            return

        # Market is now open - run open callbacks
        logger.info("Market is open - starting monitoring")
        await self._run_callbacks(self._on_market_open_callbacks)

        # Track last run times for interval tasks
        last_run = {i: 0.0 for i in range(len(self._interval_callbacks))}

        while self._running and is_market_open():
            current_time = asyncio.get_event_loop().time()

            # Run interval tasks that are due
            for i, (callback, interval) in enumerate(self._interval_callbacks):
                if current_time - last_run[i] >= interval:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback()
                        else:
                            callback()
                        last_run[i] = current_time
                    except Exception as e:
                        logger.error(f"Interval task error: {e}")

            # Sleep for 1 second before checking again
            await asyncio.sleep(1)

        # Market closed - run close callbacks
        if self._running:
            logger.info("Market closed - stopping monitoring")
            await self._run_callbacks(self._on_market_close_callbacks)

    async def _scheduler_loop(self) -> None:
        """Continuous scheduler loop that handles daily market cycles."""
        while self._running:
            await self._run_interval_loop()

            # If still running, wait for next market open
            if self._running:
                wait_seconds = seconds_until_market_open()
                if wait_seconds > 0:
                    logger.info(f"Waiting {wait_seconds/3600:.1f} hours until next market open")

                    # Sleep in chunks to allow for cancellation
                    while wait_seconds > 0 and self._running:
                        sleep_time = min(wait_seconds, 3600)  # Check every hour max
                        await asyncio.sleep(sleep_time)
                        wait_seconds = seconds_until_market_open()

    async def start(self) -> None:
        """Start the scheduler."""
        if self._running:
            logger.warning("Scheduler already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
        logger.info("Market hours scheduler started")

    async def stop(self) -> None:
        """Stop the scheduler."""
        if not self._running:
            return

        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        logger.info("Market hours scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

    def get_status(self) -> dict:
        """Get scheduler status.

        Returns:
            Dictionary with scheduler status information
        """
        market_status = get_market_status()

        return {
            "running": self._running,
            "market_open": market_status.get("is_open", False),
            "next_open": market_status.get("next_open") if not market_status.get("is_open") else None,
            "close_time": market_status.get("next_close") if market_status.get("is_open") else None,
            "seconds_until_open": market_status.get("seconds_until_open", 0),
            "seconds_until_close": market_status.get("seconds_until_close", 0),
            "data_source": market_status.get("data_source", "unknown"),
            "registered_callbacks": {
                "on_open": len(self._on_market_open_callbacks),
                "on_close": len(self._on_market_close_callbacks),
                "interval_tasks": len(self._interval_callbacks),
            },
        }


# Singleton instance
_scheduler: Optional[MarketHoursScheduler] = None


def get_scheduler() -> MarketHoursScheduler:
    """Get the singleton scheduler instance."""
    global _scheduler
    if _scheduler is None:
        _scheduler = MarketHoursScheduler()
    return _scheduler
