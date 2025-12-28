"""Market hours scheduler.

Manages starting and stopping of monitoring during US market hours.
Handles market holidays and early closes.
"""

import logging
from datetime import datetime, time, timedelta
from typing import Callable, Optional, List
from zoneinfo import ZoneInfo
import asyncio

logger = logging.getLogger(__name__)

# US Eastern timezone (handles DST automatically)
ET = ZoneInfo("America/New_York")

# Regular market hours
MARKET_OPEN = time(9, 30)  # 9:30 AM ET
MARKET_CLOSE = time(16, 0)  # 4:00 PM ET

# 2024-2025 US Market Holidays (NYSE/NASDAQ)
MARKET_HOLIDAYS = {
    # 2024
    "2024-01-01",  # New Year's Day
    "2024-01-15",  # Martin Luther King Jr. Day
    "2024-02-19",  # Presidents Day
    "2024-03-29",  # Good Friday
    "2024-05-27",  # Memorial Day
    "2024-06-19",  # Juneteenth
    "2024-07-04",  # Independence Day
    "2024-09-02",  # Labor Day
    "2024-11-28",  # Thanksgiving
    "2024-12-25",  # Christmas
    # 2025
    "2025-01-01",  # New Year's Day
    "2025-01-20",  # Martin Luther King Jr. Day
    "2025-02-17",  # Presidents Day
    "2025-04-18",  # Good Friday
    "2025-05-26",  # Memorial Day
    "2025-06-19",  # Juneteenth
    "2025-07-04",  # Independence Day
    "2025-09-01",  # Labor Day
    "2025-11-27",  # Thanksgiving
    "2025-12-25",  # Christmas
}

# Early close days (1:00 PM ET close)
EARLY_CLOSE_DAYS = {
    "2024-07-03",  # Day before Independence Day
    "2024-11-29",  # Day after Thanksgiving
    "2024-12-24",  # Christmas Eve
    "2025-07-03",  # Day before Independence Day
    "2025-11-28",  # Day after Thanksgiving
    "2025-12-24",  # Christmas Eve
}


def is_market_open() -> bool:
    """Check if the US stock market is currently open.

    Returns:
        True if market is open, False otherwise
    """
    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")

    # Check if today is a holiday
    if today_str in MARKET_HOLIDAYS:
        return False

    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False

    # Get market close time (early close or regular)
    close_time = time(13, 0) if today_str in EARLY_CLOSE_DAYS else MARKET_CLOSE

    # Check if within market hours
    current_time = now.time()
    return MARKET_OPEN <= current_time < close_time


def get_next_market_open() -> datetime:
    """Get the next market open time.

    Returns:
        datetime of next market open in ET
    """
    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")

    # If market is open now, return now
    if is_market_open():
        return now

    # Start checking from today
    check_date = now.date()

    # Check up to 10 days ahead (handles long weekends + holidays)
    for _ in range(10):
        date_str = check_date.strftime("%Y-%m-%d")

        # Skip weekends and holidays
        if check_date.weekday() < 5 and date_str not in MARKET_HOLIDAYS:
            open_datetime = datetime.combine(check_date, MARKET_OPEN, tzinfo=ET)

            # If it's today and past market open but before close, return today
            if check_date == now.date() and now.time() >= MARKET_OPEN:
                close_time = time(13, 0) if date_str in EARLY_CLOSE_DAYS else MARKET_CLOSE
                if now.time() < close_time:
                    return now
                # Market closed today, check tomorrow
                check_date += timedelta(days=1)
                continue

            # If it's today and before market open, return today's open
            if check_date == now.date() and now.time() < MARKET_OPEN:
                return open_datetime

            # Future date
            if check_date > now.date():
                return open_datetime

        check_date += timedelta(days=1)

    # Fallback: next Monday at market open
    days_until_monday = (7 - now.weekday()) % 7 or 7
    next_monday = now.date() + timedelta(days=days_until_monday)
    return datetime.combine(next_monday, MARKET_OPEN, tzinfo=ET)


def get_market_close_time() -> Optional[datetime]:
    """Get today's market close time if market is open.

    Returns:
        datetime of today's close in ET, or None if market is closed
    """
    if not is_market_open():
        return None

    now = datetime.now(ET)
    today_str = now.strftime("%Y-%m-%d")
    close_time = time(13, 0) if today_str in EARLY_CLOSE_DAYS else MARKET_CLOSE

    return datetime.combine(now.date(), close_time, tzinfo=ET)


def seconds_until_market_open() -> float:
    """Get seconds until next market open.

    Returns:
        Seconds until market open (0 if currently open)
    """
    if is_market_open():
        return 0

    next_open = get_next_market_open()
    now = datetime.now(ET)
    delta = next_open - now

    return max(0, delta.total_seconds())


def seconds_until_market_close() -> float:
    """Get seconds until market close.

    Returns:
        Seconds until close (0 if market is closed)
    """
    close_time = get_market_close_time()
    if not close_time:
        return 0

    now = datetime.now(ET)
    delta = close_time - now

    return max(0, delta.total_seconds())


class MarketHoursScheduler:
    """Manages scheduling of tasks during market hours.

    Automatically starts tasks when market opens and stops when it closes.
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
            await asyncio.sleep(wait_seconds)

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
                    await asyncio.sleep(min(wait_seconds, 3600))  # Check every hour max

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
        return {
            "running": self._running,
            "market_open": is_market_open(),
            "next_open": get_next_market_open().isoformat() if not is_market_open() else None,
            "close_time": get_market_close_time().isoformat() if is_market_open() else None,
            "seconds_until_open": seconds_until_market_open(),
            "seconds_until_close": seconds_until_market_close(),
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
