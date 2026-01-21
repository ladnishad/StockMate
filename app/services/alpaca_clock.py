"""Alpaca Market Clock Service.

Provides accurate market status by fetching from Alpaca's /v2/clock endpoint.
Includes caching to minimize API calls and fallback logic for resilience.
"""

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

from alpaca.trading.client import TradingClient
from alpaca.trading.models import Clock
from alpaca.common.exceptions import APIError

from app.config import get_settings

logger = logging.getLogger(__name__)

# US Eastern timezone (handles DST automatically)
ET = ZoneInfo("America/New_York")

# Cache settings
CACHE_TTL_SECONDS = 30  # How long to cache clock data
CACHE_TTL_MARKET_CLOSED_SECONDS = 300  # Cache longer when market is closed (5 minutes)


@dataclass
class MarketClockData:
    """Cached market clock data."""

    timestamp: datetime
    is_open: bool
    next_open: datetime
    next_close: datetime
    fetched_at: datetime
    source: str  # "alpaca" or "fallback"

    def is_cache_valid(self) -> bool:
        """Check if cached data is still valid."""
        now = datetime.now(ET)
        elapsed = (now - self.fetched_at).total_seconds()

        # Use longer TTL when market is closed (less frequent changes expected)
        ttl = CACHE_TTL_MARKET_CLOSED_SECONDS if not self.is_open else CACHE_TTL_SECONDS

        return elapsed < ttl


class AlpacaClockService:
    """Service for fetching and caching Alpaca market clock data.

    This service:
    - Fetches market status from Alpaca's authoritative /v2/clock endpoint
    - Caches results to minimize API calls (30s when open, 5min when closed)
    - Provides fallback logic using local timezone calculations if API fails
    - Is thread-safe for concurrent access
    """

    def __init__(self):
        """Initialize the clock service."""
        self._cache: Optional[MarketClockData] = None
        self._lock = threading.Lock()
        self._client: Optional[TradingClient] = None
        self._client_initialized = False
        self._consecutive_failures = 0
        self._max_consecutive_failures = 5  # After this, use fallback only

    def _get_client(self) -> Optional[TradingClient]:
        """Get or create the Alpaca trading client."""
        if self._client_initialized:
            return self._client

        settings = get_settings()

        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            logger.warning("Alpaca API keys not configured - using fallback market hours")
            self._client_initialized = True
            self._client = None
            return None

        try:
            is_paper = "paper" in settings.alpaca_base_url.lower()
            self._client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=is_paper
            )
            self._client_initialized = True
            logger.info(f"Alpaca clock client initialized (paper={is_paper})")
            return self._client
        except Exception as e:
            logger.error(f"Failed to initialize Alpaca client: {e}")
            self._client_initialized = True
            self._client = None
            return None

    def _fetch_from_alpaca(self) -> Optional[MarketClockData]:
        """Fetch clock data from Alpaca API."""
        client = self._get_client()
        if not client:
            return None

        # Skip API call if we've had too many consecutive failures
        if self._consecutive_failures >= self._max_consecutive_failures:
            logger.debug(f"Skipping Alpaca API due to {self._consecutive_failures} consecutive failures")
            return None

        try:
            clock: Clock = client.get_clock()

            # Reset failure counter on success
            self._consecutive_failures = 0

            # Ensure datetimes are timezone-aware
            timestamp = clock.timestamp
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=ET)

            next_open = clock.next_open
            if next_open.tzinfo is None:
                next_open = next_open.replace(tzinfo=ET)

            next_close = clock.next_close
            if next_close.tzinfo is None:
                next_close = next_close.replace(tzinfo=ET)

            data = MarketClockData(
                timestamp=timestamp,
                is_open=clock.is_open,
                next_open=next_open,
                next_close=next_close,
                fetched_at=datetime.now(ET),
                source="alpaca"
            )

            logger.debug(
                f"Fetched clock from Alpaca: is_open={data.is_open}, "
                f"next_open={data.next_open.isoformat()}, next_close={data.next_close.isoformat()}"
            )

            return data

        except APIError as e:
            self._consecutive_failures += 1
            logger.warning(f"Alpaca API error fetching clock (attempt {self._consecutive_failures}): {e}")
            return None
        except Exception as e:
            self._consecutive_failures += 1
            logger.error(f"Error fetching clock from Alpaca (attempt {self._consecutive_failures}): {e}")
            return None

    def _calculate_fallback(self) -> MarketClockData:
        """Calculate market status using local timezone-based fallback logic.

        This fallback uses hardcoded market hours and known US market holidays.
        It should only be used when the Alpaca API is unavailable.
        """
        from app.services.market_holidays import (
            MARKET_HOLIDAYS,
            EARLY_CLOSE_DAYS,
            MARKET_OPEN_TIME,
            MARKET_CLOSE_TIME,
            EARLY_CLOSE_TIME,
        )

        now = datetime.now(ET)
        today_str = now.strftime("%Y-%m-%d")

        # Determine if market is currently open
        is_holiday = today_str in MARKET_HOLIDAYS
        is_weekend = now.weekday() >= 5  # Saturday = 5, Sunday = 6

        # Get today's close time (early close or regular)
        close_time = EARLY_CLOSE_TIME if today_str in EARLY_CLOSE_DAYS else MARKET_CLOSE_TIME

        # Check if within market hours
        current_time = now.time()
        is_open = (
            not is_holiday
            and not is_weekend
            and MARKET_OPEN_TIME <= current_time < close_time
        )

        # Calculate next close (today's close if market is open)
        if is_open:
            next_close = datetime.combine(now.date(), close_time, tzinfo=ET)
        else:
            # Find next trading day's close
            next_trading_day = self._find_next_trading_day(now)
            next_close_time = (
                EARLY_CLOSE_TIME
                if next_trading_day.strftime("%Y-%m-%d") in EARLY_CLOSE_DAYS
                else MARKET_CLOSE_TIME
            )
            next_close = datetime.combine(next_trading_day, next_close_time, tzinfo=ET)

        # Calculate next open
        if is_open:
            # If market is open, next open is tomorrow (or next trading day)
            next_trading_day = self._find_next_trading_day(now + timedelta(days=1))
            next_open = datetime.combine(next_trading_day, MARKET_OPEN_TIME, tzinfo=ET)
        else:
            # If market is closed, find when it opens next
            next_trading_day = self._find_next_trading_day(now)
            next_open = datetime.combine(next_trading_day, MARKET_OPEN_TIME, tzinfo=ET)

            # If it's today but before market open
            if next_trading_day == now.date() and current_time < MARKET_OPEN_TIME:
                next_open = datetime.combine(now.date(), MARKET_OPEN_TIME, tzinfo=ET)

        logger.debug(f"Fallback calculation: is_open={is_open}, next_open={next_open}, next_close={next_close}")

        return MarketClockData(
            timestamp=now,
            is_open=is_open,
            next_open=next_open,
            next_close=next_close,
            fetched_at=now,
            source="fallback"
        )

    def _find_next_trading_day(self, start: datetime) -> datetime:
        """Find the next trading day from the given start date.

        Args:
            start: Starting datetime to search from

        Returns:
            Date of the next trading day
        """
        from app.services.market_holidays import MARKET_HOLIDAYS

        check_date = start.date() if isinstance(start, datetime) else start

        # Check up to 10 days ahead (handles long weekends + holidays)
        for _ in range(10):
            date_str = check_date.strftime("%Y-%m-%d")

            # Valid trading day: weekday and not a holiday
            if check_date.weekday() < 5 and date_str not in MARKET_HOLIDAYS:
                return check_date

            check_date += timedelta(days=1)

        # Fallback: return next weekday (shouldn't happen in practice)
        while check_date.weekday() >= 5:
            check_date += timedelta(days=1)

        return check_date

    def get_clock(self, force_refresh: bool = False) -> MarketClockData:
        """Get current market clock data.

        This method:
        1. Returns cached data if valid and not force_refresh
        2. Attempts to fetch from Alpaca API
        3. Falls back to local calculation if API fails

        Args:
            force_refresh: If True, ignore cache and fetch fresh data

        Returns:
            MarketClockData with current market status
        """
        with self._lock:
            # Check cache first (unless force refresh)
            if not force_refresh and self._cache and self._cache.is_cache_valid():
                return self._cache

            # Try fetching from Alpaca
            data = self._fetch_from_alpaca()

            # Fall back to local calculation if Alpaca fails
            if data is None:
                data = self._calculate_fallback()
                if self._consecutive_failures > 0:
                    logger.warning(
                        f"Using fallback market hours (Alpaca API failures: {self._consecutive_failures})"
                    )

            # Update cache
            self._cache = data

            return data

    def reset_failure_counter(self) -> None:
        """Reset the consecutive failure counter.

        Call this to retry the Alpaca API after a period of using fallback.
        """
        with self._lock:
            self._consecutive_failures = 0
            logger.info("Alpaca clock failure counter reset")

    def get_status(self) -> dict:
        """Get service status for diagnostics.

        Returns:
            Dictionary with service status information
        """
        with self._lock:
            return {
                "client_initialized": self._client_initialized,
                "client_available": self._client is not None,
                "consecutive_failures": self._consecutive_failures,
                "using_fallback_only": self._consecutive_failures >= self._max_consecutive_failures,
                "cache_valid": self._cache.is_cache_valid() if self._cache else False,
                "cache_source": self._cache.source if self._cache else None,
                "cache_age_seconds": (
                    (datetime.now(ET) - self._cache.fetched_at).total_seconds()
                    if self._cache else None
                ),
            }


# Singleton instance
_clock_service: Optional[AlpacaClockService] = None


def get_clock_service() -> AlpacaClockService:
    """Get the singleton clock service instance."""
    global _clock_service
    if _clock_service is None:
        _clock_service = AlpacaClockService()
    return _clock_service


# Convenience functions for direct use

def get_market_clock(force_refresh: bool = False) -> MarketClockData:
    """Get current market clock data.

    Convenience function that uses the singleton service.

    Args:
        force_refresh: If True, ignore cache and fetch fresh data

    Returns:
        MarketClockData with current market status
    """
    return get_clock_service().get_clock(force_refresh)


def is_market_open_live() -> bool:
    """Check if the market is currently open using live Alpaca data.

    This is the recommended way to check market status as it uses
    Alpaca's authoritative clock endpoint.

    Returns:
        True if market is open, False otherwise
    """
    return get_market_clock().is_open


def get_next_market_open_live() -> datetime:
    """Get the next market open time using live Alpaca data.

    Returns:
        datetime of next market open (timezone-aware, ET)
    """
    clock = get_market_clock()
    if clock.is_open:
        # If market is open, find the NEXT open (after today's close)
        # Force refresh to ensure we get accurate next_open
        clock = get_market_clock(force_refresh=True)
    return clock.next_open


def get_next_market_close_live() -> Optional[datetime]:
    """Get the next market close time using live Alpaca data.

    Returns:
        datetime of market close if market is open, None otherwise
    """
    clock = get_market_clock()
    if not clock.is_open:
        return None
    return clock.next_close


def seconds_until_market_open_live() -> float:
    """Get seconds until market opens using live Alpaca data.

    Returns:
        Seconds until market open (0 if currently open)
    """
    clock = get_market_clock()
    if clock.is_open:
        return 0

    now = datetime.now(ET)
    delta = clock.next_open - now
    return max(0, delta.total_seconds())


def seconds_until_market_close_live() -> float:
    """Get seconds until market closes using live Alpaca data.

    Returns:
        Seconds until market close (0 if market is closed)
    """
    clock = get_market_clock()
    if not clock.is_open:
        return 0

    now = datetime.now(ET)
    delta = clock.next_close - now
    return max(0, delta.total_seconds())
