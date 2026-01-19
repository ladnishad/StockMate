"""Tests for Alpaca market clock service and market hours functionality.

This module tests:
- AlpacaClockService with mocked Alpaca API
- Fallback logic when API is unavailable
- Market holiday data accuracy
- Caching behavior
- Scheduler integration
"""

import pytest
from datetime import datetime, time, timedelta
from unittest.mock import Mock, patch, MagicMock
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")


class TestMarketHolidays:
    """Tests for market_holidays.py module."""

    def test_market_holidays_includes_2026(self):
        """Verify 2026 holidays are included."""
        from app.services.market_holidays import MARKET_HOLIDAYS, is_year_covered

        assert is_year_covered(2026), "2026 holidays should be covered"

        # Check specific 2026 holidays
        assert "2026-01-01" in MARKET_HOLIDAYS  # New Year's Day
        assert "2026-01-19" in MARKET_HOLIDAYS  # MLK Day
        assert "2026-07-03" in MARKET_HOLIDAYS  # Independence Day (observed)
        assert "2026-11-26" in MARKET_HOLIDAYS  # Thanksgiving
        assert "2026-12-25" in MARKET_HOLIDAYS  # Christmas

    def test_market_holidays_includes_2027(self):
        """Verify 2027 holidays are included."""
        from app.services.market_holidays import MARKET_HOLIDAYS, is_year_covered

        assert is_year_covered(2027), "2027 holidays should be covered"

        # Check specific 2027 holidays
        assert "2027-01-01" in MARKET_HOLIDAYS  # New Year's Day
        assert "2027-01-18" in MARKET_HOLIDAYS  # MLK Day
        assert "2027-07-05" in MARKET_HOLIDAYS  # Independence Day (observed - July 4 is Sunday)

    def test_early_close_days_2026(self):
        """Verify 2026 early close days are correct."""
        from app.services.market_holidays import EARLY_CLOSE_DAYS

        assert "2026-11-27" in EARLY_CLOSE_DAYS  # Day after Thanksgiving
        assert "2026-12-24" in EARLY_CLOSE_DAYS  # Christmas Eve
        # July 3 is a full closure in 2026, not early close
        assert "2026-07-03" not in EARLY_CLOSE_DAYS

    def test_is_market_holiday(self):
        """Test is_market_holiday function."""
        from app.services.market_holidays import is_market_holiday

        assert is_market_holiday("2026-01-01") is True
        assert is_market_holiday("2026-01-02") is False
        assert is_market_holiday("2026-12-25") is True

    def test_is_early_close_day(self):
        """Test is_early_close_day function."""
        from app.services.market_holidays import is_early_close_day

        assert is_early_close_day("2026-11-27") is True
        assert is_early_close_day("2026-11-26") is False  # Thanksgiving itself is full close

    def test_market_close_time_for_date(self):
        """Test get_market_close_time_for_date function."""
        from app.services.market_holidays import (
            get_market_close_time_for_date,
            MARKET_CLOSE_TIME,
            EARLY_CLOSE_TIME,
        )

        # Regular day
        assert get_market_close_time_for_date("2026-01-05") == MARKET_CLOSE_TIME

        # Early close day
        assert get_market_close_time_for_date("2026-11-27") == EARLY_CLOSE_TIME

    def test_years_covered(self):
        """Test get_years_covered function."""
        from app.services.market_holidays import get_years_covered

        years = get_years_covered()
        assert 2024 in years
        assert 2025 in years
        assert 2026 in years
        assert 2027 in years
        assert 2028 in years


class TestMarketClockData:
    """Tests for MarketClockData dataclass."""

    def test_cache_validity_when_market_open(self):
        """Test cache is valid within TTL when market is open."""
        from app.services.alpaca_clock import MarketClockData, CACHE_TTL_SECONDS

        now = datetime.now(ET)
        data = MarketClockData(
            timestamp=now,
            is_open=True,
            next_open=now + timedelta(days=1),
            next_close=now + timedelta(hours=4),
            fetched_at=now,
            source="alpaca"
        )

        # Should be valid immediately
        assert data.is_cache_valid() is True

    def test_cache_validity_when_market_closed(self):
        """Test cache uses longer TTL when market is closed."""
        from app.services.alpaca_clock import MarketClockData, CACHE_TTL_MARKET_CLOSED_SECONDS

        now = datetime.now(ET)
        # Simulate data fetched 2 minutes ago
        fetched_at = now - timedelta(seconds=120)

        data = MarketClockData(
            timestamp=now,
            is_open=False,
            next_open=now + timedelta(hours=16),
            next_close=now + timedelta(hours=20),
            fetched_at=fetched_at,
            source="alpaca"
        )

        # Should still be valid (closed market TTL is 5 minutes)
        assert data.is_cache_valid() is True

    def test_cache_expired(self):
        """Test cache invalidation after TTL."""
        from app.services.alpaca_clock import MarketClockData, CACHE_TTL_SECONDS

        now = datetime.now(ET)
        # Simulate data fetched 60 seconds ago (past 30s TTL for open market)
        fetched_at = now - timedelta(seconds=60)

        data = MarketClockData(
            timestamp=now,
            is_open=True,
            next_open=now + timedelta(days=1),
            next_close=now + timedelta(hours=4),
            fetched_at=fetched_at,
            source="alpaca"
        )

        # Should be invalid (past 30s TTL)
        assert data.is_cache_valid() is False


class TestAlpacaClockService:
    """Tests for AlpacaClockService."""

    @pytest.fixture
    def mock_trading_client(self):
        """Create a mock TradingClient."""
        mock_clock = Mock()
        mock_clock.timestamp = datetime.now(ET)
        mock_clock.is_open = True
        mock_clock.next_open = datetime.now(ET) + timedelta(days=1)
        mock_clock.next_close = datetime.now(ET) + timedelta(hours=4)

        mock_client = Mock()
        mock_client.get_clock.return_value = mock_clock
        return mock_client

    def test_service_uses_alpaca_when_available(self, mock_trading_client):
        """Test that service uses Alpaca API when available."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        with patch.object(service, '_get_client', return_value=mock_trading_client):
            clock = service.get_clock()

            assert clock.source == "alpaca"
            assert clock.is_open is True
            mock_trading_client.get_clock.assert_called_once()

    def test_service_falls_back_when_api_unavailable(self):
        """Test that service falls back to local logic when API unavailable."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        with patch.object(service, '_get_client', return_value=None):
            clock = service.get_clock()

            assert clock.source == "fallback"

    def test_service_caches_results(self, mock_trading_client):
        """Test that service caches results."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        with patch.object(service, '_get_client', return_value=mock_trading_client):
            # First call
            clock1 = service.get_clock()
            # Second call should use cache
            clock2 = service.get_clock()

            # Should only have called the API once
            assert mock_trading_client.get_clock.call_count == 1
            assert clock1.source == clock2.source

    def test_service_force_refresh(self, mock_trading_client):
        """Test that force_refresh bypasses cache."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        with patch.object(service, '_get_client', return_value=mock_trading_client):
            # First call
            service.get_clock()
            # Force refresh should call API again
            service.get_clock(force_refresh=True)

            assert mock_trading_client.get_clock.call_count == 2

    def test_service_tracks_consecutive_failures(self):
        """Test that service tracks consecutive API failures."""
        from app.services.alpaca_clock import AlpacaClockService
        from alpaca.common.exceptions import APIError

        service = AlpacaClockService()
        mock_client = Mock()
        mock_client.get_clock.side_effect = APIError("Test error")

        with patch.object(service, '_get_client', return_value=mock_client):
            # Multiple failed calls
            for _ in range(3):
                clock = service.get_clock(force_refresh=True)
                assert clock.source == "fallback"

            assert service._consecutive_failures == 3

    def test_service_status(self, mock_trading_client):
        """Test get_status method."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        with patch.object(service, '_get_client', return_value=mock_trading_client):
            service.get_clock()
            status = service.get_status()

            assert "client_initialized" in status
            assert "consecutive_failures" in status
            assert "cache_valid" in status
            assert status["cache_source"] == "alpaca"


class TestSchedulerFunctions:
    """Tests for scheduler.py convenience functions."""

    @pytest.fixture
    def mock_clock_data(self):
        """Create mock clock data."""
        from app.services.alpaca_clock import MarketClockData

        now = datetime.now(ET)
        return MarketClockData(
            timestamp=now,
            is_open=True,
            next_open=now + timedelta(days=1),
            next_close=now + timedelta(hours=4),
            fetched_at=now,
            source="alpaca"
        )

    def test_is_market_open(self, mock_clock_data):
        """Test is_market_open function."""
        from app.services.scheduler import is_market_open

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            assert is_market_open() is True

            # Test when market is closed
            mock_clock_data.is_open = False
            assert is_market_open() is False

    def test_get_next_market_open_when_open(self, mock_clock_data):
        """Test get_next_market_open returns now when market is open."""
        from app.services.scheduler import get_next_market_open

        mock_clock_data.is_open = True

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = get_next_market_open()
            # Should be approximately now
            assert (datetime.now(ET) - result).total_seconds() < 5

    def test_get_next_market_open_when_closed(self, mock_clock_data):
        """Test get_next_market_open returns next_open when market is closed."""
        from app.services.scheduler import get_next_market_open

        mock_clock_data.is_open = False
        expected_open = datetime.now(ET) + timedelta(hours=16)
        mock_clock_data.next_open = expected_open

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = get_next_market_open()
            assert result == expected_open

    def test_get_market_close_time_when_open(self, mock_clock_data):
        """Test get_market_close_time returns close time when market is open."""
        from app.services.scheduler import get_market_close_time

        mock_clock_data.is_open = True
        expected_close = datetime.now(ET) + timedelta(hours=4)
        mock_clock_data.next_close = expected_close

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = get_market_close_time()
            assert result == expected_close

    def test_get_market_close_time_when_closed(self, mock_clock_data):
        """Test get_market_close_time returns None when market is closed."""
        from app.services.scheduler import get_market_close_time

        mock_clock_data.is_open = False

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = get_market_close_time()
            assert result is None

    def test_seconds_until_market_open_when_open(self, mock_clock_data):
        """Test seconds_until_market_open returns 0 when market is open."""
        from app.services.scheduler import seconds_until_market_open

        mock_clock_data.is_open = True

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = seconds_until_market_open()
            assert result == 0

    def test_seconds_until_market_open_when_closed(self, mock_clock_data):
        """Test seconds_until_market_open returns positive value when closed."""
        from app.services.scheduler import seconds_until_market_open

        mock_clock_data.is_open = False
        mock_clock_data.next_open = datetime.now(ET) + timedelta(hours=2)

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = seconds_until_market_open()
            # Should be approximately 2 hours (7200 seconds)
            assert 7000 < result < 7400

    def test_seconds_until_market_close_when_closed(self, mock_clock_data):
        """Test seconds_until_market_close returns 0 when market is closed."""
        from app.services.scheduler import seconds_until_market_close

        mock_clock_data.is_open = False

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            result = seconds_until_market_close()
            assert result == 0

    def test_get_market_status(self, mock_clock_data):
        """Test get_market_status returns comprehensive data."""
        from app.services.scheduler import get_market_status
        from app.services.alpaca_clock import AlpacaClockService

        mock_service = Mock(spec=AlpacaClockService)
        mock_service.get_status.return_value = {"client_initialized": True}

        with patch('app.services.scheduler.get_market_clock', return_value=mock_clock_data):
            with patch('app.services.scheduler.get_clock_service', return_value=mock_service):
                result = get_market_status()

                assert "is_open" in result
                assert "timestamp" in result
                assert "next_open" in result
                assert "next_close" in result
                assert "data_source" in result
                assert result["data_source"] == "alpaca"


class TestFallbackLogic:
    """Tests for fallback logic in alpaca_clock.py."""

    def test_fallback_weekend_detection(self):
        """Test that fallback correctly identifies weekends."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        # Mock a Saturday
        saturday = datetime(2026, 1, 3, 12, 0, 0, tzinfo=ET)  # January 3, 2026 is Saturday

        with patch('app.services.alpaca_clock.datetime') as mock_datetime:
            mock_datetime.now.return_value = saturday
            mock_datetime.combine = datetime.combine

            with patch.object(service, '_get_client', return_value=None):
                clock = service.get_clock(force_refresh=True)

                assert clock.is_open is False
                assert clock.source == "fallback"

    def test_fallback_holiday_detection(self):
        """Test that fallback correctly identifies holidays."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        # Mock MLK Day 2026 (January 19) during market hours
        mlk_day = datetime(2026, 1, 19, 12, 0, 0, tzinfo=ET)

        with patch('app.services.alpaca_clock.datetime') as mock_datetime:
            mock_datetime.now.return_value = mlk_day
            mock_datetime.combine = datetime.combine

            with patch.object(service, '_get_client', return_value=None):
                clock = service.get_clock(force_refresh=True)

                assert clock.is_open is False
                assert clock.source == "fallback"

    def test_fallback_market_hours(self):
        """Test that fallback correctly identifies market hours."""
        from app.services.alpaca_clock import AlpacaClockService

        service = AlpacaClockService()

        # Mock a regular Tuesday at 10:00 AM ET
        tuesday_morning = datetime(2026, 1, 6, 10, 0, 0, tzinfo=ET)

        with patch('app.services.alpaca_clock.datetime') as mock_datetime:
            mock_datetime.now.return_value = tuesday_morning
            mock_datetime.combine = datetime.combine

            with patch.object(service, '_get_client', return_value=None):
                clock = service.get_clock(force_refresh=True)

                assert clock.is_open is True
                assert clock.source == "fallback"


class TestSchedulerClass:
    """Tests for MarketHoursScheduler class."""

    @pytest.fixture
    def scheduler(self):
        """Create a fresh scheduler instance."""
        from app.services.scheduler import MarketHoursScheduler
        return MarketHoursScheduler()

    def test_scheduler_initialization(self, scheduler):
        """Test scheduler initializes correctly."""
        assert scheduler.is_running is False
        assert scheduler._on_market_open_callbacks == []
        assert scheduler._on_market_close_callbacks == []
        assert scheduler._interval_callbacks == []

    def test_register_callbacks(self, scheduler):
        """Test callback registration."""
        async def on_open():
            pass

        async def on_close():
            pass

        async def interval_task():
            pass

        scheduler.on_market_open(on_open)
        scheduler.on_market_close(on_close)
        scheduler.add_interval_task(interval_task, 60)

        assert len(scheduler._on_market_open_callbacks) == 1
        assert len(scheduler._on_market_close_callbacks) == 1
        assert len(scheduler._interval_callbacks) == 1

    def test_scheduler_status(self, scheduler):
        """Test get_status method."""
        from app.services.alpaca_clock import MarketClockData

        mock_clock = MarketClockData(
            timestamp=datetime.now(ET),
            is_open=False,
            next_open=datetime.now(ET) + timedelta(hours=16),
            next_close=datetime.now(ET) + timedelta(hours=20),
            fetched_at=datetime.now(ET),
            source="alpaca"
        )

        with patch('app.services.scheduler.get_market_status') as mock_status:
            mock_status.return_value = {
                "is_open": False,
                "next_open": "2026-01-06T09:30:00-05:00",
                "next_close": "2026-01-06T16:00:00-05:00",
                "seconds_until_open": 3600,
                "seconds_until_close": 0,
                "data_source": "alpaca"
            }

            status = scheduler.get_status()

            assert "running" in status
            assert "market_open" in status
            assert "data_source" in status
            assert "registered_callbacks" in status


class TestIntegration:
    """Integration tests that verify the full flow."""

    def test_singleton_instances(self):
        """Test that singleton instances are properly maintained."""
        from app.services.alpaca_clock import get_clock_service
        from app.services.scheduler import get_scheduler

        service1 = get_clock_service()
        service2 = get_clock_service()
        assert service1 is service2

        scheduler1 = get_scheduler()
        scheduler2 = get_scheduler()
        assert scheduler1 is scheduler2

    def test_services_module_exports(self):
        """Test that all expected items are exported from services module."""
        from app.services import (
            is_market_open,
            get_next_market_open,
            get_market_close_time,
            seconds_until_market_open,
            seconds_until_market_close,
            get_market_status,
            get_clock_service,
            get_market_clock,
            MarketClockData,
        )

        # All these should be importable
        assert callable(is_market_open)
        assert callable(get_next_market_open)
        assert callable(get_market_close_time)
        assert callable(seconds_until_market_open)
        assert callable(seconds_until_market_close)
        assert callable(get_market_status)
        assert callable(get_clock_service)
        assert callable(get_market_clock)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
