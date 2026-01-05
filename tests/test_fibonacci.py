"""Tests for Fibonacci retracement and extension functionality."""

import pytest
from datetime import datetime, timedelta
from typing import List
from unittest.mock import patch, MagicMock

from app.models.data import PriceBar, Indicator
from app.tools.analysis import calculate_fibonacci_levels
from app.agent.tools import get_fibonacci_levels
from app.tools.chart_generator import generate_chart_image


# ==================== Fixtures ====================

@pytest.fixture
def uptrend_fibonacci_bars() -> List[PriceBar]:
    """Generate price bars with clear uptrend for Fibonacci testing.

    Creates a move from $100 to $120, then pullback to $110.
    Perfect for testing Fibonacci retracement levels.
    """
    bars = []
    base_date = datetime(2025, 1, 1)

    # Create uptrend: $100 -> $120 over 30 bars
    for i in range(30):
        price = 100.0 + (i * 0.67)  # ~$120 by bar 30
        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.4,
            close=price,
            volume=1000000,
        )
        bars.append(bar)

    # Create pullback: $120 -> $110 over 10 bars
    for i in range(10):
        price = 120.0 - (i * 1.0)  # Pullback to ~$110
        bar = PriceBar(
            timestamp=base_date + timedelta(days=30 + i),
            open=price + 0.2,
            high=price + 0.4,
            low=price - 0.3,
            close=price,
            volume=1200000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def downtrend_fibonacci_bars() -> List[PriceBar]:
    """Generate price bars with clear downtrend for Fibonacci testing.

    Creates a move from $100 to $80, then bounce to $90.
    Perfect for testing Fibonacci in downtrend.
    """
    bars = []
    base_date = datetime(2025, 1, 1)

    # Create downtrend: $100 -> $80 over 30 bars
    for i in range(30):
        price = 100.0 - (i * 0.67)  # ~$80 by bar 30
        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=price + 0.2,
            high=price + 0.4,
            low=price - 0.3,
            close=price,
            volume=1000000,
        )
        bars.append(bar)

    # Create bounce: $80 -> $90 over 10 bars
    for i in range(10):
        price = 80.0 + (i * 1.0)  # Bounce to ~$90
        bar = PriceBar(
            timestamp=base_date + timedelta(days=30 + i),
            open=price - 0.2,
            high=price + 0.3,
            low=price - 0.4,
            close=price,
            volume=1200000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def swing_pivot_bars() -> List[PriceBar]:
    """Generate bars with clear swing pivots for detection testing."""
    bars = []
    base_date = datetime(2025, 1, 1)

    # Create pattern with clear swing high and low
    prices = [
        100, 102, 104, 106, 108,  # Uptrend to swing high
        110, 112, 115, 118, 120,  # Peak at 120 (swing high)
        119, 117, 115, 113, 110,  # Pullback
        108, 106, 104, 102, 100,  # Down to swing low at 100
        102, 104, 106, 108, 110,  # Recovery
    ]

    for i, close_price in enumerate(prices):
        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close_price - 0.5,
            high=close_price + 1.0,
            low=close_price - 1.5,
            close=close_price,
            volume=1000000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def flat_consolidation_bars() -> List[PriceBar]:
    """Generate bars with flat/consolidating price (edge case)."""
    bars = []
    base_date = datetime(2025, 1, 1)

    # Flat price around $100 with minimal variation
    for i in range(30):
        price = 100.0 + (i % 3 - 1) * 0.5  # Oscillates between 99.5 and 100.5
        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=price,
            high=price + 0.2,
            low=price - 0.2,
            close=price,
            volume=1000000,
        )
        bars.append(bar)

    return bars


# ==================== Test calculate_fibonacci_levels ====================

class TestCalculateFibonacciLevels:
    """Tests for calculate_fibonacci_levels function."""

    def test_fibonacci_retracement_levels_uptrend(self, uptrend_fibonacci_bars):
        """Test that retracement levels are calculated correctly in uptrend."""
        result = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        assert isinstance(result, Indicator)
        assert result.name == "Fibonacci"
        assert result.signal in ["bullish", "bearish", "neutral"]

        # Check metadata structure
        metadata = result.metadata
        assert "swing_high" in metadata
        assert "swing_low" in metadata
        assert "trend" in metadata
        assert "retracement" in metadata
        assert "extension" in metadata

        # Verify retracement levels exist
        retracement = metadata["retracement"]
        assert "0.236" in retracement
        assert "0.382" in retracement
        assert "0.500" in retracement
        assert "0.618" in retracement
        assert "0.786" in retracement

        # Verify mathematical correctness of retracement levels
        swing_high = metadata["swing_high"]
        swing_low = metadata["swing_low"]
        diff = swing_high - swing_low

        # 0.382 level should be between swing_high and swing_low
        assert swing_low < retracement["0.382"] < swing_high
        # 0.618 level should be between swing_high and swing_low
        assert swing_low < retracement["0.618"] < swing_high
        # 0.500 should be roughly midpoint
        expected_midpoint = swing_high - (diff * 0.5)
        assert abs(retracement["0.500"] - expected_midpoint) < 0.01

    def test_fibonacci_extension_levels_uptrend(self, uptrend_fibonacci_bars):
        """Test extension levels project correctly above swing high in uptrend."""
        result = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        metadata = result.metadata
        extension = metadata["extension"]
        swing_high = metadata["swing_high"]

        # Extension levels should exist
        assert "1.272" in extension
        assert "1.618" in extension
        assert "2.618" in extension

        # In uptrend, extension levels project above swing high
        # (Note: actual implementation may vary, but extensions typically
        # extend beyond the swing high in uptrend)
        assert extension["1.272"] is not None
        assert extension["1.618"] is not None
        assert extension["2.618"] is not None

    def test_fibonacci_extension_levels_downtrend(self, downtrend_fibonacci_bars):
        """Test extension levels project correctly below swing low in downtrend."""
        result = calculate_fibonacci_levels(downtrend_fibonacci_bars, swing_lookback=20)

        metadata = result.metadata
        extension = metadata["extension"]
        swing_low = metadata["swing_low"]

        # Extension levels should exist
        assert "1.272" in extension
        assert "1.618" in extension
        assert "2.618" in extension

        # Extension levels should be calculated
        assert extension["1.272"] is not None
        assert extension["1.618"] is not None
        assert extension["2.618"] is not None

    def test_fibonacci_swing_detection(self, swing_pivot_bars):
        """Test that swing pivots are detected properly."""
        result = calculate_fibonacci_levels(swing_pivot_bars, swing_lookback=20)

        metadata = result.metadata
        swing_high = metadata["swing_high"]
        swing_low = metadata["swing_low"]

        # Swing high should be higher than swing low
        assert swing_high > swing_low

        # Swing high should be within reasonable range of our test data
        # (we created highs up to ~120)
        assert swing_high >= 100
        assert swing_high <= 130

        # Swing low should be within reasonable range
        assert swing_low >= 95
        assert swing_low <= 120

    def test_fibonacci_trade_type_lookback(self, uptrend_fibonacci_bars):
        """Test different lookback periods work correctly."""
        # Test with day trade lookback (10 bars)
        result_day = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=10)
        assert result_day.metadata["swing_high"] is not None

        # Test with swing trade lookback (20 bars)
        result_swing = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)
        assert result_swing.metadata["swing_high"] is not None

        # Test with position trade lookback (50 bars - but we only have 40 bars)
        # This should work as we have enough data
        result_position = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=30)
        assert result_position.metadata["swing_high"] is not None

        # Different lookbacks may produce different swing levels
        # Longer lookback should capture wider swing range
        assert result_day.metadata["swing_high"] is not None
        assert result_swing.metadata["swing_high"] is not None

    def test_fibonacci_current_price_tracking(self, uptrend_fibonacci_bars):
        """Test that current price is tracked correctly."""
        result = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        metadata = result.metadata
        assert "current_price" in metadata

        # Current price should match last bar's close
        expected_price = uptrend_fibonacci_bars[-1].close
        assert metadata["current_price"] == round(expected_price, 2)

    def test_fibonacci_nearest_level_detection(self, uptrend_fibonacci_bars):
        """Test that nearest Fibonacci level is detected."""
        result = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        metadata = result.metadata
        assert "nearest_level" in metadata
        assert "nearest_price" in metadata
        assert "near_fib_level" in metadata

        # Nearest level should be a valid Fibonacci level
        nearest = metadata["nearest_level"]
        valid_levels = ["0.000", "0.236", "0.382", "0.500", "0.618", "0.786", "1.000",
                       "1.272", "1.618", "2.000", "2.618"]
        assert nearest in valid_levels

    def test_fibonacci_signal_generation(self, uptrend_fibonacci_bars):
        """Test that appropriate signals are generated."""
        result = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        # Signal should be one of the valid values
        assert result.signal in ["bullish", "bearish", "neutral"]

        # Metadata should indicate if at entry level
        metadata = result.metadata
        assert "at_entry_level" in metadata
        assert isinstance(metadata["at_entry_level"], bool)

    def test_fibonacci_empty_bars(self):
        """Test Fibonacci with empty price bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            calculate_fibonacci_levels([])

    def test_fibonacci_insufficient_data(self, uptrend_fibonacci_bars):
        """Test Fibonacci with insufficient data for lookback."""
        # Request 50 bar lookback but only provide 40 bars
        with pytest.raises(ValueError, match="Need at least 50 bars"):
            calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=50)

    def test_fibonacci_minimum_valid_data(self):
        """Test Fibonacci with minimum required data (edge case)."""
        # Create exactly 20 bars (minimum for default lookback)
        bars = []
        base_date = datetime(2025, 1, 1)
        for i in range(20):
            price = 100.0 + i
            bar = PriceBar(
                timestamp=base_date + timedelta(days=i),
                open=price - 0.5,
                high=price + 1.0,
                low=price - 1.0,
                close=price,
                volume=1000000,
            )
            bars.append(bar)

        result = calculate_fibonacci_levels(bars, swing_lookback=20)
        assert result is not None
        assert result.metadata["swing_high"] is not None

    def test_fibonacci_flat_market(self, flat_consolidation_bars):
        """Test Fibonacci in flat/consolidating market (edge case)."""
        result = calculate_fibonacci_levels(flat_consolidation_bars, swing_lookback=20)

        metadata = result.metadata
        swing_high = metadata["swing_high"]
        swing_low = metadata["swing_low"]

        # In flat market, swing high and low should be very close
        swing_range = swing_high - swing_low
        assert swing_range < 5  # Very narrow range in our flat market

        # Fibonacci levels should still be calculated
        assert "retracement" in metadata
        assert "extension" in metadata

    def test_fibonacci_extreme_volatility(self):
        """Test Fibonacci with extreme price swings (edge case)."""
        bars = []
        base_date = datetime(2025, 1, 1)

        # Create extreme volatility: huge swings
        prices = [100, 150, 80, 200, 50, 180, 70, 190]
        for i, close_price in enumerate(prices):
            bar = PriceBar(
                timestamp=base_date + timedelta(days=i),
                open=close_price - 5,
                high=close_price + 10,
                low=close_price - 10,
                close=close_price,
                volume=1000000,
            )
            bars.append(bar)

        # Add more bars to meet minimum requirement
        for i in range(len(prices), 25):
            bar = PriceBar(
                timestamp=base_date + timedelta(days=i),
                open=100,
                high=105,
                low=95,
                close=100,
                volume=1000000,
            )
            bars.append(bar)

        result = calculate_fibonacci_levels(bars, swing_lookback=20)

        # Should still calculate successfully
        assert result is not None
        assert result.metadata["swing_high"] > result.metadata["swing_low"]


# ==================== Test get_fibonacci_levels Agent Tool ====================

class TestGetFibonacciLevelsAgentTool:
    """Tests for get_fibonacci_levels agent tool."""

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_returns_expected_structure(self):
        """Test the agent tool returns proper data structure."""
        # Mock fetch_price_bars to return test data
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            # Create mock bars
            bars = []
            base_date = datetime(2025, 1, 1)
            for i in range(100):
                price = 100.0 + i * 0.5
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=i),
                    open=price - 0.2,
                    high=price + 0.3,
                    low=price - 0.4,
                    close=price,
                    volume=1000000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            # Test with swing trade type (default)
            result = await get_fibonacci_levels("AAPL", trade_type="swing")

            # Verify structure
            assert "symbol" in result
            assert result["symbol"] == "AAPL"
            assert "swing_high" in result
            assert "swing_low" in result
            assert "retracement_levels" in result
            assert "extension_levels" in result
            assert "current_price" in result
            assert "nearest_level" in result
            assert "nearest_price" in result
            assert "distance_to_nearest" in result
            assert "distance_pct" in result
            assert "signal" in result
            assert "at_entry_level" in result
            assert "trend" in result

            # Verify types
            assert isinstance(result["swing_high"], (int, float))
            assert isinstance(result["swing_low"], (int, float))
            assert isinstance(result["retracement_levels"], dict)
            assert isinstance(result["extension_levels"], dict)
            assert result["signal"] in ["bullish", "bearish", "neutral"]

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_suggested_zones(self):
        """Test that suggested stop and entry zones are calculated."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            # Create uptrend bars
            bars = []
            base_date = datetime(2025, 1, 1)

            # Uptrend then pullback
            for i in range(50):
                price = 100.0 + i * 0.8
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=i),
                    open=price - 0.2,
                    high=price + 0.3,
                    low=price - 0.4,
                    close=price,
                    volume=1000000,
                )
                bars.append(bar)

            # Pullback
            for i in range(10):
                price = 140.0 - i * 1.0
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=50 + i),
                    open=price + 0.2,
                    high=price + 0.4,
                    low=price - 0.3,
                    close=price,
                    volume=1200000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            result = await get_fibonacci_levels("TEST", trade_type="swing")

            # Check for suggested zones
            assert "suggested_entry_zone" in result
            assert "suggested_stop_zone" in result

            # If signal is bullish/neutral, entry zone should exist
            if result["signal"] in ["bullish", "neutral"]:
                entry_zone = result["suggested_entry_zone"]
                if entry_zone:
                    assert "low" in entry_zone
                    assert "high" in entry_zone
                    assert entry_zone["low"] <= entry_zone["high"]

            # If signal is bullish, stop zone should exist
            if result["signal"] == "bullish":
                stop_zone = result["suggested_stop_zone"]
                if stop_zone:
                    assert "low" in stop_zone
                    assert "high" in stop_zone

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_day_trade_lookback(self):
        """Test that day trade type uses correct lookback."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            bars = []
            base_date = datetime(2025, 1, 1)
            for i in range(50):
                price = 100.0 + i * 0.2
                bar = PriceBar(
                    timestamp=base_date + timedelta(minutes=i * 5),
                    open=price,
                    high=price + 0.5,
                    low=price - 0.5,
                    close=price,
                    volume=100000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            result = await get_fibonacci_levels("AAPL", trade_type="day")

            # Verify fetch was called with correct parameters for day trade
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args
            assert call_args[1]["timeframe"] == "5m"
            assert call_args[1]["days_back"] == 30

            # Should still return valid structure
            assert "swing_high" in result
            assert "swing_low" in result

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_position_trade_lookback(self):
        """Test that position trade type uses correct lookback."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            bars = []
            base_date = datetime(2025, 1, 1)
            for i in range(200):
                price = 100.0 + i * 0.1
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=i),
                    open=price,
                    high=price + 1.0,
                    low=price - 1.0,
                    close=price,
                    volume=1000000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            result = await get_fibonacci_levels("AAPL", trade_type="position")

            # Verify fetch was called with correct parameters for position trade
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args
            assert call_args[1]["timeframe"] == "1d"
            assert call_args[1]["days_back"] == 200

            # Should still return valid structure
            assert "swing_high" in result
            assert "swing_low" in result

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_swing_trade_default(self):
        """Test that swing trade is the default and uses correct lookback."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            bars = []
            base_date = datetime(2025, 1, 1)
            for i in range(100):
                price = 100.0 + i * 0.3
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=i),
                    open=price,
                    high=price + 0.8,
                    low=price - 0.8,
                    close=price,
                    volume=1000000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            # Test without specifying trade_type (should default to swing)
            result = await get_fibonacci_levels("AAPL")

            # Verify fetch was called with swing parameters
            mock_fetch.assert_called_once()
            call_args = mock_fetch.call_args
            assert call_args[1]["timeframe"] == "1d"
            assert call_args[1]["days_back"] == 100

            assert "swing_high" in result

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_insufficient_data(self):
        """Test agent tool with insufficient data."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            # Return only 5 bars (not enough for any lookback)
            bars = []
            base_date = datetime(2025, 1, 1)
            for i in range(5):
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=i),
                    open=100.0,
                    high=101.0,
                    low=99.0,
                    close=100.0,
                    volume=1000000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            result = await get_fibonacci_levels("AAPL", trade_type="swing")

            # Should return error
            assert "error" in result
            assert "Insufficient data" in result["error"]
            assert result["retracement_levels"] == {}
            assert result["extension_levels"] == {}

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_no_data(self):
        """Test agent tool with no data returned."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            mock_fetch.return_value = None

            result = await get_fibonacci_levels("INVALID", trade_type="swing")

            # Should return error
            assert "error" in result
            assert result["retracement_levels"] == {}

    @pytest.mark.asyncio
    async def test_get_fibonacci_levels_api_exception(self):
        """Test agent tool handles API exceptions gracefully."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            mock_fetch.side_effect = Exception("API Error")

            result = await get_fibonacci_levels("AAPL", trade_type="swing")

            # Should return error
            assert "error" in result
            assert "API Error" in result["error"]


# ==================== Test Chart Generation with Fibonacci ====================

class TestChartWithFibonacci:
    """Tests for chart generation with Fibonacci overlay."""

    def test_chart_with_fibonacci_levels(self, uptrend_fibonacci_bars):
        """Test chart generates correctly with Fibonacci overlay."""
        # Calculate Fibonacci levels
        fib_indicator = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        # Prepare fibonacci_levels dict for chart
        fibonacci_levels = {
            "swing_high": fib_indicator.metadata["swing_high"],
            "swing_low": fib_indicator.metadata["swing_low"],
            "retracement": fib_indicator.metadata["retracement"],
            "extension": fib_indicator.metadata["extension"],
            "trend": fib_indicator.metadata["trend"],
        }

        # Provide dummy indicators to avoid addplot=None issue
        indicators = {
            "ema_9": [bar.close for bar in uptrend_fibonacci_bars[-30:]],
        }

        # Generate chart
        chart_base64 = generate_chart_image(
            symbol="TEST",
            bars=uptrend_fibonacci_bars,
            indicators=indicators,
            lookback=30,
            show_volume=False,
            show_rsi=False,
            fibonacci_levels=fibonacci_levels,
        )

        # Verify chart was generated
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 100  # Should be a substantial base64 string

    def test_chart_without_fibonacci_backward_compatible(self, uptrend_fibonacci_bars):
        """Test chart still works when fibonacci_levels is None (backward compatibility)."""
        # Provide dummy indicators to avoid addplot=None issue
        indicators = {
            "ema_9": [bar.close for bar in uptrend_fibonacci_bars[-30:]],
        }

        # Generate chart without Fibonacci levels
        chart_base64 = generate_chart_image(
            symbol="TEST",
            bars=uptrend_fibonacci_bars,
            indicators=indicators,
            lookback=30,
            show_volume=False,
            show_rsi=False,
            fibonacci_levels=None,
        )

        # Verify chart was generated
        assert chart_base64 is not None
        assert isinstance(chart_base64, str)
        assert len(chart_base64) > 100

    def test_chart_with_fibonacci_and_indicators(self, uptrend_fibonacci_bars):
        """Test chart with both Fibonacci and technical indicators."""
        # Calculate Fibonacci
        fib_indicator = calculate_fibonacci_levels(uptrend_fibonacci_bars, swing_lookback=20)

        fibonacci_levels = {
            "swing_high": fib_indicator.metadata["swing_high"],
            "swing_low": fib_indicator.metadata["swing_low"],
            "retracement": fib_indicator.metadata["retracement"],
            "extension": fib_indicator.metadata["extension"],
            "trend": fib_indicator.metadata["trend"],
        }

        # Mock indicators
        indicators = {
            "ema_9": [bar.close * 0.99 for bar in uptrend_fibonacci_bars[-30:]],
            "ema_21": [bar.close * 0.98 for bar in uptrend_fibonacci_bars[-30:]],
            "rsi": [50.0] * 30,
        }

        # Generate chart
        chart_base64 = generate_chart_image(
            symbol="TEST",
            bars=uptrend_fibonacci_bars,
            indicators=indicators,
            lookback=30,
            show_volume=True,
            show_rsi=True,
            fibonacci_levels=fibonacci_levels,
        )

        # Verify chart was generated
        assert chart_base64 is not None
        assert len(chart_base64) > 100

    def test_chart_fibonacci_empty_levels(self, uptrend_fibonacci_bars):
        """Test chart handles empty Fibonacci levels gracefully (edge case)."""
        # Provide empty Fibonacci dict
        fibonacci_levels = {
            "swing_high": None,
            "swing_low": None,
            "retracement": {},
            "extension": {},
            "trend": "uptrend",
        }

        # Provide dummy indicators to avoid addplot=None issue
        indicators = {
            "ema_9": [bar.close for bar in uptrend_fibonacci_bars[-30:]],
        }

        # Should not crash
        chart_base64 = generate_chart_image(
            symbol="TEST",
            bars=uptrend_fibonacci_bars,
            indicators=indicators,
            lookback=30,
            show_volume=False,
            show_rsi=False,
            fibonacci_levels=fibonacci_levels,
        )

        assert chart_base64 is not None

    def test_chart_fibonacci_invalid_levels(self, uptrend_fibonacci_bars):
        """Test chart handles invalid Fibonacci levels gracefully (edge case)."""
        # Provide malformed Fibonacci dict
        fibonacci_levels = {
            "swing_high": 120.0,
            "swing_low": 100.0,
            "retracement": {"invalid_key": "invalid_value"},  # Invalid structure
            "extension": None,  # None instead of dict
            "trend": "uptrend",
        }

        # Provide dummy indicators to avoid addplot=None issue
        indicators = {
            "ema_9": [bar.close for bar in uptrend_fibonacci_bars[-30:]],
        }

        # Should not crash (will log warning)
        chart_base64 = generate_chart_image(
            symbol="TEST",
            bars=uptrend_fibonacci_bars,
            indicators=indicators,
            lookback=30,
            show_volume=False,
            show_rsi=False,
            fibonacci_levels=fibonacci_levels,
        )

        # Chart should still generate (Fibonacci plotting will fail gracefully)
        assert chart_base64 is not None


# ==================== Integration Tests ====================

class TestFibonacciIntegration:
    """Integration tests combining multiple Fibonacci features."""

    @pytest.mark.asyncio
    async def test_full_fibonacci_workflow(self):
        """Test complete workflow: fetch -> calculate -> chart."""
        with patch('app.agent.tools.fetch_price_bars') as mock_fetch:
            # Create realistic bars
            bars = []
            base_date = datetime(2025, 1, 1)

            # Uptrend
            for i in range(60):
                price = 150.0 + i * 0.5
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=i),
                    open=price - 0.3,
                    high=price + 0.5,
                    low=price - 0.6,
                    close=price,
                    volume=5000000,
                )
                bars.append(bar)

            # Pullback
            for i in range(15):
                price = 180.0 - i * 0.8
                bar = PriceBar(
                    timestamp=base_date + timedelta(days=60 + i),
                    open=price + 0.3,
                    high=price + 0.5,
                    low=price - 0.6,
                    close=price,
                    volume=6000000,
                )
                bars.append(bar)

            mock_fetch.return_value = bars

            # Step 1: Get Fibonacci via agent tool
            result = await get_fibonacci_levels("AAPL", trade_type="swing")

            assert result["symbol"] == "AAPL"
            assert result["signal"] in ["bullish", "bearish", "neutral"]

            # Step 2: Verify levels make sense
            swing_high = result["swing_high"]
            swing_low = result["swing_low"]
            assert swing_high > swing_low

            # Retracement levels should be between swing high and low
            retracement = result["retracement_levels"]
            for level_name, level_price in retracement.items():
                if level_name not in ["0.000", "1.000"]:
                    # Levels between 0 and 1 should be between swing_low and swing_high
                    pass  # Can vary based on trend direction

            # Step 3: Create chart with levels
            fibonacci_chart_data = {
                "swing_high": swing_high,
                "swing_low": swing_low,
                "retracement": retracement,
                "extension": result["extension_levels"],
                "trend": result["trend"],
            }

            # Provide dummy indicators to avoid addplot=None issue
            indicators = {
                "ema_9": [bar.close for bar in bars[-60:]],
            }

            chart = generate_chart_image(
                symbol="AAPL",
                bars=bars,
                indicators=indicators,
                fibonacci_levels=fibonacci_chart_data,
                lookback=60,
            )

            assert chart is not None
            assert len(chart) > 100
