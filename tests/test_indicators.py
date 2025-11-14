"""Tests for technical indicator calculation tools."""

import pytest
from app.tools.indicators import (
    calculate_vwap,
    calculate_ema,
    calculate_rsi,
    detect_divergences,
)


class TestVWAP:
    """Tests for VWAP calculation."""

    def test_calculate_vwap_basic(self, sample_price_bars):
        """Test basic VWAP calculation."""
        vwap = calculate_vwap(sample_price_bars)

        assert vwap.name == "VWAP"
        assert vwap.value > 0
        assert vwap.signal in ["bullish", "bearish", "neutral"]
        assert "current_price" in vwap.metadata
        assert "price_to_vwap_ratio" in vwap.metadata

    def test_calculate_vwap_uptrend(self, uptrend_price_bars):
        """Test VWAP in uptrend (should be bullish)."""
        vwap = calculate_vwap(uptrend_price_bars)

        # In uptrend, price should be above VWAP
        assert vwap.signal == "bullish"
        assert vwap.metadata["price_to_vwap_ratio"] > 1.0

    def test_calculate_vwap_empty_bars(self):
        """Test VWAP with empty price bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            calculate_vwap([])

    def test_calculate_vwap_zero_volume(self):
        """Test VWAP with all zero volume bars (edge case)."""
        from datetime import datetime
        from app.models.data import PriceBar

        zero_volume_bars = [
            PriceBar(
                timestamp=datetime(2024, 1, i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=0  # Zero volume
            )
            for i in range(1, 25)
        ]

        # Should raise ValueError for zero total volume
        with pytest.raises(ValueError, match="Total volume is zero"):
            calculate_vwap(zero_volume_bars)


class TestEMA:
    """Tests for EMA calculation."""

    def test_calculate_ema_basic(self, sample_price_bars):
        """Test basic EMA calculation."""
        ema = calculate_ema(sample_price_bars, period=20)

        assert ema.name == "EMA_20"
        assert ema.value > 0
        assert ema.signal in ["bullish", "bearish", "neutral"]
        assert ema.metadata["period"] == 20
        assert "current_price" in ema.metadata
        assert "slope_5bar_pct" in ema.metadata

    def test_calculate_ema_uptrend(self, uptrend_price_bars):
        """Test EMA in uptrend (should be bullish)."""
        ema = calculate_ema(uptrend_price_bars, period=20)

        # In uptrend, price should be above EMA
        assert ema.signal == "bullish"

    def test_calculate_ema_different_periods(self, sample_price_bars):
        """Test EMA with different periods."""
        ema_9 = calculate_ema(sample_price_bars, period=9)
        ema_50 = calculate_ema(sample_price_bars, period=50)

        assert ema_9.name == "EMA_9"
        assert ema_50.name == "EMA_50"
        assert ema_9.metadata["period"] == 9
        assert ema_50.metadata["period"] == 50

    def test_calculate_ema_insufficient_data(self, sample_price_bars):
        """Test EMA with insufficient data."""
        with pytest.raises(ValueError, match="Need at least"):
            calculate_ema(sample_price_bars[:10], period=20)

    def test_calculate_ema_invalid_period(self, sample_price_bars):
        """Test EMA with invalid period."""
        with pytest.raises(ValueError, match="period must be >= 1"):
            calculate_ema(sample_price_bars, period=0)


class TestRSI:
    """Tests for RSI calculation."""

    def test_calculate_rsi_basic(self, sample_price_bars):
        """Test basic RSI calculation."""
        rsi = calculate_rsi(sample_price_bars, period=14)

        assert rsi.name == "RSI_14"
        assert 0 <= rsi.value <= 100
        assert rsi.signal in ["bullish", "bearish", "neutral"]
        assert rsi.metadata["period"] == 14
        assert "interpretation" in rsi.metadata

    def test_calculate_rsi_uptrend(self, uptrend_price_bars):
        """Test RSI in uptrend (should be bullish)."""
        rsi = calculate_rsi(uptrend_price_bars, period=14)

        # In strong uptrend, RSI should be elevated
        assert rsi.value > 50
        assert rsi.signal == "bullish"

    def test_calculate_rsi_downtrend(self, downtrend_price_bars):
        """Test RSI in downtrend (should be bearish)."""
        rsi = calculate_rsi(downtrend_price_bars, period=14)

        # In downtrend, RSI should be low
        assert rsi.value < 50
        assert rsi.signal == "bearish"

    def test_calculate_rsi_different_periods(self, sample_price_bars):
        """Test RSI with different periods."""
        rsi_7 = calculate_rsi(sample_price_bars, period=7)
        rsi_21 = calculate_rsi(sample_price_bars, period=21)

        assert rsi_7.name == "RSI_7"
        assert rsi_21.name == "RSI_21"

    def test_calculate_rsi_insufficient_data(self, sample_price_bars):
        """Test RSI with insufficient data."""
        with pytest.raises(ValueError, match="Need at least"):
            calculate_rsi(sample_price_bars[:10], period=14)

    def test_calculate_rsi_invalid_period(self, sample_price_bars):
        """Test RSI with invalid period."""
        with pytest.raises(ValueError, match="period must be >= 2"):
            calculate_rsi(sample_price_bars, period=1)


class TestDivergenceDetection:
    """Tests for divergence detection."""

    def test_detect_divergences_rsi_basic(self, sample_price_bars):
        """Test basic RSI divergence detection."""
        # Need enough data for divergence detection
        if len(sample_price_bars) >= 30:
            divergence = detect_divergences(
                sample_price_bars,
                indicator_type="rsi",
                lookback_period=14,
                swing_detection_window=5
            )

            assert divergence.name == "Divergence_RSI"
            assert divergence.value >= 0  # Divergence count
            assert divergence.signal in ["bullish", "bearish", "neutral"]
            assert "regular_bullish" in divergence.metadata
            assert "regular_bearish" in divergence.metadata
            assert "divergence_count" in divergence.metadata
            assert "strength" in divergence.metadata

    def test_detect_divergences_macd_basic(self, sample_price_bars):
        """Test basic MACD divergence detection."""
        if len(sample_price_bars) >= 40:
            divergence = detect_divergences(
                sample_price_bars,
                indicator_type="macd",
                lookback_period=14,
                swing_detection_window=5
            )

            assert divergence.name == "Divergence_MACD"
            assert divergence.value >= 0
            assert divergence.signal in ["bullish", "bearish", "neutral"]

    def test_detect_divergences_insufficient_data(self):
        """Test divergence detection with insufficient data."""
        from app.models.data import PriceBar
        from datetime import datetime

        # Create minimal price bars (not enough for divergence)
        bars = [
            PriceBar(
                timestamp=datetime(2024, 1, i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000000
            )
            for i in range(1, 20)
        ]

        with pytest.raises(ValueError, match="Need at least"):
            detect_divergences(bars, indicator_type="rsi")

    def test_detect_divergences_invalid_indicator(self, sample_price_bars):
        """Test divergence detection with invalid indicator type."""
        with pytest.raises(ValueError, match="indicator_type must be"):
            detect_divergences(sample_price_bars, indicator_type="invalid")
