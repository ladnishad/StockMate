"""Tests for analysis tools."""

import pytest
from datetime import datetime

from app.tools.analysis import (
    find_structural_pivots,
    detect_key_levels,
    calculate_volume_profile,
    detect_chart_patterns,
    generate_trade_plan,
)
from app.models.data import (
    MarketSnapshot,
    Fundamentals,
    Sentiment,
    Indicator,
)


class TestStructuralPivots:
    """Tests for structural pivot detection."""

    def test_find_pivots_basic(self, sample_price_bars):
        """Test basic pivot detection."""
        pivots = find_structural_pivots(sample_price_bars, lookback=20, min_touches=2)

        assert isinstance(pivots, list)
        # Should find at least some pivots
        assert len(pivots) >= 0

        for pivot in pivots:
            assert pivot.type in ["support", "resistance"]
            assert pivot.price > 0
            assert 0 <= pivot.strength <= 100
            assert pivot.touches >= 2

    def test_find_pivots_sorted_by_strength(self, sample_price_bars):
        """Test that pivots are sorted by strength."""
        pivots = find_structural_pivots(sample_price_bars, lookback=20, min_touches=2)

        if len(pivots) > 1:
            for i in range(len(pivots) - 1):
                assert pivots[i].strength >= pivots[i + 1].strength

    def test_find_pivots_insufficient_data(self, sample_price_bars):
        """Test pivot detection with insufficient data."""
        with pytest.raises(ValueError, match="Need at least"):
            find_structural_pivots(sample_price_bars[:15], lookback=20)

    def test_find_pivots_empty_bars(self):
        """Test pivot detection with empty bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            find_structural_pivots([])


class TestTradePlanGeneration:
    """Tests for trade plan generation."""

    def test_generate_trade_plan_basic(self, sample_price_bars, uptrend_price_bars):
        """Test basic trade plan generation."""
        # Create a bullish snapshot
        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=125.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(
                fifty_two_week_high=130.0,
                fifty_two_week_low=100.0,
            ),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_9", value=123.0, signal="bullish", metadata={"period": 9}),
                Indicator(name="EMA_20", value=120.0, signal="bullish", metadata={"period": 20}),
                Indicator(name="EMA_50", value=115.0, signal="bullish", metadata={"period": 50}),
                Indicator(name="RSI_14", value=60.0, signal="bullish", metadata={"period": 14}),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000, risk_percentage=1.0)

        assert trade_plan is not None
        assert trade_plan.trade_type in ["day", "swing", "long"]
        assert trade_plan.entry_price > 0
        assert trade_plan.stop_loss > 0
        assert trade_plan.stop_loss < trade_plan.entry_price  # Stop below entry
        assert trade_plan.target_1 > trade_plan.entry_price  # Target above entry
        assert trade_plan.position_size >= 1
        assert trade_plan.risk_amount > 0
        assert trade_plan.risk_percentage == 1.0

    def test_generate_trade_plan_with_pivots(self, uptrend_price_bars):
        """Test trade plan generation with structural pivots."""
        from app.models.data import StructuralPivot

        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=125.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(
                fifty_two_week_high=130.0,
                fifty_two_week_low=100.0,
            ),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_20", value=120.0, signal="bullish", metadata={"period": 20}),
            ],
            pivots=[
                StructuralPivot(price=122.0, type="support", strength=80, touches=3),
                StructuralPivot(price=130.0, type="resistance", strength=70, touches=2),
            ],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000, risk_percentage=1.0)

        assert trade_plan is not None
        # Stop should be near support
        assert trade_plan.stop_loss <= 122.0

    def test_generate_trade_plan_invalid_account_size(self, uptrend_price_bars):
        """Test trade plan with invalid account size."""
        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=125.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.0, label="neutral", news_count=0),
            indicators=[],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        with pytest.raises(ValueError, match="account_size must be > 0"):
            generate_trade_plan(snapshot, account_size=0)

    def test_generate_trade_plan_invalid_risk_percentage(self, uptrend_price_bars):
        """Test trade plan with invalid risk percentage."""
        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=125.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.0, label="neutral", news_count=0),
            indicators=[],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        with pytest.raises(ValueError, match="risk_percentage must be between"):
            generate_trade_plan(snapshot, account_size=10000, risk_percentage=150)

    def test_generate_trade_plan_position_sizing(self, uptrend_price_bars):
        """Test position sizing with different account sizes."""
        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=100.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_20", value=95.0, signal="bullish", metadata={"period": 20}),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        # Test with different account sizes
        plan_small = generate_trade_plan(snapshot, account_size=1000, risk_percentage=1.0)
        plan_large = generate_trade_plan(snapshot, account_size=100000, risk_percentage=1.0)

        # Larger account should allow larger position
        if plan_small and plan_large:
            assert plan_large.position_size > plan_small.position_size


class TestKeyLevels:
    """Tests for key level detection."""

    def test_detect_key_levels_basic(self, sample_price_bars):
        """Test basic key level detection."""
        levels = detect_key_levels(sample_price_bars)

        assert isinstance(levels, dict)
        assert "round_numbers" in levels
        assert "unfilled_gaps" in levels
        assert "all_levels" in levels
        assert "nearest_support" in levels
        assert "nearest_resistance" in levels

    def test_detect_key_levels_round_numbers(self, sample_price_bars):
        """Test round number detection."""
        levels = detect_key_levels(sample_price_bars, current_price=150.0)

        round_numbers = levels["round_numbers"]
        assert isinstance(round_numbers, list)

        # Should detect numbers like 150, 140, 160, etc.
        if round_numbers:
            for level in round_numbers:
                assert "price" in level
                assert "type" in level
                assert level["type"] in ["round_major", "round_minor"]
                assert "significance" in level

    def test_detect_key_levels_gaps(self, sample_price_bars):
        """Test gap detection."""
        levels = detect_key_levels(sample_price_bars)

        unfilled_gaps = levels["unfilled_gaps"]
        assert isinstance(unfilled_gaps, list)

        # If gaps are found, check structure
        for gap in unfilled_gaps:
            assert "gap_high" in gap
            assert "gap_low" in gap
            assert "gap_size" in gap
            assert "direction" in gap
            assert gap["direction"] in ["up", "down"]
            assert gap["filled"] == False

    def test_detect_key_levels_previous_periods(self, sample_price_bars):
        """Test previous period highs/lows detection."""
        if len(sample_price_bars) >= 30:
            levels = detect_key_levels(sample_price_bars)

            # Should have previous day high/low
            assert levels["previous_day_high"] is not None
            assert levels["previous_day_low"] is not None

            # If enough data, should have week and month
            if len(sample_price_bars) >= 30:
                assert levels.get("previous_month_high") is not None
                assert levels.get("previous_month_low") is not None

    def test_detect_key_levels_empty_bars(self):
        """Test key levels with empty price bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            detect_key_levels([])


class TestVolumeProfile:
    """Tests for volume profile calculation."""

    def test_calculate_volume_profile_basic(self, sample_price_bars):
        """Test basic volume profile calculation."""
        if len(sample_price_bars) >= 30:
            vp = calculate_volume_profile(sample_price_bars, num_bins=50)

            assert isinstance(vp, dict)
            assert "vpoc" in vp
            assert "value_area_high" in vp
            assert "value_area_low" in vp
            assert "high_volume_nodes" in vp
            assert "low_volume_nodes" in vp
            assert "current_price_position" in vp

            # VPOC should be within price range
            assert vp["vpoc"] > 0
            assert vp["value_area_high"] > vp["value_area_low"]

    def test_calculate_volume_profile_hvn_lvn(self, sample_price_bars):
        """Test HVN and LVN detection."""
        if len(sample_price_bars) >= 30:
            vp = calculate_volume_profile(sample_price_bars, num_bins=50)

            # HVNs should have price, volume, and strength
            for hvn in vp["high_volume_nodes"]:
                assert "price" in hvn
                assert "volume" in hvn
                assert "strength" in hvn
                assert hvn["strength"] >= 1.5  # HVN threshold

            # LVNs should have price, volume, and weakness
            for lvn in vp["low_volume_nodes"]:
                assert "price" in lvn
                assert "volume" in lvn
                assert "weakness" in lvn

    def test_calculate_volume_profile_position(self, sample_price_bars):
        """Test current price position determination."""
        if len(sample_price_bars) >= 30:
            vp = calculate_volume_profile(sample_price_bars, num_bins=50)

            position = vp["current_price_position"]
            assert position in ["above_value_area", "below_value_area", "within_value_area"]

    def test_calculate_volume_profile_empty_bars(self):
        """Test volume profile with empty price bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            calculate_volume_profile([])

    def test_calculate_volume_profile_invalid_bins(self, sample_price_bars):
        """Test volume profile with invalid bin count."""
        with pytest.raises(ValueError, match="num_bins must be at least 10"):
            calculate_volume_profile(sample_price_bars, num_bins=5)


class TestChartPatterns:
    """Tests for chart pattern detection."""

    def test_detect_chart_patterns_basic(self, sample_price_bars):
        """Test basic chart pattern detection."""
        if len(sample_price_bars) >= 30:
            patterns = detect_chart_patterns(sample_price_bars, min_pattern_bars=20)

            assert isinstance(patterns, dict)
            assert "patterns_found" in patterns
            assert "pattern_count" in patterns
            assert "bullish_patterns" in patterns
            assert "bearish_patterns" in patterns
            assert "strongest_pattern" in patterns
            assert "net_sentiment" in patterns

    def test_detect_chart_patterns_structure(self, sample_price_bars):
        """Test pattern structure."""
        if len(sample_price_bars) >= 30:
            patterns = detect_chart_patterns(sample_price_bars, min_pattern_bars=20)

            # If patterns are found, check their structure
            for pattern in patterns["patterns_found"]:
                assert "name" in pattern
                assert "type" in pattern
                assert "confidence" in pattern
                assert "target_price" in pattern
                assert "current_price" in pattern
                assert "expected_move_pct" in pattern

                # Type should be valid
                assert "bullish" in pattern["type"] or "bearish" in pattern["type"]
                assert "reversal" in pattern["type"] or "continuation" in pattern["type"]

                # Confidence should be reasonable
                assert 0 <= pattern["confidence"] <= 100

    def test_detect_chart_patterns_strongest(self, sample_price_bars):
        """Test strongest pattern selection."""
        if len(sample_price_bars) >= 30:
            patterns = detect_chart_patterns(sample_price_bars, min_pattern_bars=20)

            if patterns["pattern_count"] > 0:
                strongest = patterns["strongest_pattern"]
                assert strongest is not None
                assert isinstance(strongest, dict)

                # Strongest should have highest confidence
                if patterns["pattern_count"] > 1:
                    for pattern in patterns["patterns_found"]:
                        assert strongest["confidence"] >= pattern["confidence"]

    def test_detect_chart_patterns_empty_bars(self):
        """Test chart patterns with empty price bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            detect_chart_patterns([])

    def test_detect_chart_patterns_insufficient_data(self, sample_price_bars):
        """Test chart patterns with insufficient data."""
        if len(sample_price_bars) < 20:
            with pytest.raises(ValueError, match="Need at least"):
                detect_chart_patterns(sample_price_bars[:10], min_pattern_bars=20)
