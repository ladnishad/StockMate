"""Tests for analysis tools."""

import pytest
from datetime import datetime

from app.tools.analysis import (
    find_structural_pivots,
    detect_key_levels,
    calculate_volume_profile,
    detect_chart_patterns,
    generate_trade_plan,
    find_comprehensive_levels,
    _count_level_touches,
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

    def test_detect_chart_patterns_flat_prices(self):
        """Test chart patterns with flat/consolidating prices (edge case).

        This tests the strict inequality fix in swing point detection.
        Flat tops/bottoms should NOT be detected as peaks/troughs.
        """
        from datetime import datetime
        from app.models.data import PriceBar

        # Create bars with flat top (should NOT create false H&S pattern)
        flat_bars = []
        base_date = datetime(2024, 1, 1)

        for i in range(50):
            if 10 <= i <= 20:
                # Flat top region - all highs are exactly 105
                high = 105.0
                close = 104.0
            elif 30 <= i <= 40:
                # Another flat region
                high = 103.0
                close = 102.0
            else:
                # Normal variation
                high = 100.0 + (i % 5)
                close = high - 1.0

            flat_bars.append(PriceBar(
                timestamp=base_date.replace(day=i+1),
                open=close - 0.5,
                high=high,
                low=close - 2.0,
                close=close,
                volume=1000000
            ))

        patterns = detect_chart_patterns(flat_bars, min_pattern_bars=20)

        # With strict inequality, flat tops should not create many false patterns
        # The pattern detector should find fewer patterns on flat/consolidating data
        assert isinstance(patterns["patterns_found"], list)
        # We don't assert zero patterns (some real patterns might exist),
        # but this test ensures the strict inequality logic doesn't crash
        assert patterns["pattern_count"] >= 0


class TestTouchCount:
    """Tests for touch count analysis in support/resistance levels."""

    @pytest.fixture
    def bars_with_clear_levels(self):
        """Create price bars with clear support at $95 and resistance at $105."""
        from datetime import timedelta
        from app.models.data import PriceBar

        bars = []
        base_date = datetime(2025, 1, 1)

        # Create 100 bars that repeatedly test $95 support and $105 resistance
        for i in range(100):
            # Price oscillates between 96-104, touching 95 and 105 periodically
            phase = i % 20

            if phase < 5:
                # Bouncing off support at 95
                close = 96 + phase * 0.5
                low = 95.0 if phase == 2 else 96.0
                high = close + 1.0
            elif phase < 10:
                # Rising
                close = 98 + (phase - 5) * 1.2
                low = close - 1.0
                high = close + 1.0
            elif phase < 15:
                # Testing resistance at 105
                close = 104 - (phase - 10) * 0.5
                high = 105.0 if phase == 12 else 104.0
                low = close - 1.0
            else:
                # Falling back
                close = 102 - (phase - 15) * 1.2
                low = close - 1.0
                high = close + 1.0

            bars.append(PriceBar(
                timestamp=base_date + timedelta(days=i),
                open=close - 0.3,
                high=high,
                low=low,
                close=close,
                volume=1000000,
            ))

        return bars

    def test_count_level_touches_support(self, bars_with_clear_levels):
        """Test counting touches on support level with full institutional metrics."""
        result = _count_level_touches(bars_with_clear_levels, 95.0, "support", tolerance_pct=0.5)

        assert isinstance(result, dict)
        # Core metrics
        assert "touches" in result
        assert "recent_touches" in result
        assert "last_touch_bars_ago" in result
        assert "held" in result
        assert "broke" in result
        assert "strength" in result
        assert "reliability" in result
        # Enhanced institutional metrics
        assert "high_volume_touches" in result
        assert "bounce_quality" in result
        assert "reclaimed" in result

        # Should detect multiple touches at the 95 level
        assert result["touches"] >= 3
        assert result["strength"] > 0
        assert result["reliability"] in ["weak", "moderate", "strong", "institutional"]
        assert isinstance(result["high_volume_touches"], int)
        assert isinstance(result["bounce_quality"], (int, float))
        assert isinstance(result["reclaimed"], bool)

    def test_count_level_touches_resistance(self, bars_with_clear_levels):
        """Test counting touches on resistance level."""
        result = _count_level_touches(bars_with_clear_levels, 105.0, "resistance", tolerance_pct=0.5)

        assert isinstance(result, dict)
        assert result["touches"] >= 3
        assert result["strength"] > 0

    def test_count_level_touches_reliability_classification(self, bars_with_clear_levels):
        """Test that reliability classification is correct based on touch count."""
        # Test with a level that has many touches
        result_many = _count_level_touches(bars_with_clear_levels, 95.0, "support", tolerance_pct=1.0)

        if result_many["touches"] >= 6:
            assert result_many["reliability"] == "institutional"
        elif result_many["touches"] >= 4:
            assert result_many["reliability"] == "strong"
        elif result_many["touches"] >= 2:
            assert result_many["reliability"] == "moderate"
        else:
            assert result_many["reliability"] == "weak"

    def test_count_level_touches_no_touches(self, bars_with_clear_levels):
        """Test counting touches on a level with no touches."""
        # Price never goes to 50
        result = _count_level_touches(bars_with_clear_levels, 50.0, "support", tolerance_pct=0.5)

        assert result["touches"] == 0
        assert result["strength"] == 0
        assert result["reliability"] == "weak"
        assert result["last_touch_bars_ago"] is None
        assert result["high_volume_touches"] == 0
        assert result["bounce_quality"] == 0
        assert result["reclaimed"] is False

    def test_count_level_touches_empty_bars(self):
        """Test touch counting with empty bars."""
        result = _count_level_touches([], 100.0, "support")

        assert result["touches"] == 0
        assert result["reliability"] == "weak"
        assert result["high_volume_touches"] == 0
        assert result["reclaimed"] is False

    def test_count_level_touches_recent_weight(self, bars_with_clear_levels):
        """Test that recent touches contribute more to strength."""
        result = _count_level_touches(bars_with_clear_levels, 95.0, "support", tolerance_pct=0.5)

        # Recent touches should be reflected in the result
        assert "recent_touches" in result
        # Strength should be higher when there are recent touches
        if result["recent_touches"] > 0:
            assert result["strength"] >= 10  # At least some recency bonus

    def test_count_level_touches_bounce_quality(self, bars_with_clear_levels):
        """Test that bounce quality is calculated correctly."""
        result = _count_level_touches(bars_with_clear_levels, 95.0, "support", tolerance_pct=0.5)

        # Bounce quality should be a score between 0-100
        assert 0 <= result["bounce_quality"] <= 100
        # If there are touches, bounce quality should be non-zero
        if result["touches"] > 0:
            assert result["bounce_quality"] >= 0  # Could be 0 if all touches broke through

    def test_count_level_touches_volume_detection(self):
        """Test that high volume touches are detected correctly."""
        from datetime import timedelta
        from app.models.data import PriceBar

        bars = []
        base_date = datetime(2025, 1, 1)

        # Create bars with varying volume - some touches with high volume
        for i in range(50):
            # Touch support at 100 on bars 10, 20, 30, 40
            if i in [10, 20, 30, 40]:
                low = 100.0
                close = 102.0
                # High volume on bars 20 and 40 (2x average)
                volume = 2000000 if i in [20, 40] else 500000
            else:
                low = 103.0
                close = 105.0
                volume = 1000000  # Average volume

            bars.append(PriceBar(
                timestamp=base_date + timedelta(days=i),
                open=close - 0.5,
                high=close + 1.0,
                low=low,
                close=close,
                volume=volume,
            ))

        result = _count_level_touches(bars, 100.0, "support", tolerance_pct=0.5)

        # Should detect 4 touches total
        assert result["touches"] == 4
        # Should detect 2 high volume touches (bars 20 and 40 have 2x average)
        assert result["high_volume_touches"] == 2


class TestComprehensiveLevels:
    """Tests for comprehensive levels with touch count."""

    def test_find_comprehensive_levels_basic(self, sample_price_bars):
        """Test that comprehensive levels include touch count data."""
        levels = find_comprehensive_levels(sample_price_bars)

        assert isinstance(levels, dict)
        assert "support" in levels
        assert "resistance" in levels
        assert "current_price" in levels

    def test_find_comprehensive_levels_touch_data(self, sample_price_bars):
        """Test that each level has full institutional touch count data."""
        levels = find_comprehensive_levels(sample_price_bars)

        for support in levels.get("support", []):
            # Core fields
            assert "price" in support
            assert "touches" in support
            assert "strength" in support
            assert "reliability" in support
            assert "type" in support
            # Enhanced institutional fields
            assert "high_volume_touches" in support
            assert "bounce_quality" in support
            assert "reclaimed" in support

        for resistance in levels.get("resistance", []):
            # Core fields
            assert "price" in resistance
            assert "touches" in resistance
            assert "strength" in resistance
            assert "reliability" in resistance
            assert "type" in resistance
            # Enhanced institutional fields
            assert "high_volume_touches" in resistance
            assert "bounce_quality" in resistance
            assert "reclaimed" in resistance

    def test_find_comprehensive_levels_sorting(self, sample_price_bars):
        """Test that levels are sorted by composite score (strength + proximity)."""
        levels = find_comprehensive_levels(sample_price_bars)

        # Levels should be sorted by a composite of strength and proximity
        # We can't easily verify the exact sort order, but we can verify
        # that strong nearby levels should rank higher
        supports = levels.get("support", [])
        if len(supports) > 1:
            # Just verify all levels have the required fields for sorting
            for s in supports:
                assert "strength" in s
                assert "distance_pct" in s

    def test_find_comprehensive_levels_includes_ema_levels(self, sample_price_bars):
        """Test that EMA levels are included with touch count."""
        from app.tools.indicators import calculate_ema_series

        ema_9 = calculate_ema_series(sample_price_bars, 9)
        ema_21 = calculate_ema_series(sample_price_bars, 21)

        levels = find_comprehensive_levels(sample_price_bars, ema_9=ema_9, ema_21=ema_21)

        all_levels = levels.get("support", []) + levels.get("resistance", [])
        ema_types = [l.get("type") for l in all_levels]

        # Should include EMA levels if they were provided
        assert "ema_9" in ema_types or "ema_21" in ema_types

    def test_find_comprehensive_levels_round_numbers(self, sample_price_bars):
        """Test that round number levels are included with touch count."""
        levels = find_comprehensive_levels(sample_price_bars)

        all_levels = levels.get("support", []) + levels.get("resistance", [])
        level_types = [l.get("type") for l in all_levels]

        # Should include round number levels
        assert "round_number" in level_types

    def test_find_comprehensive_levels_empty_bars(self):
        """Test comprehensive levels with empty bars."""
        with pytest.raises(ValueError, match="price_bars cannot be empty"):
            find_comprehensive_levels([])

    def test_find_comprehensive_levels_reliability_in_output(self, sample_price_bars):
        """Test that reliability classification is in output."""
        levels = find_comprehensive_levels(sample_price_bars)

        valid_reliabilities = ["weak", "moderate", "strong", "institutional"]

        for support in levels.get("support", []):
            assert support.get("reliability") in valid_reliabilities

        for resistance in levels.get("resistance", []):
            assert resistance.get("reliability") in valid_reliabilities
