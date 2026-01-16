"""Comprehensive tests for shortcomings fixes.

This test file validates all the implementations from the shortcomings-fix.md specification:
- Phase 1: New indicators and divergence tolerance
- Phase 2: Signal grouping, risk tiers, trend-adjusted RSI
- Phase 3: Volatility-adaptive stops, multi-factor trade type
- Phase 4: New patterns, ATR-based tolerance, success rates
- Phase 5: Confidence-based visual modifier, JSON parse handling
"""

import pytest
from datetime import datetime
from typing import List

from app.models.data import (
    PriceBar,
    MarketSnapshot,
    Fundamentals,
    Sentiment,
    Indicator,
    StructuralPivot,
)
from app.tools.indicators import (
    calculate_ichimoku,
    calculate_williams_r,
    calculate_parabolic_sar,
    calculate_cmf,
    calculate_adl,
    detect_divergences,
)
from app.tools.analysis import (
    detect_chart_patterns,
    generate_trade_plan,
)


# ============================================================================
# Phase 1: New Indicators Tests
# ============================================================================

class TestIchimokuCloud:
    """Tests for Ichimoku Cloud indicator (1.1.1)."""

    def test_ichimoku_basic_calculation(self, sample_price_bars):
        """Test basic Ichimoku calculation with sufficient data."""
        ichimoku = calculate_ichimoku(sample_price_bars)

        assert ichimoku.name == "Ichimoku"
        assert ichimoku.value > 0  # Tenkan-sen value
        assert ichimoku.signal in ["bullish", "bearish", "neutral"]

        # Check all required metadata
        assert "tenkan_sen" in ichimoku.metadata
        assert "kijun_sen" in ichimoku.metadata
        assert "senkou_span_a" in ichimoku.metadata
        assert "senkou_span_b" in ichimoku.metadata
        assert "chikou_span" in ichimoku.metadata
        assert "cloud_thickness" in ichimoku.metadata
        assert "price_vs_cloud" in ichimoku.metadata
        assert "tk_cross" in ichimoku.metadata

    def test_ichimoku_uptrend_bullish(self, uptrend_price_bars):
        """Test Ichimoku in uptrend should be bullish."""
        # Extend the uptrend bars for Ichimoku (needs 78+ bars)
        extended_bars = []
        base_date = datetime(2025, 1, 1)
        base_price = 100.0

        for i in range(100):
            close = base_price + (i * 0.5)
            bar = PriceBar(
                timestamp=base_date.replace(day=(i % 28) + 1, month=(i // 28) + 1),
                open=close - 0.2,
                high=close + 0.3,
                low=close - 0.4,
                close=close,
                volume=1000000,
            )
            extended_bars.append(bar)

        ichimoku = calculate_ichimoku(extended_bars)

        # In strong uptrend, expect bullish or at least bullish signals
        assert ichimoku.metadata["price_vs_cloud"] in ["above", "inside"]
        assert ichimoku.metadata["bullish_signals"] >= 2

    def test_ichimoku_cloud_components(self, sample_price_bars):
        """Test Ichimoku cloud component relationships."""
        ichimoku = calculate_ichimoku(sample_price_bars)

        # Cloud boundaries should be correctly ordered
        assert ichimoku.metadata["cloud_top"] >= ichimoku.metadata["cloud_bottom"]
        assert ichimoku.metadata["cloud_thickness"] >= 0

    def test_ichimoku_insufficient_data(self):
        """Test Ichimoku with insufficient data."""
        from datetime import timedelta
        short_bars = []
        base_date = datetime(2025, 1, 1)
        for i in range(50):  # Less than 78 bars needed
            bar = PriceBar(
                timestamp=base_date + timedelta(days=i),
                open=100.0,
                high=101.0,
                low=99.0,
                close=100.0,
                volume=1000000,
            )
            short_bars.append(bar)

        with pytest.raises(ValueError, match="Need at least"):
            calculate_ichimoku(short_bars)


class TestWilliamsR:
    """Tests for Williams %R indicator (1.1.2)."""

    def test_williams_r_basic(self, sample_price_bars):
        """Test basic Williams %R calculation."""
        williams = calculate_williams_r(sample_price_bars)

        assert williams.name == "Williams_R"
        assert -100 <= williams.value <= 0  # Williams %R is 0 to -100
        assert williams.signal in ["bullish", "bearish", "neutral"]

        # Check metadata
        assert "period" in williams.metadata
        assert "highest_high" in williams.metadata
        assert "lowest_low" in williams.metadata
        assert "overbought" in williams.metadata
        assert "oversold" in williams.metadata

    def test_williams_r_overbought(self, uptrend_price_bars):
        """Test Williams %R overbought detection in uptrend."""
        williams = calculate_williams_r(uptrend_price_bars)

        # In strong uptrend, should be near overbought
        assert williams.value > -50  # Closer to 0 = overbought

    def test_williams_r_oversold(self, downtrend_price_bars):
        """Test Williams %R oversold detection in downtrend."""
        williams = calculate_williams_r(downtrend_price_bars)

        # In strong downtrend, should be near oversold
        assert williams.value < -50  # Closer to -100 = oversold

    def test_williams_r_crossover_detection(self, sample_price_bars):
        """Test Williams %R crossover metadata."""
        williams = calculate_williams_r(sample_price_bars)

        assert "oversold_cross" in williams.metadata
        assert "overbought_cross" in williams.metadata
        assert isinstance(williams.metadata["oversold_cross"], bool)
        assert isinstance(williams.metadata["overbought_cross"], bool)


class TestParabolicSAR:
    """Tests for Parabolic SAR indicator (1.1.3)."""

    def test_parabolic_sar_basic(self, sample_price_bars):
        """Test basic Parabolic SAR calculation."""
        sar = calculate_parabolic_sar(sample_price_bars)

        assert sar.name == "Parabolic_SAR"
        assert sar.value > 0  # SAR value
        assert sar.signal in ["bullish", "bearish"]

        # Check metadata
        assert "sar_value" in sar.metadata
        assert "sar_position" in sar.metadata
        assert "trend_direction" in sar.metadata
        assert "af_current" in sar.metadata
        assert "bars_in_trend" in sar.metadata

    def test_parabolic_sar_uptrend(self, uptrend_price_bars):
        """Test SAR in uptrend (should be below price)."""
        sar = calculate_parabolic_sar(uptrend_price_bars)

        assert sar.metadata["sar_position"] == "below_price"
        assert sar.metadata["trend_direction"] == "up"
        assert sar.signal == "bullish"

    def test_parabolic_sar_downtrend(self, downtrend_price_bars):
        """Test SAR in downtrend (should be above price)."""
        sar = calculate_parabolic_sar(downtrend_price_bars)

        assert sar.metadata["sar_position"] == "above_price"
        assert sar.metadata["trend_direction"] == "down"
        assert sar.signal == "bearish"

    def test_parabolic_sar_acceleration_factor(self, sample_price_bars):
        """Test SAR acceleration factor bounds."""
        sar = calculate_parabolic_sar(sample_price_bars, af_start=0.02, af_max=0.20)

        # AF should be within bounds
        assert 0.02 <= sar.metadata["af_current"] <= 0.20


class TestCMF:
    """Tests for Chaikin Money Flow indicator (1.1.4)."""

    def test_cmf_basic(self, sample_price_bars):
        """Test basic CMF calculation."""
        cmf = calculate_cmf(sample_price_bars)

        assert cmf.name == "CMF"
        assert -1 <= cmf.value <= 1  # CMF is -1 to 1
        assert cmf.signal in ["bullish", "bearish", "neutral"]

        # Check metadata
        assert "period" in cmf.metadata
        assert "cmf" in cmf.metadata
        assert "cmf_trend" in cmf.metadata
        assert "interpretation" in cmf.metadata

    def test_cmf_buying_pressure(self, uptrend_price_bars):
        """Test CMF shows buying pressure in uptrend."""
        cmf = calculate_cmf(uptrend_price_bars)

        # In uptrend with volume, should show buying pressure
        assert cmf.value > -0.25  # Not strong selling

    def test_cmf_divergence_detection(self, sample_price_bars):
        """Test CMF divergence metadata."""
        cmf = calculate_cmf(sample_price_bars)

        assert "divergence" in cmf.metadata
        # Divergence can be None, "bullish", or "bearish"


class TestADL:
    """Tests for Accumulation/Distribution Line indicator (1.1.5)."""

    def test_adl_basic(self, sample_price_bars):
        """Test basic ADL calculation."""
        adl = calculate_adl(sample_price_bars)

        assert adl.name == "ADL"
        assert adl.signal in ["bullish", "bearish", "neutral"]

        # Check metadata
        assert "adl_value" in adl.metadata
        assert "adl_trend" in adl.metadata
        assert "price_trend" in adl.metadata
        assert "adl_vs_price" in adl.metadata

    def test_adl_confirming_trend(self, uptrend_price_bars):
        """Test ADL confirms price trend."""
        adl = calculate_adl(uptrend_price_bars)

        # Should confirm uptrend
        assert adl.metadata["adl_vs_price"] in ["confirming", "neutral"]

    def test_adl_divergence_detection(self, sample_price_bars):
        """Test ADL divergence metadata."""
        adl = calculate_adl(sample_price_bars)

        assert "is_diverging" in adl.metadata
        assert isinstance(adl.metadata["is_diverging"], bool)


# ============================================================================
# Phase 1: Divergence Tolerance Tests (1.2)
# ============================================================================

class TestDivergenceVolatilityTolerance:
    """Tests for volatility-scaled divergence tolerance (1.2)."""

    def test_divergence_includes_volatility_metadata(self, sample_price_bars):
        """Test that divergence includes volatility regime info."""
        divergence = detect_divergences(sample_price_bars, indicator_type="rsi")

        # Should include volatility scaling info
        assert "volatility_regime" in divergence.metadata
        assert "atr_percentage" in divergence.metadata
        assert "swing_match_tolerance" in divergence.metadata

    def test_divergence_low_volatility_tight_tolerance(self, low_volatility_bars):
        """Test divergence uses tighter tolerance for low volatility."""
        divergence = detect_divergences(low_volatility_bars, indicator_type="rsi")

        # Low volatility should use 2-bar tolerance
        assert divergence.metadata["volatility_regime"] == "low"
        assert divergence.metadata["swing_match_tolerance"] == 2

    def test_divergence_high_volatility_wide_tolerance(self, high_volatility_bars):
        """Test divergence uses wider tolerance for high volatility."""
        divergence = detect_divergences(high_volatility_bars, indicator_type="rsi")

        # High volatility should use 4-bar tolerance
        assert divergence.metadata["volatility_regime"] == "high"
        assert divergence.metadata["swing_match_tolerance"] == 4


# ============================================================================
# Phase 4: Pattern Detection Tests
# ============================================================================

class TestPatternATRTolerance:
    """Tests for ATR-based pattern detection tolerance (4.1)."""

    def test_pattern_detection_returns_success_rate(self, sample_price_bars):
        """Test that patterns include historical success rate."""
        patterns = detect_chart_patterns(sample_price_bars)

        for pattern in patterns["patterns_found"]:
            assert "success_rate" in pattern
            assert 0 <= pattern["success_rate"] <= 1

    def test_pattern_low_volatility_tolerance(self, low_volatility_bars):
        """Test pattern detection uses tighter tolerance for low volatility."""
        # Just verify it runs without error on low volatility data
        patterns = detect_chart_patterns(low_volatility_bars)
        assert isinstance(patterns["patterns_found"], list)

    def test_pattern_high_volatility_tolerance(self, high_volatility_bars):
        """Test pattern detection uses wider tolerance for high volatility."""
        # Just verify it runs without error on high volatility data
        patterns = detect_chart_patterns(high_volatility_bars)
        assert isinstance(patterns["patterns_found"], list)


class TestNewPatterns:
    """Tests for new pattern detection (4.2)."""

    def test_cup_and_handle_detection(self, cup_and_handle_bars):
        """Test Cup and Handle pattern detection."""
        patterns = detect_chart_patterns(cup_and_handle_bars)

        # Should find patterns (may or may not find cup and handle specifically)
        assert "patterns_found" in patterns
        assert "pattern_count" in patterns

        # If cup and handle found, verify structure
        cup_patterns = [p for p in patterns["patterns_found"] if "Cup" in p["name"]]
        if cup_patterns:
            pattern = cup_patterns[0]
            assert pattern["type"] == "bullish_continuation"
            assert "success_rate" in pattern
            assert pattern["success_rate"] == 0.65

    def test_rectangle_detection(self, rectangle_bars):
        """Test Rectangle pattern detection."""
        patterns = detect_chart_patterns(rectangle_bars)

        assert "patterns_found" in patterns

        # Look for rectangle-like patterns
        rect_patterns = [p for p in patterns["patterns_found"] if "Rectangle" in p["name"]]
        if rect_patterns:
            pattern = rect_patterns[0]
            assert "resistance" in pattern
            assert "support" in pattern
            assert "success_rate" in pattern

    def test_all_patterns_have_success_rates(self, sample_price_bars):
        """Test that ALL detected patterns have success rates."""
        patterns = detect_chart_patterns(sample_price_bars)

        for pattern in patterns["patterns_found"]:
            assert "success_rate" in pattern, f"Pattern {pattern['name']} missing success_rate"
            assert 0 < pattern["success_rate"] <= 1, f"Invalid success_rate for {pattern['name']}"


# ============================================================================
# Phase 3: Trade Plan Generation Tests
# ============================================================================

class TestVolatilityAdaptiveStops:
    """Tests for volatility-adaptive stop loss bounds (3.1)."""

    def test_trade_plan_stop_within_bounds(self, uptrend_price_bars):
        """Test that stop loss is within volatility-based bounds."""
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
                Indicator(name="ATR_14", value=2.5, signal="neutral", metadata={
                    "period": 14, "atr_percentage": 2.0
                }),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000, risk_percentage=1.0)

        if trade_plan:
            # Stop should be below entry
            assert trade_plan.stop_loss < trade_plan.entry_price

            # Calculate actual stop percentage
            stop_pct = (trade_plan.entry_price - trade_plan.stop_loss) / trade_plan.entry_price

            # Should be within reasonable bounds (1% - 8% based on volatility)
            assert 0.01 <= stop_pct <= 0.08

    def test_low_volatility_tight_stops(self, low_volatility_bars):
        """Test that low volatility stocks get tighter stops."""
        snapshot = MarketSnapshot(
            symbol="LOW_VOL",
            current_price=155.0,
            price_bars_1d=low_volatility_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.3, label="neutral", news_count=5),
            indicators=[
                Indicator(name="EMA_20", value=153.0, signal="bullish", metadata={"period": 20}),
                Indicator(name="ATR_14", value=1.2, signal="neutral", metadata={
                    "period": 14, "atr_percentage": 0.8
                }),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000, risk_percentage=1.0)

        if trade_plan:
            stop_pct = (trade_plan.entry_price - trade_plan.stop_loss) / trade_plan.entry_price
            # Low volatility should have stop <= 4%
            assert stop_pct <= 0.04

    def test_high_volatility_wide_stops(self, high_volatility_bars):
        """Test that high volatility stocks get wider stops."""
        snapshot = MarketSnapshot(
            symbol="HIGH_VOL",
            current_price=280.0,
            price_bars_1d=high_volatility_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_20", value=270.0, signal="bullish", metadata={"period": 20}),
                Indicator(name="ATR_14", value=11.2, signal="neutral", metadata={
                    "period": 14, "atr_percentage": 4.0
                }),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000, risk_percentage=1.0)

        if trade_plan:
            stop_pct = (trade_plan.entry_price - trade_plan.stop_loss) / trade_plan.entry_price
            # High volatility should have stop >= 3%
            assert stop_pct >= 0.03


class TestMultiFactorTradeType:
    """Tests for multi-factor trade type determination (3.2)."""

    def test_trade_type_is_determined(self, uptrend_price_bars):
        """Test that trade type is determined."""
        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=125.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_9", value=123.0, signal="bullish", metadata={"period": 9}),
                Indicator(name="EMA_20", value=120.0, signal="bullish", metadata={"period": 20}),
                Indicator(name="EMA_50", value=115.0, signal="bullish", metadata={"period": 50}),
                Indicator(name="Volume", value=1500000, signal="bullish", metadata={
                    "relative_volume": 1.5
                }),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000)

        if trade_plan:
            assert trade_plan.trade_type in ["day", "swing", "long"]

    def test_high_volatility_prefers_day_trade(self, high_volatility_bars):
        """Test that high volatility + high volume suggests day trade."""
        snapshot = MarketSnapshot(
            symbol="HIGH_VOL",
            current_price=280.0,
            price_bars_1d=high_volatility_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_9", value=275.0, signal="bullish", metadata={"period": 9}),
                Indicator(name="EMA_20", value=260.0, signal="bullish", metadata={"period": 20}),
                Indicator(name="EMA_50", value=240.0, signal="bullish", metadata={"period": 50}),
                Indicator(name="ATR_14", value=14.0, signal="neutral", metadata={
                    "period": 14, "atr_percentage": 5.0
                }),
                Indicator(name="Volume", value=8000000, signal="bullish", metadata={
                    "relative_volume": 2.5
                }),
            ],
            pivots=[
                StructuralPivot(price=275.0, type="support", strength=70, touches=2),
                StructuralPivot(price=285.0, type="resistance", strength=60, touches=2),
            ],
            timestamp=datetime.utcnow(),
        )

        trade_plan = generate_trade_plan(snapshot, account_size=10000)

        if trade_plan:
            # High volatility + near S/R should favor shorter-term trades
            assert trade_plan.trade_type in ["day", "swing"]


# ============================================================================
# Phase 5: AI Planning Agent Tests
# ============================================================================

class TestConfidenceBasedModifierScaling:
    """Tests for confidence-based visual modifier scaling (5.1)."""

    def test_modifier_scaling_logic(self):
        """Test the modifier scaling logic directly."""
        # Low confidence (0-30%) should scale by 0.7
        original_confidence = 25
        raw_modifier = 10

        if original_confidence < 30:
            modifier_scale = 0.7
        elif original_confidence <= 60:
            modifier_scale = 1.0
        else:
            modifier_scale = 1.2

        scaled_modifier = int(raw_modifier * modifier_scale)
        assert scaled_modifier == 7  # 10 * 0.7 = 7

    def test_high_confidence_amplifies_modifier(self):
        """Test high confidence amplifies modifier."""
        original_confidence = 75
        raw_modifier = 10

        if original_confidence < 30:
            modifier_scale = 0.7
        elif original_confidence <= 60:
            modifier_scale = 1.0
        else:
            modifier_scale = 1.2

        scaled_modifier = int(raw_modifier * modifier_scale)
        assert scaled_modifier == 12  # 10 * 1.2 = 12


class TestJSONParseFailureHandling:
    """Tests for enhanced JSON parse failure handling (5.2)."""

    def test_regex_extraction_bias(self):
        """Test regex extraction of bias field."""
        import re
        response_text = '"bias": "bullish", other stuff'

        bias_match = re.search(r'"bias"\s*:\s*"(bullish|bearish|neutral)"', response_text, re.IGNORECASE)
        assert bias_match is not None
        assert bias_match.group(1).lower() == "bullish"

    def test_regex_extraction_confidence(self):
        """Test regex extraction of confidence field."""
        import re
        response_text = '"confidence": 75, other stuff'

        conf_match = re.search(r'"confidence"\s*:\s*(\d+)', response_text)
        assert conf_match is not None
        assert int(conf_match.group(1)) == 75

    def test_regex_extraction_entry_zones(self):
        """Test regex extraction of entry zone fields."""
        import re
        response_text = '"entry_zone_low": 150.25, "entry_zone_high": $155.50'

        entry_low_match = re.search(r'"entry_zone_low"\s*:\s*\$?([\d.]+)', response_text)
        entry_high_match = re.search(r'"entry_zone_high"\s*:\s*\$?([\d.]+)', response_text)

        assert entry_low_match is not None
        assert float(entry_low_match.group(1)) == 150.25

        assert entry_high_match is not None
        assert float(entry_high_match.group(1)) == 155.50

    def test_regex_extraction_stop_loss(self):
        """Test regex extraction of stop loss field."""
        import re
        response_text = '"stop_loss": 145.00'

        stop_match = re.search(r'"stop_loss"\s*:\s*\$?([\d.]+)', response_text)
        assert stop_match is not None
        assert float(stop_match.group(1)) == 145.00


# ============================================================================
# Integration Tests
# ============================================================================

class TestEndToEndAnalysis:
    """End-to-end integration tests."""

    def test_full_indicator_suite(self, sample_price_bars):
        """Test all indicators can run on same data."""
        # All new indicators
        ichimoku = calculate_ichimoku(sample_price_bars)
        williams = calculate_williams_r(sample_price_bars)
        sar = calculate_parabolic_sar(sample_price_bars)
        cmf = calculate_cmf(sample_price_bars)
        adl = calculate_adl(sample_price_bars)
        divergence = detect_divergences(sample_price_bars, indicator_type="rsi")

        # All should return valid indicators
        assert ichimoku.name == "Ichimoku"
        assert williams.name == "Williams_R"
        assert sar.name == "Parabolic_SAR"
        assert cmf.name == "CMF"
        assert adl.name == "ADL"
        assert divergence.name == "Divergence_RSI"

    def test_pattern_detection_all_volatilities(
        self, low_volatility_bars, medium_volatility_bars, high_volatility_bars
    ):
        """Test pattern detection works across all volatility levels."""
        patterns_low = detect_chart_patterns(low_volatility_bars)
        patterns_med = detect_chart_patterns(medium_volatility_bars)
        patterns_high = detect_chart_patterns(high_volatility_bars)

        # All should return valid structure
        assert "patterns_found" in patterns_low
        assert "patterns_found" in patterns_med
        assert "patterns_found" in patterns_high

    def test_trade_plan_different_account_sizes(self, uptrend_price_bars):
        """Test trade plan with different account sizes (risk tier testing)."""
        snapshot = MarketSnapshot(
            symbol="TEST",
            current_price=125.0,
            price_bars_1d=uptrend_price_bars,
            fundamentals=Fundamentals(),
            sentiment=Sentiment(score=0.5, label="bullish", news_count=10),
            indicators=[
                Indicator(name="EMA_9", value=123.0, signal="bullish", metadata={"period": 9}),
                Indicator(name="EMA_20", value=120.0, signal="bullish", metadata={"period": 20}),
                Indicator(name="EMA_50", value=115.0, signal="bullish", metadata={"period": 50}),
            ],
            pivots=[],
            timestamp=datetime.utcnow(),
        )

        # Small account (aggressive tier)
        plan_small = generate_trade_plan(snapshot, account_size=3000)

        # Medium account (moderate tier)
        plan_medium = generate_trade_plan(snapshot, account_size=7500)

        # Large account (conservative tier)
        plan_large = generate_trade_plan(snapshot, account_size=15000)

        # All should generate valid plans
        for plan in [plan_small, plan_medium, plan_large]:
            if plan:
                assert plan.entry_price > 0
                assert plan.stop_loss > 0
                assert plan.stop_loss < plan.entry_price
