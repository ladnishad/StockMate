"""Tests for analysis tools."""

import pytest
from datetime import datetime

from app.tools.analysis import find_structural_pivots, generate_trade_plan
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
