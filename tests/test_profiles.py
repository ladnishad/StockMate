"""Tests for trader profile system."""

import pytest

from app.models.profile import (
    TraderProfile,
    TraderProfileType,
    TimeframeConfig,
    RiskConfig,
    TargetConfig,
    ScoringWeights,
)
from app.models.profile_presets import (
    get_profile,
    list_profiles,
    PROFILE_REGISTRY,
    DAY_TRADER,
    SWING_TRADER,
    POSITION_TRADER,
    LONG_TERM_INVESTOR,
)


class TestProfileRegistry:
    """Tests for profile registry and lookup."""

    def test_all_profiles_exist(self):
        """Test that all 4 profiles are registered."""
        profiles = list_profiles()
        assert len(profiles) == 4
        assert "day_trader" in profiles
        assert "swing_trader" in profiles
        assert "position_trader" in profiles
        assert "long_term_investor" in profiles

    def test_get_profile_valid(self):
        """Test getting a valid profile."""
        profile = get_profile("swing_trader")
        assert profile.name == "Swing Trader"
        assert profile.profile_type == TraderProfileType.SWING_TRADER

    def test_get_profile_invalid(self):
        """Test getting an invalid profile raises error."""
        with pytest.raises(ValueError, match="Unknown profile type"):
            get_profile("invalid_profile")

    def test_registry_contains_all_types(self):
        """Test that registry contains all profile types."""
        for profile_type in TraderProfileType:
            assert profile_type.value in PROFILE_REGISTRY


class TestDayTraderProfile:
    """Tests for Day Trader profile configuration."""

    def test_profile_type(self):
        """Test profile type is correct."""
        assert DAY_TRADER.profile_type == TraderProfileType.DAY_TRADER

    def test_timeframes(self):
        """Test day trading timeframes."""
        assert DAY_TRADER.timeframes.primary == "15m"
        assert DAY_TRADER.timeframes.confirmation == "1h"
        assert DAY_TRADER.timeframes.entry == "5m"

    def test_allowed_trade_types(self):
        """Test only day trades allowed."""
        assert DAY_TRADER.allowed_trade_types == ["day"]

    def test_risk_config(self):
        """Test day trader risk config (tight stops)."""
        assert DAY_TRADER.risk.stop_method == "atr"
        assert DAY_TRADER.risk.atr_multiplier == 1.5  # Tighter stops
        assert DAY_TRADER.risk.risk_percentage == 0.5

    def test_vwap_volume_emphasis(self):
        """Test that VWAP and volume are heavily weighted."""
        assert DAY_TRADER.weights.vwap == 15.0
        assert DAY_TRADER.weights.volume == 15.0

    def test_high_confidence_threshold(self):
        """Test day trader has highest confidence threshold."""
        assert DAY_TRADER.buy_confidence_threshold == 70.0

    def test_fibonacci_not_used(self):
        """Test Fibonacci not used for day trading."""
        assert DAY_TRADER.weights.fibonacci == 0.0


class TestSwingTraderProfile:
    """Tests for Swing Trader profile configuration."""

    def test_profile_type(self):
        """Test profile type is correct."""
        assert SWING_TRADER.profile_type == TraderProfileType.SWING_TRADER

    def test_timeframes(self):
        """Test swing trading timeframes."""
        assert SWING_TRADER.timeframes.primary == "1d"
        assert SWING_TRADER.timeframes.confirmation == "1h"
        assert SWING_TRADER.timeframes.entry == "15m"

    def test_holding_period(self):
        """Test holding period for swing trades."""
        assert SWING_TRADER.min_holding_period == "2d"
        assert SWING_TRADER.max_holding_period == "3w"

    def test_risk_config(self):
        """Test swing trader uses structure-based stops."""
        assert SWING_TRADER.risk.stop_method == "structure"
        assert SWING_TRADER.risk.atr_multiplier == 2.0

    def test_fibonacci_weighted(self):
        """Test that Fibonacci is weighted for swing trading."""
        assert SWING_TRADER.weights.fibonacci == 5.0

    def test_uses_fibonacci_extensions(self):
        """Test swing trader uses Fibonacci extensions for targets."""
        assert SWING_TRADER.targets.use_fibonacci_extensions is True
        assert SWING_TRADER.targets.method == "fibonacci"

    def test_divergence_weighted(self):
        """Test divergence is important for swing trading."""
        assert SWING_TRADER.weights.divergence == 8.0


class TestPositionTraderProfile:
    """Tests for Position Trader profile configuration."""

    def test_profile_type(self):
        """Test profile type is correct."""
        assert POSITION_TRADER.profile_type == TraderProfileType.POSITION_TRADER

    def test_timeframes(self):
        """Test position trading timeframes (weekly confirmation)."""
        assert POSITION_TRADER.timeframes.primary == "1d"
        assert POSITION_TRADER.timeframes.confirmation == "1w"

    def test_allowed_trade_types(self):
        """Test can hold swing or long positions."""
        assert "swing" in POSITION_TRADER.allowed_trade_types
        assert "long" in POSITION_TRADER.allowed_trade_types

    def test_ema_trend_emphasis(self):
        """Test EMA trend is heavily weighted."""
        assert POSITION_TRADER.weights.ema_trend == 15.0

    def test_multi_timeframe_emphasis(self):
        """Test multi-timeframe alignment is important."""
        assert POSITION_TRADER.weights.multi_tf == 10.0

    def test_wider_stops(self):
        """Test wider ATR multiplier for position trades."""
        assert POSITION_TRADER.risk.atr_multiplier == 2.5


class TestLongTermInvestorProfile:
    """Tests for Long-Term Investor profile configuration."""

    def test_profile_type(self):
        """Test profile type is correct."""
        assert LONG_TERM_INVESTOR.profile_type == TraderProfileType.LONG_TERM_INVESTOR

    def test_holding_period(self):
        """Test long holding period."""
        assert LONG_TERM_INVESTOR.min_holding_period == "1m"
        assert LONG_TERM_INVESTOR.max_holding_period == "2y"

    def test_allowed_trade_types(self):
        """Test only long positions allowed."""
        assert LONG_TERM_INVESTOR.allowed_trade_types == ["long"]

    def test_risk_config(self):
        """Test percentage-based stops with wide multiplier."""
        assert LONG_TERM_INVESTOR.risk.stop_method == "percentage"
        assert LONG_TERM_INVESTOR.risk.atr_multiplier == 3.0

    def test_sentiment_emphasis(self):
        """Test sentiment/fundamentals heavily weighted."""
        assert LONG_TERM_INVESTOR.weights.sentiment == 15.0

    def test_ema_trend_most_important(self):
        """Test EMA trend has highest weight."""
        assert LONG_TERM_INVESTOR.weights.ema_trend == 18.0

    def test_lowest_confidence_threshold(self):
        """Test long-term has lowest threshold (more forgiving)."""
        assert LONG_TERM_INVESTOR.buy_confidence_threshold == 60.0

    def test_vwap_not_used(self):
        """Test VWAP not relevant for long-term investing."""
        assert LONG_TERM_INVESTOR.weights.vwap == 0.0

    def test_stochastic_not_used(self):
        """Test stochastic not used for long-term."""
        assert LONG_TERM_INVESTOR.weights.stochastic == 0.0


class TestScoringWeightsSum:
    """Tests to verify scoring weights are reasonable."""

    @pytest.mark.parametrize("profile_type", [
        "day_trader",
        "swing_trader",
        "position_trader",
        "long_term_investor",
    ])
    def test_weights_sum_approximately_100(self, profile_type):
        """Test that all weights sum to approximately 100."""
        profile = get_profile(profile_type)
        weights = profile.weights

        total = (
            weights.sentiment +
            weights.ema_trend +
            weights.rsi +
            weights.vwap +
            weights.volume +
            weights.macd +
            weights.bollinger +
            weights.multi_tf +
            weights.support_resistance +
            weights.divergence +
            weights.volume_profile +
            weights.chart_patterns +
            weights.fibonacci +
            weights.adx +
            weights.stochastic
        )

        # Allow some flexibility (90-110 range)
        assert 90 <= total <= 110, f"{profile_type} weights sum to {total}, expected ~100"

    @pytest.mark.parametrize("profile_type", [
        "day_trader",
        "swing_trader",
        "position_trader",
        "long_term_investor",
    ])
    def test_no_negative_weights(self, profile_type):
        """Test that no weights are negative."""
        profile = get_profile(profile_type)
        weights = profile.weights

        assert weights.sentiment >= 0
        assert weights.ema_trend >= 0
        assert weights.rsi >= 0
        assert weights.vwap >= 0
        assert weights.volume >= 0
        assert weights.macd >= 0
        assert weights.bollinger >= 0
        assert weights.multi_tf >= 0
        assert weights.support_resistance >= 0
        assert weights.divergence >= 0
        assert weights.volume_profile >= 0
        assert weights.chart_patterns >= 0
        assert weights.fibonacci >= 0
        assert weights.adx >= 0
        assert weights.stochastic >= 0


class TestProfileValidation:
    """Tests for profile model validation."""

    def test_timeframe_config_validation(self):
        """Test TimeframeConfig requires primary timeframe."""
        config = TimeframeConfig(primary="1d")
        assert config.primary == "1d"
        assert config.confirmation is None

    def test_risk_config_validation(self):
        """Test RiskConfig validation bounds."""
        config = RiskConfig(
            risk_percentage=1.0,
            stop_method="atr",
            atr_multiplier=2.0,
            max_position_percent=20.0
        )
        assert config.risk_percentage == 1.0

    def test_target_config_validation(self):
        """Test TargetConfig validation."""
        config = TargetConfig(
            method="rr_ratio",
            rr_ratios=[1.5, 2.5, 3.5]
        )
        assert config.method == "rr_ratio"
        assert len(config.rr_ratios) == 3
