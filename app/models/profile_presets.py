"""Preset trader profile configurations.

This module contains the 4 predefined trader profiles:
- DAY_TRADER: Intraday trading with tight stops and VWAP focus
- SWING_TRADER: Multi-day trades with Fibonacci and structure-based analysis
- POSITION_TRADER: Multi-week momentum trades
- LONG_TERM_INVESTOR: Multi-month positions with trend focus

Each profile is carefully tuned for its trading style with appropriate
scoring weights, risk parameters, and thresholds.
"""

from typing import Dict

from app.models.profile import (
    TraderProfile,
    TraderProfileType,
    TimeframeConfig,
    RiskConfig,
    TargetConfig,
    ScoringWeights,
)


# =============================================================================
# Day Trader Profile
# =============================================================================
DAY_TRADER = TraderProfile(
    name="Day Trader",
    profile_type=TraderProfileType.DAY_TRADER,
    description="Intraday trades closed before market close. Focus on momentum, "
    "VWAP, and volume confirmation. Tight stops with quick profit targets.",
    timeframes=TimeframeConfig(
        primary="15m",
        confirmation="1h",
        entry="5m",
    ),
    min_holding_period="5m",
    max_holding_period="1d",
    allowed_trade_types=["day"],
    risk=RiskConfig(
        risk_percentage=0.5,  # Lower risk per trade, more trades
        stop_method="atr",
        atr_multiplier=1.5,  # Tighter stops for intraday
        max_position_percent=15.0,
    ),
    targets=TargetConfig(
        method="rr_ratio",
        rr_ratios=[1.0, 1.5, 2.0],  # Quick scalps
        use_fibonacci_extensions=False,
        validate_against_resistance=True,
    ),
    weights=ScoringWeights(
        sentiment=5.0,  # Less important intraday
        ema_trend=10.0,
        rsi=8.0,
        vwap=15.0,  # CRITICAL for day trading
        volume=15.0,  # CRITICAL for day trading
        macd=8.0,
        bollinger=10.0,
        multi_tf=5.0,  # Less focus on higher TFs
        support_resistance=8.0,
        divergence=5.0,
        volume_profile=8.0,
        chart_patterns=3.0,
        fibonacci=0.0,  # Not used for day trading
        adx=0.0,
        stochastic=0.0,
    ),
    buy_confidence_threshold=70.0,  # Higher threshold for day trades
    rsi_overbought=75.0,
    rsi_oversold=25.0,
    adx_trend_threshold=20.0,
)


# =============================================================================
# Swing Trader Profile
# =============================================================================
SWING_TRADER = TraderProfile(
    name="Swing Trader",
    profile_type=TraderProfileType.SWING_TRADER,
    description="Multi-day to multi-week trades capturing price swings. "
    "Focus on Fibonacci retracements, structure-based entries, and divergences.",
    timeframes=TimeframeConfig(
        primary="1d",
        confirmation="1h",
        entry="15m",
    ),
    min_holding_period="2d",
    max_holding_period="3w",
    allowed_trade_types=["swing"],
    risk=RiskConfig(
        risk_percentage=1.0,
        stop_method="structure",  # Use support levels
        atr_multiplier=2.0,
        max_position_percent=20.0,
    ),
    targets=TargetConfig(
        method="fibonacci",
        rr_ratios=[1.5, 2.5, 3.5],
        use_fibonacci_extensions=True,  # Key for swing trading
        validate_against_resistance=True,
    ),
    weights=ScoringWeights(
        sentiment=8.0,
        ema_trend=12.0,
        rsi=6.0,
        vwap=5.0,
        volume=8.0,
        macd=10.0,
        bollinger=6.0,
        multi_tf=8.0,
        support_resistance=7.0,
        divergence=8.0,  # Important for reversals
        volume_profile=5.0,
        chart_patterns=7.0,
        fibonacci=5.0,  # KEY for swing trading
        adx=3.0,  # Trend strength filter
        stochastic=2.0,  # Timing tool
    ),
    buy_confidence_threshold=65.0,
    rsi_overbought=70.0,
    rsi_oversold=30.0,
    adx_trend_threshold=25.0,
)


# =============================================================================
# Position Trader Profile
# =============================================================================
POSITION_TRADER = TraderProfile(
    name="Position Trader",
    profile_type=TraderProfileType.POSITION_TRADER,
    description="Multi-week to multi-month positions. Momentum-driven with "
    "strong trend confirmation. Combines technical and fundamental factors.",
    timeframes=TimeframeConfig(
        primary="1d",
        confirmation="1w",  # Weekly confirmation
        entry="1d",
    ),
    min_holding_period="1w",
    max_holding_period="3m",
    allowed_trade_types=["swing", "long"],  # Can hold either
    risk=RiskConfig(
        risk_percentage=1.5,
        stop_method="structure",
        atr_multiplier=2.5,
        max_position_percent=25.0,
    ),
    targets=TargetConfig(
        method="structure",
        rr_ratios=[2.0, 3.0, 5.0],  # Larger moves expected
        use_fibonacci_extensions=True,
        validate_against_resistance=True,
    ),
    weights=ScoringWeights(
        sentiment=12.0,  # More important for longer holds
        ema_trend=15.0,  # TREND IS KING
        rsi=5.0,
        vwap=3.0,
        volume=8.0,
        macd=12.0,  # Weekly MACD important
        bollinger=5.0,
        multi_tf=10.0,  # Multi-TF alignment critical
        support_resistance=8.0,
        divergence=7.0,
        volume_profile=5.0,
        chart_patterns=5.0,
        fibonacci=3.0,
        adx=2.0,
        stochastic=0.0,
    ),
    buy_confidence_threshold=65.0,
    rsi_overbought=75.0,
    rsi_oversold=25.0,
    adx_trend_threshold=25.0,
)


# =============================================================================
# Long-Term Investor Profile
# =============================================================================
LONG_TERM_INVESTOR = TraderProfile(
    name="Long-Term Investor",
    profile_type=TraderProfileType.LONG_TERM_INVESTOR,
    description="Multi-month to multi-year positions. Focus on major trends "
    "and value. More forgiving thresholds with wider stops for volatility.",
    timeframes=TimeframeConfig(
        primary="1d",
        confirmation="1w",
        entry="1d",
    ),
    min_holding_period="1m",
    max_holding_period="2y",
    allowed_trade_types=["long"],
    risk=RiskConfig(
        risk_percentage=2.0,
        stop_method="percentage",  # Wider percentage-based stops
        atr_multiplier=3.0,  # Wide stops for volatility
        max_position_percent=30.0,
    ),
    targets=TargetConfig(
        method="structure",
        rr_ratios=[3.0, 5.0, 10.0],  # Big moves
        use_fibonacci_extensions=False,
        validate_against_resistance=False,  # Less important for long term
    ),
    weights=ScoringWeights(
        sentiment=15.0,  # Fundamentals/sentiment key
        ema_trend=18.0,  # LONG-TERM TREND
        rsi=3.0,
        vwap=0.0,  # Not relevant for long term
        volume=5.0,
        macd=10.0,
        bollinger=3.0,
        multi_tf=12.0,
        support_resistance=10.0,
        divergence=5.0,
        volume_profile=5.0,
        chart_patterns=4.0,
        fibonacci=0.0,
        adx=5.0,  # Trend strength matters
        stochastic=0.0,
    ),
    buy_confidence_threshold=60.0,  # Lower threshold, more forgiving
    rsi_overbought=80.0,
    rsi_oversold=20.0,
    adx_trend_threshold=20.0,
)


# =============================================================================
# Profile Registry
# =============================================================================
PROFILE_REGISTRY: Dict[str, TraderProfile] = {
    TraderProfileType.DAY_TRADER.value: DAY_TRADER,
    TraderProfileType.SWING_TRADER.value: SWING_TRADER,
    TraderProfileType.POSITION_TRADER.value: POSITION_TRADER,
    TraderProfileType.LONG_TERM_INVESTOR.value: LONG_TERM_INVESTOR,
}


def get_profile(profile_type: str) -> TraderProfile:
    """Get a trader profile by type name.

    Args:
        profile_type: Profile type string (e.g., 'swing_trader')

    Returns:
        TraderProfile instance for the requested type

    Raises:
        ValueError: If profile_type is not a valid profile

    Example:
        >>> profile = get_profile("swing_trader")
        >>> print(profile.name)
        "Swing Trader"
        >>> print(profile.weights.fibonacci)
        5.0
    """
    if profile_type not in PROFILE_REGISTRY:
        valid_profiles = list(PROFILE_REGISTRY.keys())
        raise ValueError(
            f"Unknown profile type: '{profile_type}'. "
            f"Available profiles: {valid_profiles}"
        )
    return PROFILE_REGISTRY[profile_type]


def list_profiles() -> list:
    """List all available profile types.

    Returns:
        List of profile type strings
    """
    return list(PROFILE_REGISTRY.keys())
