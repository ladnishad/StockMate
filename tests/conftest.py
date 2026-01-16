"""Pytest configuration and shared fixtures."""

import pytest
from datetime import datetime, timedelta
from typing import List

from app.models.data import PriceBar


@pytest.fixture
def sample_price_bars() -> List[PriceBar]:
    """Generate sample price bars for testing."""
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 100.0

    for i in range(100):
        # Create slightly trending upward prices with some volatility
        trend = i * 0.1
        volatility = (i % 5) - 2  # -2 to +2
        close = base_price + trend + volatility

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - 0.5,
            high=close + 1.0,
            low=close - 1.5,
            close=close,
            volume=1000000 + (i * 10000),
        )
        bars.append(bar)

    return bars


@pytest.fixture
def uptrend_price_bars() -> List[PriceBar]:
    """Generate price bars showing clear uptrend."""
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 100.0

    for i in range(50):
        close = base_price + (i * 0.5)  # Clear uptrend

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - 0.2,
            high=close + 0.3,
            low=close - 0.4,
            close=close,
            volume=1000000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def downtrend_price_bars() -> List[PriceBar]:
    """Generate price bars showing clear downtrend."""
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 100.0

    for i in range(50):
        close = base_price - (i * 0.3)  # Clear downtrend

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close + 0.2,
            high=close + 0.4,
            low=close - 0.3,
            close=close,
            volume=1000000,
        )
        bars.append(bar)

    return bars


# ============================================================================
# Volatility-Specific Fixtures for Shortcomings Fix Testing
# ============================================================================

@pytest.fixture
def low_volatility_bars() -> List[PriceBar]:
    """Generate price bars simulating low volatility stock (like PG, JNJ).

    Characteristics:
    - ATR ~0.5-0.8% of price (below 1.0% threshold)
    - Very tight daily ranges
    - Steady, predictable movements
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 150.0  # Higher price stock

    for i in range(80):
        # Very small daily movements (low volatility)
        trend = i * 0.05  # Gentle uptrend
        noise = (i % 7 - 3) * 0.15  # Very small noise (reduced from 0.3)
        close = base_price + trend + noise

        # Very tight ranges: high-low spread is only ~0.6% of price
        daily_range = close * 0.006  # 0.6% range (reduced from 1%)

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - daily_range * 0.2,
            high=close + daily_range * 0.5,
            low=close - daily_range * 0.5,
            close=close,
            volume=800000 + (i * 5000),
        )
        bars.append(bar)

    return bars


@pytest.fixture
def medium_volatility_bars() -> List[PriceBar]:
    """Generate price bars simulating medium volatility stock (like AAPL, MSFT).

    Characteristics:
    - ATR ~1.5-2.5% of price
    - Moderate daily ranges
    - Normal market movements
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 180.0

    for i in range(80):
        # Moderate daily movements
        trend = i * 0.15  # Moderate uptrend
        noise = (i % 5 - 2) * 1.5  # Moderate noise
        close = base_price + trend + noise

        # Medium ranges: high-low spread is ~2% of price
        daily_range = close * 0.02

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - daily_range * 0.3,
            high=close + daily_range * 0.6,
            low=close - daily_range * 0.6,
            close=close,
            volume=2000000 + (i * 20000),
        )
        bars.append(bar)

    return bars


@pytest.fixture
def high_volatility_bars() -> List[PriceBar]:
    """Generate price bars simulating high volatility stock (like TSLA, NVDA).

    Characteristics:
    - ATR ~3.5-5% of price
    - Wide daily ranges
    - Large swings
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 250.0

    for i in range(80):
        # Large daily movements (high volatility)
        trend = i * 0.4  # Strong uptrend
        noise = (i % 3 - 1) * 8  # Large noise
        close = base_price + trend + noise

        # Wide ranges: high-low spread is ~4% of price
        daily_range = close * 0.04

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - daily_range * 0.4,
            high=close + daily_range * 0.7,
            low=close - daily_range * 0.7,
            close=close,
            volume=5000000 + (i * 50000),
        )
        bars.append(bar)

    return bars


@pytest.fixture
def cup_and_handle_bars() -> List[PriceBar]:
    """Generate price bars that form a cup and handle pattern.

    Pattern characteristics:
    - U-shaped bottom (cup)
    - Small consolidation (handle)
    - Breakout above rim
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 100.0

    for i in range(60):
        if i < 10:
            # Left rim of cup
            close = base_price + (10 - i) * 0.3  # Declining into cup
        elif i < 25:
            # Cup bottom (U-shape)
            cup_position = i - 10
            cup_depth = 8 - abs(cup_position - 7.5) * 0.8  # U-shape
            close = base_price - cup_depth
        elif i < 40:
            # Right rim rising
            close = base_price + (i - 25) * 0.2
        elif i < 50:
            # Handle (small pullback)
            close = base_price + 3 - (i - 40) * 0.15
        else:
            # Breakout
            close = base_price + 3 + (i - 50) * 0.4

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - 0.3,
            high=close + 0.5,
            low=close - 0.6,
            close=close,
            volume=1000000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def rising_wedge_bars() -> List[PriceBar]:
    """Generate price bars that form a rising wedge (bearish reversal).

    Pattern characteristics:
    - Both highs and lows rising
    - Converging trendlines (lows rising faster)
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 100.0

    for i in range(50):
        # Rising wedge: support rises faster than resistance
        support_level = base_price + i * 0.6  # Rising faster
        resistance_level = base_price + 5 + i * 0.4  # Rising slower

        # Price oscillates between support and resistance
        if i % 5 == 0:
            close = support_level + 0.5  # Near support
        elif i % 5 == 2:
            close = resistance_level - 0.5  # Near resistance
        else:
            close = (support_level + resistance_level) / 2  # Middle

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - 0.3,
            high=min(close + 1.0, resistance_level),
            low=max(close - 1.0, support_level),
            close=close,
            volume=1000000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def channel_bars() -> List[PriceBar]:
    """Generate price bars forming an ascending channel.

    Pattern characteristics:
    - Parallel higher highs
    - Parallel higher lows
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    base_price = 100.0
    channel_width = 5.0

    for i in range(50):
        # Channel midline rises steadily
        midline = base_price + i * 0.3

        # Price oscillates within channel
        if i % 6 < 3:
            close = midline + (i % 6) * (channel_width / 3)  # Rising in channel
        else:
            close = midline + channel_width - (i % 6 - 3) * (channel_width / 3)  # Falling

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - 0.3,
            high=close + 0.8,
            low=close - 0.8,
            close=close,
            volume=1000000,
        )
        bars.append(bar)

    return bars


@pytest.fixture
def rectangle_bars() -> List[PriceBar]:
    """Generate price bars forming a rectangle (consolidation).

    Pattern characteristics:
    - Horizontal resistance
    - Horizontal support
    - Price bouncing between them
    """
    bars = []
    base_date = datetime(2025, 1, 1)
    support = 95.0
    resistance = 105.0

    for i in range(50):
        # Price oscillates between support and resistance
        cycle = i % 10
        if cycle < 5:
            close = support + cycle * 2  # Rising to resistance
        else:
            close = resistance - (cycle - 5) * 2  # Falling to support

        bar = PriceBar(
            timestamp=base_date + timedelta(days=i),
            open=close - 0.3,
            high=min(close + 1.0, resistance + 0.5),
            low=max(close - 1.0, support - 0.5),
            close=close,
            volume=1000000,
        )
        bars.append(bar)

    return bars
