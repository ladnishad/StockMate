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
