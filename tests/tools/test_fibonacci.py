#!/usr/bin/env python3
"""Test script for Fibonacci SDK tool.

Usage:
    python tests/tools/test_fibonacci.py AAPL
    python tests/tools/test_fibonacci.py NVDA
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_fibonacci(symbol: str):
    """Test Fibonacci levels for all trade types.

    Args:
        symbol: Stock ticker symbol
    """
    from app.agent.sdk import tools as sdk_tools

    print("=" * 70)
    print(f"FIBONACCI ANALYSIS TEST: {symbol}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Trade type configurations
    trade_configs = [
        ("5m", 3, "day", "Day Trade"),
        ("1d", 100, "swing", "Swing Trade"),
        ("1w", 365, "position", "Position Trade"),
    ]

    for timeframe, days_back, trade_type, description in trade_configs:
        print(f"\n{'─' * 70}")
        print(f"  {description} Fibonacci Analysis")
        print(f"  Timeframe: {timeframe} | Trade Type: {trade_type}")
        print(f"{'─' * 70}")

        try:
            # First get price bars for this timeframe
            bars_data = await sdk_tools.get_price_bars(symbol, timeframe, days_back)
            bars = bars_data.get("bars", [])

            if not bars:
                print(f"  ERROR: No price bars available")
                continue

            print(f"\n  Data: {len(bars)} bars fetched")

            # Get Fibonacci levels
            fib_result = await sdk_tools.get_fibonacci_levels(symbol, bars, trade_type)

            if "error" in fib_result and fib_result["error"]:
                print(f"  ERROR: {fib_result['error']}")
                continue

            # Swing Detection
            swing_high = fib_result.get('swing_high')
            swing_low = fib_result.get('swing_low')
            swing_range = fib_result.get('swing_range', 0)

            print(f"\n  Swing Detection:")
            print(f"    Swing High:   ${swing_high:.2f}" if swing_high else "    Swing High:   N/A")
            print(f"    Swing Low:    ${swing_low:.2f}" if swing_low else "    Swing Low:    N/A")
            print(f"    Swing Range:  ${swing_range:.2f}" if swing_range else "    Swing Range:  N/A")

            # Trend & Signal
            print(f"\n  Analysis:")
            print(f"    Trend:        {fib_result.get('trend', 'N/A').upper()}")
            print(f"    Signal:       {fib_result.get('signal', 'N/A').upper()}")
            print(f"    Current Price: ${fib_result.get('current_price', 0):.2f}")
            print(f"    At Entry Level: {'YES' if fib_result.get('at_entry_level') else 'NO'}")

            # Nearest Level
            nearest_level = fib_result.get('nearest_level', 'N/A')
            nearest_price = fib_result.get('nearest_price')
            print(f"\n  Nearest Fibonacci Level:")
            print(f"    Level:        {nearest_level}")
            if nearest_price:
                print(f"    Price:        ${nearest_price:.2f}")

            # Key Retracement Levels
            retracements = fib_result.get('retracement_levels', {})
            if retracements:
                print(f"\n  Retracement Levels (Entry Zones):")
                key_levels = ['0.382', '0.500', '0.618', '0.786']
                for level in key_levels:
                    if level in retracements and retracements[level]:
                        price = retracements[level]
                        current = fib_result.get('current_price', 0)
                        distance = ((price - current) / current * 100) if current > 0 else 0
                        print(f"    {level}:        ${price:.2f}  ({distance:+.1f}%)")

            # Extension Targets
            extensions = fib_result.get('extension_levels', {})
            if extensions:
                print(f"\n  Extension Levels (Targets):")
                ext_levels = ['1.272', '1.618', '2.000', '2.618']
                for level in ext_levels:
                    if level in extensions and extensions[level]:
                        price = extensions[level]
                        current = fib_result.get('current_price', 0)
                        distance = ((price - current) / current * 100) if current > 0 else 0
                        print(f"    {level}:        ${price:.2f}  ({distance:+.1f}%)")

            # Suggested Zones
            entry_zone = fib_result.get('suggested_entry_zone', {})
            stop_zone = fib_result.get('suggested_stop_zone', {})

            if entry_zone and entry_zone.get('low') and entry_zone.get('high'):
                print(f"\n  Suggested Entry Zone:")
                print(f"    Low:          ${entry_zone['low']:.2f}")
                print(f"    High:         ${entry_zone['high']:.2f}")

            if stop_zone and stop_zone.get('low') and stop_zone.get('high'):
                print(f"\n  Suggested Stop Zone:")
                print(f"    Low:          ${stop_zone['low']:.2f}")
                print(f"    High:         ${stop_zone['high']:.2f}")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test Fibonacci SDK tool for a stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/tools/test_fibonacci.py AAPL
    python tests/tools/test_fibonacci.py TSLA
    python tests/tools/test_fibonacci.py NVDA
        """
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    args = parser.parse_args()

    asyncio.run(test_fibonacci(args.symbol.upper()))
