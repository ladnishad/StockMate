#!/usr/bin/env python3
"""Test script for price bars SDK tool.

Usage:
    python tests/tools/test_price_bars.py AAPL
    python tests/tools/test_price_bars.py NVDA
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_price_bars(symbol: str):
    """Test price bars for multiple timeframes.

    Args:
        symbol: Stock ticker symbol
    """
    from app.agent.sdk import tools as sdk_tools

    print("=" * 70)
    print(f"PRICE BARS TEST: {symbol}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Timeframe configurations
    timeframes = [
        ("5m", 3, "Day Trade (Intraday)"),
        ("1d", 100, "Swing Trade (Daily)"),
        ("1w", 365, "Position Trade (Weekly)"),
    ]

    for tf, days_back, description in timeframes:
        print(f"\n{'─' * 70}")
        print(f"  {description}")
        print(f"  Timeframe: {tf} | Days Back: {days_back}")
        print(f"{'─' * 70}")

        try:
            result = await sdk_tools.get_price_bars(symbol, tf, days_back)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue

            # Basic Info
            print(f"\n  Bar Count:      {result.get('bars_count', 'N/A')}")
            print(f"  Current Price:  ${result.get('current_price', 0):.2f}")

            # Price Range
            print(f"\n  Price Range (last 20 bars):")
            print(f"    High:         ${result.get('high', 0):.2f}")
            print(f"    Low:          ${result.get('low', 0):.2f}")

            # Volatility (ATR)
            atr = result.get('atr', 0)
            atr_pct = result.get('atr_pct', 0)
            print(f"\n  ATR (Volatility):")
            print(f"    ATR:          ${atr:.2f}")
            print(f"    ATR%:         {atr_pct:.2f}%")

            # Volume
            avg_vol = result.get('average_volume', 0)
            print(f"\n  Volume:")
            print(f"    Average:      {avg_vol:,.0f}")

            # Last Bar OHLCV
            last_bar = result.get('last_bar', {})
            if last_bar:
                print(f"\n  Last Bar (OHLCV):")
                print(f"    Open:         ${last_bar.get('open', 0):.2f}")
                print(f"    High:         ${last_bar.get('high', 0):.2f}")
                print(f"    Low:          ${last_bar.get('low', 0):.2f}")
                print(f"    Close:        ${last_bar.get('close', 0):.2f}")
                print(f"    Volume:       {last_bar.get('volume', 0):,.0f}")

            # Show a few recent bars if available
            bars = result.get('bars', [])
            if bars and len(bars) >= 3:
                print(f"\n  Recent Bars (last 3):")
                for i, bar in enumerate(bars[-3:]):
                    ts = bar.timestamp.strftime("%Y-%m-%d %H:%M") if hasattr(bar.timestamp, 'strftime') else str(bar.timestamp)
                    print(f"    [{i+1}] {ts}: O=${bar.open:.2f} H=${bar.high:.2f} L=${bar.low:.2f} C=${bar.close:.2f} V={bar.volume:,.0f}")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test price bars SDK tool for a stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/tools/test_price_bars.py AAPL
    python tests/tools/test_price_bars.py TSLA
    python tests/tools/test_price_bars.py NVDA
        """
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    args = parser.parse_args()

    asyncio.run(test_price_bars(args.symbol.upper()))
