#!/usr/bin/env python3
"""Test script for chart generator SDK tool.

Generates charts for multiple timeframes and saves PNG files.

Usage:
    python tests/tools/test_chart_generator.py AAPL
    python tests/tools/test_chart_generator.py TSLA
"""

import argparse
import asyncio
import base64
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Output directory for charts
OUTPUT_DIR = Path(__file__).parent.parent / "output" / "charts"


def save_chart_png(chart_base64: str, symbol: str, timeframe: str) -> Path:
    """Save base64 chart image to PNG file.

    Args:
        chart_base64: Base64 encoded PNG image
        symbol: Stock ticker symbol
        timeframe: Chart timeframe

    Returns:
        Path to saved file
    """
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{timeframe}_{timestamp}.png"
    filepath = OUTPUT_DIR / filename

    # Decode and save
    image_data = base64.b64decode(chart_base64)
    filepath.write_bytes(image_data)

    return filepath


async def test_chart_generator(symbol: str):
    """Test chart generation for multiple timeframes.

    Args:
        symbol: Stock ticker symbol
    """
    from app.agent.sdk import tools as sdk_tools

    print("=" * 70)
    print(f"CHART GENERATOR TEST: {symbol}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print("=" * 70)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Chart configurations
    chart_configs = [
        ("5m", 3, "Day Trade (5-minute)"),
        ("1d", 100, "Swing Trade (Daily)"),
        ("1w", 365, "Position Trade (Weekly)"),
    ]

    saved_files = []

    for timeframe, days_back, description in chart_configs:
        print(f"\n{'─' * 70}")
        print(f"  {description}")
        print(f"  Timeframe: {timeframe} | Days Back: {days_back}")
        print(f"{'─' * 70}")

        try:
            result = await sdk_tools.generate_chart(symbol, timeframe, days_back)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue

            chart_base64 = result.get('chart_image_base64')
            if not chart_base64:
                print("  ERROR: No chart image returned")
                continue

            # Chart metadata
            bars_plotted = result.get('bars_plotted', 'N/A')
            base64_size = len(chart_base64)

            print(f"\n  Chart Generated:")
            print(f"    Bars Plotted: {bars_plotted}")
            print(f"    Base64 Size:  {base64_size:,} characters")

            # Save to PNG file
            filepath = save_chart_png(chart_base64, symbol, timeframe)
            file_size = filepath.stat().st_size
            saved_files.append(filepath)

            print(f"\n  Saved to PNG:")
            print(f"    File Path:    {filepath}")
            print(f"    File Size:    {file_size:,} bytes ({file_size / 1024:.1f} KB)")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("SAVED FILES SUMMARY")
    print("=" * 70)

    if saved_files:
        total_size = 0
        for filepath in saved_files:
            size = filepath.stat().st_size
            total_size += size
            print(f"  {filepath.name}: {size:,} bytes")
        print(f"\n  Total: {len(saved_files)} files, {total_size:,} bytes ({total_size / 1024:.1f} KB)")
    else:
        print("  No files saved.")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test chart generator SDK tool for a stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/tools/test_chart_generator.py AAPL
    python tests/tools/test_chart_generator.py TSLA
    python tests/tools/test_chart_generator.py NVDA

Output:
    Charts are saved to tests/output/charts/{symbol}_{timeframe}_{timestamp}.png
        """
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    args = parser.parse_args()

    asyncio.run(test_chart_generator(args.symbol.upper()))
