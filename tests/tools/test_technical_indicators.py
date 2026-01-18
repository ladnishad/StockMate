#!/usr/bin/env python3
"""Test script for technical indicators SDK tool.

Usage:
    python tests/tools/test_technical_indicators.py AAPL
    python tests/tools/test_technical_indicators.py MSFT
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


async def test_technical_indicators(symbol: str):
    """Test technical indicators for all trade styles.

    Args:
        symbol: Stock ticker symbol
    """
    from app.agent.sdk import tools as sdk_tools

    print("=" * 70)
    print(f"TECHNICAL INDICATORS TEST: {symbol}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # EMA configurations per trade style
    indicator_configs = [
        ([5, 9, 20], "5m", "Day Trade"),
        ([9, 21, 50], "1d", "Swing Trade"),
        ([21, 50, 200], "1w", "Position Trade"),
    ]

    for ema_periods, timeframe, description in indicator_configs:
        print(f"\n{'─' * 70}")
        print(f"  {description} Indicators")
        print(f"  EMA Periods: {ema_periods}, Timeframe: {timeframe}")
        print(f"{'─' * 70}")

        try:
            result = await sdk_tools.get_technical_indicators(symbol, ema_periods, rsi_period=14, timeframe=timeframe)

            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue

            # Current Price
            current_price = result.get('current_price', 0)
            print(f"\n  Current Price: ${current_price:.2f}")

            # EMAs
            emas = result.get('emas', {})
            ema_trend = result.get('ema_trend', 'N/A')
            print(f"\n  EMAs:")
            print(f"    Trend Alignment: {ema_trend.upper()}")
            for period in ema_periods:
                key = f"ema_{period}"
                value = emas.get(key)
                if value:
                    vs_price = "above" if current_price > value else "below"
                    print(f"    EMA({period}):        ${value:.2f}  (price {vs_price})")
                else:
                    print(f"    EMA({period}):        N/A")

            # RSI
            rsi = result.get('rsi', {})
            rsi_value = rsi.get('value')
            rsi_signal = rsi.get('signal', 'N/A')
            print(f"\n  RSI (14):")
            if rsi_value:
                print(f"    Value:        {rsi_value:.2f}")
                print(f"    Signal:       {rsi_signal.upper()}")
            else:
                print(f"    Value:        N/A")

            # MACD
            macd = result.get('macd', {})
            macd_value = macd.get('value')
            macd_signal = macd.get('signal', 'N/A')
            macd_histogram = macd.get('histogram')
            print(f"\n  MACD:")
            if macd_value is not None:
                print(f"    Line:         {macd_value:.4f}")
                print(f"    Signal:       {macd_signal.upper()}")
                if macd_histogram is not None:
                    print(f"    Histogram:    {macd_histogram:.4f}")
            else:
                print(f"    Value:        N/A")

            # Bollinger Bands
            bollinger = result.get('bollinger', {})
            bb_upper = bollinger.get('upper')
            bb_middle = bollinger.get('middle')
            bb_lower = bollinger.get('lower')
            bb_position = bollinger.get('position', 'N/A')
            print(f"\n  Bollinger Bands:")
            if bb_upper and bb_middle and bb_lower:
                print(f"    Upper:        ${bb_upper:.2f}")
                print(f"    Middle:       ${bb_middle:.2f}")
                print(f"    Lower:        ${bb_lower:.2f}")
                print(f"    Position:     {bb_position}")
            else:
                print(f"    Value:        N/A")

            # VWAP
            vwap = result.get('vwap', {})
            vwap_value = vwap.get('value')
            vwap_position = vwap.get('price_vs_vwap', 'N/A')
            vwap_distance = vwap.get('distance_pct')
            print(f"\n  VWAP:")
            if vwap_value:
                print(f"    Value:        ${vwap_value:.2f}")
                print(f"    Price vs VWAP: {vwap_position.upper()}")
                if vwap_distance is not None:
                    print(f"    Distance:     {vwap_distance:+.2f}%")
            else:
                print(f"    Value:        N/A (intraday only)")

            # ADX (Trend Strength)
            adx = result.get('adx', {})
            adx_value = adx.get('value')
            adx_strength = adx.get('strength', 'N/A')
            adx_trending = adx.get('trending', False)
            print(f"\n  ADX (Trend Strength):")
            if adx_value:
                print(f"    Value:        {adx_value:.2f}")
                print(f"    Strength:     {adx_strength.upper()}")
                print(f"    Trending:     {'YES' if adx_trending else 'NO'}")
            else:
                print(f"    Value:        N/A")

            # ATR
            atr = result.get('atr', {})
            atr_value = atr.get('value')
            atr_pct = atr.get('pct')
            atr_regime = atr.get('volatility_regime', 'N/A')
            print(f"\n  ATR (Volatility):")
            if atr_value:
                print(f"    Value:        ${atr_value:.2f}")
                print(f"    ATR%:         {atr_pct:.2f}%")
                print(f"    Regime:       {atr_regime.upper()}")
            else:
                print(f"    Value:        N/A")

            # Volume Analysis
            volume = result.get('volume', {})
            vol_current = volume.get('current', 0)
            vol_average = volume.get('average', 0)
            vol_relative = volume.get('relative')
            vol_trend = volume.get('trend', 'N/A')
            print(f"\n  Volume:")
            print(f"    Current:      {vol_current:,.0f}")
            print(f"    Average:      {vol_average:,.0f}")
            if vol_relative is not None:
                print(f"    Relative Vol: {vol_relative:.2f}x")
            print(f"    Trend:        {vol_trend}")

            # Institutional Indicators (if available)
            ichimoku = result.get('ichimoku', {})
            if ichimoku.get('available'):
                print(f"\n  Ichimoku Cloud:")
                print(f"    Signal:       {ichimoku.get('signal', 'N/A').upper()}")
                print(f"    Price vs Cloud: {ichimoku.get('price_vs_cloud', 'N/A')}")
                print(f"    TK Cross:     {ichimoku.get('tk_cross', 'N/A')}")
                print(f"    Cloud Color:  {ichimoku.get('cloud_color', 'N/A')}")

            williams = result.get('williams_r', {})
            if williams.get('available'):
                print(f"\n  Williams %R:")
                print(f"    Value:        {williams.get('value', 'N/A')}")
                print(f"    Signal:       {williams.get('signal', 'N/A').upper()}")

            psar = result.get('parabolic_sar', {})
            if psar.get('available'):
                print(f"\n  Parabolic SAR:")
                print(f"    Value:        ${psar.get('value', 'N/A')}")
                print(f"    Trend:        {psar.get('trend_direction', 'N/A')}")
                print(f"    Signal:       {psar.get('signal', 'N/A').upper()}")

            cmf = result.get('cmf', {})
            if cmf.get('available'):
                print(f"\n  Chaikin Money Flow:")
                print(f"    Value:        {cmf.get('value', 'N/A')}")
                print(f"    Signal:       {cmf.get('signal', 'N/A').upper()}")

            adl = result.get('adl', {})
            if adl.get('available'):
                print(f"\n  Accumulation/Distribution:")
                print(f"    Signal:       {adl.get('signal', 'N/A').upper()}")
                print(f"    Trend:        {adl.get('trend', 'N/A')}")
                print(f"    Divergence:   {'YES' if adl.get('divergence') else 'NO'}")

        except Exception as e:
            print(f"  EXCEPTION: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test technical indicators SDK tool for a stock",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/tools/test_technical_indicators.py AAPL
    python tests/tools/test_technical_indicators.py TSLA
    python tests/tools/test_technical_indicators.py MSFT
        """
    )
    parser.add_argument(
        "symbol",
        nargs="?",
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    args = parser.parse_args()

    asyncio.run(test_technical_indicators(args.symbol.upper()))
