#!/usr/bin/env python3
"""Detailed test to show ALL data gathered for market overview analysis.

Usage:
    python tests/test_market_overview.py              # Full analysis
    python tests/test_market_overview.py --quick      # Quick snapshot only
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a section separator with title."""
    print(f"\n{char * width}")
    print(f"{title}")
    print(f"{char * width}")


def print_index_data(index_data: dict, show_timeframes: bool = True):
    """Pretty print data for a single index."""
    symbol = index_data.get("symbol", "???")
    name = index_data.get("name", "Unknown")

    # Basic info
    print(f"\n  {'─' * 70}")
    print(f"  {symbol} - {name}")
    print(f"  {'─' * 70}")

    # Price and signal
    price = index_data.get("price", 0)
    change_pct = index_data.get("change_pct", 0)
    signal = index_data.get("signal", "unknown")
    signal_strength = index_data.get("signal_strength", 0)

    change_indicator = "▲" if change_pct > 0 else "▼" if change_pct < 0 else "─"
    signal_color = "🟢" if signal == "bullish" else "🔴" if signal == "bearish" else "🟡"

    print(f"    Price: ${price:.2f} ({change_indicator} {change_pct:+.2f}%)")
    print(f"    Signal: {signal_color} {signal.upper()} (strength: {signal_strength}%)")

    # Core indicators
    rsi = index_data.get("rsi", 0)
    above_ema20 = index_data.get("above_ema20", False)
    above_ema50 = index_data.get("above_ema50")

    print(f"\n    Core Indicators:")
    print(f"      RSI(14): {rsi:.1f}")
    print(f"      Above EMA(20): {'✓' if above_ema20 else '✗'}")
    print(f"      Above EMA(50): {'✓' if above_ema50 else '✗' if above_ema50 is not None else 'N/A'}")

    # Enhanced indicators
    macd_signal = index_data.get("macd_signal")
    macd_histogram = index_data.get("macd_histogram")
    stochastic_k = index_data.get("stochastic_k")
    atr_pct = index_data.get("atr_pct")
    volatility = index_data.get("volatility")

    print(f"\n    Enhanced Indicators:")
    print(f"      MACD Signal: {macd_signal or 'N/A'}")
    if macd_histogram is not None:
        print(f"      MACD Histogram: {macd_histogram:.4f}")
    print(f"      Stochastic %K: {stochastic_k or 'N/A'}")
    print(f"      ATR%: {atr_pct or 'N/A'}{'%' if atr_pct else ''}")
    print(f"      Volatility: {volatility or 'N/A'}")

    # ADX / Trend strength
    adx = index_data.get("adx")
    if adx:
        print(f"\n    Trend Analysis (ADX):")
        print(f"      ADX Value: {adx.get('adx', 'N/A')}")
        print(f"      Trend Strength: {adx.get('strength', 'N/A')}")
        print(f"      Trend Direction: {adx.get('direction', 'N/A')}")
        print(f"      Is Trending: {'✓' if adx.get('is_trending') else '✗'}")

    # Momentum
    momentum = index_data.get("momentum")
    if momentum:
        print(f"\n    Momentum Score:")
        print(f"      Score: {momentum.get('score', 'N/A')}/100")
        print(f"      Label: {momentum.get('label', 'N/A')}")
        print(f"      MACD Contribution: {momentum.get('macd_contribution', 'N/A')}")
        print(f"      Stochastic Contribution: {momentum.get('stochastic_contribution', 'N/A')}")
        print(f"      RSI Contribution: {momentum.get('rsi_contribution', 'N/A')}")

    # Multi-timeframe analysis
    timeframes = index_data.get("timeframes")
    if timeframes and show_timeframes:
        print(f"\n    Multi-Timeframe Analysis:")

        for tf_name, tf_key in [("Daily (1d)", "daily"), ("Hourly (1h)", "hourly"), ("15-Min", "m15")]:
            tf_data = timeframes.get(tf_key)
            if tf_data:
                tf_signal = tf_data.get("signal", "N/A")
                tf_strength = tf_data.get("signal_strength", "N/A")
                tf_icon = "🟢" if tf_signal == "bullish" else "🔴" if tf_signal == "bearish" else "🟡"
                print(f"      {tf_name}: {tf_icon} {tf_signal} ({tf_strength}%)")

        confluence = timeframes.get("confluence")
        if confluence:
            print(f"\n      Confluence:")
            print(f"        Alignment: {confluence.get('alignment', 'N/A')}")
            print(f"        Score: {confluence.get('confluence_score', 'N/A')}%")
            print(f"        Aligned TFs: {confluence.get('aligned_timeframes', [])}")
            print(f"        Conflicting TFs: {confluence.get('conflicting_timeframes', [])}")

    # Warnings
    warnings = index_data.get("warnings", [])
    if warnings:
        print(f"\n    ⚠️  Warnings ({len(warnings)}):")
        for w in warnings:
            severity = w.get("severity", "medium")
            severity_icon = "🔴" if severity == "high" else "🟡"
            print(f"      {severity_icon} [{w.get('type', 'unknown')}] {w.get('message', '')}")
            print(f"         Action: {w.get('action', 'N/A')}")


def test_market_overview_detailed():
    """Run detailed market overview test showing all gathered data."""
    from app.tools.market_scanner import (
        get_market_overview,
        MARKET_INDICES,
        CORE_INDICES,
    )

    print_separator("MARKET OVERVIEW DETAILED TEST", "=", 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Indices to analyze: {list(MARKET_INDICES.keys())}")
    print(f"Core indices (for signals): {CORE_INDICES}")

    # Step 1: Run the market overview
    print_separator("STEP 1: RUNNING ENHANCED MARKET OVERVIEW", "-", 80)
    print("Fetching data for all indices with multi-timeframe analysis...")
    print("This may take 15-30 seconds due to multiple API calls.\n")

    try:
        result = get_market_overview()  # Uses default days_back=45
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 2: Show individual index analysis
    print_separator("STEP 2: INDIVIDUAL INDEX ANALYSIS", "-", 80)

    indices = result.get("indices", [])
    for index_data in indices:
        symbol = index_data.get("symbol", "")
        # Show full timeframe data only for core indices
        show_tf = symbol in CORE_INDICES
        print_index_data(index_data, show_timeframes=show_tf)

    # Step 3: Overall market signal
    print_separator("STEP 3: OVERALL MARKET SIGNAL", "-", 80)

    market_signal = result.get("market_signal", "unknown")
    signal_icon = "🟢" if market_signal == "bullish" else "🔴" if market_signal == "bearish" else "🟡"

    print(f"\n  Market Signal: {signal_icon} {market_signal.upper()}")
    print(f"  Summary: {result.get('summary', 'N/A')}")
    print(f"\n  Core Index Counts:")
    print(f"    Bullish: {result.get('bullish_count', 0)}")
    print(f"    Bearish: {result.get('bearish_count', 0)}")
    print(f"    Neutral: {result.get('neutral_count', 0)}")
    print(f"    Total Indices: {result.get('total_indices', 0)}")

    # Step 4: Volatility regime
    print_separator("STEP 4: VOLATILITY REGIME (VXX)", "-", 80)

    vol_regime = result.get("volatility_regime", {})
    regime = vol_regime.get("regime", "unknown")
    regime_icons = {
        "high_fear": "🔴",
        "elevated": "🟠",
        "normal": "🟢",
        "complacent": "🟡",
        "unknown": "⚪"
    }

    print(f"\n  Regime: {regime_icons.get(regime, '⚪')} {regime.upper()}")
    print(f"  VXX RSI: {vol_regime.get('vxx_rsi', 'N/A')}")
    print(f"  VXX Change%: {vol_regime.get('vxx_change_pct', 'N/A')}")
    print(f"  Interpretation: {vol_regime.get('interpretation', 'N/A')}")

    # Step 5: Aggregate momentum
    print_separator("STEP 5: AGGREGATE MOMENTUM SCORE", "-", 80)

    momentum = result.get("momentum_score", {})
    mom_score = momentum.get("score", 50)
    mom_label = momentum.get("label", "neutral")

    # Visual bar
    filled = int(mom_score / 5)
    bar = "█" * filled + "░" * (20 - filled)

    print(f"\n  Momentum: [{bar}] {mom_score}%")
    print(f"  Label: {mom_label.upper()}")

    # Step 6: Trend strength
    print_separator("STEP 6: AGGREGATE TREND STRENGTH", "-", 80)

    trend = result.get("trend_strength", {})
    avg_adx = trend.get("average_adx", 0)
    is_trending = trend.get("is_trending", False)
    strength = trend.get("strength", "weak")

    print(f"\n  Average ADX: {avg_adx:.1f}")
    print(f"  Is Trending: {'✓ Yes' if is_trending else '✗ No'}")
    print(f"  Strength: {strength.upper()}")

    # Step 7: Timeframe alignment
    print_separator("STEP 7: TIMEFRAME ALIGNMENT", "-", 80)

    tf_alignment = result.get("timeframe_alignment", {})
    daily = tf_alignment.get("daily_consensus", "unknown")
    hourly = tf_alignment.get("hourly_consensus", "unknown")
    m15 = tf_alignment.get("m15_consensus", "unknown")
    full_align = tf_alignment.get("full_alignment", False)

    def tf_icon(signal):
        return "🟢" if signal == "bullish" else "🔴" if signal == "bearish" else "🟡"

    print(f"\n  Daily Consensus:  {tf_icon(daily)} {daily}")
    print(f"  Hourly Consensus: {tf_icon(hourly)} {hourly}")
    print(f"  15-Min Consensus: {tf_icon(m15)} {m15}")
    print(f"\n  Full Alignment: {'✓ YES - All timeframes agree' if full_align else '✗ NO - Timeframes diverge'}")

    # Per-index confluence
    per_index = tf_alignment.get("per_index", {})
    if per_index:
        print(f"\n  Per-Index Confluence:")
        for sym, conf in per_index.items():
            align = conf.get("alignment", "unknown")
            score = conf.get("confluence_score", 0)
            print(f"    {sym}: {align} ({score}%)")

    # Step 8: Risk context
    print_separator("STEP 8: RISK CONTEXT (TLT)", "-", 80)

    risk = result.get("risk_context", "neutral")
    risk_icons = {"risk_on": "🟢", "risk_off": "🔴", "neutral": "🟡"}

    print(f"\n  Risk Context: {risk_icons.get(risk, '⚪')} {risk.upper()}")

    if risk == "risk_off":
        print("  Interpretation: Bonds rising while equities fall - flight to safety")
    elif risk == "risk_on":
        print("  Interpretation: Bonds falling while equities rise - risk appetite strong")
    else:
        print("  Interpretation: No clear risk-on/risk-off signal")

    # Step 9: All warnings
    print_separator("STEP 9: ALL WARNINGS", "-", 80)

    all_warnings = result.get("warnings", [])
    if all_warnings:
        print(f"\n  Total Warnings: {len(all_warnings)}")

        # Group by severity
        high = [w for w in all_warnings if w.get("severity") == "high"]
        medium = [w for w in all_warnings if w.get("severity") == "medium"]

        if high:
            print(f"\n  🔴 HIGH SEVERITY ({len(high)}):")
            for w in high:
                print(f"    [{w.get('symbol', '???')}] {w.get('type', 'unknown')}: {w.get('message', '')}")

        if medium:
            print(f"\n  🟡 MEDIUM SEVERITY ({len(medium)}):")
            for w in medium:
                print(f"    [{w.get('symbol', '???')}] {w.get('type', 'unknown')}: {w.get('message', '')}")
    else:
        print("\n  ✓ No warnings detected")

    # Final summary
    print_separator("FINAL SUMMARY", "=", 80)

    print(f"""
  Market Signal:     {signal_icon} {market_signal.upper()}
  Volatility:        {regime_icons.get(regime, '⚪')} {regime}
  Momentum:          {mom_score}% ({mom_label})
  Trend Strength:    {strength} (ADX: {avg_adx:.1f})
  TF Alignment:      {'✓ Full' if full_align else '✗ Partial'}
  Risk Context:      {risk_icons.get(risk, '⚪')} {risk}
  Warnings:          {len(all_warnings)} active

  {result.get('summary', '')}
""")

    print_separator("TEST COMPLETE", "=", 80)


def test_quick_market_status():
    """Run quick market status test (snapshot-based, faster)."""
    from app.tools.market_scanner import get_quick_market_status, MARKET_INDICES

    print_separator("QUICK MARKET STATUS TEST", "=", 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Using batch snapshots for {list(MARKET_INDICES.keys())}")

    try:
        result = get_quick_market_status()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return

    if "error" in result:
        print(f"\n❌ API Error: {result['error']}")
        return

    print_separator("INDICES STATUS", "-", 80)

    for idx in result.get("indices", []):
        symbol = idx.get("symbol", "???")
        name = idx.get("name", "Unknown")
        price = idx.get("price", 0)
        change_pct = idx.get("change_pct", 0)

        indicator = "▲" if change_pct > 0 else "▼" if change_pct < 0 else "─"
        color = "🟢" if change_pct > 0 else "🔴" if change_pct < 0 else "⚪"

        print(f"  {color} {symbol:<5} {name:<25} ${price:>8.2f}  {indicator} {change_pct:+6.2f}%")

    print_separator("SUMMARY", "-", 80)

    direction = result.get("market_direction", "mixed")
    dir_icon = "🟢" if direction == "up" else "🔴" if direction == "down" else "🟡"

    print(f"\n  Direction: {dir_icon} {direction.upper()}")
    print(f"  Up: {result.get('up_count', 0)} | Down: {result.get('down_count', 0)}")
    print(f"  Average Change: {result.get('average_change_pct', 0):+.2f}%")

    market_status = result.get("market_status", {})
    if market_status:
        is_open = market_status.get("is_open", False)
        next_event = market_status.get("next_event", "N/A")
        next_type = market_status.get("next_event_type", "N/A")

        print(f"\n  Market Open: {'✓ Yes' if is_open else '✗ No'}")
        print(f"  Next {next_type}: {next_event}")

    print_separator("TEST COMPLETE", "=", 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Detailed market overview test",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_market_overview.py           # Full detailed analysis
    python tests/test_market_overview.py --quick   # Quick snapshot only
        """
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick snapshot test instead of full analysis"
    )
    args = parser.parse_args()

    if args.quick:
        test_quick_market_status()
    else:
        test_market_overview_detailed()
