"""Comprehensive AI Integration Tests.

This test suite validates that all shortcomings fixes are properly integrated
into the AI-powered analysis flow. Run with:

    pytest tests/test_ai_integration.py -v

Or for a quick manual test:

    python tests/test_ai_integration.py AAPL
"""

import sys
import asyncio
import argparse
from typing import Dict, Any, Optional


def setup_path():
    """Ensure app is in path."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


setup_path()

from app.agent.tools import get_technical_indicators, get_chart_patterns
from app.agent.planning_agent import StockPlanningAgent


# ============================================================================
# Test Functions
# ============================================================================

async def test_technical_indicators(symbol: str) -> Dict[str, Any]:
    """Test that all technical indicators are calculated and returned.

    Validates:
    - Core indicators (RSI, MACD, EMAs, Volume, Bollinger, ATR)
    - New institutional indicators (Ichimoku, Williams %R, PSAR, CMF, ADL)
    - Volatility regime classification
    """
    print(f"\n{'='*60}")
    print(f"TESTING TECHNICAL INDICATORS: {symbol}")
    print(f"{'='*60}")

    tech = await get_technical_indicators(symbol)

    if 'error' in tech:
        print(f"ERROR: {tech['error']}")
        return {"passed": False, "error": tech['error']}

    results = {
        "passed": True,
        "core_indicators": {},
        "institutional_indicators": {},
        "volatility": {},
    }

    # Core indicators
    print("\n--- CORE INDICATORS ---")

    rsi = tech.get('rsi', {})
    rsi_ok = rsi.get('value') is not None
    results['core_indicators']['rsi'] = rsi_ok
    print(f"  RSI: {rsi.get('value', 'N/A'):.1f} ({rsi.get('signal', 'N/A')}) {'✓' if rsi_ok else '✗'}")

    macd = tech.get('macd', {})
    macd_ok = macd.get('signal') is not None
    results['core_indicators']['macd'] = macd_ok
    print(f"  MACD: {macd.get('signal', 'N/A')} {'✓' if macd_ok else '✗'}")

    emas = tech.get('emas', {})
    emas_ok = emas.get('trend') is not None
    results['core_indicators']['emas'] = emas_ok
    print(f"  EMA Trend: {emas.get('trend', 'N/A')} ({emas.get('bullish_count', 0)}/3) {'✓' if emas_ok else '✗'}")

    vol = tech.get('volume', {})
    vol_ok = vol.get('relative') is not None
    results['core_indicators']['volume'] = vol_ok
    rel_vol = vol.get('relative', 0)
    print(f"  Volume: {rel_vol:.2f}x ({vol.get('signal', 'N/A')}) {'✓' if vol_ok else '✗'}")

    # Volatility
    print("\n--- VOLATILITY ---")

    atr = tech.get('atr', {})
    atr_ok = atr.get('volatility_regime') is not None
    results['volatility']['atr'] = atr_ok
    print(f"  ATR: ${atr.get('value', 0):.2f} ({atr.get('percentage', 0):.2f}%) {'✓' if atr_ok else '✗'}")
    print(f"  Volatility Regime: {atr.get('volatility_regime', 'N/A').upper()} {'✓' if atr_ok else '✗'}")

    # Institutional indicators
    print("\n--- INSTITUTIONAL INDICATORS (Shortcomings Fix) ---")

    ich = tech.get('ichimoku', {})
    ich_ok = ich.get('available', False)
    results['institutional_indicators']['ichimoku'] = ich_ok
    print(f"  Ichimoku: {ich.get('signal', 'N/A')} (vs cloud: {ich.get('price_vs_cloud', 'N/A')}) {'✓' if ich_ok else '✗'}")

    will = tech.get('williams_r', {})
    will_ok = will.get('available', False)
    results['institutional_indicators']['williams_r'] = will_ok
    will_val = will.get('value')
    will_str = f"{will_val:.1f}" if will_val is not None else "N/A"
    print(f"  Williams %R: {will_str} ({will.get('signal', 'N/A')}) {'✓' if will_ok else '✗'}")

    psar = tech.get('parabolic_sar', {})
    psar_ok = psar.get('available', False)
    results['institutional_indicators']['parabolic_sar'] = psar_ok
    print(f"  Parabolic SAR: {psar.get('signal', 'N/A')} (trend: {psar.get('trend_direction', 'N/A')}) {'✓' if psar_ok else '✗'}")

    cmf = tech.get('cmf', {})
    cmf_ok = cmf.get('available', False)
    results['institutional_indicators']['cmf'] = cmf_ok
    cmf_val = cmf.get('value')
    cmf_str = f"{cmf_val:.3f}" if cmf_val is not None else "N/A"
    print(f"  CMF: {cmf_str} ({cmf.get('signal', 'N/A')}) {'✓' if cmf_ok else '✗'}")

    adl = tech.get('adl', {})
    adl_ok = adl.get('available', False)
    results['institutional_indicators']['adl'] = adl_ok
    print(f"  ADL: {adl.get('signal', 'N/A')} (divergence: {adl.get('divergence', False)}) {'✓' if adl_ok else '✗'}")

    # Summary
    all_core = all(results['core_indicators'].values())
    all_inst = all(results['institutional_indicators'].values())
    results['passed'] = all_core and all_inst

    print(f"\n--- SUMMARY ---")
    print(f"  Core Indicators: {'✓ ALL PASS' if all_core else '✗ SOME FAILED'}")
    print(f"  Institutional Indicators: {'✓ ALL PASS' if all_inst else '✗ SOME FAILED'}")

    return results


async def test_chart_patterns(symbol: str) -> Dict[str, Any]:
    """Test chart pattern detection with success rates.

    Validates:
    - Pattern detection working
    - Success rates are included (not None/0)
    - ATR tolerance is being used
    """
    print(f"\n{'='*60}")
    print(f"TESTING CHART PATTERNS: {symbol}")
    print(f"{'='*60}")

    patterns = await get_chart_patterns(symbol)

    if 'error' in patterns:
        print(f"ERROR: {patterns['error']}")
        return {"passed": False, "error": patterns['error']}

    results = {
        "passed": True,
        "pattern_count": patterns.get('pattern_count', 0),
        "patterns_with_success_rate": 0,
        "atr_tolerance_used": patterns.get('atr_tolerance_used'),
    }

    print(f"\n  ATR Tolerance Used: {results['atr_tolerance_used']}")
    print(f"  Patterns Detected: {results['pattern_count']}")

    if patterns.get('patterns'):
        print("\n--- DETECTED PATTERNS ---")
        for p in patterns['patterns']:
            sr = p.get('success_rate')
            has_sr = sr is not None and sr > 0
            if has_sr:
                results['patterns_with_success_rate'] += 1

            print(f"  - {p['name']}: {p['type']}")
            print(f"      Confidence: {p['confidence']}%")
            print(f"      Success Rate: {sr}% {'✓' if has_sr else '✗ MISSING'}")
            if p.get('target_price'):
                print(f"      Target: ${p['target_price']:.2f}")
    else:
        print("\n  No patterns detected (this may be normal for some stocks)")

    # Check that success rates are present
    if results['pattern_count'] > 0:
        results['passed'] = results['patterns_with_success_rate'] == results['pattern_count']

    print(f"\n--- SUMMARY ---")
    print(f"  Patterns with Success Rates: {results['patterns_with_success_rate']}/{results['pattern_count']}")

    return results


async def test_ai_plan_generation(symbol: str) -> Dict[str, Any]:
    """Test full AI trading plan generation.

    Validates:
    - AI can gather comprehensive data
    - AI generates a valid trading plan
    - New indicators are mentioned in thesis (when relevant)
    """
    print(f"\n{'='*60}")
    print(f"TESTING AI PLAN GENERATION: {symbol}")
    print(f"{'='*60}")

    agent = StockPlanningAgent(symbol, user_id='test_user')

    results = {
        "passed": False,
        "data_gathered": False,
        "plan_generated": False,
        "has_thesis": False,
        "has_setup": False,
        "uses_new_indicators": False,
    }

    # Step 1: Gather data
    print("\n[1] Gathering comprehensive data...")
    try:
        data = await agent.gather_comprehensive_data()
        results['data_gathered'] = True

        # Check new indicators in data
        tech = data.get('technical', {})
        new_indicators_present = (
            tech.get('ichimoku', {}).get('available', False) or
            tech.get('williams_r', {}).get('available', False) or
            tech.get('cmf', {}).get('available', False) or
            tech.get('adl', {}).get('available', False)
        )
        print(f"    Data gathered: ✓")
        print(f"    New indicators in data: {'✓' if new_indicators_present else '✗'}")
    except Exception as e:
        print(f"    ERROR gathering data: {e}")
        return results

    # Step 2: Generate AI plan
    print("\n[2] Generating AI trading plan...")
    try:
        plan = await agent.generate_smart_plan()
        results['plan_generated'] = True

        print(f"\n{'='*60}")
        print(f"AI TRADING PLAN FOR {symbol}")
        print(f"{'='*60}")

        # Bias and Confidence
        print(f"\n┌─────────────────────────────────────────────────────────┐")
        print(f"│  RECOMMENDATION: {plan.bias.upper():^40} │")
        print(f"│  CONFIDENCE: {plan.confidence}%{' '*44}│")
        print(f"└─────────────────────────────────────────────────────────┘")

        # Trade Style
        if hasattr(plan, 'trade_style') and plan.trade_style:
            ts = plan.trade_style
            print(f"\n--- TRADE STYLE ---")
            print(f"    Style: {ts.recommended_style.upper()}")
            print(f"    Holding Period: {ts.holding_period}")
            print(f"    Reasoning: {ts.reasoning}")

        # Entry Setup
        print(f"\n--- ENTRY SETUP ---")
        if plan.entry_zone_low and plan.entry_zone_high:
            print(f"    Entry Zone: ${plan.entry_zone_low:.2f} - ${plan.entry_zone_high:.2f}")
        else:
            print(f"    Entry Zone: Not specified (neutral bias)")

        # Stop Loss
        print(f"\n--- STOP LOSS ---")
        if plan.stop_loss:
            print(f"    Stop Price: ${plan.stop_loss:.2f}")
            if plan.stop_reasoning:
                print(f"    Reasoning: {plan.stop_reasoning}")
        else:
            print(f"    Stop Loss: Not specified")

        # Targets
        print(f"\n--- PRICE TARGETS ---")
        if plan.targets:
            for i, target in enumerate(plan.targets, 1):
                print(f"    Target {i}: ${target.price:.2f}")
                print(f"       Reasoning: {target.reasoning}")
        else:
            print(f"    Targets: Not specified")

        # Risk Management
        print(f"\n--- RISK MANAGEMENT ---")
        if plan.risk_reward:
            print(f"    Risk/Reward Ratio: {plan.risk_reward}")
        if plan.position_size_pct:
            print(f"    Position Size: {plan.position_size_pct}% of portfolio")

        # Thesis
        results['has_thesis'] = bool(plan.thesis)
        print(f"\n--- THESIS ---")
        print(f"    {plan.thesis}")

        # Key Levels
        print(f"\n--- KEY LEVELS ---")
        if plan.key_supports:
            print(f"    Support Levels: {['$' + str(s) for s in plan.key_supports]}")
        if plan.key_resistances:
            print(f"    Resistance Levels: {['$' + str(r) for r in plan.key_resistances]}")

        # Invalidation
        print(f"\n--- INVALIDATION CRITERIA ---")
        if plan.invalidation_criteria:
            print(f"    {plan.invalidation_criteria}")

        # Educational content if available
        if hasattr(plan, 'educational') and plan.educational:
            print(f"\n--- EDUCATIONAL ---")
            if hasattr(plan.educational, 'setup_explanation'):
                print(f"    Setup: {plan.educational.setup_explanation[:200]}...")

        # Check for setup completeness
        results['has_setup'] = bool(plan.entry_zone_low or plan.stop_loss or plan.targets)

        # Check if thesis mentions new indicators
        thesis_lower = (plan.thesis or "").lower()
        new_indicator_keywords = ['cmf', 'money flow', 'a/d', 'accumulation', 'distribution',
                                  'ichimoku', 'cloud', 'williams', 'parabolic', 'sar', 'divergen']
        results['uses_new_indicators'] = any(kw in thesis_lower for kw in new_indicator_keywords)

        print(f"\n{'='*60}")
        print(f"VALIDATION")
        print(f"{'='*60}")
        print(f"    ✓ Plan Generated: {results['plan_generated']}")
        print(f"    {'✓' if results['has_thesis'] else '✗'} Has Thesis: {results['has_thesis']}")
        print(f"    {'✓' if results['has_setup'] else '✗'} Has Setup: {results['has_setup']}")
        print(f"    {'✓' if results['uses_new_indicators'] else '○'} Uses New Indicators: {results['uses_new_indicators']} (may not always be relevant)")

        results['passed'] = results['plan_generated'] and results['has_thesis']

    except Exception as e:
        print(f"    ERROR generating plan: {e}")
        import traceback
        traceback.print_exc()
        return results

    return results


async def run_full_integration_test(symbol: str) -> Dict[str, Any]:
    """Run all integration tests for a symbol.

    Args:
        symbol: Stock ticker to test

    Returns:
        Dictionary with all test results
    """
    print("\n" + "="*70)
    print(f"FULL AI INTEGRATION TEST: {symbol}")
    print("="*70)
    print("Testing all shortcomings fixes are integrated into AI flow...")

    results = {
        "symbol": symbol,
        "technical_indicators": None,
        "chart_patterns": None,
        "ai_plan": None,
        "all_passed": False,
    }

    # Test 1: Technical Indicators
    results['technical_indicators'] = await test_technical_indicators(symbol)

    # Test 2: Chart Patterns
    results['chart_patterns'] = await test_chart_patterns(symbol)

    # Test 3: AI Plan Generation
    results['ai_plan'] = await test_ai_plan_generation(symbol)

    # Summary
    print("\n" + "="*70)
    print("INTEGRATION TEST SUMMARY")
    print("="*70)

    tech_pass = results['technical_indicators'].get('passed', False)
    pattern_pass = results['chart_patterns'].get('passed', False)
    ai_pass = results['ai_plan'].get('passed', False)

    print(f"\n  Technical Indicators: {'✓ PASS' if tech_pass else '✗ FAIL'}")
    print(f"  Chart Patterns: {'✓ PASS' if pattern_pass else '✗ FAIL'}")
    print(f"  AI Plan Generation: {'✓ PASS' if ai_pass else '✗ FAIL'}")

    results['all_passed'] = tech_pass and pattern_pass and ai_pass

    print(f"\n  OVERALL: {'✓ ALL TESTS PASSED' if results['all_passed'] else '✗ SOME TESTS FAILED'}")
    print("="*70 + "\n")

    return results


# ============================================================================
# Pytest Test Classes
# ============================================================================

class TestTechnicalIndicators:
    """Test technical indicators integration."""

    def test_core_indicators_available(self, sample_symbol="AAPL"):
        """Test that core indicators are calculated."""
        async def run():
            tech = await get_technical_indicators(sample_symbol)
            assert 'error' not in tech, f"Error: {tech.get('error')}"
            assert tech.get('rsi', {}).get('value') is not None
            assert tech.get('macd', {}).get('signal') is not None
            assert tech.get('emas', {}).get('trend') is not None
        asyncio.run(run())

    def test_institutional_indicators_available(self, sample_symbol="AAPL"):
        """Test that new institutional indicators are calculated."""
        async def run():
            tech = await get_technical_indicators(sample_symbol)
            assert 'error' not in tech, f"Error: {tech.get('error')}"
            # At least some should be available
            available_count = sum([
                tech.get('ichimoku', {}).get('available', False),
                tech.get('williams_r', {}).get('available', False),
                tech.get('parabolic_sar', {}).get('available', False),
                tech.get('cmf', {}).get('available', False),
                tech.get('adl', {}).get('available', False),
            ])
            assert available_count >= 3, f"Only {available_count}/5 institutional indicators available"
        asyncio.run(run())

    def test_volatility_regime_classification(self, sample_symbol="AAPL"):
        """Test that volatility regime is classified."""
        async def run():
            tech = await get_technical_indicators(sample_symbol)
            assert 'error' not in tech
            atr = tech.get('atr', {})
            assert atr.get('volatility_regime') in ['low', 'moderate', 'high']
        asyncio.run(run())


class TestChartPatterns:
    """Test chart pattern detection integration."""

    def test_patterns_have_success_rates(self, sample_symbol="SPY"):
        """Test that detected patterns include success rates."""
        async def run():
            patterns = await get_chart_patterns(sample_symbol)
            if patterns.get('patterns'):
                for p in patterns['patterns']:
                    assert p.get('success_rate') is not None, f"Pattern {p['name']} missing success_rate"
                    assert p.get('success_rate') > 0, f"Pattern {p['name']} has invalid success_rate"
        asyncio.run(run())


class TestAIPlanGeneration:
    """Test AI plan generation integration."""

    def test_ai_generates_plan(self, sample_symbol="AAPL"):
        """Test that AI can generate a trading plan."""
        async def run():
            agent = StockPlanningAgent(sample_symbol, user_id='test')
            plan = await agent.generate_smart_plan()
            assert plan is not None
            assert plan.bias in ['bullish', 'bearish', 'neutral']
            assert plan.confidence >= 0
        asyncio.run(run())


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Run integration tests from command line."""
    parser = argparse.ArgumentParser(
        description="Run AI integration tests for a stock symbol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_ai_integration.py AAPL        # Test with AAPL
  python tests/test_ai_integration.py RXRX        # Test with RXRX
  python tests/test_ai_integration.py TSLA --quick  # Quick test (indicators only)
        """
    )
    parser.add_argument('symbol', help='Stock ticker symbol to test')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test (indicators only, skip AI generation)')

    args = parser.parse_args()
    symbol = args.symbol.upper()

    async def run_tests():
        if args.quick:
            print(f"Running quick test for {symbol}...")
            await test_technical_indicators(symbol)
            await test_chart_patterns(symbol)
        else:
            await run_full_integration_test(symbol)

    asyncio.run(run_tests())


if __name__ == "__main__":
    main()
