#!/usr/bin/env python3
"""Detailed test to show ALL data gathered and sent for ALCY analysis."""

import asyncio
import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_detailed_workflow():
    """Test with detailed output of all data gathered."""
    from app.agent.sdk import tools as sdk_tools
    from app.agent.sdk.orchestrator import TradePlanOrchestrator
    from app.agent.providers.factory import get_provider_config, get_user_provider
    from app.agent.providers import ModelProvider
    from app.agent.providers.grok_provider import get_x_search_parameters
    from app.agent.prompts.day_trade_prompt import build_day_trade_prompt
    from app.agent.prompts.swing_trade_prompt import build_swing_trade_prompt
    from app.agent.prompts.position_trade_prompt import build_position_trade_prompt

    symbol = "ALCY"
    user_id = "test_user"

    print("=" * 80)
    print(f"DETAILED WORKFLOW TEST FOR {symbol}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Step 1: Provider Configuration
    print("\n" + "=" * 80)
    print("STEP 1: PROVIDER CONFIGURATION")
    print("=" * 80)
    config = get_provider_config(ModelProvider.GROK)
    print(f"  Provider: GROK")
    print(f"  Planning Model: {config.planning_model}")
    print(f"  Fast Model: {config.fast_model}")
    print(f"  API Key: {'Configured (' + config.api_key[:8] + '...)' if config.api_key else 'MISSING!'}")

    search_params = get_x_search_parameters()
    print(f"\n  X Search Parameters:")
    print(f"    Mode: {search_params.mode}")
    print(f"    Sources: {search_params.sources}")
    print(f"    Return Citations: {search_params.return_citations}")

    # Step 2: Data Gathering - Price Bars
    print("\n" + "=" * 80)
    print("STEP 2: PRICE BARS DATA")
    print("=" * 80)

    timeframes = [
        ("5m", 3, "Day Trade"),
        ("1d", 100, "Swing Trade"),
        ("1w", 52, "Position Trade"),
    ]

    for tf, days, style in timeframes:
        print(f"\n  --- {style} ({tf}, {days} days back) ---")
        try:
            bars_data = await sdk_tools.get_price_bars(symbol, tf, days)
            print(f"    Bars Count: {bars_data.get('bars_count', 'N/A')}")
            print(f"    Current Price: ${bars_data.get('current_price', 'N/A')}")
            print(f"    High (20 bars): ${bars_data.get('high', 'N/A')}")
            print(f"    Low (20 bars): ${bars_data.get('low', 'N/A')}")
            print(f"    ATR: ${bars_data.get('atr', 'N/A')}")
            print(f"    ATR%: {bars_data.get('atr_pct', 'N/A')}%")
            print(f"    Avg Volume: {bars_data.get('average_volume', 'N/A'):,.0f}")
            if bars_data.get('last_bar'):
                lb = bars_data['last_bar']
                print(f"    Last Bar: O=${lb['open']:.2f} H=${lb['high']:.2f} L=${lb['low']:.2f} C=${lb['close']:.2f} V={lb['volume']:,.0f}")
        except Exception as e:
            print(f"    Error: {e}")

    # Step 3: Technical Indicators
    print("\n" + "=" * 80)
    print("STEP 3: TECHNICAL INDICATORS")
    print("=" * 80)

    ema_configs = [
        ([5, 9, 20], "Day Trade"),
        ([9, 21, 50], "Swing Trade"),
        ([21, 50, 200], "Position Trade"),
    ]

    for ema_periods, style in ema_configs:
        print(f"\n  --- {style} (EMAs: {ema_periods}) ---")
        try:
            indicators = await sdk_tools.get_technical_indicators(symbol, ema_periods, 14)
            print(f"    EMA Trend: {indicators.get('ema_trend', 'N/A')}")
            for period in ema_periods:
                ema_key = f"ema_{period}"
                if ema_key in indicators:
                    print(f"    EMA({period}): ${indicators[ema_key]}")
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                print(f"    RSI(14): {rsi.get('value', 'N/A')} - {rsi.get('signal', 'N/A')}")
            if 'macd' in indicators:
                macd = indicators['macd']
                print(f"    MACD: {macd.get('macd_line', 'N/A'):.4f}, Signal: {macd.get('signal_line', 'N/A'):.4f}")
                print(f"    MACD Histogram: {macd.get('histogram', 'N/A'):.4f} ({macd.get('histogram_trend', 'N/A')})")
        except Exception as e:
            print(f"    Error: {e}")

    # Step 4: Support/Resistance
    print("\n" + "=" * 80)
    print("STEP 4: SUPPORT/RESISTANCE LEVELS")
    print("=" * 80)

    sr_types = ["intraday", "daily", "weekly"]
    for sr_type in sr_types:
        print(f"\n  --- {sr_type.capitalize()} S/R ---")
        try:
            sr_levels = await sdk_tools.get_support_resistance(symbol, sr_type)
            supports = sr_levels.get('support', [])
            resistances = sr_levels.get('resistance', [])
            print(f"    Supports: {[f'${s['price']:.2f}' for s in supports[:3]]}")
            print(f"    Resistances: {[f'${r['price']:.2f}' for r in resistances[:3]]}")
        except Exception as e:
            print(f"    Error: {e}")

    # Step 5: Chart Generation
    print("\n" + "=" * 80)
    print("STEP 5: CHART GENERATION")
    print("=" * 80)

    chart_configs = [
        ("5m", 3, "Day Trade"),
        ("1d", 100, "Swing Trade"),
        ("1w", 52, "Position Trade"),
    ]

    for tf, days, style in chart_configs:
        print(f"\n  --- {style} Chart ({tf}) ---")
        try:
            chart_result = await sdk_tools.generate_chart(symbol, tf, days)
            if 'chart_image_base64' in chart_result:
                img_len = len(chart_result['chart_image_base64'])
                print(f"    Chart Generated: {img_len:,} bytes (base64)")
                print(f"    Timeframe: {chart_result.get('timeframe', 'N/A')}")
                print(f"    Bars Used: {chart_result.get('bars_count', 'N/A')}")
            else:
                print(f"    Error: {chart_result.get('error', 'Unknown')}")
        except Exception as e:
            print(f"    Error: {e}")

    # Step 6: Vision Analysis (requires provider)
    print("\n" + "=" * 80)
    print("STEP 6: VISION ANALYSIS (Chart Reading)")
    print("=" * 80)

    provider = await get_user_provider(user_id)
    print(f"  Using Provider: {type(provider).__name__}")

    for tf, days, style in [("1d", 100, "Swing Trade")]:  # Just test one
        print(f"\n  --- {style} Vision Analysis ---")
        try:
            chart_result = await sdk_tools.generate_chart(symbol, tf, days)
            if 'chart_image_base64' in chart_result:
                vision_result = await sdk_tools.analyze_chart_vision(
                    symbol,
                    chart_result['chart_image_base64'],
                    style.lower().replace(" ", "_").replace("_trade", ""),
                    provider
                )
                print(f"    Trend Quality: {vision_result.get('trend_quality', 'N/A')}")
                print(f"    Visual Patterns: {vision_result.get('visual_patterns', [])}")
                print(f"    Candlestick Patterns: {vision_result.get('candlestick_patterns', [])}")
                print(f"    EMA Structure: {vision_result.get('ema_structure', 'N/A')}")
                print(f"    Volume Confirmation: {vision_result.get('volume_confirmation', 'N/A')}")
                print(f"    Warning Signs: {vision_result.get('warning_signs', [])}")
                print(f"    Confidence Modifier: {vision_result.get('confidence_modifier', 0)}")
                print(f"    Summary: {vision_result.get('summary', 'N/A')}")
        except Exception as e:
            print(f"    Error: {e}")

    # Step 7: Show the Prompts
    print("\n" + "=" * 80)
    print("STEP 7: AGENT PROMPTS (What's sent to Grok)")
    print("=" * 80)

    position_context = "No existing position."
    news_context = "News Sentiment: bullish | Recent Headlines: Market optimism continues"

    prompts = [
        ("Day Trade", build_day_trade_prompt(symbol, position_context, news_context)),
        ("Swing Trade", build_swing_trade_prompt(symbol, position_context, news_context)),
        ("Position Trade", build_position_trade_prompt(symbol, position_context, news_context)),
    ]

    for name, prompt in prompts:
        print(f"\n  --- {name} Prompt (first 1500 chars) ---")
        print(f"  {prompt[:1500]}...")
        print(f"\n  [Total prompt length: {len(prompt)} characters]")

    # Step 8: Full Orchestrator Run
    print("\n" + "=" * 80)
    print("STEP 8: FULL ORCHESTRATOR RUN")
    print("=" * 80)

    orchestrator = TradePlanOrchestrator(user_id=user_id)

    final_result = None
    all_events = []

    try:
        async for event in orchestrator.generate_plan_stream(
            symbol=symbol,
            user_id=user_id,
            force_new=True,
        ):
            event_data = event.model_dump() if hasattr(event, 'model_dump') else event
            event_type = event_data.get("type", "unknown")
            all_events.append(event_data)

            if event_type == "orchestrator_step":
                step = event_data.get("step", "unknown")
                status = event_data.get("status", "")
                findings = event_data.get("findings", [])
                print(f"\n  >> {step.upper().replace('_', ' ')} [{status}]")
                for f in findings[:5]:
                    print(f"     - {f}")

            elif event_type == "subagent_complete":
                agent_name = event_data.get("agent_name", "")
                findings = event_data.get("findings", [])
                print(f"\n  >> AGENT COMPLETE: {agent_name}")
                for f in findings[:5]:
                    print(f"     - {f}")

            elif event_type == "final_result":
                final_result = event_data
                print("\n  >> FINAL RESULT RECEIVED")

            elif event_type == "error":
                print(f"\n  >> ERROR: {event_data.get('message', 'Unknown')}")

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()

    # Step 9: Final Results Summary
    if final_result:
        print("\n" + "=" * 80)
        print("STEP 9: FINAL RESULTS SUMMARY")
        print("=" * 80)

        plan = final_result.get("plan", {})
        alternatives = final_result.get("alternatives", [])
        all_citations = final_result.get("all_citations", [])

        print(f"\n  Selected Style: {final_result.get('selected_style', 'N/A').upper()}")
        print(f"  Symbol: {plan.get('symbol', 'N/A')}")
        print(f"  Bias: {plan.get('bias', 'N/A')}")
        print(f"  Confidence: {plan.get('confidence', 'N/A')}%")
        print(f"  Suitable: {plan.get('suitable', 'N/A')}")
        print(f"  ATR%: {plan.get('atr_percent', 'N/A')}")

        print(f"\n  Entry: ${plan.get('entry_zone_low', 'N/A')} - ${plan.get('entry_zone_high', 'N/A')}")
        print(f"  Stop: ${plan.get('stop_loss', 'N/A')}")
        print(f"  Targets: ${plan.get('target_1', 'N/A')}, ${plan.get('target_2', 'N/A')}, ${plan.get('target_3', 'N/A')}")
        print(f"  R:R: {plan.get('risk_reward', 'N/A')}")

        print(f"\n  THESIS:")
        thesis = plan.get('thesis', 'N/A')
        # Wrap thesis for readability
        words = thesis.split()
        line = "    "
        for word in words:
            if len(line) + len(word) > 100:
                print(line)
                line = "    " + word + " "
            else:
                line += word + " "
        if line.strip():
            print(line)

        print(f"\n  X/SOCIAL CITATIONS: {len(all_citations)} total")
        x_posts = [c for c in all_citations if "x.com" in c]
        web_sources = [c for c in all_citations if "x.com" not in c]
        print(f"    X Posts: {len(x_posts)}")
        for url in x_posts[:5]:
            print(f"      - {url}")
        print(f"    Web Sources: {len(web_sources)}")
        for url in web_sources[:3]:
            print(f"      - {url}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_detailed_workflow())
