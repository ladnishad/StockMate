#!/usr/bin/env python3
"""Test script to verify the complete plan generation workflow with Grok for ALCY."""

import asyncio
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

async def test_plan_generation():
    """Test the complete plan generation workflow."""
    from app.agent.sdk.orchestrator import TradePlanOrchestrator
    from app.agent.providers.factory import get_provider_config
    from app.agent.providers import ModelProvider

    symbol = "ALCY"
    user_id = "test_user"

    print("=" * 70)
    print(f"TESTING PLAN GENERATION FOR {symbol} WITH GROK")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 70)

    # Check provider configuration
    print("\n[STEP 1] Checking Provider Configuration...")
    config = get_provider_config(ModelProvider.GROK)
    print(f"  Provider: GROK")
    print(f"  Planning Model: {config.planning_model}")
    print(f"  Fast Model: {config.fast_model}")
    print(f"  API Key: {'Configured' if config.api_key else 'MISSING!'}")

    print("\n[STEP 2] Initializing Orchestrator...")
    orchestrator = TradePlanOrchestrator(user_id=user_id)

    print("\n[STEP 3] Running Plan Generation Stream...")
    print("-" * 70)

    final_result = None

    try:
        async for event in orchestrator.generate_plan_stream(
            symbol=symbol,
            user_id=user_id,
            force_new=True,
        ):
            event_data = event.model_dump() if hasattr(event, 'model_dump') else event
            event_type = event_data.get("type", "unknown")

            if event_type == "orchestrator_step":
                step = event_data.get("step", "unknown")
                status = event_data.get("status", "")
                findings = event_data.get("findings", [])
                print(f"\n  >> {step.upper().replace('_', ' ')} [{status}]")
                if findings:
                    for f in findings:
                        print(f"     - {f}")

            elif event_type == "subagent_progress":
                progress = event_data.get("progress", {})
                for agent_name, agent_progress in progress.items():
                    status = agent_progress.get("status", "")
                    current_step = agent_progress.get("current_step", "")
                    if current_step:
                        print(f"     [{agent_name}] {status}: {current_step}")

            elif event_type == "subagent_complete":
                agent_name = event_data.get("agent_name", "")
                findings = event_data.get("findings", [])
                print(f"\n  >> COMPLETED: {agent_name}")
                for f in findings:
                    print(f"     - {f}")

            elif event_type == "final_result":
                final_result = event_data
                print("\n" + "=" * 70)
                print("FINAL RESULT RECEIVED")
                print("=" * 70)

            elif event_type == "error":
                print(f"\n  >> ERROR: {event_data.get('message', 'Unknown error')}")

    except Exception as e:
        print(f"\n  EXCEPTION: {e}")
        import traceback
        traceback.print_exc()
        return

    # Print final analysis
    if final_result:
        plan = final_result.get("plan", {})
        alternatives = final_result.get("alternatives", [])
        selected_style = final_result.get("selected_style", "")
        selection_reasoning = final_result.get("selection_reasoning", "")

        print(f"\n{'='*70}")
        print(f"SELECTED: {selected_style.upper()} TRADE")
        print(f"{'='*70}")

        print(f"\nSelection Reasoning:")
        print(f"  {selection_reasoning[:500]}...")

        print(f"\n--- PLAN DETAILS ---")
        print(f"  Symbol:       {plan.get('symbol', 'N/A')}")
        print(f"  Bias:         {plan.get('bias', 'N/A')}")
        print(f"  Confidence:   {plan.get('confidence', 'N/A')}%")
        print(f"  Suitable:     {plan.get('suitable', 'N/A')}")
        print(f"  ATR%:         {plan.get('atr_percent', 'N/A')}")

        print(f"\n--- ENTRY/EXIT ---")
        print(f"  Entry Zone:   ${plan.get('entry_zone_low', 'N/A')} - ${plan.get('entry_zone_high', 'N/A')}")
        print(f"  Stop Loss:    ${plan.get('stop_loss', 'N/A')}")
        print(f"  Target 1:     ${plan.get('target_1', 'N/A')}")
        print(f"  Target 2:     ${plan.get('target_2', 'N/A')}")
        print(f"  Target 3:     ${plan.get('target_3', 'N/A')}")
        print(f"  Risk/Reward:  {plan.get('risk_reward', 'N/A')}")
        print(f"  Hold Period:  {plan.get('holding_period', 'N/A')}")

        print(f"\n--- THESIS ---")
        thesis = plan.get('thesis', 'N/A')
        print(f"  {thesis}")

        print(f"\n--- WHAT TO WATCH ---")
        for item in plan.get('what_to_watch', [])[:5]:
            print(f"  - {item}")

        print(f"\n--- RISK WARNINGS ---")
        for item in plan.get('risk_warnings', [])[:5]:
            print(f"  - {item}")

        print(f"\n--- VISION ANALYSIS ---")
        vision = plan.get('vision_analysis', {})
        print(f"  Trend Quality:    {vision.get('trend_quality', 'N/A')}")
        print(f"  Visual Patterns:  {vision.get('visual_patterns', [])}")
        print(f"  Warning Signs:    {vision.get('warning_signs', [])}")
        print(f"  Conf. Modifier:   {vision.get('confidence_modifier', 'N/A')}")

        print(f"\n--- ALTERNATIVES ({len(alternatives)}) ---")
        for alt in alternatives:
            style = alt.get('trade_style', 'N/A').upper()
            bias = alt.get('bias', 'N/A')
            conf = alt.get('confidence', 'N/A')
            why_not = alt.get('why_not_selected', 'N/A')
            print(f"  {style}: {bias}, {conf}% confidence")
            print(f"    Why not: {why_not}")

        # Show X/Social citations
        all_citations = final_result.get('all_citations', [])
        if all_citations:
            print(f"\n--- X/SOCIAL CITATIONS ({len(all_citations)}) ---")
            # Show X posts first (x.com URLs)
            x_posts = [c for c in all_citations if "x.com" in c]
            other = [c for c in all_citations if "x.com" not in c]
            for url in x_posts[:10]:  # Show first 10 X posts
                print(f"  X: {url}")
            if len(x_posts) > 10:
                print(f"  ... and {len(x_posts) - 10} more X posts")
            for url in other[:5]:  # Show first 5 other sources
                print(f"  Web: {url}")
            if len(other) > 5:
                print(f"  ... and {len(other) - 5} more web sources")
        else:
            print(f"\n--- X/SOCIAL CITATIONS ---")
            print("  No citations returned (Grok X search may not have been used)")

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_plan_generation())
