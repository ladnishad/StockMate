"""Test script for the new agentic mode analyzer.

Run with: python test_agentic_mode.py NBIS
"""

import asyncio
import json
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()


async def test_agentic_mode(symbol: str):
    """Run agentic analysis on a symbol and display progress."""

    from app.agent.sdk.orchestrator import TradePlanOrchestrator

    print(f"\n{'='*60}")
    print(f"  AGENTIC MODE TEST - {symbol}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    orchestrator = TradePlanOrchestrator()

    iteration_count = 0
    tool_count = 0

    try:
        async for event in orchestrator.generate_plan_stream(
            symbol=symbol,
            user_id="test-user",
            agentic_mode=True,
        ):
            event_type = event.type if hasattr(event, 'type') else event.get('type', 'unknown')

            # Handle different event types
            if event_type == "orchestrator_step":
                step = event.step_type if hasattr(event, 'step_type') else event.data.get('step', '')
                status = event.step_status if hasattr(event, 'step_status') else event.data.get('status', '')
                print(f"\n📍 Orchestrator: {step} ({status})")

            elif event_type == "agent_thinking":
                thinking = event.thinking if hasattr(event, 'thinking') else event.data.get('thinking', '')
                iteration = event.iteration if hasattr(event, 'iteration') else event.data.get('iteration', '')
                if thinking:
                    iteration_count = iteration or iteration_count + 1
                    print(f"\n💭 AI Thinking (iteration {iteration_count}):")
                    # Truncate long thinking to first 500 chars
                    display_text = thinking[:500] + "..." if len(thinking) > 500 else thinking
                    print(f"   {display_text}")

            elif event_type == "tool_call":
                tool_name = event.tool_name if hasattr(event, 'tool_name') else event.data.get('tool', '')
                args = event.tool_arguments if hasattr(event, 'tool_arguments') else event.data.get('arguments', {})
                tool_count += 1
                print(f"\n🔧 Tool Call #{tool_count}: {tool_name}")
                if args:
                    args_str = json.dumps(args, indent=2)
                    print(f"   Args: {args_str}")

            elif event_type == "tool_result":
                tool_name = event.tool_name if hasattr(event, 'tool_name') else event.data.get('tool', '')
                result = event.tool_result if hasattr(event, 'tool_result') else event.data.get('result', {})
                print(f"\n📊 Tool Result: {tool_name}")
                # Pretty print result (truncated)
                result_str = json.dumps(result, indent=2, default=str)
                if len(result_str) > 800:
                    result_str = result_str[:800] + "\n   ... (truncated)"
                print(f"   {result_str}")

            elif event_type == "final_result":
                plan = event.plan if hasattr(event, 'plan') else event.data.get('plan', {})
                iterations = event.data.get('iterations', iteration_count) if hasattr(event, 'data') else iteration_count
                tools_called = event.data.get('tools_called', tool_count) if hasattr(event, 'data') else tool_count
                elapsed = event.data.get('elapsed_seconds', 0) if hasattr(event, 'data') else 0

                print(f"\n{'='*60}")
                print(f"  ✅ FINAL RESULT")
                print(f"{'='*60}")
                print(f"\n📈 Symbol: {symbol}")
                print(f"⏱️  Iterations: {iterations}")
                print(f"🔧 Tools called: {tools_called}")
                print(f"⏰ Time: {elapsed:.2f}s")

                if plan:
                    print(f"\n🎯 Recommended Style: {plan.get('recommended_style', 'N/A').upper()}")
                    print(f"\n📝 Reasoning:")
                    reasoning = plan.get('recommendation_reasoning', 'N/A')
                    print(f"   {reasoning[:500]}..." if len(reasoning) > 500 else f"   {reasoning}")

                    # Show each plan summary
                    for style in ['day_trade_plan', 'swing_trade_plan', 'position_trade_plan']:
                        style_plan = plan.get(style, {})
                        if style_plan:
                            print(f"\n{'─'*40}")
                            print(f"📋 {style.replace('_', ' ').title()}")
                            print(f"   Suitable: {style_plan.get('suitable', 'N/A')}")
                            print(f"   Conviction: {style_plan.get('conviction', 'N/A').upper() if style_plan.get('conviction') else 'N/A'}")
                            print(f"   Bias: {style_plan.get('bias', 'N/A')}")
                            conviction_reason = style_plan.get('conviction_reasoning', '')
                            if conviction_reason:
                                print(f"   Why: {conviction_reason[:200]}...")
                            entry = style_plan.get('entry_zone', [])
                            if entry:
                                print(f"   Entry: ${entry[0]:.2f} - ${entry[1]:.2f}" if len(entry) >= 2 else f"   Entry: {entry}")
                            stop = style_plan.get('stop_loss')
                            if stop:
                                print(f"   Stop: ${stop:.2f}")
                            targets = style_plan.get('targets', [])
                            if targets:
                                target_str = ", ".join([f"${t:.2f}" if isinstance(t, (int, float)) else str(t) for t in targets[:3]])
                                print(f"   Targets: {target_str}")

            elif event_type == "error":
                error_msg = event.error_message if hasattr(event, 'error_message') else event.data.get('error', 'Unknown error')
                print(f"\n❌ Error: {error_msg}")

            else:
                # Handle generic events with data field
                if hasattr(event, 'data') and event.data:
                    data = event.data
                    if 'thinking' in data:
                        print(f"\n💭 {data['thinking'][:300]}...")
                    elif 'tool' in data:
                        print(f"\n🔧 Tool: {data['tool']}")

    except Exception as e:
        print(f"\n❌ Exception: {e}")
        import traceback
        traceback.print_exc()

    print(f"\n{'='*60}")
    print(f"  TEST COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NBIS"
    asyncio.run(test_agentic_mode(symbol.upper()))
