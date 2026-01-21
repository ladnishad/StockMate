#!/usr/bin/env python3
"""
Test script to generate and save charts for each trade style,
then run vision analysis and display results.
"""
import asyncio
import logging
import base64
import os
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def generate_and_save_charts(symbol: str, output_dir: str = "test_charts"):
    """Generate charts for each trade style and save them as images."""
    from app.agent.sdk import tools as sdk_tools

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Trade style configurations
    trade_configs = {
        "day": {"timeframe": "5m", "days_back": 3},
        "swing": {"timeframe": "1d", "days_back": 100},
        "position": {"timeframe": "1w", "days_back": 364},  # 52 weeks = 364 days
    }

    results = {}

    print(f"\n{'='*70}")
    print(f"GENERATING CHARTS FOR {symbol}")
    print(f"{'='*70}")

    for trade_style, config in trade_configs.items():
        print(f"\n--- {trade_style.upper()} TRADE CHART ({config['timeframe']}, {config['days_back']} periods) ---")

        try:
            # Generate chart
            chart_result = await sdk_tools.generate_chart(
                symbol,
                config["timeframe"],
                config["days_back"]
            )

            if "error" in chart_result:
                print(f"  ERROR: {chart_result['error']}")
                results[trade_style] = {"error": chart_result["error"]}
                continue

            # Save chart as image
            chart_base64 = chart_result.get("chart_image_base64", "")
            if chart_base64:
                filename = f"{output_dir}/{symbol}_{trade_style}_{config['timeframe']}_chart.png"
                with open(filename, "wb") as f:
                    f.write(base64.b64decode(chart_base64))
                print(f"  ✓ Chart saved to: {filename}")
                print(f"  Bars plotted: {chart_result.get('bars_plotted', 'N/A')}")

            results[trade_style] = {
                "chart_saved": filename,
                "bars_plotted": chart_result.get("bars_plotted"),
                "chart_base64": chart_base64,
            }

        except Exception as e:
            print(f"  ERROR generating chart: {e}")
            results[trade_style] = {"error": str(e)}

    return results


async def run_vision_analysis(symbol: str, chart_results: dict):
    """Run vision analysis on the generated charts."""
    from app.agent.sdk import tools as sdk_tools
    from app.agent.providers.factory import get_provider, get_default_provider

    print(f"\n{'='*70}")
    print(f"VISION ANALYSIS FOR {symbol}")
    print(f"{'='*70}")

    # Get the AI provider (Grok or Claude)
    default_provider = get_default_provider()
    provider = get_provider(default_provider)
    print(f"Using provider: {provider.__class__.__name__}")

    vision_results = {}

    for trade_style, chart_data in chart_results.items():
        if "error" in chart_data:
            print(f"\n--- {trade_style.upper()} TRADE: Skipped (chart error) ---")
            continue

        chart_base64 = chart_data.get("chart_base64", "")
        if not chart_base64:
            print(f"\n--- {trade_style.upper()} TRADE: Skipped (no chart data) ---")
            continue

        print(f"\n--- {trade_style.upper()} TRADE VISION ANALYSIS ---")

        try:
            vision_result = await sdk_tools.analyze_chart_vision(
                symbol,
                chart_base64,
                trade_style,
                provider
            )

            vision_results[trade_style] = vision_result

            # Display results
            print(f"  Trend Quality: {vision_result.get('trend_quality', 'N/A')}")
            print(f"  Visual Patterns: {', '.join(vision_result.get('visual_patterns', [])) or 'None detected'}")
            print(f"  Candlestick Patterns: {', '.join(vision_result.get('candlestick_patterns', [])) or 'None detected'}")
            print(f"  EMA Structure: {vision_result.get('ema_structure', 'N/A')}")
            print(f"  Volume Confirmation: {vision_result.get('volume_confirmation', 'N/A')}")
            print(f"  Warning Signs: {', '.join(vision_result.get('warning_signs', [])) or 'None'}")
            print(f"  Confidence Modifier: {vision_result.get('confidence_modifier', 0):+d}")
            print(f"\n  Summary: {vision_result.get('summary', 'N/A')}")

        except Exception as e:
            print(f"  ERROR in vision analysis: {e}")
            import traceback
            traceback.print_exc()
            vision_results[trade_style] = {"error": str(e)}

    return vision_results


async def run_full_orchestrator(symbol: str, output_dir: str = "test_charts"):
    """Run the full orchestrator and capture all intermediate results."""
    from app.agent.sdk.orchestrator import TradePlanOrchestrator

    print(f"\n{'='*70}")
    print(f"FULL ORCHESTRATOR ANALYSIS FOR {symbol}")
    print(f"{'='*70}")

    orchestrator = TradePlanOrchestrator(user_id="test_user")

    agent_details = {}
    final_result = None

    try:
        async for event in orchestrator.generate_plan_stream(symbol, "test_user", force_new=True):
            event_type = event.type

            if event_type == "orchestrator_step":
                step = event.step_type or ""
                status = event.step_status or ""
                print(f"\n[ORCHESTRATOR] {step}: {status}")

            elif event_type == "subagent_complete":
                agent_name = event.agent_name or "unknown"
                findings = event.agent_findings or []
                print(f"\n[{agent_name.upper()}] COMPLETED")
                for f in findings:
                    print(f"  - {f}")

                # Get the full report from orchestrator
                if agent_name in orchestrator.subagent_reports:
                    report = orchestrator.subagent_reports[agent_name]
                    agent_details[agent_name] = {
                        "trade_style": report.trade_style,
                        "suitable": report.suitable,
                        "confidence": report.confidence,
                        "bias": report.bias,
                        "vision_analysis": report.vision_analysis,
                        "thesis": report.thesis,
                        "entry_zone": (report.entry_zone_low, report.entry_zone_high),
                        "stop_loss": report.stop_loss,
                        "targets": [(t.price, t.reasoning) for t in report.targets] if report.targets else [],
                        "technical_summary": report.technical_summary,
                    }

            elif event_type == "final_result":
                final_result = {
                    "plan": event.plan,
                    "alternatives": event.alternatives or [],
                    "selected_style": event.selected_style,
                    "selection_reasoning": event.selection_reasoning,
                }

        # Display detailed agent results
        print(f"\n{'='*70}")
        print("DETAILED AGENT RESULTS")
        print(f"{'='*70}")

        for agent_name, details in agent_details.items():
            print(f"\n{'='*50}")
            print(f"{agent_name.upper()}")
            print(f"{'='*50}")

            print(f"\n  Trade Style: {details['trade_style']}")
            print(f"  Suitable: {details['suitable']}")
            print(f"  Confidence: {details['confidence']}%")
            print(f"  Bias: {details['bias']}")
            print(f"  Technical Summary: {details['technical_summary']}")

            print(f"\n  Entry Zone: ${details['entry_zone'][0]:.2f} - ${details['entry_zone'][1]:.2f}")
            print(f"  Stop Loss: ${details['stop_loss']:.2f}")

            print(f"\n  Targets:")
            for i, (price, reasoning) in enumerate(details['targets'], 1):
                print(f"    {i}. ${price:.2f} - {reasoning}")

            # Vision Analysis
            vision = details.get('vision_analysis')
            if vision:
                print(f"\n  VISION ANALYSIS:")
                print(f"    Trend Quality: {vision.trend_quality}")
                print(f"    Visual Patterns: {', '.join(vision.visual_patterns) if vision.visual_patterns else 'None'}")
                print(f"    Candlestick Patterns: {', '.join(vision.candlestick_patterns) if vision.candlestick_patterns else 'None'}")
                print(f"    EMA Structure: {vision.ema_structure}")
                print(f"    Volume Confirmation: {vision.volume_confirmation}")
                print(f"    Warning Signs: {', '.join(vision.warning_signs) if vision.warning_signs else 'None'}")
                print(f"    Confidence Modifier: {vision.confidence_modifier:+d}")
                print(f"    Summary: {vision.summary}")

            print(f"\n  THESIS:")
            print(f"    {details['thesis'][:500]}..." if len(details['thesis']) > 500 else f"    {details['thesis']}")

        # Final selected plan
        if final_result:
            print(f"\n{'='*70}")
            print("FINAL SELECTED PLAN")
            print(f"{'='*70}")

            plan = final_result.get("plan", {}) or {}
            print(f"\n  Selected Style: {final_result.get('selected_style', 'N/A').upper()}")
            print(f"  Confidence: {plan.get('confidence', 0)}%")
            print(f"  Bias: {plan.get('bias', 'N/A')}")
            print(f"  Suitable: {plan.get('suitable', False)}")
            print(f"\n  Selection Reasoning: {final_result.get('selection_reasoning', 'N/A')}")

        return agent_details, final_result

    except Exception as e:
        logger.error(f"Error in orchestrator: {e}")
        import traceback
        traceback.print_exc()
        return {}, None


async def main():
    """Run the full chart generation and analysis test."""
    symbol = "OSCR"
    output_dir = "test_charts"

    print(f"\n{'#'*70}")
    print(f"# COMPREHENSIVE CHART & VISION ANALYSIS TEST")
    print(f"# Symbol: {symbol}")
    print(f"# Time: {datetime.now().isoformat()}")
    print(f"{'#'*70}")

    # Step 1: Generate and save charts
    chart_results = await generate_and_save_charts(symbol, output_dir)

    # Step 2: Run vision analysis on charts
    vision_results = await run_vision_analysis(symbol, chart_results)

    # Step 3: Run full orchestrator to see complete flow
    agent_details, final_result = await run_full_orchestrator(symbol, output_dir)

    # Summary
    print(f"\n{'#'*70}")
    print("# TEST COMPLETE")
    print(f"{'#'*70}")

    print(f"\nCharts saved to: {output_dir}/")
    for trade_style, data in chart_results.items():
        if "chart_saved" in data:
            print(f"  - {data['chart_saved']}")

    print(f"\nTo view the charts, open the PNG files in the '{output_dir}' directory.")


if __name__ == "__main__":
    asyncio.run(main())
