"""Test full plan generation for a symbol with detailed output."""
import asyncio
import sys
import json
import logging

sys.path.insert(0, '.')

# Set up logging to see orchestrator details
logging.basicConfig(
    level=logging.INFO,
    format='%(name)s - %(message)s'
)

async def test_plan_generation(symbol: str):
    """Run full plan generation and display all details."""
    from app.agent.sdk.orchestrator import TradePlanOrchestrator
    from app.agent.sdk.tools import get_news_sentiment, get_fundamentals
    
    print("=" * 80)
    print(f"FULL PLAN GENERATION TEST FOR ${symbol}")
    print("=" * 80)
    
    # =========================================================================
    # Step 1: Test News Fetch Directly
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 1: NEWS DATA (from Finnhub)")
    print("=" * 80)
    
    news_data = await get_news_sentiment(symbol)
    print(f"\n📰 Sentiment: {news_data.get('sentiment', 'N/A')} (score: {news_data.get('sentiment_score', 0):.3f})")
    print(f"📰 Articles: {news_data.get('article_count', 0)}")
    print(f"📰 Breaking News: {news_data.get('has_breaking_news', False)}")
    print(f"📰 Key Themes: {news_data.get('key_themes', [])}")
    print(f"📰 Summary: {news_data.get('summary', 'N/A')}")
    
    if news_data.get('headlines'):
        print("\n📰 Headlines:")
        for h in news_data['headlines'][:5]:
            print(f"   • {h[:75]}{'...' if len(h) > 75 else ''}")
    
    # =========================================================================
    # Step 2: Test Fundamentals Fetch Directly
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 2: FUNDAMENTALS DATA (from Finnhub)")
    print("=" * 80)
    
    fund_data = await get_fundamentals(symbol)
    if fund_data.get('error'):
        print(f"\n❌ Error: {fund_data['error']}")
    else:
        print(f"\n💰 Valuation:")
        print(f"   P/E Ratio: {fund_data.get('pe_ratio', 'N/A')}")
        print(f"   P/B Ratio: {fund_data.get('pb_ratio', 'N/A')}")
        print(f"   P/S Ratio: {fund_data.get('ps_ratio', 'N/A')}")
        print(f"   Market Cap: ${fund_data.get('market_cap', 0):,.0f}M" if fund_data.get('market_cap') else "   Market Cap: N/A")
        
        print(f"\n📈 Growth:")
        print(f"   EPS Growth YoY: {fund_data.get('eps_growth_yoy', 'N/A')}")
        print(f"   Revenue Growth YoY: {fund_data.get('revenue_growth_yoy', 'N/A')}")
        
        print(f"\n🏥 Financial Health:")
        print(f"   Health Score: {fund_data.get('financial_health', 'N/A')}")
        print(f"   Debt/Equity: {fund_data.get('debt_to_equity', 'N/A')}")
        print(f"   Current Ratio: {fund_data.get('current_ratio', 'N/A')}")
        
        print(f"\n📅 Earnings:")
        print(f"   Has Earnings Risk: {fund_data.get('has_earnings_risk', False)}")
        print(f"   Days Until Earnings: {fund_data.get('days_until_earnings', 'N/A')}")
        print(f"   Beat Rate: {fund_data.get('earnings_beat_rate', 'N/A')}")
    
    # =========================================================================
    # Step 3: Run Full Orchestrator
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: RUNNING ORCHESTRATOR")
    print("=" * 80)
    
    orchestrator = TradePlanOrchestrator(user_id="test-user")
    
    events = []
    final_response = None
    
    print("\n🔄 Streaming events...")
    async for event in orchestrator.generate_plan_stream(symbol, "test-user"):
        events.append(event)
        
        # Print key events (StreamEvent uses 'type' not 'event_type')
        if event.type == "orchestrator_step":
            step = event.step_type or ''
            status = event.step_status or ''
            findings = event.step_findings or []
            print(f"\n   [{step}] {status}")
            if findings:
                for f in findings:
                    print(f"      • {f}")
        
        elif event.type == "subagent_progress":
            if event.subagents:
                for agent, info in event.subagents.items():
                    status = info.status if hasattr(info, 'status') else 'unknown'
                    if status != 'pending':
                        print(f"   [{agent}] {status}")
        
        elif event.type == "subagent_complete":
            print(f"\n   ✅ [{event.agent_name}] COMPLETE")
            if event.agent_findings:
                for f in event.agent_findings[:3]:
                    print(f"      • {f}")
        
        elif event.type == "final_result":
            final_response = event
    
    # =========================================================================
    # Step 4: Display DataContext passed to agents
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: DATA CONTEXT (Passed to All Agents)")
    print("=" * 80)
    
    ctx = orchestrator.context
    if ctx:
        print(f"\n📊 Price: ${ctx.current_price:.2f}")
        print(f"📊 Market Direction: {ctx.market_direction} ({ctx.bullish_indices}/4 indices bullish)")
        print(f"📊 Has Position: {ctx.has_position}")
        
        print(f"\n📰 News Context:")
        print(f"   Sentiment: {ctx.news_sentiment}")
        print(f"   Score: {ctx.news_score}")
        print(f"   Articles: {ctx.news_article_count}")
        print(f"   Breaking: {ctx.news_has_breaking}")
        print(f"   Themes: {ctx.news_key_themes}")
        
        if ctx.fundamentals:
            print(f"\n💰 Fundamentals Context:")
            print(f"   P/E: {ctx.fundamentals.pe_ratio}")
            print(f"   Health: {ctx.fundamentals.get_financial_health_score()}")
            print(f"   Valuation: {ctx.fundamentals.get_valuation_assessment()}")
        
        print(f"\n⚠️  Risk Flags:")
        print(f"   Earnings Risk: {ctx.has_earnings_risk}")
        print(f"   Days Until Earnings: {ctx.days_until_earnings}")
    
    # =========================================================================
    # Step 5: Display Each Agent's Analysis from final response
    # =========================================================================
    if final_response:
        print("\n" + "=" * 80)
        print("STEP 5: SUB-AGENT REPORTS")
        print("=" * 80)
        
        # Get reports from final_response attributes
        all_reports = []
        if hasattr(final_response, 'selected_plan') and final_response.selected_plan:
            all_reports.append(('SELECTED', final_response.selected_plan))
        if hasattr(final_response, 'alternatives'):
            for alt in (final_response.alternatives or []):
                all_reports.append(('ALTERNATIVE', alt))
        
        for label, report in all_reports:
            print(f"\n{'━' * 40}")
            style = (report.trade_style if hasattr(report, 'trade_style') else 'unknown').upper()
            print(f"[{label}] {style} TRADE ANALYZER")
            print(f"{'━' * 40}")
            
            suitable = report.suitable if hasattr(report, 'suitable') else 'N/A'
            bias = report.bias if hasattr(report, 'bias') else 'N/A'
            confidence = report.confidence if hasattr(report, 'confidence') else 'N/A'
            rr = report.risk_reward if hasattr(report, 'risk_reward') else 'N/A'
            holding = report.holding_period if hasattr(report, 'holding_period') else 'N/A'
            
            print(f"\n   Suitable: {suitable}")
            print(f"   Bias: {bias}")
            print(f"   Confidence: {confidence}%")
            print(f"   Risk/Reward: {rr}")
            print(f"   Holding Period: {holding}")
            
            thesis = report.thesis if hasattr(report, 'thesis') else 'N/A'
            print(f"\n   📝 Thesis:")
            # Word wrap thesis
            words = thesis.split()
            line = "      "
            for word in words:
                if len(line) + len(word) > 75:
                    print(line)
                    line = "      " + word + " "
                else:
                    line += word + " "
            if line.strip():
                print(line)
            
            entry = report.entry_zone if hasattr(report, 'entry_zone') else {}
            stop = report.stop_loss if hasattr(report, 'stop_loss') else 'N/A'
            print(f"\n   🎯 Entry: ${entry.get('ideal', 'N/A') if isinstance(entry, dict) else entry}")
            print(f"   🛑 Stop: ${stop}")
            
            targets = report.targets if hasattr(report, 'targets') else []
            if targets:
                print(f"   🎯 Targets: {', '.join([f'${t}' for t in targets[:3]])}")
            
            warnings = report.risk_warnings if hasattr(report, 'risk_warnings') else []
            if warnings:
                print(f"\n   ⚠️  Risk Warnings:")
                for w in warnings[:3]:
                    print(f"      • {w[:70]}{'...' if len(w) > 70 else ''}")
        
        # =========================================================================
        # Step 6: Final Selection
        # =========================================================================
        print("\n" + "=" * 80)
        print("STEP 6: FINAL RECOMMENDATION")
        print("=" * 80)
        
        sel_style = final_response.selected_style if hasattr(final_response, 'selected_style') else 'N/A'
        sel_reason = final_response.selection_reasoning if hasattr(final_response, 'selection_reasoning') else 'N/A'
        
        print(f"\n✅ Selected Style: {sel_style.upper() if sel_style else 'N/A'}")
        print(f"✅ Selection Reasoning: {sel_reason}")
        
        if hasattr(final_response, 'selected_plan') and final_response.selected_plan:
            selected = final_response.selected_plan
            entry = selected.entry_zone if hasattr(selected, 'entry_zone') else {}
            print(f"\n📊 Final Plan Summary:")
            print(f"   Bias: {selected.bias if hasattr(selected, 'bias') else 'N/A'}")
            print(f"   Confidence: {selected.confidence if hasattr(selected, 'confidence') else 'N/A'}%")
            print(f"   Entry: ${entry.get('ideal', 'N/A') if isinstance(entry, dict) else entry}")
            print(f"   Stop: ${selected.stop_loss if hasattr(selected, 'stop_loss') else 'N/A'}")
            print(f"   R:R: {selected.risk_reward if hasattr(selected, 'risk_reward') else 'N/A'}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    symbol = sys.argv[1] if len(sys.argv) > 1 else "NBIS"
    asyncio.run(test_plan_generation(symbol))
