# V2 Trading Plan Generation API

## Overview

The V2 Planning API introduces a parallel sub-agent architecture using the Claude Agent SDK pattern. Instead of a single monolithic analysis, three specialized agents analyze the stock simultaneously for different trade styles, then an orchestrator synthesizes the best recommendation.

## Architecture

```
                    ┌─────────────────────────────────────┐
                    │         TradePlanOrchestrator       │
                    │                                     │
                    │  1. Gather Common Data              │
                    │     - Current price                 │
                    │     - Position status               │
                    │     - Market context                │
                    │     - News sentiment                │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────┴──────────────────────┐
                    │     2. Parallel Sub-Agent Dispatch   │
                    └──────────────┬──────────────────────┘
                                   │
          ┌────────────────────────┼────────────────────────┐
          │                        │                        │
          ▼                        ▼                        ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Day Trade Agent │    │ Swing Trade Agent│    │Position Trade   │
│                 │    │                 │    │     Agent       │
│ - 5m/15m bars   │    │ - Daily bars    │    │ - Weekly bars   │
│ - EMA 5,9,20    │    │ - EMA 9,21,50   │    │ - EMA 21,50,200 │
│ - Intraday S/R  │    │ - Daily S/R     │    │ - Weekly S/R    │
│ - 5-min chart   │    │ - Daily chart   │    │ - Weekly chart  │
│ - Vision AI     │    │ - Vision AI     │    │ - Vision AI     │
└────────┬────────┘    └────────┬────────┘    └────────┬────────┘
         │                      │                      │
         └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │   3. Claude Synthesis  │
                    │                       │
                    │ - Compare all 3 reports│
                    │ - Select best plan    │
                    │ - Generate targets    │
                    │ - Create thesis       │
                    │ - Add news context    │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │   4. Final Response    │
                    │                       │
                    │ - Selected plan       │
                    │ - Alternatives        │
                    │ - Selection reasoning │
                    └───────────────────────┘
```

## API Endpoint

### Generate Trading Plan (V2)

```
POST /plan/{symbol}/generate/v2?user_id={user_id}
```

**Response**: Server-Sent Events (SSE) stream

## Streaming Events

The API streams progress events to the iOS app in real-time:

### 1. Orchestrator Step Events

```json
{
  "type": "orchestrator_step",
  "step_type": "gathering_common_data",
  "status": "active" | "completed",
  "findings": ["Price: $450.07", "Market: Bullish (3/4)", "Position: None", "News: Neutral"]
}
```

Step types:
- `gathering_common_data` - Fetching price, position, market, news
- `spawning_subagents` - Starting parallel analyzers
- `selecting_best` - Synthesizing final plan
- `complete` - Generation finished

### 2. Sub-Agent Progress Events

```json
{
  "type": "subagent_progress",
  "subagents": {
    "day-trade-analyzer": {
      "agent_name": "day-trade-analyzer",
      "display_name": "Day Trade",
      "status": "gathering_data" | "generating_chart" | "analyzing_chart" | "generating_plan",
      "current_step": "Gathering 5-min bars",
      "steps_completed": ["Started", "Bars fetched"],
      "findings": [],
      "elapsed_ms": 1234
    },
    "swing-trade-analyzer": { ... },
    "position-trade-analyzer": { ... }
  }
}
```

Sub-agent statuses:
- `pending` - Not started
- `running` - In progress
- `gathering_data` - Fetching price bars
- `generating_chart` - Creating candlestick chart
- `analyzing_chart` - Vision AI analysis
- `generating_plan` - Claude generating analysis
- `completed` - Done
- `failed` - Error occurred

### 3. Sub-Agent Complete Events

```json
{
  "type": "subagent_complete",
  "agent_name": "swing-trade-analyzer",
  "findings": ["Bullish", "78% confidence", "Setup: Yes"]
}
```

### 4. Final Result Event

```json
{
  "type": "final_result",
  "selected_style": "swing",
  "selection_reasoning": "Swing trade selected due to higher confidence...",
  "plan": {
    "trade_style": "swing",
    "symbol": "TSLA",
    "suitable": true,
    "confidence": 78,
    "bias": "bullish",
    "thesis": "Detailed 3-4 sentence thesis with news context...",
    "entry_zone_low": 445.00,
    "entry_zone_high": 448.50,
    "entry_reasoning": "Near support at $445...",
    "stop_loss": 438.00,
    "stop_reasoning": "Below key support...",
    "targets": [
      {"price": 465.00, "reasoning": "First resistance level"},
      {"price": 480.00, "reasoning": "Prior swing high"},
      {"price": 495.00, "reasoning": "52-week high approach"}
    ],
    "risk_reward": 2.5,
    "holding_period": "3-7 days",
    "what_to_watch": [
      "Break above $455 triggers acceleration",
      "Volume needs to exceed 50M on breakout",
      "Watch for earnings on Jan 15"
    ],
    "risk_warnings": [
      "Broad market weakness could drag stock down",
      "RSI approaching overbought at 65"
    ],
    "position_recommendation": "hold" | "trim" | "reduce" | "exit" | null,
    "position_aligned": true
  },
  "alternatives": [
    {
      "trade_style": "day",
      "bias": "bullish",
      "confidence": 65,
      "suitable": true,
      "brief_thesis": "Intraday momentum play...",
      "why_not_selected": "Lower confidence (65% vs 78%)"
    }
  ]
}
```

## Data Gathered Per Trade Style

| Data Type | Day Trade | Swing Trade | Position Trade |
|-----------|-----------|-------------|----------------|
| **Bars** | 5-min (3 days) | Daily (100 days) | Weekly (52 weeks) |
| **Chart** | 5-min candles | Daily candles | Weekly candles |
| **EMAs** | 5, 9, 20 | 9, 21, 50 | 21, 50, 200 |
| **S/R Levels** | Intraday | Daily | Weekly |
| **ATR Focus** | > 2.5% | 1-3.5% | < 2% |

## News & Sentiment Integration

The orchestrator gathers news sentiment from Alpaca News API:

```python
context = DataContext(
    # ... other fields
    news_sentiment="bullish" | "bearish" | "neutral",
    news_summary="10 articles analyzed. News volume: increasing",
    recent_headlines=["TSLA announces new factory...", "Analyst upgrade..."]
)
```

News is:
1. Shown in orchestrator findings
2. Passed to all sub-agents via prompt context
3. Used in final Claude synthesis for thesis generation
4. Affects what_to_watch and risk_warnings

## Final Synthesis Step

After all 3 sub-agents complete, the orchestrator calls Claude to synthesize a comprehensive plan:

1. **Selection Logic** (programmatic):
   - Position alignment (never recommend opposite direction)
   - Setup suitability (valid setup for style)
   - Confidence score (higher is better)
   - Risk/reward ratio

2. **Claude Synthesis** (AI-generated):
   - Comprehensive thesis incorporating all analyses
   - Price targets with reasoning (3 levels)
   - What to watch items (5-7 specific triggers)
   - Risk warnings (3-5 specific risks)
   - Entry/stop reasoning
   - News impact assessment

## Position Management

When user has an existing position, each sub-agent evaluates:

| P&L Range | Bullish Technicals | Bearish Technicals |
|-----------|-------------------|-------------------|
| +50% or more | trim | trim |
| +20% to +50% | hold (tighten stop) | trim |
| 0% to +20% | hold | reduce |
| -10% to 0% | hold | reduce |
| < -10% | hold (reassess) | exit |

The `position_recommendation` field indicates: `hold`, `trim`, `reduce`, or `exit`.

## Files Structure

### Backend (Python)

```
app/
├── agent/
│   ├── sdk/
│   │   ├── __init__.py
│   │   ├── orchestrator.py      # Main TradePlanOrchestrator class
│   │   ├── tools.py             # SDK tools (get_price_bars, get_news_sentiment, etc.)
│   │   ├── subagent_definitions.py  # Agent definitions
│   │   └── streaming.py         # StreamEvent models
│   ├── schemas/
│   │   ├── final_response.py    # FinalPlanResponse, DataContext, AlternativePlan
│   │   ├── subagent_report.py   # SubAgentReport model
│   │   └── streaming.py         # SubAgentProgress, SubAgentStatus
│   └── prompts/
│       ├── day_trade_prompt.py
│       ├── swing_trade_prompt.py
│       └── position_trade_prompt.py
└── main.py                      # /plan/{symbol}/generate/v2 endpoint
```

### iOS (Swift)

```
ios/StockMate/StockMate/
├── ViewModels/
│   ├── PlanGenerationManager.swift   # Singleton for background generation
│   └── TradingPlanViewModel.swift    # ViewModel with orchestrator steps
├── Views/Plan/
│   └── SimplifiedPlanView.swift      # UI with V2SubAgentsView, OrchestratorStepRow
└── Services/
    └── APIService.swift              # SSE streaming support
```

## Key Models

### SubAgentReport

```python
class SubAgentReport(BaseModel):
    trade_style: Literal["day", "swing", "position"]
    symbol: str
    suitable: bool
    confidence: int  # 0-100
    bias: Literal["bullish", "bearish", "neutral"]
    thesis: str
    vision_analysis: VisionAnalysisResult
    entry_zone_low: float
    entry_zone_high: float
    entry_reasoning: str
    stop_loss: float
    stop_reasoning: str
    targets: List[PriceTargetWithReasoning]
    risk_reward: float
    holding_period: str
    what_to_watch: List[str]
    risk_warnings: List[str]
    position_recommendation: Optional[str]  # hold/trim/reduce/exit
    position_aligned: bool
```

### OrchestratorStep (iOS)

```swift
struct OrchestratorStep: Identifiable, Equatable {
    let stepType: String
    var status: Status  // .active or .completed
    var findings: [String]

    var displayName: String {
        switch stepType {
        case "gathering_common_data": return "Gathering Market Data"
        case "spawning_subagents": return "Starting Analyzers"
        case "selecting_best": return "Selecting Best Plan"
        case "complete": return "Complete"
        }
    }
}
```

## Environment Variables

```bash
# Default: false (use real sub-agents)
USE_SIMULATED_SUBAGENTS=false

# Required for Claude API calls
CLAUDE_API_KEY=sk-ant-...
```

## Testing

To test with simulated sub-agents (faster, no API calls):
```bash
export USE_SIMULATED_SUBAGENTS=true
```

## Migration from V1

The V1 endpoint (`/chat/{symbol}/plan/stream`) remains available for backward compatibility. V2 provides:

1. **Parallel analysis** - 3 trade styles analyzed simultaneously
2. **Better UI feedback** - Orchestrator + sub-agent progress
3. **News integration** - Sentiment affects thesis and recommendations
4. **Position-aware recommendations** - Hold/trim/reduce/exit guidance
5. **Synthesized targets** - AI-generated price targets with reasoning

## Changelog

### 2024-12-31

- **Initial V2 Implementation**
  - Created TradePlanOrchestrator with parallel sub-agents
  - Added orchestrator step streaming for iOS UI
  - Integrated news sentiment from Alpaca API

- **Fixes**
  - Fixed simulation mode default (USE_SIMULATED_SUBAGENTS=false)
  - Added real Claude API calls for plan generation
  - Fixed null value handling when Claude returns null fields

- **Enhancements**
  - Added final Claude synthesis step for comprehensive plans
  - Added news_sentiment, news_summary, recent_headlines to DataContext
  - Added targets with reasoning to synthesis
  - Added news_impact to thesis generation
  - iOS UI now shows orchestrator steps alongside sub-agent cards
