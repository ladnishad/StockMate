# StockMate - Intelligent Stock Analysis Backend

A production-ready FastAPI backend for stock analysis and trading recommendations, designed with LLM-agent-ready tools and mobile app integration.

## Features

- **Comprehensive Stock Analysis**: Multi-timeframe price action analysis (daily, hourly, 15-minute)
- **Advanced Technical Indicators**:
  - Trend: VWAP, EMA (9/20/50), Bollinger Bands
  - Momentum: RSI, MACD (with crossover detection)
  - Volume: Volume analysis, OBV, relative volume, volume spikes
  - Volatility: ATR (Average True Range) for dynamic stop losses
- **Divergence Detection**: RSI and MACD divergence analysis for reversal signals
  - Regular bullish/bearish divergences (reversal patterns)
  - Hidden divergences (continuation patterns)
  - Automatic swing point detection
- **Enhanced Key Level Detection**: Comprehensive support/resistance identification
  - Round number psychology (100.00, 150.00, etc.)
  - Unfilled gap detection and tracking
  - Previous period highs/lows (day, week, month)
  - Nearest support/resistance calculation
- **News Sentiment Analysis**: Real-time news integration via Alpaca News API
  - Article sentiment scoring
  - News volume trends
  - Weighted recent news more heavily
  - Graceful fallback to price-based sentiment
- **Volume Profile Analysis** (Institutional-Grade): Where the big money is positioned
  - VPOC (Volume Point of Control) - highest traded price level
  - Value Area calculation (70% volume range)
  - High Volume Nodes (HVN) - strong support/resistance from institutions
  - Low Volume Nodes (LVN) - liquidity gaps where price moves fast
  - Current price position relative to institutional levels
- **Chart Pattern Recognition** (Professional-Grade): Geometric pattern detection
  - Reversal Patterns: Head & Shoulders, Inverse H&S, Double Tops/Bottoms
  - Continuation Patterns: Ascending/Descending Triangles, Bull/Bear Flags
  - Pattern confidence scoring (0-100%)
  - Auto-calculated price targets based on pattern geometry
  - Pattern completion status tracking
- **Market Scanner** (Top-Down Analysis): Professional trader workflow for stock selection
  - Market Overview: S&P 500, Nasdaq, Dow Jones, Russell 2000 health analysis
  - Sector Performance: 11 SPDR sector ETFs ranked by performance/strength/volume
  - Sector Leaders: Find top stocks within strongest sectors
  - Complete Market Scan: Automated Market → Sectors → Stocks workflow
  - Sector Rotation Detection: Identify risk-on vs risk-off market environment
- **Multi-Timeframe Confluence**: Analyzes alignment across daily, hourly, and 15-minute timeframes
- **Candlestick Pattern Recognition**: Doji, hammer, engulfing, shooting star, etc.
- **Fibonacci Levels**: Automatic retracement and extension calculations
- **Structural Analysis**: Automatic support/resistance level detection with volume confirmation
- **Smart Recommendations**: BUY or NO_BUY with confidence scoring (uses 13 factors)
- **Professional Trade Plans**:
  - ATR-based stop losses (volatility-adjusted)
  - Multiple price targets based on resistance levels
  - Risk-based position sizing (1% account risk)
  - Trade type classification (day/swing/long)
- **LLM-Ready Tools**: 23 functions designed with clear signatures for AI agent integration
- **Production-Ready**: Comprehensive error handling, logging, and validation
- **Alpaca Markets Integration**: Official alpaca-py SDK with IEX (free) and SIP (paid) support

## Architecture

```
StockMate/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application and endpoints
│   ├── config.py               # Configuration management
│   ├── models/                 # Pydantic data models
│   │   ├── __init__.py
│   │   ├── request.py         # API request models
│   │   ├── response.py        # API response models
│   │   └── data.py            # Internal data models
│   └── tools/                  # Analysis tools (LLM-ready)
│       ├── __init__.py
│       ├── market_data.py     # Data fetching tools
│       ├── indicators.py      # Technical indicator tools
│       ├── analysis.py        # Analysis and trading logic tools
│       └── market_scanner.py  # Market-wide and sector scanning tools
├── tests/                      # Test suite
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## Installation

### Prerequisites

- Python 3.10 or higher
- Alpaca Markets API account (free paper trading account available at https://alpaca.markets)
  - See [Alpaca Integration Guide](docs/ALPACA_INTEGRATION.md) for detailed setup instructions

### Setup

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd StockMate
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment variables**:
   ```bash
   cp .env.example .env
   ```

   Edit `.env` and add your Alpaca API credentials:
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ```

   **Get Alpaca API credentials**:
   - Sign up at https://alpaca.markets
   - Navigate to "Your API Keys" in the dashboard
   - Generate new API keys (use Paper Trading for testing)

5. **Run the server**:
   ```bash
   # Development mode (with auto-reload)
   python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

   # Or using the main.py directly
   python app/main.py
   ```

6. **Access the API**:
   - API: http://localhost:8000
   - Interactive docs: http://localhost:8000/docs
   - Alternative docs: http://localhost:8000/redoc

## API Usage

### Analyze Stock Endpoint

**Endpoint**: `POST /analyze`

**Request**:
```json
{
  "symbol": "AAPL",
  "account_size": 10000.0,
  "use_ai": false
}
```

**Response (BUY recommendation)**:
```json
{
  "symbol": "AAPL",
  "recommendation": "BUY",
  "confidence": 78.5,
  "trade_plan": {
    "trade_type": "swing",
    "entry_price": 175.50,
    "stop_loss": 172.00,
    "target_1": 180.00,
    "target_2": 185.00,
    "target_3": 190.00,
    "position_size": 28,
    "risk_amount": 100.00,
    "risk_percentage": 1.0
  },
  "reasoning": "Bullish sentiment | Price above key EMAs (3/3 bullish) | RSI in bullish zone (58.3)",
  "timestamp": "2025-11-13T10:30:00Z"
}
```

**Response (NO_BUY recommendation)**:
```json
{
  "symbol": "AAPL",
  "recommendation": "NO_BUY",
  "confidence": 45.2,
  "trade_plan": null,
  "reasoning": "Bearish sentiment | Price below key EMAs | Confidence too low (45.2%) - wait for better setup",
  "timestamp": "2025-11-13T10:30:00Z"
}
```

### Example cURL Request

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "account_size": 10000.0,
    "use_ai": false
  }'
```

### Example Python Client

```python
import requests

response = requests.post(
    "http://localhost:8000/analyze",
    json={
        "symbol": "AAPL",
        "account_size": 10000.0,
        "use_ai": False
    }
)

result = response.json()
print(f"Recommendation: {result['recommendation']}")
print(f"Confidence: {result['confidence']}%")

if result['trade_plan']:
    plan = result['trade_plan']
    print(f"Entry: ${plan['entry_price']}")
    print(f"Stop Loss: ${plan['stop_loss']}")
    print(f"Target 1: ${plan['target_1']}")
    print(f"Position Size: {plan['position_size']} shares")
```

### Market Scanner Endpoints (Top-Down Analysis)

#### 1. GET `/market` - Market Overview

Get health of major indices (S&P 500, Nasdaq, Dow, Russell 2000):

```bash
curl "http://localhost:8000/market?days_back=30"
```

**Response**:
```json
{
  "indices": [
    {
      "symbol": "SPY",
      "name": "S&P 500",
      "price": 450.25,
      "change_pct": 2.5,
      "signal": "bullish",
      "signal_strength": 80,
      "rsi": 58.5,
      "above_ema20": true,
      "above_ema50": true
    }
  ],
  "market_signal": "bullish",
  "bullish_count": 3,
  "bearish_count": 0,
  "summary": "3/4 indices bullish - strong market"
}
```

#### 2. GET `/sectors` - Sector Performance

Analyze all 11 SPDR sectors:

```bash
curl "http://localhost:8000/sectors?days_back=30&sort_by=performance"
```

**Response**:
```json
{
  "sectors": [
    {
      "symbol": "XLK",
      "name": "Technology",
      "price": 180.50,
      "change_pct": 5.2,
      "signal": "bullish",
      "signal_strength": 85,
      "rsi": 62.3,
      "relative_volume": 1.3
    },
    {
      "symbol": "XLF",
      "name": "Financials",
      "price": 38.75,
      "change_pct": 3.1,
      "signal": "bullish",
      "signal_strength": 70
    }
  ],
  "leading_sectors": ["Technology", "Healthcare", "Financials"],
  "lagging_sectors": ["Utilities", "Real Estate", "Energy"],
  "rotation_signal": "risk_on"
}
```

#### 3. GET `/sectors/{symbol}/leaders` - Sector Leaders

Find top stocks in a sector:

```bash
curl "http://localhost:8000/sectors/XLK/leaders?min_score=70&max_results=5"
```

**Response**:
```json
{
  "sector_name": "Technology",
  "sector_etf": "XLK",
  "leaders": [
    {
      "symbol": "NVDA",
      "score": 85,
      "recommendation": "BUY",
      "current_price": 495.50,
      "reasons": [
        "Strong volume confirmation (2.5x avg)",
        "MACD bullish crossover",
        "Price above key EMAs (3/3 bullish)"
      ],
      "trade_plan": {
        "entry": 495.50,
        "stop": 485.00,
        "target": 510.00
      }
    }
  ],
  "stocks_analyzed": 10,
  "stocks_above_threshold": 3,
  "average_score": 68.5
}
```

#### 4. GET `/market/scan` - Complete Market Scan

Run full top-down analysis (Market → Sectors → Stocks):

```bash
curl "http://localhost:8000/market/scan?min_stock_score=70&top_sectors=3&stocks_per_sector=3"
```

**Response**:
```json
{
  "market": {
    "market_signal": "bullish",
    "bullish_count": 3,
    "summary": "3/4 indices bullish - strong market"
  },
  "sectors": {
    "leading_sectors": ["Technology", "Healthcare", "Financials"],
    "rotation_signal": "risk_on"
  },
  "top_stocks": [
    {
      "sector_name": "Technology",
      "leaders": [
        {"symbol": "NVDA", "score": 85, "recommendation": "BUY"},
        {"symbol": "MSFT", "score": 78, "recommendation": "BUY"},
        {"symbol": "AAPL", "score": 72, "recommendation": "BUY"}
      ]
    }
  ]
}
```

### Python Client - Market Scanner Example

```python
import requests

BASE_URL = "http://localhost:8000"

# Step 1: Check market health
market = requests.get(f"{BASE_URL}/market").json()
print(f"Market: {market['market_signal']}")

# Step 2: Find leading sectors
sectors = requests.get(f"{BASE_URL}/sectors").json()
print(f"Leading sectors: {sectors['leading_sectors']}")

# Step 3: Find top stocks in leading sector
sector_etf = sectors['sectors'][0]['symbol']  # Top sector
leaders = requests.get(f"{BASE_URL}/sectors/{sector_etf}/leaders").json()

print(f"\nTop stocks in {leaders['sector_name']}:")
for stock in leaders['leaders']:
    print(f"  {stock['symbol']}: {stock['score']}% - {stock['recommendation']}")

# Or run complete scan in one call
scan = requests.get(f"{BASE_URL}/market/scan?min_stock_score=70").json()
print(f"\nComplete scan found {len(scan['top_stocks'])} sectors with leaders")
```

## LLM-Ready Tools

All analysis functions are designed to be used by LLM agents with clear signatures and comprehensive docstrings:

### Market Data Tools (4 functions)

```python
from app.tools import (
    fetch_price_bars,
    fetch_fundamentals,
    fetch_sentiment,
    fetch_news_sentiment,
)

# Fetch price data (multiple timeframes with IEX/SIP feed support)
bars_daily = fetch_price_bars("AAPL", timeframe="1d", days_back=100)
bars_hourly = fetch_price_bars("AAPL", timeframe="1h", days_back=30)
bars_15min = fetch_price_bars("AAPL", timeframe="15m", days_back=7)

# Fetch fundamentals
fundamentals = fetch_fundamentals("AAPL")

# Fetch basic sentiment (price-based)
sentiment = fetch_sentiment("AAPL")

# Fetch news sentiment (real articles from Alpaca News API)
news = fetch_news_sentiment("AAPL", days_back=7)
print(f"News sentiment: {news['sentiment_label']} ({news['sentiment_score']:.2f})")
print(f"Analyzed {news['article_count']} articles")
```

### Technical Indicator Tools (8 functions)

```python
from app.tools import (
    calculate_vwap,
    calculate_ema,
    calculate_rsi,
    analyze_volume,
    calculate_macd,
    calculate_atr,
    calculate_bollinger_bands,
    detect_divergences,
)

# Trend indicators
vwap = calculate_vwap(price_bars)
ema_20 = calculate_ema(price_bars, period=20)
bb = calculate_bollinger_bands(price_bars, period=20)

# Momentum indicators
rsi = calculate_rsi(price_bars, period=14)
macd = calculate_macd(price_bars)  # Detects crossovers!

# Divergence detection (powerful reversal signals)
rsi_div = detect_divergences(price_bars, indicator_type="rsi")
macd_div = detect_divergences(price_bars, indicator_type="macd")
if rsi_div.metadata['regular_bullish']:
    print("Bullish divergence detected - potential reversal up!")

# Volume analysis (CRITICAL)
volume = analyze_volume(price_bars)  # OBV, relative volume, spikes

# Volatility indicator
atr = calculate_atr(price_bars, period=14)  # For stop loss calculation
```

### Analysis Tools (7 functions)

```python
from app.tools import (
    find_structural_pivots,
    detect_key_levels,
    calculate_volume_profile,
    detect_chart_patterns,
    calculate_fibonacci_levels,
    analyze_multi_timeframe_confluence,
    detect_candlestick_patterns,
    build_snapshot,
    generate_trade_plan,
    run_analysis,
)

# Find support/resistance levels
pivots = find_structural_pivots(price_bars)

# Detect key psychological and technical levels
key_levels = detect_key_levels(price_bars)
print(f"Round numbers: {[l['price'] for l in key_levels['round_numbers'][:3]]}")
print(f"Nearest support: ${key_levels['nearest_support']['price']:.2f}")
print(f"Unfilled gaps: {len(key_levels['unfilled_gaps'])}")

# Volume Profile Analysis (Institutional-Grade)
volume_profile = calculate_volume_profile(price_bars, num_bins=50)
print(f"VPOC (institutions positioned): ${volume_profile['vpoc']:.2f}")
print(f"Value Area: ${volume_profile['value_area_low']:.2f} - ${volume_profile['value_area_high']:.2f}")
print(f"HVN support levels: {[hvn['price'] for hvn in volume_profile['high_volume_nodes'][:3]]}")
print(f"Current position: {volume_profile['current_price_position']}")

# Chart Pattern Recognition (Professional-Grade)
chart_patterns = detect_chart_patterns(price_bars, min_pattern_bars=20)
if chart_patterns['strongest_pattern']:
    p = chart_patterns['strongest_pattern']
    print(f"Pattern: {p['name']} ({p['type']}, {p['confidence']:.0f}% confidence)")
    print(f"Target: ${p['target_price']:.2f} ({p['expected_move_pct']:.1f}% move)")

# Calculate Fibonacci levels
fib = calculate_fibonacci_levels(price_bars)

# Detect candlestick patterns
patterns = detect_candlestick_patterns(price_bars)

# Analyze timeframe alignment
confluence = analyze_multi_timeframe_confluence(snapshot)

# Build complete market snapshot
snapshot = build_snapshot("AAPL")

# Generate trade plan (with ATR-based stops)
trade_plan = generate_trade_plan(snapshot, account_size=10000)

# Run complete analysis (main orchestrator)
result = run_analysis("AAPL", account_size=10000)
```

### Market Scanner Tools (4 functions)

```python
from app.tools import (
    get_market_overview,
    get_sector_performance,
    find_sector_leaders,
    run_market_scan,
)

# Step 1: Check overall market health (S&P 500, Nasdaq, Dow, Russell 2000)
market = get_market_overview(days_back=30)
print(f"Market: {market['market_signal']}")  # bullish/bearish/neutral
print(f"Summary: {market['summary']}")  # "3/4 indices bullish - strong market"

for index in market['indices']:
    print(f"{index['name']}: {index['signal']} ({index['change_pct']}%)")

# Step 2: Analyze sector performance (11 SPDR sectors)
sectors = get_sector_performance(days_back=30, sort_by="performance")
print(f"Leading sectors: {sectors['leading_sectors']}")  # ["Technology", "Healthcare"]
print(f"Sector rotation: {sectors['rotation_signal']}")  # risk_on/risk_off/neutral

for sector in sectors['sectors'][:3]:  # Top 3 sectors
    print(f"{sector['name']}: {sector['change_pct']}% ({sector['signal']})")

# Step 3: Find top stocks in leading sectors
tech_leaders = find_sector_leaders(
    sector_symbol="XLK",  # Technology sector
    min_score=65,  # Only BUY candidates
    max_results=5
)

print(f"\nTop stocks in {tech_leaders['sector_name']}:")
for stock in tech_leaders['leaders']:
    print(f"  {stock['symbol']}: {stock['score']}% - {stock['recommendation']}")
    print(f"    Entry: ${stock['trade_plan']['entry']}, Stop: ${stock['trade_plan']['stop']}")
    print(f"    Top reasons: {', '.join(stock['reasons'][:2])}")

# Complete top-down scan in one call
scan = run_market_scan(
    min_sector_change=0.0,  # All sectors
    min_stock_score=70,     # Only strong candidates
    top_sectors=3,          # Top 3 sectors
    stocks_per_sector=3     # Top 3 stocks per sector
)

print(f"\nMarket Scan Results:")
print(f"  Market: {scan['market']['market_signal']}")
print(f"  Leading sectors: {scan['sectors']['leading_sectors']}")
print(f"  Found {len(scan['top_stocks'])} sectors with leaders")

for sector_stocks in scan['top_stocks']:
    print(f"\n  {sector_stocks['sector_name']}:")
    for stock in sector_stocks['leaders']:
        print(f"    {stock['symbol']}: {stock['score']}%")
```

## Analysis Algorithm

The recommendation engine uses a comprehensive weighted scoring system with **13 factors**:

| Factor | Weight | Details |
|--------|--------|---------|
| Sentiment | 20% | Based on price momentum and volume trends |
| EMA Trend | 25% | Multiple EMAs (9, 20, 50) for trend confirmation |
| RSI | 15% | Momentum indicator (optimal range: 40-70) |
| VWAP | 15% | Price relative to volume-weighted average |
| **Volume** | **20%** | **OBV, relative volume, accumulation/distribution** |
| **MACD** | **15%** | **Crossovers and momentum direction** |
| **Bollinger Bands** | **10%** | **Volatility, overbought/oversold, squeeze patterns** |
| **Multi-Timeframe Confluence** | **15%** | **Alignment across daily/hourly/15min** |
| **Support/Resistance** | **10%** | **Enhanced with round numbers, gaps, period highs/lows** |
| **Divergence Detection** | **15%** | **RSI/MACD divergences for reversal signals** |
| ATR Volatility | 5% | Risk assessment based on volatility |
| **Volume Profile** | **10%** | **VPOC, value area, HVN/LVN institutional positioning** |
| **Chart Patterns** | **15%** | **H&S, Double Tops/Bottoms, Triangles, Flags** |

**Enhanced Features**:
- **MACD Crossover Detection**: +20 points for bullish crossover (strong entry signal)
- **Volume Confirmation**: +20 points for high volume accumulation
- **Divergence Signals**: +15 points for regular bullish divergence (reversal up)
- **Key Level Detection**: Enhanced support/resistance with psychological levels and gaps
- **Volume Profile**: +10 points for price near VPOC institutional support
- **Chart Patterns**: Up to +15 points for high-confidence reversal patterns (H&S, Double Bottoms)
- **Timeframe Alignment**: +15 points when all timeframes agree
- **Bollinger Squeeze**: +5 points for potential breakout setups
- **ATR-Based Stop Losses**: Dynamic stops based on volatility (2x ATR for swing trades)

**Recommendation Thresholds**:
- **BUY**: Confidence ≥ 65%
- **NO_BUY**: Confidence < 65%

## Trade Plan Generation

When a BUY recommendation is issued, the system generates a detailed trade plan:

1. **Trade Type Determination**:
   - **Long**: Strong multi-timeframe uptrend (EMA 9 > 20 > 50)
   - **Swing**: Moderate trend alignment (days to weeks)
   - **Day**: Short-term momentum only (intraday)

2. **Entry Price**: Slightly below current price (0.2%) for better entry

3. **Stop Loss** (ATR-Based - Professional Approach):
   - Primary: ATR-based (1.5x ATR for day trades, 2x ATR for swing trades)
   - Secondary: Below nearest strong support level
   - Fallback: 2-3% below entry if no ATR data

4. **Price Targets**:
   - Based on resistance levels or risk/reward ratios (1.5:1, 2.5:1, 3.5:1)
   - Up to 3 targets for scaling out

5. **Position Sizing**:
   - Risk-based: 1% of account at risk per trade
   - Respects maximum position size (20% of account)

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test file
pytest tests/test_indicators.py

# Run with verbose output
pytest -v
```

## Mobile App Integration

The API is designed for mobile app integration with:

- **CORS enabled**: Configure `allow_origins` in `app/main.py` for production
- **Structured JSON responses**: Consistent Pydantic models
- **Error handling**: Comprehensive HTTP status codes and error messages
- **Fast responses**: Optimized for low latency

### Mobile Integration Example (React Native)

```javascript
const analyzeStock = async (symbol, accountSize) => {
  try {
    const response = await fetch('http://your-api-url/analyze', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        symbol: symbol,
        account_size: accountSize,
        use_ai: false,
      }),
    });

    const data = await response.json();

    if (data.recommendation === 'BUY') {
      // Show trade plan to user
      console.log(`Entry: $${data.trade_plan.entry_price}`);
      console.log(`Stop Loss: $${data.trade_plan.stop_loss}`);
      console.log(`Position Size: ${data.trade_plan.position_size} shares`);
    }

    return data;
  } catch (error) {
    console.error('Analysis error:', error);
  }
};
```

## Production Deployment

### Using Docker

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t stockmate .
docker run -p 8000:8000 --env-file .env stockmate
```

### Using Docker Compose

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    restart: unless-stopped
```

Run:
```bash
docker-compose up -d
```

### Environment Variables for Production

```bash
APP_ENV=production
LOG_LEVEL=INFO
ALPACA_API_KEY=your_production_key
ALPACA_SECRET_KEY=your_production_secret
ALPACA_BASE_URL=https://api.alpaca.markets  # Live trading
```

## Limitations & Future Enhancements

### Current Limitations

1. **Fundamental Data**: Limited fundamental metrics from Alpaca's free tier
   - Enhancement: Integrate Financial Modeling Prep, Alpha Vantage, or IEX Cloud

2. **Sentiment Analysis**: Currently based on price action only
   - Enhancement: Integrate Alpaca News API, MarketPsych, or social sentiment APIs

3. **Options Analysis**: Not currently supported
   - Enhancement: Add options chain analysis for enhanced strategies

4. **Backtesting**: No historical performance tracking
   - Enhancement: Add backtesting framework with performance metrics

### Future Enhancements

- [ ] Real-time WebSocket streaming for live price updates
- [ ] Portfolio management and tracking
- [ ] Advanced AI/ML models for prediction (use_ai flag)
- [ ] Multi-asset support (crypto, forex, commodities)
- [ ] Custom alert system
- [ ] Integration with brokerage APIs for order execution
- [ ] Performance analytics and reporting

## Contributing

Contributions are welcome! Please ensure:

1. All new functions include comprehensive docstrings
2. Tests are added for new features
3. Code follows PEP 8 style guidelines
4. API changes are documented in README

## License

MIT License - See LICENSE file for details

## Support

For issues, questions, or feature requests:
- Create an issue on GitHub
- Email: support@stockmate.example.com

## Documentation

- **[README.md](README.md)**: Main documentation (you are here)
- **[QUICKSTART.md](QUICKSTART.md)**: 5-minute setup guide
- **[Alpaca Integration Guide](docs/ALPACA_INTEGRATION.md)**: Detailed Alpaca API documentation
- **API Docs**: Interactive documentation at `/docs` endpoint when running

## Acknowledgments

- **Alpaca Markets**: Market data provider via official alpaca-py SDK
- **FastAPI**: Modern Python web framework
- **Pydantic**: Data validation library

---

**Disclaimer**: This software is for educational and informational purposes only. It does not constitute financial advice. Trading stocks involves risk, including the loss of principal. Always do your own research and consult with a qualified financial advisor before making investment decisions.
