# StockMate - Intelligent Stock Analysis Backend

A production-ready FastAPI backend for stock analysis and trading recommendations, designed with LLM-agent-ready tools and mobile app integration.

## Features

- **Comprehensive Stock Analysis**: Multi-timeframe price action analysis (daily, hourly, 15-minute)
- **Technical Indicators**: VWAP, EMA (multiple periods), RSI
- **Structural Analysis**: Automatic support/resistance level detection
- **Sentiment Analysis**: Market sentiment based on price action and volume
- **Smart Recommendations**: BUY or NO_BUY with confidence scoring
- **Detailed Trade Plans**: Entry price, stop loss, multiple price targets, position sizing
- **LLM-Ready Tools**: All functions designed with clear signatures for AI agent integration
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
│       └── analysis.py        # Analysis and trading logic tools
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

## LLM-Ready Tools

All analysis functions are designed to be used by LLM agents with clear signatures and comprehensive docstrings:

### Market Data Tools

```python
from app.tools import fetch_price_bars, fetch_fundamentals, fetch_sentiment

# Fetch price data
bars = fetch_price_bars("AAPL", timeframe="1d", days_back=100)

# Fetch fundamentals
fundamentals = fetch_fundamentals("AAPL")

# Fetch sentiment
sentiment = fetch_sentiment("AAPL")
```

### Technical Indicator Tools

```python
from app.tools import calculate_vwap, calculate_ema, calculate_rsi

# Calculate VWAP
vwap = calculate_vwap(price_bars)

# Calculate EMA
ema_20 = calculate_ema(price_bars, period=20)

# Calculate RSI
rsi = calculate_rsi(price_bars, period=14)
```

### Analysis Tools

```python
from app.tools import find_structural_pivots, build_snapshot, generate_trade_plan, run_analysis

# Find support/resistance levels
pivots = find_structural_pivots(price_bars)

# Build complete market snapshot
snapshot = build_snapshot("AAPL")

# Generate trade plan
trade_plan = generate_trade_plan(snapshot, account_size=10000)

# Run complete analysis (main orchestrator)
result = run_analysis("AAPL", account_size=10000)
```

## Analysis Algorithm

The recommendation engine uses a weighted scoring system:

| Factor | Weight | Details |
|--------|--------|---------|
| Sentiment | 20% | Based on price momentum and volume trends |
| EMA Trend | 25% | Multiple EMAs (9, 20, 50) for trend confirmation |
| RSI | 15% | Momentum indicator (optimal range: 40-70) |
| VWAP | 15% | Price relative to volume-weighted average |
| Support/Resistance | 15% | Proximity to structural levels |
| 52-Week Range | 10% | Position within yearly range |

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

3. **Stop Loss**:
   - Primary: Below nearest strong support level
   - Fallback: 2-3% below entry (depending on trade type)

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
