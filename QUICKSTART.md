# StockMate - Quick Start Guide

Get up and running with StockMate in 5 minutes!

## Step 1: Get Alpaca API Credentials (2 minutes)

1. Go to https://alpaca.markets and sign up for a free account
2. After signing in, navigate to "Your API Keys" in the dashboard
3. Click "Generate New Keys" and choose **Paper Trading** (free, no real money)
4. Copy your API Key and Secret Key

## Step 2: Setup Environment (1 minute)

```bash
# Clone and enter directory
cd StockMate

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
```

Now edit `.env` and paste your Alpaca credentials:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

## Step 3: Run the Server (30 seconds)

```bash
# Start the API server
python run.py
```

You should see:
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

## Step 4: Test It Out (1 minute)

Open another terminal and try it:

```bash
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "account_size": 10000.0,
    "use_ai": false
  }'
```

Or visit the interactive docs at: **http://localhost:8000/docs**

Click on "POST /analyze" → "Try it out" → Fill in:
- symbol: `AAPL`
- account_size: `10000`
- use_ai: `false`

Click "Execute" and see your analysis!

## What You'll Get

### BUY Recommendation Example:
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
    "position_size": 28,
    "risk_amount": 100.00
  },
  "reasoning": "Bullish sentiment | Price above EMAs | RSI in zone"
}
```

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Run tests: `pytest`
- Explore the tools in `app/tools/` for LLM integration
- Integrate with your mobile app

## Troubleshooting

**Error: "Alpaca API credentials not configured"**
- Make sure you created `.env` file and added your API keys

**Error: "No data available for symbol"**
- Try a different symbol (AAPL, TSLA, MSFT, GOOGL)
- Check if market is open (US market hours: 9:30 AM - 4:00 PM ET)

**ModuleNotFoundError**
- Make sure you activated the virtual environment: `source venv/bin/activate`
- Run `pip install -r requirements.txt` again

## Support

Having issues? Check:
1. Is your virtual environment activated?
2. Are your Alpaca API credentials correct in `.env`?
3. Is the symbol valid? (try AAPL first)
4. Check the terminal output for error messages

Happy trading! 📈
