# Alpaca Markets API Integration Guide

## Overview

StockMate uses the official **alpaca-py** SDK to fetch market data from Alpaca Markets. This integration supports both free and paid subscription tiers.

## Official SDK

- **Package**: `alpaca-py` (version 0.20.2+)
- **GitHub**: https://github.com/alpacahq/alpaca-py
- **Documentation**: https://alpaca.markets/sdks/python/

## Data Feeds

Alpaca provides two data feed options:

### IEX Feed (Free Tier)
- **Source**: Investors Exchange (single exchange)
- **History**: ~5 years of historical data
- **Cost**: FREE with any Alpaca account
- **Best For**: Development, testing, retail trading
- **Usage**: Automatic on free accounts

### SIP Feed (Paid Subscription)
- **Source**: Securities Information Processor (all US exchanges)
- **History**: ~7 years of historical data
- **Cost**: Requires paid subscription
- **Best For**: Professional trading, production apps
- **Note**: Real-time SIP data requires active subscription

### Auto-Selection
By default, StockMate automatically uses the best available feed based on your Alpaca subscription.

## API Credentials

### Getting Your API Keys

1. Sign up at https://alpaca.markets (free)
2. Choose **Paper Trading** (free, no real money required)
3. Navigate to "Your API Keys" in dashboard
4. Generate new keys and copy:
   - API Key
   - Secret Key

### Configuration

Add credentials to `.env`:

```bash
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_BASE_URL=https://paper-api.alpaca.markets
```

**Important**:
- Use `https://paper-api.alpaca.markets` for paper trading
- Use `https://api.alpaca.markets` for live trading (requires funding)

## Implementation Details

### Client Initialization

```python
from alpaca.data.historical import StockHistoricalDataClient

client = StockHistoricalDataClient(
    api_key="your_api_key",
    secret_key="your_secret_key"
)
```

### Fetching Stock Bars

```python
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

# Create request
request_params = StockBarsRequest(
    symbol_or_symbols="AAPL",
    timeframe=TimeFrame.Day,
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    feed="iex"  # Optional: specify feed
)

# Fetch data
bars = client.get_stock_bars(request_params)
```

### Available Timeframes

StockMate supports multiple timeframes:

- **Daily**: `TimeFrame.Day` → `timeframe="1d"`
- **Hourly**: `TimeFrame.Hour` → `timeframe="1h"`
- **15-minute**: `TimeFrame(15, TimeFrame.Minute)` → `timeframe="15m"`
- **5-minute**: `TimeFrame(5, TimeFrame.Minute)` → `timeframe="5m"`

## StockMate Integration

### Using the Tools

```python
from app.tools import fetch_price_bars

# Fetch daily bars (automatic feed selection)
bars = fetch_price_bars("AAPL", timeframe="1d", days_back=100)

# Explicitly use IEX feed
bars = fetch_price_bars("AAPL", timeframe="1d", days_back=100, feed="iex")

# Use SIP feed (requires paid subscription)
bars = fetch_price_bars("AAPL", timeframe="1d", days_back=100, feed="sip")
```

### Error Handling

The integration handles common Alpaca API errors:

```python
from alpaca.common.exceptions import APIError

try:
    bars = fetch_price_bars("AAPL", timeframe="1d", days_back=100)
except ValueError as e:
    # Invalid symbol or no data available
    print(f"Validation error: {e}")
except APIError as e:
    # Alpaca API errors (auth, rate limits, subscription)
    print(f"API error: {e}")
```

## Rate Limits

### Free Tier (IEX)
- **Historical Data**: 200 requests/minute
- **Real-time Data**: Not available on free tier
- **Concurrent Requests**: Limited

### Paid Tier (SIP)
- Higher rate limits
- Real-time data access
- More concurrent connections

**Best Practices**:
- Cache data when possible
- Batch requests for multiple symbols
- Implement exponential backoff on rate limit errors

## Data Availability

### Market Hours
- **Regular Hours**: 9:30 AM - 4:00 PM ET (Monday-Friday)
- **Pre-market**: 4:00 AM - 9:30 AM ET
- **After-hours**: 4:00 PM - 8:00 PM PM ET

### Historical Data
- **IEX**: Up to 5 years of minute-level bars
- **SIP**: Up to 7 years of minute-level bars
- **Daily bars**: Available for full history

### Data Delay
- **Historical queries**: Must be at least 15 minutes old for SIP without subscription
- **Latest endpoints**: Require subscription for SIP feed
- **IEX**: Available immediately on free tier

## Features Used in StockMate

### ✅ Implemented
- [x] Historical stock bars (OHLCV)
- [x] Multiple timeframes (1d, 1h, 15m, 5m)
- [x] Automatic feed selection
- [x] Error handling for API errors
- [x] Data conversion to internal models

### 🔜 Future Enhancements
- [ ] Alpaca News API integration (sentiment)
- [ ] Real-time WebSocket streaming
- [ ] Options data (if available)
- [ ] Corporate actions (splits, dividends)
- [ ] Snapshot endpoint (latest quote + bars)

## Troubleshooting

### "No data available for symbol"
- Verify symbol is valid (try AAPL, TSLA, MSFT)
- Check if market is open or use historical dates
- Ensure you're querying during valid market hours

### "Alpaca API error: Unauthorized"
- Check `.env` file has correct API keys
- Verify keys are not expired
- Ensure using paper trading URL for paper keys

### "Alpaca API error: Rate limit exceeded"
- Reduce request frequency
- Implement caching
- Consider upgrading to paid tier

### "Feed parameter not working"
- Free accounts default to IEX automatically
- SIP feed requires paid subscription
- Use `feed=None` or omit parameter for auto-selection

## Additional Resources

- **Alpaca Documentation**: https://docs.alpaca.markets
- **API Reference**: https://docs.alpaca.markets/reference
- **Python SDK Examples**: https://github.com/alpacahq/alpaca-py/tree/master/examples
- **Community Forum**: https://forum.alpaca.markets
- **Status Page**: https://status.alpaca.markets

## Support

For Alpaca-specific issues:
- Forum: https://forum.alpaca.markets
- Email: support@alpaca.markets
- Discord: https://alpaca.markets/discord

For StockMate integration issues:
- Check logs: Application logs show detailed error messages
- Verify configuration: Ensure `.env` is properly configured
- Test credentials: Use the `/health` endpoint to verify Alpaca connection

---

**Last Updated**: 2025-11-13
**Alpaca SDK Version**: alpaca-py 0.20.2
**API Version**: Alpaca Markets Data API v2
