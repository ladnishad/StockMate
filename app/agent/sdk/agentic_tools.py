"""Tool definitions for the agentic stock analyzer.

These tools are exposed to Claude/Grok for iterative tool-calling.
The AI decides which tools to call based on its investigation needs.
"""

from typing import List, Dict, Any


# Tool definitions in Claude/OpenAI-compatible format
AGENTIC_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "get_price",
        "description": "Get the current stock price, bid/ask spread, and today's price change. Use this first to understand current market conditions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g., NVDA, AAPL, TSLA)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_market_context",
        "description": "Get overall market direction by checking major indices (SPY, QQQ, DIA, IWM). Helps understand if the market is bullish, bearish, or mixed.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "get_position",
        "description": "Check if the user has an existing position in this stock. Returns position direction (long/short), entry price, current P&L, and any existing stop loss or targets.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_and_analyze_chart",
        "description": "Generate a candlestick chart AND analyze it for patterns, trends, EMA structure, volume confirmation, and warning signs. Returns analysis summary (not the image). Use this for visual chart analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["5m", "15m", "1h", "1d", "1w"],
                    "description": "Chart timeframe: 5m for intraday/day trades, 1d for swing trades, 1w for position trades"
                },
                "trade_style": {
                    "type": "string",
                    "enum": ["day", "swing", "position"],
                    "description": "Trading style context for the analysis"
                },
                "days_back": {
                    "type": "integer",
                    "description": "Number of days of data to include (default: 100 for daily, 3 for intraday, 365 for weekly)"
                }
            },
            "required": ["symbol", "timeframe", "trade_style"]
        }
    },
    {
        "name": "get_technicals",
        "description": "Get technical indicators: EMAs (trend), RSI (momentum), MACD (momentum crossovers), Bollinger Bands (volatility), and volume analysis.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "ema_periods": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "EMA periods to calculate (e.g., [9, 21, 50] for swing, [5, 9, 20] for day, [21, 50, 200] for position)"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["5m", "1d", "1w"],
                    "description": "Timeframe for indicator calculation"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_support_resistance",
        "description": "Get key support and resistance levels based on price structure, swing highs/lows, and volume nodes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["intraday", "daily", "weekly"],
                    "description": "Timeframe for level calculation"
                }
            },
            "required": ["symbol", "timeframe"]
        }
    },
    {
        "name": "get_fibonacci",
        "description": "Get Fibonacci retracement levels (for entries) and extension levels (for targets) based on recent swing high/low.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "trade_type": {
                    "type": "string",
                    "enum": ["day", "swing", "position"],
                    "description": "Trade type determines the swing lookback period"
                }
            },
            "required": ["symbol", "trade_type"]
        }
    },
    {
        "name": "get_fundamentals",
        "description": "Get fundamental data: P/E ratio, EPS growth, revenue growth, margins, debt levels, financial health score, and earnings risk (upcoming earnings date).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_news",
        "description": "Get recent news headlines, sentiment analysis, and key themes. Identifies breaking news and potential catalysts.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "days_back": {
                    "type": "integer",
                    "description": "Number of days to look back for news (default: 7)"
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "search_x",
        "description": "Search X/Twitter for real-time trader sentiment, discussions, and breaking news. Great for understanding current market mood and finding catalysts not in traditional news.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query (e.g., '$NVDA sentiment', '$AAPL earnings')"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "get_atr",
        "description": "Get Average True Range (ATR) to understand volatility. ATR% helps determine if a stock is suitable for day trading (>3%), swing trading (1-3%), or position trading (<1.5%).",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Stock ticker symbol"
                },
                "timeframe": {
                    "type": "string",
                    "enum": ["5m", "1d", "1w"],
                    "description": "Timeframe for ATR calculation"
                }
            },
            "required": ["symbol"]
        }
    }
]


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get a tool definition by name."""
    for tool in AGENTIC_TOOLS:
        if tool["name"] == name:
            return tool
    return None


def get_tool_names() -> List[str]:
    """Get list of all tool names."""
    return [tool["name"] for tool in AGENTIC_TOOLS]
