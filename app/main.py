"""FastAPI application for StockMate - Intelligent Stock Analysis Backend."""

import logging
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.models.request import AnalysisRequest
from app.models.response import AnalysisResponse
from app.models.profile_presets import list_profiles, get_profile, PROFILE_REGISTRY
from app.tools.analysis import run_analysis
from app.tools.market_scanner import (
    get_market_overview,
    get_sector_performance,
    find_sector_leaders,
    run_market_scan,
    get_quick_market_status,
    get_quick_sector_status,
)
from app.tools.market_data import (
    fetch_latest_quote,
    fetch_latest_trade,
    fetch_snapshots,
)
from app.tools.streaming import get_streamer
from app.config import get_settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting StockMate backend...")
    settings = get_settings()
    logger.info(f"Environment: {settings.app_env}")
    logger.info(f"Alpaca API configured: {bool(settings.alpaca_api_key)}")
    logger.info(f"Data feed: {settings.alpaca_data_feed.upper()} (AlgoTrader Plus: {settings.alpaca_data_feed == 'sip'})")
    yield
    logger.info("Shutting down StockMate backend...")


# Initialize FastAPI app
app = FastAPI(
    title="StockMate API",
    description="Intelligent Stock Analysis Backend with AI-Ready Tools",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware for mobile app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint - API health check."""
    settings = get_settings()
    return {
        "service": "StockMate API",
        "version": "1.1.0",
        "status": "operational",
        "data_feed": settings.alpaca_data_feed.upper(),
        "algotrader_plus": settings.alpaca_data_feed == "sip",
        "endpoints": {
            "analyze": "/analyze",
            "market": "/market",
            "market_quick": "/market/quick",
            "sectors": "/sectors",
            "sectors_quick": "/sectors/quick",
            "sector_leaders": "/sectors/{symbol}/leaders",
            "market_scan": "/market/scan",
            "watchlist_smart": "/watchlist/smart",
            "profiles": "/profiles",
            "quote": "/quote/{symbol}",
            "trade": "/trade/{symbol}",
            "snapshots": "/snapshots",
            "streaming_status": "/streaming/status",
            "health": "/health",
            "docs": "/docs",
        },
        "websockets": {
            "quotes": "/ws/quotes/{symbol}",
            "trades": "/ws/trades/{symbol}",
            "bars": "/ws/bars/{symbol}",
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    settings = get_settings()
    return {
        "status": "healthy",
        "alpaca_configured": bool(settings.alpaca_api_key and settings.alpaca_secret_key),
    }


@app.get(
    "/profiles",
    summary="List available trader profiles",
    description="""
    Get all available trader profiles with their configurations.

    Each profile customizes the analysis pipeline:
    - **Day Trader**: Intraday focus, tight 1.5x ATR stops, VWAP/volume emphasis (70% confidence threshold)
    - **Swing Trader**: Multi-day holds, Fibonacci levels, structure-based stops (65% threshold)
    - **Position Trader**: Multi-week holds, trend-following, momentum focus (65% threshold)
    - **Long-Term Investor**: Multi-month holds, fundamentals, wide 3x ATR stops (60% threshold)

    Returns detailed configuration including:
    - Timeframe settings (primary, confirmation, entry)
    - Risk parameters (stop method, ATR multiplier, max position size)
    - Target methodology (R:R ratios, Fibonacci extensions, structure-based)
    - Scoring weights for all 15 technical factors
    """,
)
async def get_profiles():
    """List all available trader profiles with configurations."""
    profiles = []
    for profile_type in list_profiles():
        profile = get_profile(profile_type)
        profiles.append({
            "type": profile_type,
            "name": profile.name,
            "description": profile.description,
            "timeframes": {
                "primary": profile.timeframes.primary,
                "confirmation": profile.timeframes.confirmation,
                "entry": profile.timeframes.entry,
            },
            "holding_period": {
                "min": profile.min_holding_period,
                "max": profile.max_holding_period,
            },
            "allowed_trade_types": profile.allowed_trade_types,
            "risk": {
                "risk_percentage": profile.risk.risk_percentage,
                "stop_method": profile.risk.stop_method,
                "atr_multiplier": profile.risk.atr_multiplier,
                "max_position_percent": profile.risk.max_position_percent,
            },
            "targets": {
                "method": profile.targets.method,
                "rr_ratios": profile.targets.rr_ratios,
                "use_fibonacci_extensions": profile.targets.use_fibonacci_extensions,
                "validate_against_resistance": profile.targets.validate_against_resistance,
            },
            "thresholds": {
                "buy_confidence": profile.buy_confidence_threshold,
                "rsi_overbought": profile.rsi_overbought,
                "rsi_oversold": profile.rsi_oversold,
                "adx_trend": profile.adx_trend_threshold,
            },
            "weights": {
                "sentiment": profile.weights.sentiment,
                "ema_trend": profile.weights.ema_trend,
                "rsi": profile.weights.rsi,
                "vwap": profile.weights.vwap,
                "volume": profile.weights.volume,
                "macd": profile.weights.macd,
                "bollinger": profile.weights.bollinger,
                "multi_tf": profile.weights.multi_tf,
                "support_resistance": profile.weights.support_resistance,
                "divergence": profile.weights.divergence,
                "volume_profile": profile.weights.volume_profile,
                "chart_patterns": profile.weights.chart_patterns,
                "fibonacci": profile.weights.fibonacci,
                "adx": profile.weights.adx,
                "stochastic": profile.weights.stochastic,
            },
        })

    return {
        "profiles": profiles,
        "count": len(profiles),
        "usage": "Pass the 'type' value as the 'trader_profile' parameter in /analyze requests",
    }


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze stock and generate trading recommendation",
    description="""
    Analyze a stock ticker and generate a comprehensive trading recommendation.

    The analysis includes:
    - Price action across multiple timeframes
    - Technical indicators (VWAP, EMA, RSI, MACD, Bollinger, ADX, Stochastic, Fibonacci)
    - Structural support/resistance levels
    - Sentiment analysis
    - Fundamental context

    **Trader Profiles** (optional):
    - `day_trader`: Intraday focus, tight stops, VWAP/volume emphasis
    - `swing_trader`: Multi-day holds, Fibonacci levels, structure-based stops
    - `position_trader`: Multi-week holds, trend-following, momentum focus
    - `long_term_investor`: Multi-month holds, fundamentals, wide stops

    Returns:
    - BUY or NO_BUY recommendation
    - Confidence score (0-100)
    - If BUY: Complete trade plan with entry, stop loss, targets, and position size
    - Reasoning for the recommendation (customized based on profile)
    """,
    responses={
        200: {
            "description": "Analysis completed successfully",
            "content": {
                "application/json": {
                    "example": {
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
                }
            }
        },
        400: {"description": "Invalid request parameters"},
        500: {"description": "Internal server error during analysis"},
    }
)
async def analyze_stock(request: AnalysisRequest):
    """
    Analyze a stock and generate trading recommendation.

    Args:
        request: AnalysisRequest containing symbol, account_size, use_ai flag, and optional trader_profile

    Returns:
        AnalysisResponse with recommendation and optional trade plan
    """
    profile_str = request.trader_profile if request.trader_profile else "default"
    logger.info(f"Received analysis request for {request.symbol} (profile: {profile_str})")

    try:
        # Validate Alpaca configuration
        settings = get_settings()
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        # Run analysis with optional trader profile
        result = run_analysis(
            symbol=request.symbol,
            account_size=request.account_size,
            use_ai=request.use_ai,
            trader_profile=request.trader_profile,
        )

        logger.info(
            f"Analysis complete for {request.symbol}: "
            f"{result.recommendation} (confidence: {result.confidence}%)"
        )

        return result

    except ValueError as e:
        logger.error(f"Validation error for {request.symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Error analyzing {request.symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


@app.get(
    "/market",
    summary="Get market overview",
    description="Analyze major market indices (S&P 500, Nasdaq, Dow) to determine overall market health.",
)
async def market_overview(days_back: int = 30):
    """Get overview of major market indices."""
    try:
        result = get_market_overview(days_back=days_back)
        logger.info(f"Market overview: {result['market_signal']}")
        return result
    except Exception as e:
        logger.error(f"Error getting market overview: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get market overview: {str(e)}"
        )


@app.get(
    "/sectors",
    summary="Get sector performance",
    description="Analyze all 11 SPDR sectors and rank them by performance, strength, or volume.",
)
async def sector_performance(
    days_back: int = 30,
    sort_by: str = "performance"
):
    """Get performance of all sectors."""
    try:
        if sort_by not in ["performance", "strength", "volume"]:
            raise ValueError("sort_by must be 'performance', 'strength', or 'volume'")

        result = get_sector_performance(days_back=days_back, sort_by=sort_by)
        logger.info(f"Sector performance: {result['leading_sectors']}")
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error getting sector performance: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get sector performance: {str(e)}"
        )


@app.get(
    "/sectors/{sector_symbol}/leaders",
    summary="Find sector leaders",
    description="Find top stocks within a specific sector using full StockMate analysis.",
)
async def sector_leaders(
    sector_symbol: str,
    min_score: int = 65,
    max_results: int = 5
):
    """Find top stocks in a sector."""
    try:
        result = find_sector_leaders(
            sector_symbol=sector_symbol.upper(),
            min_score=min_score,
            max_results=max_results
        )
        logger.info(
            f"Found {len(result['leaders'])} leaders in {result['sector_name']}"
        )
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error finding sector leaders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to find sector leaders: {str(e)}"
        )


@app.get(
    "/market/scan",
    summary="Complete market scan",
    description="Run complete top-down analysis: Market → Sectors → Stocks. Professional trader workflow.",
)
async def market_scan(
    min_sector_change: float = 0.0,
    min_stock_score: int = 65,
    top_sectors: int = 3,
    stocks_per_sector: int = 3
):
    """Run complete top-down market scan."""
    try:
        result = run_market_scan(
            min_sector_change=min_sector_change,
            min_stock_score=min_stock_score,
            top_sectors=top_sectors,
            stocks_per_sector=stocks_per_sector
        )
        logger.info(
            f"Market scan complete: {result['market']['market_signal']} market, "
            f"{len(result['top_stocks'])} sectors scanned"
        )
        return result
    except Exception as e:
        logger.error(f"Error running market scan: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market scan failed: {str(e)}"
        )


@app.get(
    "/watchlist/smart",
    summary="Smart watchlist for mobile app",
    description="""
    Get a smart watchlist of top stocks matching a trader profile.

    This endpoint is optimized for mobile app consumption, returning:
    - Flattened list of top stocks across all leading sectors
    - Profile-specific filtering based on confidence thresholds
    - Sorted by score (highest first)

    **Profile Thresholds:**
    - day_trader: 70% confidence
    - swing_trader: 65% confidence
    - position_trader: 65% confidence
    - long_term_investor: 60% confidence

    Returns a simplified response perfect for rendering in a watchlist UI.
    """,
)
async def smart_watchlist(
    profile: str = "swing_trader",
    max_results: int = 10,
):
    """Get smart watchlist of top stocks for a trader profile."""
    try:
        # Validate profile
        valid_profiles = ["day_trader", "swing_trader", "position_trader", "long_term_investor"]
        if profile not in valid_profiles:
            raise ValueError(f"Invalid profile. Must be one of: {', '.join(valid_profiles)}")

        # Get profile-specific threshold
        profile_thresholds = {
            "day_trader": 70,
            "swing_trader": 65,
            "position_trader": 65,
            "long_term_investor": 60,
        }
        min_score = profile_thresholds[profile]

        # Run market scan with profile-appropriate settings
        scan_result = run_market_scan(
            min_sector_change=-5.0,  # Include slightly negative sectors too
            min_stock_score=min_score,
            top_sectors=5,
            stocks_per_sector=5
        )

        # Flatten stocks from all sectors
        all_stocks = []
        for sector_data in scan_result.get("top_stocks", []):
            sector_name = sector_data.get("sector_name", "Unknown")
            for stock in sector_data.get("leaders", []):
                # Add sector info to stock
                stock_entry = {
                    "symbol": stock.get("symbol"),
                    "score": stock.get("score", 0),
                    "recommendation": stock.get("recommendation", "NO_BUY"),
                    "current_price": stock.get("current_price", 0),
                    "reasons": stock.get("key_reasons", []),
                    "sector_name": sector_name,
                    "trade_plan": None,
                }

                # Include trade plan if it's a BUY
                if stock.get("recommendation") == "BUY" and stock.get("trade_plan"):
                    tp = stock["trade_plan"]
                    stock_entry["trade_plan"] = {
                        "entry_price": tp.get("entry_price"),
                        "stop_loss": tp.get("stop_loss"),
                        "targets": [
                            tp.get("target_1"),
                            tp.get("target_2"),
                            tp.get("target_3"),
                        ],
                        "risk_reward_ratio": tp.get("risk_reward_ratio"),
                        "trade_type": tp.get("trade_type"),
                    }

                all_stocks.append(stock_entry)

        # Sort by score (highest first) and limit results
        all_stocks.sort(key=lambda x: x["score"], reverse=True)
        watchlist = all_stocks[:max_results]

        # Get market direction
        market_direction = scan_result.get("market", {}).get("market_signal", "mixed")

        logger.info(
            f"Smart watchlist for {profile}: {len(watchlist)} stocks "
            f"(min_score={min_score}, market={market_direction})"
        )

        return {
            "profile": profile,
            "min_score_threshold": min_score,
            "market_direction": market_direction,
            "stocks": watchlist,
            "total_found": len(all_stocks),
            "returned": len(watchlist),
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error generating smart watchlist: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart watchlist failed: {str(e)}"
        )


# ============================================================================
# Quick Status Endpoints (Using Batch Snapshots)
# ============================================================================


@app.get(
    "/market/quick",
    summary="Quick market status (real-time)",
    description="""
    Get real-time market status using batch snapshots.

    This is a fast endpoint that fetches all major indices in a single API call,
    providing instant price data without full indicator analysis.

    With AlgoTrader Plus, this provides real-time SIP data.
    """,
)
async def quick_market_status():
    """Get quick market status using real-time snapshots."""
    try:
        result = get_quick_market_status()
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        logger.info(f"Quick market: {result['market_direction']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quick market status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quick market status: {str(e)}"
        )


@app.get(
    "/sectors/quick",
    summary="Quick sector status (real-time)",
    description="""
    Get real-time sector status using batch snapshots.

    This is a fast endpoint that fetches all 11 sector ETFs in a single API call,
    providing instant price data without full indicator analysis.

    With AlgoTrader Plus, this provides real-time SIP data.
    """,
)
async def quick_sector_status():
    """Get quick sector status using real-time snapshots."""
    try:
        result = get_quick_sector_status()
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        logger.info(f"Quick sectors: Leading={result['leading']}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting quick sector status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get quick sector status: {str(e)}"
        )


# ============================================================================
# AlgoTrader Plus Real-Time Endpoints
# ============================================================================


class SnapshotsRequest(BaseModel):
    """Request body for batch snapshots endpoint."""
    symbols: List[str]


@app.get(
    "/quote/{symbol}",
    summary="Get real-time quote",
    description="""
    Get real-time bid/ask quote for a symbol.

    With AlgoTrader Plus subscription, this provides data from the SIP feed
    (all US exchanges) with no delay.

    Returns:
    - Bid/ask prices and sizes
    - Spread (absolute and percentage)
    - Mid price
    - Timestamp
    """,
)
async def get_quote(symbol: str):
    """Get real-time bid/ask quote for a symbol."""
    try:
        result = fetch_latest_quote(symbol.upper())
        logger.info(f"Quote for {symbol}: bid=${result['bid_price']:.2f} ask=${result['ask_price']:.2f}")
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching quote for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch quote: {str(e)}"
        )


@app.get(
    "/trade/{symbol}",
    summary="Get latest trade",
    description="""
    Get the most recent trade execution for a symbol.

    With AlgoTrader Plus subscription, this provides data from the SIP feed
    (all US exchanges) with no delay.

    Returns:
    - Trade price and size
    - Exchange where trade occurred
    - Timestamp
    - Trade conditions
    """,
)
async def get_trade(symbol: str):
    """Get latest trade execution for a symbol."""
    try:
        result = fetch_latest_trade(symbol.upper())
        logger.info(f"Latest trade for {symbol}: ${result['price']:.2f} x {result['size']}")
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching trade for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch trade: {str(e)}"
        )


@app.post(
    "/snapshots",
    summary="Get real-time snapshots for multiple symbols",
    description="""
    Get comprehensive real-time snapshots for multiple symbols in a single API call.

    This is more efficient than fetching data for each symbol individually.
    With AlgoTrader Plus subscription, this provides data from the SIP feed.

    Each snapshot includes:
    - Latest quote (bid/ask)
    - Latest trade
    - Current daily bar
    - Previous daily bar

    Example request body:
    ```json
    {"symbols": ["AAPL", "MSFT", "GOOGL"]}
    ```
    """,
)
async def get_snapshots(request: SnapshotsRequest):
    """Get real-time snapshots for multiple symbols."""
    try:
        if not request.symbols:
            raise ValueError("Symbols list cannot be empty")

        if len(request.symbols) > 100:
            raise ValueError("Maximum 100 symbols per request")

        symbols = [s.upper() for s in request.symbols]
        result = fetch_snapshots(symbols)
        logger.info(f"Fetched snapshots for {len(result)} symbols")
        return result
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching snapshots: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch snapshots: {str(e)}"
        )


# ============================================================================
# WebSocket Streaming Endpoints
# ============================================================================


@app.websocket("/ws/quotes/{symbol}")
async def websocket_quotes(websocket: WebSocket, symbol: str):
    """Stream real-time quotes for a symbol via WebSocket.

    Connects to Alpaca's SIP feed and forwards quote updates to the client.

    Message format:
    ```json
    {
        "type": "quote",
        "symbol": "AAPL",
        "bid_price": 175.50,
        "ask_price": 175.52,
        "bid_size": 100,
        "ask_size": 200,
        "spread": 0.02,
        "timestamp": "2025-01-10T10:30:00Z"
    }
    ```
    """
    await websocket.accept()
    symbol = symbol.upper()
    logger.info(f"WebSocket client connected for {symbol} quotes")

    streamer = get_streamer()
    connected = True

    async def quote_handler(data):
        if not connected:
            return
        try:
            bid = float(data.bid_price)
            ask = float(data.ask_price)
            spread = ask - bid

            await websocket.send_json({
                "type": "quote",
                "symbol": data.symbol,
                "bid_price": bid,
                "ask_price": ask,
                "bid_size": int(data.bid_size),
                "ask_size": int(data.ask_size),
                "spread": round(spread, 4),
                "timestamp": str(data.timestamp) if data.timestamp else None,
            })
        except Exception as e:
            logger.error(f"Error sending quote: {e}")

    try:
        streamer.subscribe_quotes(quote_handler, symbol)

        # Start streamer in background if not running
        if not streamer.is_running:
            streamer.run_in_background()

        # Keep connection alive and handle client messages
        while True:
            try:
                # Wait for any message from client (ping/pong, close, etc.)
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {symbol} quotes")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")
    finally:
        connected = False
        streamer.unsubscribe_quotes(symbol)
        logger.info(f"Cleaned up {symbol} quote subscription")


@app.websocket("/ws/trades/{symbol}")
async def websocket_trades(websocket: WebSocket, symbol: str):
    """Stream real-time trades for a symbol via WebSocket.

    Connects to Alpaca's SIP feed and forwards trade updates to the client.

    Message format:
    ```json
    {
        "type": "trade",
        "symbol": "AAPL",
        "price": 175.51,
        "size": 100,
        "exchange": "XNAS",
        "timestamp": "2025-01-10T10:30:00Z"
    }
    ```
    """
    await websocket.accept()
    symbol = symbol.upper()
    logger.info(f"WebSocket client connected for {symbol} trades")

    streamer = get_streamer()
    connected = True

    async def trade_handler(data):
        if not connected:
            return
        try:
            await websocket.send_json({
                "type": "trade",
                "symbol": data.symbol,
                "price": float(data.price),
                "size": int(data.size),
                "exchange": data.exchange if hasattr(data, 'exchange') else None,
                "timestamp": str(data.timestamp) if data.timestamp else None,
            })
        except Exception as e:
            logger.error(f"Error sending trade: {e}")

    try:
        streamer.subscribe_trades(trade_handler, symbol)

        if not streamer.is_running:
            streamer.run_in_background()

        while True:
            try:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {symbol} trades")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")
    finally:
        connected = False
        streamer.unsubscribe_trades(symbol)
        logger.info(f"Cleaned up {symbol} trade subscription")


@app.websocket("/ws/bars/{symbol}")
async def websocket_bars(websocket: WebSocket, symbol: str):
    """Stream real-time minute bars for a symbol via WebSocket.

    Connects to Alpaca's SIP feed and forwards bar updates to the client.
    Bars are emitted at the end of each minute.

    Message format:
    ```json
    {
        "type": "bar",
        "symbol": "AAPL",
        "open": 175.00,
        "high": 175.60,
        "low": 174.90,
        "close": 175.50,
        "volume": 12345,
        "vwap": 175.25,
        "timestamp": "2025-01-10T10:30:00Z"
    }
    ```
    """
    await websocket.accept()
    symbol = symbol.upper()
    logger.info(f"WebSocket client connected for {symbol} bars")

    streamer = get_streamer()
    connected = True

    async def bar_handler(data):
        if not connected:
            return
        try:
            await websocket.send_json({
                "type": "bar",
                "symbol": data.symbol,
                "open": float(data.open),
                "high": float(data.high),
                "low": float(data.low),
                "close": float(data.close),
                "volume": int(data.volume),
                "vwap": float(data.vwap) if hasattr(data, 'vwap') and data.vwap else None,
                "timestamp": str(data.timestamp) if data.timestamp else None,
            })
        except Exception as e:
            logger.error(f"Error sending bar: {e}")

    try:
        streamer.subscribe_bars(bar_handler, symbol)

        if not streamer.is_running:
            streamer.run_in_background()

        while True:
            try:
                message = await websocket.receive_text()
                if message == "ping":
                    await websocket.send_text("pong")
            except WebSocketDisconnect:
                break

    except WebSocketDisconnect:
        logger.info(f"WebSocket client disconnected from {symbol} bars")
    except Exception as e:
        logger.error(f"WebSocket error for {symbol}: {e}")
    finally:
        connected = False
        streamer.unsubscribe_bars(symbol)
        logger.info(f"Cleaned up {symbol} bar subscription")


@app.get("/streaming/status")
async def streaming_status():
    """Get the current status of the WebSocket streaming connection."""
    streamer = get_streamer()
    return streamer.get_subscription_status()


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled error: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again later."}
    )


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
