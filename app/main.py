"""FastAPI application for StockMate - Intelligent Stock Analysis Backend."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from app.models.request import AnalysisRequest
from app.models.response import AnalysisResponse
from app.tools.analysis import run_analysis
from app.tools.market_scanner import (
    get_market_overview,
    get_sector_performance,
    find_sector_leaders,
    run_market_scan,
)
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
    return {
        "service": "StockMate API",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "analyze": "/analyze",
            "market": "/market",
            "sectors": "/sectors",
            "sector_leaders": "/sectors/{symbol}/leaders",
            "market_scan": "/market/scan",
            "health": "/health",
            "docs": "/docs",
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


@app.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze stock and generate trading recommendation",
    description="""
    Analyze a stock ticker and generate a comprehensive trading recommendation.

    The analysis includes:
    - Price action across multiple timeframes
    - Technical indicators (VWAP, EMA, RSI)
    - Structural support/resistance levels
    - Sentiment analysis
    - Fundamental context

    Returns:
    - BUY or NO_BUY recommendation
    - Confidence score (0-100)
    - If BUY: Complete trade plan with entry, stop loss, targets, and position size
    - Reasoning for the recommendation
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
        request: AnalysisRequest containing symbol, account_size, and use_ai flag

    Returns:
        AnalysisResponse with recommendation and optional trade plan
    """
    logger.info(f"Received analysis request for {request.symbol}")

    try:
        # Validate Alpaca configuration
        settings = get_settings()
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        # Run analysis
        result = run_analysis(
            symbol=request.symbol,
            account_size=request.account_size,
            use_ai=request.use_ai,
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
