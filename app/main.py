"""FastAPI application for StockMate - Intelligent Stock Analysis Backend."""

import json
import logging
from typing import List, Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status, WebSocket, WebSocketDisconnect, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from app.models.request import AnalysisRequest
from app.auth import auth_router, get_current_user, get_optional_user, User
from app.models.response import AnalysisResponse, SmartAnalysisResponse
from app.agent.planning_agent import StockPlanningAgent
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
    fetch_price_bars,
)
from app.tools.streaming import get_streamer
from app.config import get_settings
from app.storage.watchlist_store import get_watchlist_store
from app.services.watchlist_service import remove_symbol_with_cleanup
from app.storage.database import init_database
from app.storage.position_store import get_position_store, Position
from app.storage.alert_history import get_alert_history, Alert
from app.storage.device_store import get_device_store, DeviceToken
from app.services.plan_evaluator import get_plan_evaluator
from app.models.watchlist import (
    WatchlistItem,
    WatchlistResponse,
    SearchResult,
    StockDetailResponse,
    PriceBarResponse,
)

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

    # Initialize SQLite database for positions, alerts, and device tokens
    await init_database()
    logger.info("Database initialized")

    # Test Alpaca API connectivity at startup
    if settings.alpaca_api_key and settings.alpaca_secret_key:
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.common.exceptions import APIError

            is_paper = "paper" in settings.alpaca_base_url.lower()
            client = TradingClient(
                settings.alpaca_api_key,
                settings.alpaca_secret_key,
                paper=is_paper
            )

            # Test connectivity by getting a single asset
            asset = client.get_asset("AAPL")
            if asset:
                logger.info(f"✓ Alpaca API connection verified (tested with AAPL: {asset.name})")
            else:
                logger.warning("⚠ Alpaca API connected but returned no data for test asset")
        except APIError as e:
            logger.error(f"✗ Alpaca API error at startup: {str(e)}")
            logger.warning("Search and live data features may use fallback data")
        except Exception as e:
            logger.error(f"✗ Failed to connect to Alpaca API: {type(e).__name__}: {str(e)}")
            logger.warning("Search and live data features may use fallback data")
    else:
        logger.warning("⚠ Alpaca API keys not configured - search will use fallback data")

    # Start plan evaluator if Claude API is configured
    plan_evaluator = None
    if settings.claude_api_key:
        try:
            from app.services.plan_evaluator import get_plan_evaluator
            plan_evaluator = get_plan_evaluator()
            await plan_evaluator.start()
            logger.info("✓ Plan evaluator started (30-min periodic + key level triggers)")
        except Exception as e:
            logger.error(f"Failed to start plan evaluator: {e}")
    else:
        logger.warning("⚠ Claude API key not configured - plan evaluator disabled")

    yield

    # Cleanup
    if plan_evaluator:
        await plan_evaluator.stop()
    logger.info("Shutting down StockMate backend...")


# Initialize FastAPI app
app = FastAPI(
    title="StockMate API",
    description="Intelligent Stock Analysis Backend with AI-Ready Tools",
    version="1.0.0",
    lifespan=lifespan,
)

# Get settings for CORS configuration
settings = get_settings()

# Configure CORS based on environment
if settings.is_production and settings.cors_origin_list:
    # Production: Use configured origins
    cors_origins = settings.cors_origin_list
else:
    # Development: Allow all origins
    cors_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include authentication router
app.include_router(auth_router)

# Add rate limiting
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.middleware.rate_limit import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Add security headers middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""

    async def dispatch(self, request, call_next) -> Response:
        response = await call_next(request)
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS header for production (HTTPS only)
        if settings.is_production:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"

        return response


app.add_middleware(SecurityHeadersMiddleware)


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
            "watchlist": "/watchlist",
            "watchlist_add": "/watchlist/{symbol}",
            "watchlist_search": "/watchlist/search",
            "stock_detail": "/stock/{symbol}/detail",
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


@app.get("/auth/callback", response_class=HTMLResponse)
async def auth_callback():
    """Handle Supabase auth redirects (email confirmation, password reset)."""
    settings = get_settings()
    return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StockMate - Auth</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600&display=swap" rel="stylesheet">
<style>
*{{margin:0;padding:0;box-sizing:border-box}}
body{{font-family:'Outfit',sans-serif;background:#06080d;color:#f1f5f9;min-height:100vh;display:flex;align-items:center;justify-content:center}}
.card{{max-width:400px;width:100%;margin:20px;padding:48px 40px;background:rgba(13,17,23,0.9);border:1px solid rgba(148,163,184,0.1);border-radius:20px;text-align:center}}
.icon{{width:72px;height:72px;margin:0 auto 24px;background:linear-gradient(135deg,rgba(14,165,233,0.15),rgba(14,165,233,0.05));border:2px solid #0ea5e9;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:32px}}
.icon.success{{background:linear-gradient(135deg,rgba(16,185,129,0.15),rgba(16,185,129,0.05));border-color:#10b981}}
h1{{font-size:24px;font-weight:600;margin-bottom:8px}}
h1.success{{color:#10b981}}
.subtitle{{color:#94a3b8;margin-bottom:28px}}
.form-group{{text-align:left;margin-bottom:20px}}
label{{display:block;font-size:14px;color:#94a3b8;margin-bottom:8px}}
input{{width:100%;padding:14px 16px;background:#0d1117;border:1px solid rgba(148,163,184,0.2);border-radius:10px;color:#f1f5f9;font-size:15px;font-family:inherit}}
input:focus{{outline:none;border-color:#0ea5e9}}
.btn{{width:100%;padding:14px;background:linear-gradient(135deg,#0ea5e9,#0284c7);border:none;border-radius:10px;color:#fff;font-size:15px;font-weight:600;cursor:pointer;margin-top:8px}}
.btn:hover{{opacity:0.9}}
.btn:disabled{{opacity:0.5;cursor:not-allowed}}
.error{{color:#ef4444;font-size:14px;margin-top:12px}}
.success-msg{{color:#10b981;font-size:14px;margin-top:12px}}
.brand{{font-size:12px;color:#64748b;letter-spacing:0.1em;text-transform:uppercase;padding-top:24px;margin-top:24px;border-top:1px solid rgba(148,163,184,0.1)}}
.icon.error{{background:linear-gradient(135deg,rgba(239,68,68,0.15),rgba(239,68,68,0.05));border-color:#ef4444}}
.error-title{{color:#ef4444}}
#reset-form,#success-view,#confirm-view,#error-view{{display:none}}
</style></head>
<body>
<div class="card">
  <!-- Password Reset Form -->
  <div id="reset-form">
    <div class="icon">🔑</div>
    <h1>Reset Password</h1>
    <p class="subtitle">Enter your new password</p>
    <div class="form-group">
      <label>New Password</label>
      <input type="password" id="password" placeholder="Enter new password" minlength="6">
    </div>
    <div class="form-group">
      <label>Confirm Password</label>
      <input type="password" id="confirm-password" placeholder="Confirm new password">
    </div>
    <button class="btn" id="submit-btn" onclick="resetPassword()">Update Password</button>
    <div id="error-msg" class="error"></div>
  </div>

  <!-- Success View -->
  <div id="success-view">
    <div class="icon success">✓</div>
    <h1 class="success">Password Updated!</h1>
    <p class="subtitle">Your password has been reset successfully. You can now open the StockMate app and sign in with your new password.</p>
  </div>

  <!-- Email Confirmed View -->
  <div id="confirm-view">
    <div class="icon success">✓</div>
    <h1 class="success">Email Confirmed!</h1>
    <p class="subtitle">Your account is verified. You can now open the StockMate app and sign in.</p>
  </div>

  <!-- Error View -->
  <div id="error-view">
    <div class="icon error">⚠</div>
    <h1 class="error-title">Link Expired</h1>
    <p class="subtitle" id="error-description">This link has expired or is invalid. Please request a new one from the app.</p>
  </div>

  <div class="brand">StockMate</div>
</div>

<script>
const SUPABASE_URL = '{settings.supabase_url}';

// Parse hash params
function getHashParams() {{
  const hash = window.location.hash.substring(1);
  const params = {{}};
  hash.split('&').forEach(p => {{
    const [k, v] = p.split('=');
    if (k) params[k] = decodeURIComponent(v || '');
  }});
  return params;
}}

// Initialize page based on URL params
const params = getHashParams();
if (params.error) {{
  // Handle error cases
  document.getElementById('error-view').style.display = 'block';
  const desc = params.error_description ? params.error_description.replace(/\+/g, ' ') : 'This link has expired or is invalid.';
  document.getElementById('error-description').textContent = desc + ' Please request a new one from the app.';
}} else if (params.type === 'recovery' && params.access_token) {{
  document.getElementById('reset-form').style.display = 'block';
}} else if (params.access_token) {{
  document.getElementById('confirm-view').style.display = 'block';
}} else {{
  document.getElementById('confirm-view').style.display = 'block';
}}

async function resetPassword() {{
  const password = document.getElementById('password').value;
  const confirmPwd = document.getElementById('confirm-password').value;
  const errorEl = document.getElementById('error-msg');
  const btn = document.getElementById('submit-btn');

  errorEl.textContent = '';

  if (password.length < 6) {{
    errorEl.textContent = 'Password must be at least 6 characters';
    return;
  }}
  if (password !== confirmPwd) {{
    errorEl.textContent = 'Passwords do not match';
    return;
  }}

  btn.disabled = true;
  btn.textContent = 'Updating...';

  try {{
    const res = await fetch(SUPABASE_URL + '/auth/v1/user', {{
      method: 'PUT',
      headers: {{
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + params.access_token,
        'apikey': '{settings.supabase_anon_key}'
      }},
      body: JSON.stringify({{ password }})
    }});

    if (res.ok) {{
      document.getElementById('reset-form').style.display = 'none';
      document.getElementById('success-view').style.display = 'block';
    }} else {{
      const data = await res.json();
      errorEl.textContent = data.message || data.error_description || 'Failed to update password';
      btn.disabled = false;
      btn.textContent = 'Update Password';
    }}
  }} catch (e) {{
    errorEl.textContent = 'Network error. Please try again.';
    btn.disabled = false;
    btn.textContent = 'Update Password';
  }}
}}
</script>
</body></html>"""


@app.get(
    "/debug/alpaca",
    summary="Alpaca API diagnostics",
    description="Debug endpoint to verify Alpaca API connectivity and configuration.",
)
async def debug_alpaca():
    """Verify Alpaca API connection and return diagnostic information."""
    from alpaca.trading.client import TradingClient
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.common.exceptions import APIError

    settings = get_settings()

    # Mask API keys for display (show first 4 and last 4 chars)
    def mask_key(key: str) -> str:
        if not key:
            return "NOT SET"
        if len(key) <= 8:
            return "***"
        return f"{key[:4]}...{key[-4:]}"

    diagnostics = {
        "configuration": {
            "api_key": mask_key(settings.alpaca_api_key),
            "secret_key": mask_key(settings.alpaca_secret_key),
            "base_url": settings.alpaca_base_url,
            "data_feed": settings.alpaca_data_feed,
            "is_paper": "paper" in settings.alpaca_base_url.lower(),
        },
        "trading_api": {
            "status": "unknown",
            "message": None,
            "account_info": None,
        },
        "data_api": {
            "status": "unknown",
            "message": None,
            "test_quote": None,
        },
        "overall_status": "unknown",
    }

    # Check if API keys are configured
    if not settings.alpaca_api_key or not settings.alpaca_secret_key:
        diagnostics["overall_status"] = "error"
        diagnostics["trading_api"]["status"] = "error"
        diagnostics["trading_api"]["message"] = "API keys not configured"
        diagnostics["data_api"]["status"] = "error"
        diagnostics["data_api"]["message"] = "API keys not configured"
        return diagnostics

    # Test Trading API
    try:
        is_paper = "paper" in settings.alpaca_base_url.lower()
        trading_client = TradingClient(
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
            paper=is_paper
        )

        # Try to get account info (simple connectivity test)
        account = trading_client.get_account()
        diagnostics["trading_api"]["status"] = "healthy"
        diagnostics["trading_api"]["message"] = "Successfully connected to Trading API"
        diagnostics["trading_api"]["account_info"] = {
            "account_number": account.account_number,
            "status": account.status.value if account.status else None,
            "buying_power": float(account.buying_power) if account.buying_power else None,
        }
    except APIError as e:
        diagnostics["trading_api"]["status"] = "error"
        diagnostics["trading_api"]["message"] = f"API Error: {str(e)}"
    except Exception as e:
        diagnostics["trading_api"]["status"] = "error"
        diagnostics["trading_api"]["message"] = f"{type(e).__name__}: {str(e)}"

    # Test Data API
    try:
        from alpaca.data.requests import StockLatestQuoteRequest

        data_client = StockHistoricalDataClient(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
        )

        # Try to get a quote for AAPL (simple connectivity test)
        request = StockLatestQuoteRequest(
            symbol_or_symbols="AAPL",
            feed=settings.alpaca_data_feed,
        )
        quotes = data_client.get_stock_latest_quote(request)

        if "AAPL" in quotes:
            quote = quotes["AAPL"]
            diagnostics["data_api"]["status"] = "healthy"
            diagnostics["data_api"]["message"] = f"Successfully connected to Data API (feed: {settings.alpaca_data_feed})"
            diagnostics["data_api"]["test_quote"] = {
                "symbol": "AAPL",
                "bid": float(quote.bid_price) if quote.bid_price else None,
                "ask": float(quote.ask_price) if quote.ask_price else None,
                "timestamp": quote.timestamp.isoformat() if quote.timestamp else None,
            }
        else:
            diagnostics["data_api"]["status"] = "warning"
            diagnostics["data_api"]["message"] = "Connected but no quote data returned"
    except APIError as e:
        diagnostics["data_api"]["status"] = "error"
        diagnostics["data_api"]["message"] = f"API Error: {str(e)}"
    except Exception as e:
        diagnostics["data_api"]["status"] = "error"
        diagnostics["data_api"]["message"] = f"{type(e).__name__}: {str(e)}"

    # Determine overall status
    trading_ok = diagnostics["trading_api"]["status"] == "healthy"
    data_ok = diagnostics["data_api"]["status"] == "healthy"

    if trading_ok and data_ok:
        diagnostics["overall_status"] = "healthy"
    elif trading_ok or data_ok:
        diagnostics["overall_status"] = "partial"
    else:
        diagnostics["overall_status"] = "error"

    logger.info(f"Alpaca diagnostics: {diagnostics['overall_status']}")
    return diagnostics


@app.get(
    "/profiles",
    summary="List trader profile reference data (informational only)",
    description="""
    **DEPRECATED**: The analysis system now uses an expert-based approach that
    automatically determines the optimal trade style based on technical analysis.
    User profile selection is no longer required or used.

    This endpoint returns reference information about different trading styles:
    - **Day Trader**: Intraday focus, tight stops, volume emphasis
    - **Swing Trader**: Multi-day holds, Fibonacci levels, structure-based stops
    - **Position Trader**: Multi-week holds, trend-following, momentum focus
    - **Long-Term Investor**: Multi-month holds, fundamentals, wide stops

    The /analyze endpoint now automatically determines the best approach for each stock.
    """,
    deprecated=True,
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
        "note": "DEPRECATED: The /analyze endpoint now uses expert-based analysis that automatically determines the optimal trade style. Profile selection is no longer used.",
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

    The agent automatically determines the optimal trade style (day/swing/position)
    based on the stock's technical setup - no user preferences needed.

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
        request: AnalysisRequest containing symbol and account_size

    Returns:
        AnalysisResponse with recommendation and optimal trade plan
    """
    logger.info(f"Received expert analysis request for {request.symbol}")

    try:
        # Validate Alpaca configuration
        settings = get_settings()
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Alpaca API credentials not configured. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables."
            )

        # Run expert analysis - trade type determined by market structure
        result = run_analysis(
            symbol=request.symbol,
            account_size=request.account_size,
            use_ai=request.use_ai,
        )

        logger.info(
            f"Expert analysis complete for {request.symbol}: "
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
    "/analyze/smart/{symbol}",
    response_model=SmartAnalysisResponse,
    summary="Smart stock analysis with AI agent",
    description="""
    **Intelligent Profile-Less Stock Analysis**

    This endpoint uses the AI planning agent to:
    1. **Determine optimal trade style** - Agent analyzes volatility, patterns, and timeframe to recommend day/swing/position trade
    2. **Generate comprehensive analysis** - Full technical analysis with precise levels
    3. **Provide educational content** - Hand-holding guidance explaining the setup in plain English
    4. **Create scenario paths** - Bullish, bearish, and sideways scenarios with probabilities

    **Key Features:**
    - No profile selection needed - agent is smart enough to determine the best approach
    - Educational explanations for each level (why support/resistance matters)
    - Multiple scenario paths with probability assessments
    - "What to watch" guidance for novice traders
    - Risk warnings and invalidation criteria

    **Response includes:**
    - Trade style recommendation (day/swing/position) with reasoning
    - Entry zone, stop loss, and targets with explanations
    - Educational content (expandable in the app)
    - Chart annotations for visualization
    """,
    responses={
        200: {"description": "Smart analysis completed successfully"},
        400: {"description": "Invalid symbol"},
        500: {"description": "Internal server error during analysis"},
    }
)
async def smart_analyze_stock(symbol: str):
    """
    Run intelligent stock analysis with the AI planning agent.

    This endpoint doesn't require a trader profile - the agent determines
    the optimal trade style based on the stock's setup.

    Args:
        symbol: Stock ticker symbol (e.g., AAPL, TSLA)

    Returns:
        SmartAnalysisResponse with trade plan and educational content
    """
    symbol = symbol.upper().strip()
    logger.info(f"Received smart analysis request for {symbol}")

    try:
        # Validate Alpaca configuration
        settings = get_settings()
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Alpaca API credentials not configured"
            )

        # Create planning agent and generate smart plan
        agent = StockPlanningAgent(symbol)
        enhanced_plan = await agent.generate_smart_plan()

        # Get current price from agent's cached data
        current_price = agent._market_data.get("price", {}).get("price", 0)

        # Determine recommendation based on confidence and entry zone
        has_entry = enhanced_plan.entry_zone_low is not None
        recommendation = "BUY" if has_entry and enhanced_plan.confidence >= 60 else "NO_BUY"

        # Build response
        from datetime import datetime
        response = SmartAnalysisResponse(
            symbol=symbol,
            current_price=current_price,
            recommendation=recommendation,
            trade_plan=enhanced_plan,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )

        logger.info(
            f"Smart analysis complete for {symbol}: "
            f"{recommendation} ({enhanced_plan.trade_style.recommended_style}, "
            f"confidence: {enhanced_plan.confidence}%)"
        )

        return response

    except ValueError as e:
        logger.error(f"Validation error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    except Exception as e:
        logger.error(f"Error in smart analysis for {symbol}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Smart analysis failed: {str(e)}"
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
    Get a smart watchlist of top stocks based on expert analysis.

    This endpoint is optimized for mobile app consumption, returning:
    - Flattened list of top stocks across all leading sectors
    - Expert-determined trade styles (day/swing/position) for each stock
    - Sorted by score (highest first)

    The agent automatically determines the optimal trade style for each stock
    based on technical analysis - no user preferences needed.

    Returns a simplified response perfect for rendering in a watchlist UI.
    """,
)
async def smart_watchlist(
    max_results: int = 10,
    min_confidence: int = 65,
):
    """Get smart watchlist of top stocks with expert-determined trade styles."""
    try:
        # Use expert threshold - balanced confidence level
        min_score = max(50, min(80, min_confidence))  # Clamp between 50-80

        # Run market scan with expert settings
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
                    "reasons": stock.get("reasons", []),
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
            f"Smart watchlist: {len(watchlist)} stocks "
            f"(min_score={min_score}, market={market_direction})"
        )

        return {
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
# User Watchlist CRUD Endpoints
# ============================================================================


@app.get(
    "/watchlist",
    response_model=WatchlistResponse,
    summary="Get user's watchlist",
    description="""
    Get the user's watchlist with latest price data.

    Returns all symbols the user has added to their watchlist,
    along with current prices and basic analysis data.

    Requires authentication.
    """,
)
async def get_user_watchlist(user: User = Depends(get_current_user)):
    """Get user's watchlist with latest quotes."""
    user_id = user.id
    try:
        store = get_watchlist_store()
        # Handle both sync (JSON) and async (PostgreSQL) stores
        items = store.get_watchlist(user_id)
        if hasattr(items, '__await__'):
            items = await items

        # Enrich with live price data
        if items:
            symbols = [item["symbol"] for item in items]
            try:
                snapshots = fetch_snapshots(symbols)
                for item in items:
                    symbol = item["symbol"]
                    if symbol in snapshots:
                        snap = snapshots[symbol]
                        item["current_price"] = snap.get("latest_trade", {}).get("price")
                        daily = snap.get("daily_bar", {})
                        prev = snap.get("prev_daily_bar", {})
                        if daily and prev and prev.get("close"):
                            change = daily.get("close", 0) - prev.get("close", 0)
                            change_pct = (change / prev["close"]) * 100
                            item["change"] = round(change, 2)
                            item["change_pct"] = round(change_pct, 2)
            except Exception as e:
                logger.warning(f"Failed to fetch snapshots for watchlist: {e}")

        watchlist_items = [WatchlistItem(**item) for item in items]

        return WatchlistResponse(
            user_id=user_id,
            items=watchlist_items,
            count=len(watchlist_items),
        )

    except Exception as e:
        logger.error(f"Error fetching watchlist: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch watchlist: {str(e)}"
        )


@app.post(
    "/watchlist/{symbol}",
    response_model=WatchlistItem,
    summary="Add ticker to watchlist",
    description="Add a stock ticker to the user's watchlist. Requires authentication.",
)
async def add_to_watchlist(
    symbol: str,
    notes: str = None,
    background_tasks: BackgroundTasks = None,
    user: User = Depends(get_current_user),
):
    """Add ticker to user's watchlist."""
    user_id = user.id
    try:
        store = get_watchlist_store()
        item = store.add_symbol(user_id, symbol.upper(), notes=notes)
        if hasattr(item, '__await__'):
            item = await item

        # Fetch current price
        try:
            quote = fetch_latest_quote(symbol.upper())
            item["current_price"] = quote.get("mid_price") or quote.get("ask_price")
        except Exception:
            pass

        # Plan generation is now manual - user triggers it from the Trading Plan view
        # This ensures the plan considers their position if they have one

        logger.info(f"Added {symbol.upper()} to watchlist for user {user_id}")
        return WatchlistItem(**item)

    except Exception as e:
        logger.error(f"Error adding to watchlist: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add to watchlist: {str(e)}"
        )


async def generate_plan_for_stock(symbol: str, user_id: str):
    """Background task to generate a trading plan for a newly added stock."""
    from app.agent.planning_agent import StockPlanningAgent

    try:
        logger.info(f"Generating initial plan for {symbol}")
        agent = StockPlanningAgent(symbol, user_id)
        plan = await agent.generate_plan(force_new=True)
        logger.info(f"Created plan for {symbol}: {plan.bias} bias")
    except Exception as e:
        logger.error(f"Failed to generate plan for {symbol}: {e}")


@app.delete(
    "/watchlist/{symbol}",
    summary="Remove ticker from watchlist",
    description="Remove a stock ticker from the user's watchlist and clean up all related data. Requires authentication.",
)
async def remove_from_watchlist(symbol: str, user: User = Depends(get_current_user)):
    """Remove ticker from user's watchlist and clean up all related data.

    This endpoint performs comprehensive cleanup including:
    - Watchlist entry removal
    - Trading plan deletion
    - Position deletion
    - Conversation history clearing
    - Alert history deletion
    - Agent context cache clearing
    """
    user_id = user.id
    try:
        # Perform comprehensive cleanup
        results = await remove_symbol_with_cleanup(user_id, symbol.upper())

        if not results["watchlist"]:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Symbol {symbol.upper()} not in watchlist"
            )

        logger.info(f"Removed {symbol.upper()} with cleanup for user {user_id}: {results}")
        return {
            "status": "removed",
            "symbol": symbol.upper(),
            "cleanup": results
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error removing from watchlist: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to remove from watchlist: {str(e)}"
        )


@app.get(
    "/watchlist/search",
    response_model=List[SearchResult],
    summary="Search tickers",
    description="""
    Search for stock tickers by symbol or company name.

    Returns matching tickers from US exchanges.
    The response includes a 'source' field indicating whether results came from Alpaca API or fallback data.
    """,
)
async def search_tickers(query: str, limit: int = 10):
    """Search for tickers by symbol or company name."""
    if not query or len(query) < 1:
        return []

    query_upper = query.upper().strip().lstrip("$")
    query_lower = query.lower().strip().lstrip("$")

    # Fallback list of popular stocks for when API fails
    POPULAR_STOCKS = [
        ("AAPL", "Apple Inc.", "NASDAQ", "stock"),
        ("MSFT", "Microsoft Corporation", "NASDAQ", "stock"),
        ("GOOGL", "Alphabet Inc.", "NASDAQ", "stock"),
        ("AMZN", "Amazon.com Inc.", "NASDAQ", "stock"),
        ("NVDA", "NVIDIA Corporation", "NASDAQ", "stock"),
        ("META", "Meta Platforms Inc.", "NASDAQ", "stock"),
        ("TSLA", "Tesla Inc.", "NASDAQ", "stock"),
        ("BRK.B", "Berkshire Hathaway Inc.", "NYSE", "stock"),
        ("JPM", "JPMorgan Chase & Co.", "NYSE", "stock"),
        ("V", "Visa Inc.", "NYSE", "stock"),
        ("JNJ", "Johnson & Johnson", "NYSE", "stock"),
        ("WMT", "Walmart Inc.", "NYSE", "stock"),
        ("MA", "Mastercard Inc.", "NYSE", "stock"),
        ("PG", "Procter & Gamble Co.", "NYSE", "stock"),
        ("HD", "Home Depot Inc.", "NYSE", "stock"),
        ("DIS", "Walt Disney Co.", "NYSE", "stock"),
        ("NFLX", "Netflix Inc.", "NASDAQ", "stock"),
        ("PYPL", "PayPal Holdings Inc.", "NASDAQ", "stock"),
        ("ADBE", "Adobe Inc.", "NASDAQ", "stock"),
        ("CRM", "Salesforce Inc.", "NYSE", "stock"),
        ("AMD", "Advanced Micro Devices Inc.", "NASDAQ", "stock"),
        ("INTC", "Intel Corporation", "NASDAQ", "stock"),
        ("CSCO", "Cisco Systems Inc.", "NASDAQ", "stock"),
        ("PEP", "PepsiCo Inc.", "NASDAQ", "stock"),
        ("KO", "Coca-Cola Co.", "NYSE", "stock"),
        ("MCD", "McDonald's Corp.", "NYSE", "stock"),
        ("NKE", "Nike Inc.", "NYSE", "stock"),
        ("BA", "Boeing Co.", "NYSE", "stock"),
        ("GS", "Goldman Sachs Group Inc.", "NYSE", "stock"),
        ("MS", "Morgan Stanley", "NYSE", "stock"),
        ("C", "Citigroup Inc.", "NYSE", "stock"),
        ("BAC", "Bank of America Corp.", "NYSE", "stock"),
        ("WFC", "Wells Fargo & Co.", "NYSE", "stock"),
        ("T", "AT&T Inc.", "NYSE", "stock"),
        ("VZ", "Verizon Communications Inc.", "NYSE", "stock"),
        ("XOM", "Exxon Mobil Corp.", "NYSE", "stock"),
        ("CVX", "Chevron Corp.", "NYSE", "stock"),
        ("PFE", "Pfizer Inc.", "NYSE", "stock"),
        ("MRK", "Merck & Co. Inc.", "NYSE", "stock"),
        ("ABBV", "AbbVie Inc.", "NYSE", "stock"),
        ("UNH", "UnitedHealth Group Inc.", "NYSE", "stock"),
        ("LLY", "Eli Lilly and Co.", "NYSE", "stock"),
        ("COST", "Costco Wholesale Corp.", "NASDAQ", "stock"),
        ("AVGO", "Broadcom Inc.", "NASDAQ", "stock"),
        ("TXN", "Texas Instruments Inc.", "NASDAQ", "stock"),
        ("QCOM", "Qualcomm Inc.", "NASDAQ", "stock"),
        ("LOW", "Lowe's Companies Inc.", "NYSE", "stock"),
        ("SBUX", "Starbucks Corp.", "NASDAQ", "stock"),
        ("HOOD", "Robinhood Markets Inc.", "NASDAQ", "stock"),
        ("PLTR", "Palantir Technologies Inc.", "NYSE", "stock"),
        ("SOFI", "SoFi Technologies Inc.", "NASDAQ", "stock"),
        ("COIN", "Coinbase Global Inc.", "NASDAQ", "stock"),
        ("RBLX", "Roblox Corp.", "NYSE", "stock"),
        ("SNAP", "Snap Inc.", "NYSE", "stock"),
        ("UBER", "Uber Technologies Inc.", "NYSE", "stock"),
        ("LYFT", "Lyft Inc.", "NASDAQ", "stock"),
        ("SQ", "Block Inc.", "NYSE", "stock"),
        ("SHOP", "Shopify Inc.", "NYSE", "stock"),
        ("ROKU", "Roku Inc.", "NASDAQ", "stock"),
        ("ZM", "Zoom Video Communications Inc.", "NASDAQ", "stock"),
        ("DOCU", "DocuSign Inc.", "NASDAQ", "stock"),
        ("SNOW", "Snowflake Inc.", "NYSE", "stock"),
        ("CRWD", "CrowdStrike Holdings Inc.", "NASDAQ", "stock"),
        ("DDOG", "Datadog Inc.", "NASDAQ", "stock"),
        ("NET", "Cloudflare Inc.", "NYSE", "stock"),
        ("PANW", "Palo Alto Networks Inc.", "NASDAQ", "stock"),
        ("ZS", "Zscaler Inc.", "NASDAQ", "stock"),
        ("OKTA", "Okta Inc.", "NASDAQ", "stock"),
        ("MDB", "MongoDB Inc.", "NASDAQ", "stock"),
        ("ABNB", "Airbnb Inc.", "NASDAQ", "stock"),
        ("DASH", "DoorDash Inc.", "NASDAQ", "stock"),
        ("SPY", "SPDR S&P 500 ETF Trust", "NYSE", "etf"),
        ("QQQ", "Invesco QQQ Trust", "NASDAQ", "etf"),
        ("IWM", "iShares Russell 2000 ETF", "NYSE", "etf"),
        ("DIA", "SPDR Dow Jones Industrial Average ETF", "NYSE", "etf"),
        ("VTI", "Vanguard Total Stock Market ETF", "NYSE", "etf"),
        ("VOO", "Vanguard S&P 500 ETF", "NYSE", "etf"),
        ("ARKK", "ARK Innovation ETF", "NYSE", "etf"),
        ("XLF", "Financial Select Sector SPDR Fund", "NYSE", "etf"),
        ("XLE", "Energy Select Sector SPDR Fund", "NYSE", "etf"),
        ("XLK", "Technology Select Sector SPDR Fund", "NYSE", "etf"),
        ("GLD", "SPDR Gold Shares", "NYSE", "etf"),
        ("SLV", "iShares Silver Trust", "NYSE", "etf"),
        ("TLT", "iShares 20+ Year Treasury Bond ETF", "NASDAQ", "etf"),
    ]

    def search_fallback(query_upper: str, query_lower: str, limit: int):
        """Search using fallback stock list."""
        results = []
        for symbol, name, exchange, asset_type in POPULAR_STOCKS:
            symbol_match = query_upper in symbol
            name_match = query_lower in name.lower()

            if symbol_match or name_match:
                priority = 0 if symbol == query_upper else (1 if symbol.startswith(query_upper) else 2)
                results.append((priority, SearchResult(
                    symbol=symbol,
                    name=name,
                    exchange=exchange,
                    asset_type=asset_type,
                    source="fallback",
                )))

        results.sort(key=lambda x: (x[0], x[1].symbol))
        return [r[1] for r in results[:limit]]

    # Try Alpaca API first
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import GetAssetsRequest
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.common.exceptions import APIError

        settings = get_settings()

        # Check if API keys are configured
        if not settings.alpaca_api_key or not settings.alpaca_secret_key:
            logger.warning("Alpaca API keys not configured, using fallback")
            return search_fallback(query_upper, query_lower, limit)

        # Determine if using paper trading based on base URL
        is_paper = "paper" in settings.alpaca_base_url.lower()

        logger.info(f"Searching Alpaca API for query: '{query}' (paper={is_paper})")

        client = TradingClient(
            settings.alpaca_api_key,
            settings.alpaca_secret_key,
            paper=is_paper
        )

        # Use GetAssetsRequest for server-side filtering (more efficient)
        request = GetAssetsRequest(
            asset_class=AssetClass.US_EQUITY,
            status=AssetStatus.ACTIVE
        )
        assets = client.get_all_assets(request)

        logger.info(f"Alpaca returned {len(assets)} active US equity assets")

        results = []
        for asset in assets:
            # Only include tradable assets
            if not asset.tradable:
                continue

            # Match by symbol or name
            symbol_match = query_upper in asset.symbol
            name_match = query_lower in (asset.name or "").lower()

            if symbol_match or name_match:
                # Prioritize exact symbol matches
                priority = 0 if asset.symbol == query_upper else (1 if asset.symbol.startswith(query_upper) else 2)
                results.append((priority, SearchResult(
                    symbol=asset.symbol,
                    name=asset.name or asset.symbol,
                    exchange=asset.exchange.value if asset.exchange else "UNKNOWN",
                    asset_type="etf" if "ETF" in (asset.name or "").upper() else "stock",
                    source="alpaca",
                )))

        # Sort by priority and limit
        results.sort(key=lambda x: (x[0], x[1].symbol))
        final_results = [r[1] for r in results[:limit]]
        logger.info(f"Search returned {len(final_results)} results from Alpaca API")
        return final_results

    except APIError as e:
        logger.error(f"Alpaca API error during search: {str(e)}")
        logger.warning("Falling back to local stock list due to Alpaca API error")
        return search_fallback(query_upper, query_lower, limit)
    except Exception as e:
        logger.error(f"Unexpected error during Alpaca search: {type(e).__name__}: {str(e)}")
        logger.warning("Falling back to local stock list due to unexpected error")
        return search_fallback(query_upper, query_lower, limit)


@app.get(
    "/stock/{symbol}/detail",
    response_model=StockDetailResponse,
    summary="Get stock detail with charts",
    description="""
    Get comprehensive stock detail for the detail page.

    Includes:
    - Current price and change
    - Analysis score and recommendation
    - Trade plan (if BUY recommendation)
    - Multi-timeframe bar data for charts (1D, 1H, 15M)
    - Support and resistance levels
    """,
)
async def get_stock_detail(symbol: str):
    """Get comprehensive stock detail with multi-timeframe bar data."""
    try:
        symbol = symbol.upper()

        # Run full analysis
        analysis = run_analysis(
            symbol=symbol,
            account_size=10000.0,  # Default account size
            use_ai=False,
        )

        # Fetch multi-timeframe bars
        bars_1d = []
        bars_1h = []
        bars_15m = []

        try:
            # Daily bars (365 days for 52-week high/low calculation)
            daily_bars = fetch_price_bars(symbol, timeframe="1d", days_back=365)
            bars_1d = [
                PriceBarResponse(
                    timestamp=str(bar.timestamp),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                for bar in daily_bars
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch daily bars for {symbol}: {e}")

        try:
            # Hourly bars (7 days)
            hourly_bars = fetch_price_bars(symbol, timeframe="1h", days_back=7)
            bars_1h = [
                PriceBarResponse(
                    timestamp=str(bar.timestamp),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                for bar in hourly_bars
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch hourly bars for {symbol}: {e}")

        try:
            # 15-minute bars (2 days)
            m15_bars = fetch_price_bars(symbol, timeframe="15m", days_back=2)
            bars_15m = [
                PriceBarResponse(
                    timestamp=str(bar.timestamp),
                    open=bar.open,
                    high=bar.high,
                    low=bar.low,
                    close=bar.close,
                    volume=bar.volume,
                )
                for bar in m15_bars
            ]
        except Exception as e:
            logger.warning(f"Failed to fetch 15m bars for {symbol}: {e}")

        # Calculate EMA series for chart overlays
        ema_9_values = []
        ema_21_values = []
        vwap_value = None

        if daily_bars and len(daily_bars) >= 9:
            try:
                from app.tools.indicators import calculate_ema_series, calculate_vwap

                ema_9_values = calculate_ema_series(daily_bars, period=9)
                logger.info(f"Calculated EMA 9 series: {len(ema_9_values)} values")

                if len(daily_bars) >= 21:
                    ema_21_values = calculate_ema_series(daily_bars, period=21)
                    logger.info(f"Calculated EMA 21 series: {len(ema_21_values)} values")

                # Calculate VWAP
                vwap_result = calculate_vwap(daily_bars)
                vwap_value = vwap_result.value
                logger.info(f"Calculated VWAP: {vwap_value}")
            except Exception as e:
                logger.warning(f"Failed to calculate indicators for {symbol}: {e}")

        # Get company name from Alpaca
        company_name = symbol
        try:
            from alpaca.trading.client import TradingClient

            settings = get_settings()
            if settings.alpaca_api_key and settings.alpaca_secret_key:
                is_paper = "paper" in settings.alpaca_base_url.lower()
                client = TradingClient(
                    settings.alpaca_api_key,
                    settings.alpaca_secret_key,
                    paper=is_paper
                )
                asset = client.get_asset(symbol)
                if asset and asset.name:
                    company_name = asset.name
                    logger.info(f"Found company name: {company_name}")
        except Exception as e:
            logger.warning(f"Failed to get company name for {symbol}: {e}")

        # Get current price from latest quote
        current_price = 0.0
        change = 0.0
        change_pct = 0.0

        try:
            quote = fetch_latest_quote(symbol)
            current_price = quote.get("mid_price") or quote.get("ask_price", 0)
        except Exception:
            if bars_1d:
                current_price = bars_1d[-1].close

        # Calculate change from previous close
        if len(bars_1d) >= 2:
            prev_close = bars_1d[-2].close
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100 if prev_close else 0

        # Calculate key statistics
        open_price = None
        high_price = None
        low_price = None
        volume = None
        fifty_two_week_high = None
        fifty_two_week_low = None
        avg_volume = None

        if daily_bars and len(daily_bars) > 0:
            # Today's OHLCV from the latest bar
            latest_bar = daily_bars[-1]
            open_price = round(latest_bar.open, 2)
            high_price = round(latest_bar.high, 2)
            low_price = round(latest_bar.low, 2)
            volume = latest_bar.volume

            # 52-week high/low from all available bars
            all_highs = [bar.high for bar in daily_bars]
            all_lows = [bar.low for bar in daily_bars]
            fifty_two_week_high = round(max(all_highs), 2) if all_highs else None
            fifty_two_week_low = round(min(all_lows), 2) if all_lows else None

            # 30-day average volume
            recent_bars = daily_bars[-30:] if len(daily_bars) >= 30 else daily_bars
            volumes = [bar.volume for bar in recent_bars]
            avg_volume = int(sum(volumes) / len(volumes)) if volumes else None

            logger.info(
                f"Key stats for {symbol}: Open={open_price}, High={high_price}, "
                f"Low={low_price}, Vol={volume}, 52WH={fifty_two_week_high}, "
                f"52WL={fifty_two_week_low}, AvgVol={avg_volume}"
            )

        # Calculate comprehensive support/resistance levels
        support_levels = []
        resistance_levels = []

        if daily_bars and len(daily_bars) >= 10:
            try:
                from app.tools.analysis import find_comprehensive_levels

                levels = find_comprehensive_levels(
                    price_bars=daily_bars,
                    current_price=current_price,
                    ema_9=ema_9_values if ema_9_values else None,
                    ema_21=ema_21_values if ema_21_values else None,
                    vwap=vwap_value,
                )
                # Extract prices from level objects
                support_levels = [lvl["price"] for lvl in levels["support"]]
                resistance_levels = [lvl["price"] for lvl in levels["resistance"]]

                # Log level details for debugging
                logger.info(f"Found {len(support_levels)} support levels: {support_levels[:5]}")
                logger.info(f"Found {len(resistance_levels)} resistance levels: {resistance_levels[:5]}")
            except Exception as e:
                logger.warning(f"Failed to calculate comprehensive levels for {symbol}: {e}")

        # Build trade plan dict
        trade_plan = None
        if analysis.trade_plan:
            tp = analysis.trade_plan
            trade_plan = {
                "trade_type": tp.trade_type,
                "entry_price": tp.entry_price,
                "stop_loss": tp.stop_loss,
                "target_1": tp.target_1,
                "target_2": tp.target_2,
                "target_3": tp.target_3,
                "position_size": tp.position_size,
                "risk_amount": tp.risk_amount,
                "risk_percentage": tp.risk_percentage,
            }
            # Add trade plan levels to support/resistance if not already present
            if tp.stop_loss and tp.stop_loss not in support_levels:
                support_levels.append(round(tp.stop_loss, 2))
            for target in [tp.target_1, tp.target_2, tp.target_3]:
                if target and target not in resistance_levels:
                    resistance_levels.append(round(target, 2))

        # Parse reasons from analysis
        reasons = analysis.reasoning.split(" | ") if analysis.reasoning else []

        logger.info(f"Stock detail for {symbol}: {analysis.recommendation} ({analysis.confidence}%)")

        return StockDetailResponse(
            symbol=symbol,
            name=company_name,
            current_price=round(current_price, 2),
            change=round(change, 2),
            change_pct=round(change_pct, 2),
            # Key statistics
            open_price=open_price,
            high_price=high_price,
            low_price=low_price,
            volume=volume,
            fifty_two_week_high=fifty_two_week_high,
            fifty_two_week_low=fifty_two_week_low,
            avg_volume=avg_volume,
            # Analysis data
            score=analysis.confidence,
            recommendation=analysis.recommendation,
            reasoning=analysis.reasoning,
            reasons=reasons[:5],  # Top 5 reasons
            trade_plan=trade_plan,
            bars_1d=bars_1d,
            bars_1h=bars_1h,
            bars_15m=bars_15m,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            ema_9=ema_9_values,
            ema_21=ema_21_values,
            vwap=vwap_value,
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching stock detail for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch stock detail: {str(e)}"
        )


@app.get(
    "/stock/{symbol}/bars",
    response_model=List[PriceBarResponse],
    summary="Get chart bars for a specific timeframe",
    description="""
    Fetch price bars for a specific timeframe on demand.

    Timeframe options:
    - 1d: Intraday (5-minute bars for 1 day)
    - 1w: 1 week (hourly bars)
    - 1m: 1 month (daily bars)
    - 3m: 3 months (daily bars)
    - 6m: 6 months (daily bars)
    - 1y: 1 year (daily bars)
    - ytd: Year-to-date (daily bars)
    - 5y: 5 years (weekly bars)
    - all: All available history (weekly bars)
    """,
)
async def get_stock_bars(
    symbol: str,
    timeframe: str = "1m",
):
    """Get chart bars for a specific timeframe."""
    try:
        symbol = symbol.upper()

        # Define timeframe configurations
        # Maps API timeframe param -> (Alpaca timeframe, days_back)
        from datetime import date

        # Calculate YTD days
        today = date.today()
        ytd_days = (today - date(today.year, 1, 1)).days + 1

        timeframe_config = {
            "1d": ("1m", 3),       # Intraday: 1-min bars (3 days to cover weekends)
            "1w": ("1h", 7),       # 1 week: hourly bars
            "1m": ("1d", 30),      # 1 month: daily bars
            "3m": ("1d", 90),      # 3 months: daily bars
            "6m": ("1d", 180),     # 6 months: daily bars
            "1y": ("1d", 365),     # 1 year: daily bars
            "ytd": ("1d", ytd_days),  # Year-to-date: daily bars
            "5y": ("1w", 1825),    # 5 years: weekly bars
            "all": ("1w", 3650),   # ~10 years: weekly bars
        }

        if timeframe not in timeframe_config:
            raise ValueError(
                f"Invalid timeframe: {timeframe}. "
                f"Valid options: {', '.join(timeframe_config.keys())}"
            )

        alpaca_tf, days_back = timeframe_config[timeframe]

        logger.info(f"Fetching {timeframe} bars for {symbol}: {alpaca_tf} bars for {days_back} days")

        bars = fetch_price_bars(symbol, timeframe=alpaca_tf, days_back=days_back)

        # For 1D (intraday), filter to only show the last real trading day (extended hours)
        if timeframe == "1d" and bars:
            from collections import defaultdict
            import pytz

            # Extended trading hours: 4:00 AM - 8:00 PM ET
            # Pre-market: 4:00 AM - 9:30 AM
            # Regular: 9:30 AM - 4:00 PM
            # After-hours: 4:00 PM - 8:00 PM
            et = pytz.timezone('America/New_York')
            extended_open_hour = 4   # 4:00 AM ET
            extended_close_hour = 20  # 8:00 PM ET

            # Group bars by date, filtering to extended trading hours
            bars_by_date = defaultdict(list)
            for bar in bars:
                # Convert to ET
                bar_et = bar.timestamp.astimezone(et)
                # Only include bars during extended trading hours (4 AM - 8 PM ET)
                if extended_open_hour <= bar_et.hour < extended_close_hour:
                    bars_by_date[bar_et.date()].append(bar)

            # Find the most recent date with meaningful data (at least 100 bars for 1-min data)
            sorted_dates = sorted(bars_by_date.keys(), reverse=True)
            last_trading_date = None
            for d in sorted_dates:
                if len(bars_by_date[d]) >= 100:
                    last_trading_date = d
                    break

            # Fallback to most recent date if no day has enough bars
            if last_trading_date is None and sorted_dates:
                last_trading_date = sorted_dates[0]

            if last_trading_date:
                bars = bars_by_date[last_trading_date]
                logger.info(f"Filtered to {len(bars)} extended-hours bars for {last_trading_date}")

        return [
            PriceBarResponse(
                timestamp=str(bar.timestamp),
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
            )
            for bar in bars
        ]

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error fetching bars for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch bars: {str(e)}"
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


# ============================================================================
# Position Management Endpoints
# ============================================================================


class CreatePositionRequest(BaseModel):
    """Request body for creating a position."""
    symbol: str
    trade_type: str = "swing"
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    notes: Optional[str] = None


class EnterPositionRequest(BaseModel):
    """Request body for entering a position."""
    entry_price: float
    size: int


class UpdatePositionRequest(BaseModel):
    """Request body for updating a position."""
    stop_loss: Optional[float] = None
    shares_sold: Optional[int] = None
    target_hit: Optional[int] = None


class AddEntryRequest(BaseModel):
    """Request body for adding an entry to a position."""
    price: float
    shares: int
    date: Optional[str] = None


class AddExitRequest(BaseModel):
    """Request body for adding an exit from a position."""
    price: float
    shares: int
    reason: str = "manual"  # target_1, target_2, target_3, stop_loss, manual
    date: Optional[str] = None


@app.post(
    "/positions",
    summary="Create a position",
    description="Create a new position to track (starts in 'watching' status).",
)
async def create_position(request: CreatePositionRequest, user: User = Depends(get_current_user)):
    """Create a new position for tracking."""
    user_id = user.id
    try:
        store = get_position_store()
        position = await store.create_position(
            user_id=user_id,
            symbol=request.symbol.upper(),
            trade_type=request.trade_type,
            stop_loss=request.stop_loss,
            target_1=request.target_1,
            target_2=request.target_2,
            target_3=request.target_3,
            notes=request.notes,
        )
        logger.info(f"Created position for {request.symbol.upper()}")
        return position.model_dump()
    except Exception as e:
        logger.error(f"Error creating position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create position: {str(e)}"
        )


@app.get(
    "/positions",
    summary="Get all positions",
    description="Get all positions for a user.",
)
async def get_positions(active_only: bool = True, user: User = Depends(get_current_user)):
    """Get all positions for a user."""
    user_id = user.id
    try:
        store = get_position_store()
        if active_only:
            positions = await store.get_active_positions(user_id)
        else:
            positions = await store.get_all_positions(user_id)
        return [p.model_dump() for p in positions]
    except Exception as e:
        logger.error(f"Error getting positions: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get positions: {str(e)}"
        )


@app.get(
    "/positions/{symbol}",
    summary="Get position for symbol",
    description="Get position details for a specific symbol.",
)
async def get_position(symbol: str, user: User = Depends(get_current_user)):
    """Get position for a specific symbol."""
    user_id = user.id
    try:
        store = get_position_store()
        position = await store.get_position(user_id, symbol.upper())
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol.upper()}"
            )
        return position.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get position: {str(e)}"
        )


@app.post(
    "/positions/{symbol}/enter",
    summary="Enter a position",
    description="Mark a position as entered with entry price and size.",
)
async def enter_position(symbol: str, request: EnterPositionRequest, user: User = Depends(get_current_user)):
    """Enter a position with price and size."""
    user_id = user.id
    try:
        store = get_position_store()
        position = await store.enter_position(
            user_id=user_id,
            symbol=symbol.upper(),
            entry_price=request.entry_price,
            size=request.size,
        )
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol.upper()}"
            )
        logger.info(f"Entered position {symbol.upper()} @ ${request.entry_price}")
        return position.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error entering position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enter position: {str(e)}"
        )


@app.post(
    "/positions/{symbol}/entries",
    summary="Add entry to position",
    description="Add an entry to an existing position (scale in or initial entry). Recalculates average entry price automatically.",
)
async def add_position_entry(symbol: str, request: AddEntryRequest, user: User = Depends(get_current_user)):
    """Add an entry to a position."""
    user_id = user.id
    try:
        store = get_position_store()
        position = await store.add_entry(
            user_id=user_id,
            symbol=symbol.upper(),
            price=request.price,
            shares=request.shares,
            date=request.date,
        )
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol.upper()}"
            )
        logger.info(f"Added entry to {symbol.upper()}: {request.shares} shares @ ${request.price}")

        return position.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding entry: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add entry: {str(e)}"
        )


@app.post(
    "/positions/{symbol}/exits",
    summary="Add exit from position",
    description="Add an exit from a position (partial or full). Calculates realized P&L automatically.",
)
async def add_position_exit(symbol: str, request: AddExitRequest, user: User = Depends(get_current_user)):
    """Add an exit from a position."""
    user_id = user.id
    try:
        store = get_position_store()
        position = await store.add_exit(
            user_id=user_id,
            symbol=symbol.upper(),
            price=request.price,
            shares=request.shares,
            reason=request.reason,
            date=request.date,
        )
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol.upper()}"
            )
        logger.info(f"Added exit from {symbol.upper()}: {request.shares} shares @ ${request.price}")

        return position.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding exit: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add exit: {str(e)}"
        )


@app.get(
    "/positions/{symbol}/pnl",
    summary="Get position with live P&L",
    description="Get position details with unrealized P&L calculated from current market price.",
)
async def get_position_with_pnl(symbol: str, user: User = Depends(get_current_user)):
    """Get position with live P&L calculated."""
    user_id = user.id
    try:
        from app.tools.market_data import fetch_latest_trade

        store = get_position_store()
        symbol_upper = symbol.upper()

        # Get current price for P&L calculation
        current_price = None
        try:
            trade_data = fetch_latest_trade(symbol_upper)
            current_price = trade_data.get("price")
        except Exception as price_error:
            logger.warning(f"Could not fetch price for {symbol_upper}: {price_error}")

        position = await store.get_position_with_pnl(
            user_id=user_id,
            symbol=symbol_upper,
            current_price=current_price,
        )
        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol_upper}"
            )

        result = position.model_dump()
        result["current_price"] = current_price
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting position with P&L: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get position with P&L: {str(e)}"
        )


@app.patch(
    "/positions/{symbol}",
    summary="Update a position",
    description="Update position (move stop loss, scale out).",
)
async def update_position(symbol: str, request: UpdatePositionRequest, user: User = Depends(get_current_user)):
    """Update a position."""
    user_id = user.id
    try:
        store = get_position_store()
        position = None

        if request.stop_loss is not None:
            position = await store.update_stop_loss(user_id, symbol.upper(), request.stop_loss)
            logger.info(f"Updated stop loss for {symbol.upper()} to ${request.stop_loss}")

        if request.shares_sold is not None and request.target_hit is not None:
            position = await store.scale_out(
                user_id, symbol.upper(), request.shares_sold, request.target_hit
            )
            logger.info(f"Scaled out {request.shares_sold} shares of {symbol.upper()}")

        if not position:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol.upper()}"
            )

        return position.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update position: {str(e)}"
        )


@app.delete(
    "/positions/{symbol}",
    summary="Delete a position",
    description="Delete a position completely, removing all entries and exits.",
)
async def delete_position_endpoint(symbol: str, user: User = Depends(get_current_user)):
    """Delete a position completely."""
    user_id = user.id
    try:
        store = get_position_store()
        deleted = await store.delete_position(user_id, symbol.upper())
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No position found for {symbol.upper()}"
            )
        logger.info(f"Deleted position {symbol.upper()}")
        return {"status": "deleted", "symbol": symbol.upper()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting position: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete position: {str(e)}"
        )


# ============================================================================
# Device Registration Endpoints (Push Notifications)
# ============================================================================


class RegisterDeviceRequest(BaseModel):
    """Request body for device registration."""
    device_token: str
    platform: str = "ios"


@app.post(
    "/devices/register",
    summary="Register device for push notifications",
    description="Register a device token for receiving push notifications.",
)
async def register_device(request: RegisterDeviceRequest, user: User = Depends(get_current_user)):
    """Register a device for push notifications."""
    user_id = user.id
    try:
        store = get_device_store()
        device = await store.register_device(
            user_id=user_id,
            device_token=request.device_token,
            platform=request.platform,
        )
        logger.info(f"Registered device for user {user_id}")
        return device.model_dump()
    except Exception as e:
        logger.error(f"Error registering device: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register device: {str(e)}"
        )


@app.delete(
    "/devices/{device_token}",
    summary="Unregister device",
    description="Remove a device from push notifications.",
)
async def unregister_device(device_token: str):
    """Unregister a device from push notifications."""
    try:
        store = get_device_store()
        removed = await store.unregister_device(device_token)
        if not removed:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Device not found"
            )
        logger.info("Unregistered device")
        return {"status": "unregistered"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unregistering device: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister device: {str(e)}"
        )


@app.get(
    "/devices",
    summary="Get registered devices",
    description="Get all registered devices for a user.",
)
async def get_devices(user: User = Depends(get_current_user)):
    """Get all registered devices for a user."""
    user_id = user.id
    try:
        store = get_device_store()
        devices = await store.get_user_devices(user_id)
        return [d.model_dump() for d in devices]
    except Exception as e:
        logger.error(f"Error getting devices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get devices: {str(e)}"
        )


# ============================================================================
# Alert History Endpoints
# ============================================================================


@app.get(
    "/alerts/history",
    summary="Get alert history",
    description="Get recent alerts for a user, optionally filtered by symbol.",
)
async def get_alert_history_endpoint(
    user_id: str = "default",
    symbol: Optional[str] = None,
    limit: int = 50,
):
    """Get alert history for a user."""
    try:
        history = get_alert_history()
        alerts = await history.get_recent_alerts(
            user_id=user_id,
            limit=limit,
            symbol=symbol.upper() if symbol else None,
        )
        return [a.model_dump() for a in alerts]
    except Exception as e:
        logger.error(f"Error getting alert history: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get alert history: {str(e)}"
        )


@app.post(
    "/alerts/{alert_id}/acknowledge",
    summary="Acknowledge an alert",
    description="Mark an alert as acknowledged.",
)
async def acknowledge_alert(alert_id: str):
    """Acknowledge an alert."""
    try:
        history = get_alert_history()
        acknowledged = await history.acknowledge_alert(alert_id)
        if not acknowledged:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Alert not found"
            )
        return {"status": "acknowledged", "alert_id": alert_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error acknowledging alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )


# ============================================================================
# Agent Control Endpoints
# ============================================================================


@app.get(
    "/agent/status",
    summary="Get agent monitoring status",
    description="Get the current status of the AI stock monitoring agent and plan evaluator.",
)
async def get_agent_status():
    """Get agent monitoring status."""
    from app.agent.master_agent import get_master_agent
    from app.services.scheduler import is_market_open, get_next_market_open
    from app.services.push_notification import get_apns_service
    from app.services.plan_evaluator import get_plan_evaluator

    agent = get_master_agent()
    apns = get_apns_service()
    evaluator = get_plan_evaluator()

    return {
        "agent": agent.get_stats(),
        "plan_evaluator": evaluator.get_status(),
        "push_notifications": apns.get_status(),
        "market_hours": {
            "is_open": is_market_open(),
            "next_open": get_next_market_open().isoformat() if not is_market_open() else None,
        },
    }


@app.post(
    "/agent/start",
    summary="Start agent monitoring",
    description="Start the AI stock monitoring agent for the current user.",
)
async def start_agent():
    """Start the agent monitoring system."""
    from app.agent.master_agent import get_master_agent
    from app.services.push_notification import send_trading_alert

    # Set up alert callback to send push notifications
    async def alert_callback(user_id, symbol, alert_type, message, price):
        await send_trading_alert(user_id, symbol, alert_type, message, price)

    agent = get_master_agent(alert_callback)

    if agent._running:
        return {"status": "already_running", "message": "Agent is already running"}

    try:
        await agent.start()
        return {
            "status": "started",
            "message": "Agent monitoring started",
            "stats": agent.get_stats(),
        }
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start agent: {str(e)}"
        )


@app.post(
    "/agent/stop",
    summary="Stop agent monitoring",
    description="Stop the AI stock monitoring agent.",
)
async def stop_agent():
    """Stop the agent monitoring system."""
    from app.agent.master_agent import get_master_agent

    agent = get_master_agent()

    if not agent._running:
        return {"status": "not_running", "message": "Agent is not running"}

    try:
        await agent.stop()
        return {
            "status": "stopped",
            "message": "Agent monitoring stopped",
        }
    except Exception as e:
        logger.error(f"Error stopping agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop agent: {str(e)}"
        )


@app.post(
    "/agent/symbols/{symbol}",
    summary="Add symbol to agent monitoring",
    description="Add a symbol to be monitored by the AI agent.",
)
async def add_symbol_to_agent(symbol: str, user: User = Depends(get_current_user)):
    """Add a symbol to agent monitoring."""
    from app.agent.master_agent import get_master_agent

    user_id = user.id
    agent = get_master_agent()

    if not agent._running:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Agent is not running. Start the agent first."
        )

    success = await agent.add_symbol(symbol.upper(), user_id)

    if success:
        return {
            "status": "added",
            "symbol": symbol.upper(),
            "monitored_symbols": list(agent._stock_agents.keys()),
        }
    else:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add {symbol} to monitoring"
        )


@app.delete(
    "/agent/symbols/{symbol}",
    summary="Remove symbol from agent monitoring",
    description="Remove a symbol from AI agent monitoring.",
)
async def remove_symbol_from_agent(symbol: str):
    """Remove a symbol from agent monitoring."""
    from app.agent.master_agent import get_master_agent

    agent = get_master_agent()
    await agent.remove_symbol(symbol.upper())

    return {
        "status": "removed",
        "symbol": symbol.upper(),
        "monitored_symbols": list(agent._stock_agents.keys()),
    }


# ============================================================================
# Chat & Planning Agent Endpoints
# ============================================================================


class ChatRequest(BaseModel):
    """Request body for chat."""
    message: str


class ChatResponse(BaseModel):
    """Response from chat."""
    symbol: str
    response: str
    context: dict
    has_plan: bool = False
    plan_status: Optional[str] = None


def _parse_price(value: Any) -> Optional[float]:
    """Parse a price value, handling dollar signs and formatting."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("$", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


class PlanResponse(BaseModel):
    """Response containing a trading plan."""
    symbol: str
    bias: str = "neutral"
    thesis: str = ""
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_loss: Optional[float] = None
    stop_reasoning: str = ""
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    target_reasoning: str = ""
    risk_reward: Optional[float] = None
    key_supports: List[float] = []
    key_resistances: List[float] = []
    invalidation_criteria: str = ""
    technical_summary: str = ""
    status: str = "draft"
    created_at: str = ""
    last_evaluation: Optional[str] = None
    evaluation_notes: Optional[str] = None
    # Trade style fields
    trade_style: Optional[str] = None
    trade_style_reasoning: Optional[str] = None
    holding_period: Optional[str] = None
    confidence: Optional[int] = None
    # External sentiment from web search
    news_summary: Optional[str] = None
    reddit_sentiment: Optional[str] = None
    reddit_buzz: Optional[str] = None

    @field_validator(
        'entry_zone_low', 'entry_zone_high', 'stop_loss',
        'target_1', 'target_2', 'target_3', 'risk_reward',
        mode='before'
    )
    @classmethod
    def parse_price_fields(cls, v):
        """Parse price fields, handling dollar signs."""
        return _parse_price(v)

    @field_validator('key_supports', 'key_resistances', mode='before')
    @classmethod
    def parse_price_lists(cls, v):
        """Parse price list fields, handling dollar signs."""
        if not v:
            return []
        if isinstance(v, list):
            return [p for p in [_parse_price(x) for x in v] if p is not None]
        return []


@app.post(
    "/chat/{symbol}",
    response_model=ChatResponse,
    summary="Chat with stock planning agent",
    description="""
    Chat with an AI planning agent about a specific stock. The agent has full context of:
    - ALL technical indicators (RSI, MACD, EMAs, Bollinger, ADX, etc.)
    - Key support/resistance levels
    - Volume analysis and chart patterns
    - Your trading plan (if any)
    - Your position status
    - Previous conversation history

    The agent can:
    - Answer questions about the stock
    - Create trading plans with entry, stop, targets
    - Evaluate plans as price progresses
    - Discuss trade management

    Example questions:
    - "Create a trading plan for this stock"
    - "What are the key levels I should watch?"
    - "Is my stop loss still valid?"
    - "Should I take profit here?"
    - "Has anything changed since you created the plan?"
    """,
)
async def chat_with_stock(symbol: str, request: ChatRequest, user: User = Depends(get_current_user)):
    """Chat with the planning agent about a specific stock."""
    from app.agent.planning_agent import StockPlanningAgent

    user_id = user.id
    symbol = symbol.upper()

    try:
        # Create planning agent
        agent = StockPlanningAgent(symbol, user_id)

        # Check for plan creation/update requests
        msg_lower = request.message.lower()

        # Phrases that trigger plan creation
        create_phrases = [
            "create a plan", "make a plan", "generate a plan", "new plan",
            "build a plan", "make me a plan", "create plan", "trading plan for",
            "give me a plan", "set up a plan", "develop a plan"
        ]

        # Phrases that trigger plan evaluation
        evaluate_phrases = [
            "evaluate", "still valid", "update the plan", "check the plan",
            "review the plan", "how's the plan", "is the plan still good",
            "reassess", "re-evaluate", "recheck", "validate the plan",
            "is my plan", "what about my plan", "plan status",
            "anything changed", "has anything changed", "what changed",
            "still on track", "how are we doing", "where are we",
            "should i adjust", "need to adjust", "modify the plan"
        ]

        evaluation_result = None

        if any(phrase in msg_lower for phrase in create_phrases):
            # Generate a new plan
            await agent.generate_plan(force_new=True)
            response_text = await agent.chat(request.message)

        elif any(phrase in msg_lower for phrase in evaluate_phrases):
            # Evaluate existing plan and get result
            evaluation_result = await agent.evaluate_plan()
            # Chat will now have access to the fresh evaluation in context
            response_text = await agent.chat(request.message)

        else:
            # Regular chat
            response_text = await agent.chat(request.message)

        # Get current plan status
        plan = await agent.get_plan()
        summary = await agent.get_summary()

        # Build context with evaluation info if available
        context = {
            "has_position": summary.get("has_position", False),
            "market_direction": summary.get("market_direction"),
            "current_price": summary.get("current_price"),
        }

        # Add evaluation result if we just evaluated
        if evaluation_result and not evaluation_result.get("error"):
            context["just_evaluated"] = True
            context["evaluation_status"] = evaluation_result.get("plan_status")
            context["price_at_creation"] = evaluation_result.get("price_at_creation")

        return ChatResponse(
            symbol=symbol,
            response=response_text,
            context=context,
            has_plan=plan is not None,
            plan_status=plan.status if plan else None,
        )

    except Exception as e:
        logger.error(f"Chat error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@app.get(
    "/chat/{symbol}/plan",
    response_model=PlanResponse,
    summary="Get existing trading plan",
    description="Get the existing AI trading plan for a stock (does not create new).",
)
async def get_plan(
    symbol: str,
    user_id: str = "default"
):
    """Get an existing trading plan for a stock."""
    from app.storage.plan_store import get_plan_store

    symbol = symbol.upper()
    plan_store = get_plan_store()
    plan = await plan_store.get_plan(user_id, symbol)

    if not plan:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No trading plan found for {symbol}"
        )

    return PlanResponse(
        symbol=plan.symbol,
        bias=plan.bias,
        thesis=plan.thesis,
        entry_zone_low=plan.entry_zone_low,
        entry_zone_high=plan.entry_zone_high,
        stop_loss=plan.stop_loss,
        stop_reasoning=plan.stop_reasoning,
        target_1=plan.target_1,
        target_2=plan.target_2,
        target_3=plan.target_3,
        target_reasoning=plan.target_reasoning,
        risk_reward=plan.risk_reward,
        key_supports=plan.key_supports,
        key_resistances=plan.key_resistances,
        invalidation_criteria=plan.invalidation_criteria,
        technical_summary=plan.technical_summary,
        status=plan.status,
        created_at=plan.created_at,
        last_evaluation=plan.last_evaluation,
        evaluation_notes=plan.evaluation_notes,
        trade_style=plan.trade_style,
        trade_style_reasoning=plan.trade_style_reasoning,
        holding_period=plan.holding_period,
        confidence=plan.confidence,
        news_summary=plan.news_summary,
        reddit_sentiment=plan.reddit_sentiment,
        reddit_buzz=plan.reddit_buzz,
    )


@app.post(
    "/chat/{symbol}/plan",
    response_model=PlanResponse,
    summary="Generate trading plan",
    description="Generate a new AI trading plan.",
)
async def create_plan(
    symbol: str,
    force_new: bool = False,
    user_id: str = "default"
):
    """Generate a trading plan for a stock."""
    from app.agent.planning_agent import StockPlanningAgent

    symbol = symbol.upper()

    try:
        agent = StockPlanningAgent(symbol, user_id)
        plan = await agent.generate_plan(force_new=force_new)

        return PlanResponse(
            symbol=plan.symbol,
            bias=plan.bias,
            thesis=plan.thesis,
            entry_zone_low=plan.entry_zone_low,
            entry_zone_high=plan.entry_zone_high,
            stop_loss=plan.stop_loss,
            stop_reasoning=plan.stop_reasoning,
            target_1=plan.target_1,
            target_2=plan.target_2,
            target_3=plan.target_3,
            target_reasoning=plan.target_reasoning,
            risk_reward=plan.risk_reward,
            key_supports=plan.key_supports,
            key_resistances=plan.key_resistances,
            invalidation_criteria=plan.invalidation_criteria,
            technical_summary=plan.technical_summary,
            status=plan.status,
            created_at=plan.created_at,
            last_evaluation=plan.last_evaluation,
            evaluation_notes=plan.evaluation_notes,
            trade_style=plan.trade_style,
            trade_style_reasoning=plan.trade_style_reasoning,
            holding_period=plan.holding_period,
            confidence=plan.confidence,
            news_summary=plan.news_summary,
            reddit_sentiment=plan.reddit_sentiment,
            reddit_buzz=plan.reddit_buzz,
        )

    except Exception as e:
        logger.error(f"Plan error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Plan generation failed: {str(e)}"
        )


@app.post(
    "/chat/{symbol}/plan/stream",
    summary="Generate trading plan with streaming",
    description="Generate a new AI trading plan with Server-Sent Events streaming.",
)
async def create_plan_stream(
    symbol: str,
    force_new: bool = True,
    user_id: str = "default"
):
    """Generate a trading plan with real-time streaming output.

    Streams detailed analysis steps with findings in Claude Code agent style.
    Each step includes: type, step_type, status, findings[], timestamp
    """
    from app.agent.planning_agent import StockPlanningAgent
    from app.tools.market_data import fetch_price_bars, fetch_latest_quote
    from app.tools.indicators import calculate_ema_series, calculate_rsi_series
    from app.tools.analysis import detect_chart_patterns
    from app.tools.chart_generator import generate_chart_image
    import time

    symbol = symbol.upper()

    def emit_step(step_type: str, status: str, findings: list = None):
        """Helper to emit a step update."""
        return f"data: {json.dumps({'type': 'step', 'step_type': step_type, 'status': status, 'findings': findings or [], 'timestamp': time.time()})}\n\n"

    async def generate_stream():
        try:
            agent = StockPlanningAgent(symbol, user_id)

            # Step 1: Gathering market data
            yield emit_step("gathering_data", "active")

            try:
                quote = fetch_latest_quote(symbol)
                price = quote.get("mid_price", 0)
                bid = quote.get("bid_price", 0)
                ask = quote.get("ask_price", 0)
                findings = [
                    f"Price: ${price:.2f}",
                    f"Bid: ${bid:.2f} | Ask: ${ask:.2f}",
                ]
            except Exception:
                findings = ["Fetching market data..."]

            yield emit_step("gathering_data", "completed", findings)

            # Step 2: Technical indicators
            yield emit_step("technical_indicators", "active")

            await agent.gather_comprehensive_data()
            tech = agent._technical_data or {}

            rsi_data = tech.get("rsi", {})
            macd_data = tech.get("macd", {})
            ema_data = tech.get("emas", {})

            tech_findings = []
            if rsi_data:
                rsi_val = rsi_data.get("value", 0)
                rsi_signal = rsi_data.get("signal", "neutral")
                tech_findings.append(f"RSI: {rsi_val:.1f} ({rsi_signal})")
            if macd_data:
                macd_signal = macd_data.get("signal", "neutral")
                tech_findings.append(f"MACD: {macd_signal.replace('_', ' ').title()}")
            if ema_data:
                above_9 = "above" if ema_data.get("above_9") else "below"
                above_21 = "above" if ema_data.get("above_21") else "below"
                above_50 = "above" if ema_data.get("above_50") else "below"
                tech_findings.append(f"EMA: Price {above_9} 9, {above_21} 21, {above_50} 50")

            yield emit_step("technical_indicators", "completed", tech_findings or ["Analysis complete"])

            # Step 3: Support & Resistance
            yield emit_step("support_resistance", "active")

            levels = agent._levels_data or {}
            supports = levels.get("support_levels", [])[:3]
            resistances = levels.get("resistance_levels", [])[:3]

            level_findings = []
            if supports:
                support_strs = [f"${s.get('price', s):.2f}" if isinstance(s, dict) else f"${s:.2f}" for s in supports[:2]]
                level_findings.append(f"Support: {', '.join(support_strs)}")
            if resistances:
                resist_strs = [f"${r.get('price', r):.2f}" if isinstance(r, dict) else f"${r:.2f}" for r in resistances[:2]]
                level_findings.append(f"Resistance: {', '.join(resist_strs)}")

            yield emit_step("support_resistance", "completed", level_findings or ["Levels identified"])

            # Step 4: Chart patterns
            yield emit_step("chart_patterns", "active")

            patterns = agent._patterns_data or {}
            pattern_list = patterns.get("patterns", [])

            pattern_findings = []
            if pattern_list:
                for p in pattern_list[:2]:
                    pname = p.get("name", "Pattern")
                    pconf = p.get("confidence", 0)
                    pattern_findings.append(f"Found: {pname} ({pconf}% confidence)")
            else:
                pattern_findings.append("No clear patterns detected")

            yield emit_step("chart_patterns", "completed", pattern_findings)

            # Step 5: Generate chart for Vision
            yield emit_step("generating_chart", "active")

            try:
                bars = fetch_price_bars(symbol, timeframe="1d", days_back=120)
                if bars and len(bars) >= 60:
                    ema_9 = calculate_ema_series(bars, 9)
                    ema_21 = calculate_ema_series(bars, 21)
                    ema_50 = calculate_ema_series(bars, 50)
                    rsi_vals = calculate_rsi_series(bars, 14)

                    indicators = {"ema_9": ema_9, "ema_21": ema_21, "ema_50": ema_50, "rsi": rsi_vals}
                    chart_image = generate_chart_image(symbol, bars, indicators, lookback=60)
                    chart_size_kb = len(chart_image) / 1024
                    chart_findings = [f"Chart rendered ({chart_size_kb:.0f}KB)", "EMAs + RSI + Volume overlays"]
                else:
                    chart_findings = ["Insufficient data for chart"]
            except Exception as e:
                chart_findings = [f"Chart generation: {str(e)[:50]}"]

            yield emit_step("generating_chart", "completed", chart_findings)

            # Step 6: Vision analysis - Actually run visual analysis on the chart
            yield emit_step("vision_analysis", "active")
            yield emit_step("vision_analysis", "active", ["Sending chart to Claude Vision..."])

            visual_analysis_result = None
            try:
                # Call the agent's visual analysis method
                visual_analysis_result = await agent._perform_visual_analysis()
                if visual_analysis_result:
                    trend = visual_analysis_result.get("trend_quality", {}).get("assessment", "unknown")
                    modifier = visual_analysis_result.get("visual_confidence_modifier", 0)
                    patterns = visual_analysis_result.get("visual_patterns_identified", [])
                    warnings = visual_analysis_result.get("warning_signs", [])

                    vision_findings = [
                        f"Trend quality: {trend.title()}",
                        f"Visual confidence: {'+' if modifier >= 0 else ''}{modifier}",
                    ]
                    if patterns:
                        vision_findings.append(f"Patterns: {', '.join(patterns[:2])}")
                    if warnings:
                        vision_findings.append(f"Warning: {warnings[0][:40]}...")

                    yield emit_step("vision_analysis", "completed", vision_findings)
                else:
                    yield emit_step("vision_analysis", "completed", ["No significant patterns found"])
            except Exception as e:
                logger.warning(f"Visual analysis failed for {symbol}: {e}")
                yield emit_step("vision_analysis", "completed", ["Visual analysis unavailable"])

            # Step 7: Generate the plan
            yield emit_step("generating_plan", "active")

            # Stream the actual plan generation
            plan_result = None
            async for chunk in agent.generate_plan_streaming(force_new=force_new):
                # Capture plan from plan_complete or existing_plan events
                chunk_type = chunk.get("type")
                if chunk_type in ("plan_complete", "existing_plan"):
                    plan_result = chunk.get("plan", {})
                    # Inject visual analysis results into the plan
                    if visual_analysis_result and plan_result:
                        modifier = visual_analysis_result.get("visual_confidence_modifier", 0)
                        plan_result["visual_analysis"] = {
                            "confidence_modifier": modifier,
                            "trend_quality": visual_analysis_result.get("trend_quality", {}),
                            "patterns_identified": visual_analysis_result.get("visual_patterns_identified", []),
                            "warning_signs": visual_analysis_result.get("warning_signs", []),
                        }
                        # Update the chunk with enriched plan
                        chunk["plan"] = plan_result
                yield f"data: {json.dumps(chunk)}\n\n"

            # Final plan step completion
            if plan_result:
                bias = plan_result.get("bias", "neutral")
                confidence = plan_result.get("confidence", 0)
                plan_findings = [
                    f"Bias: {bias.title()}",
                    f"Confidence: {confidence}%",
                ]
                yield emit_step("generating_plan", "completed", plan_findings)
            else:
                yield emit_step("generating_plan", "completed", ["Plan generated"])

            # Complete
            yield emit_step("complete", "completed", ["Trading plan ready for review"])
            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.error(f"Streaming plan error for {symbol}: {str(e)}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            yield "data: [DONE]\n\n"  # Always send DONE to close stream cleanly

    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.post(
    "/chat/{symbol}/evaluate",
    summary="Evaluate trading plan",
    description="Re-evaluate the trading plan against current market conditions.",
)
async def evaluate_plan(symbol: str, user: User = Depends(get_current_user)):
    """Evaluate an existing trading plan."""
    from app.agent.planning_agent import StockPlanningAgent

    user_id = user.id
    symbol = symbol.upper()

    try:
        agent = StockPlanningAgent(symbol, user_id)
        result = await agent.evaluate_plan()

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=result["error"]
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Evaluation error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Evaluation failed: {str(e)}"
        )


@app.delete(
    "/chat/{symbol}/conversation",
    summary="Clear conversation history",
    description="Clear the chat history for a stock.",
)
async def clear_conversation(symbol: str, user: User = Depends(get_current_user)):
    """Clear conversation history for a stock."""
    from app.agent.planning_agent import StockPlanningAgent

    user_id = user.id
    symbol = symbol.upper()

    try:
        agent = StockPlanningAgent(symbol, user_id)
        cleared = await agent.clear_conversation()

        return {"success": cleared, "symbol": symbol}

    except Exception as e:
        logger.error(f"Clear conversation error for {symbol}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear conversation: {str(e)}"
        )


# ============================================================================
# Interactive Plan Session Endpoints (Claude Code-style planning)
# ============================================================================


class PlanSessionResponse(BaseModel):
    """Response for a plan session."""
    session_id: str
    status: str  # generating, draft, refining, approved, rejected
    symbol: str
    draft_plan: Optional[PlanResponse] = None
    messages: List[dict] = []
    revision_count: int = 0
    created_at: str
    updated_at: str


class PlanFeedbackRequest(BaseModel):
    """Request to submit feedback on a draft plan."""
    feedback_type: str  # "question" or "adjust"
    content: str


class PlanFeedbackResponse(BaseModel):
    """Response after processing feedback."""
    ai_response: str
    updated_plan: Optional[PlanResponse] = None
    options: List[dict] = []  # For adjustment options
    session_status: str


@app.post(
    "/plan/{symbol}/session",
    response_model=PlanSessionResponse,
    summary="Start interactive planning session",
    description="Start a new interactive planning session. AI generates a draft plan for user review.",
)
async def start_plan_session(
    symbol: str,
    background_tasks: BackgroundTasks,
    user: User = Depends(get_current_user)
):
    """Start a new interactive planning session."""
    from app.storage.plan_session_store import get_plan_session_store
    from app.agent.planning_agent import StockPlanningAgent

    user_id = user.id
    symbol = symbol.upper()

    try:
        session_store = get_plan_session_store()

        # Check for existing active session
        existing = await session_store.get_active_session(user_id, symbol)
        if existing and existing.status in ("draft", "refining"):
            # Return existing session
            draft_plan = None
            if existing.draft_plan_data:
                draft_plan = PlanResponse(**existing.draft_plan_data)

            return PlanSessionResponse(
                session_id=existing.id,
                status=existing.status,
                symbol=symbol,
                draft_plan=draft_plan,
                messages=[m.to_dict() for m in existing.messages],
                revision_count=existing.revision_count,
                created_at=existing.created_at,
                updated_at=existing.updated_at,
            )

        # Create new session
        session = await session_store.create_session(user_id, symbol)

        # Generate plan in background and update session
        async def generate_plan_for_session():
            try:
                agent = StockPlanningAgent(symbol, user_id)
                plan = await agent.generate_plan(force_new=True)

                # Convert plan to dict for storage
                plan_dict = {
                    "symbol": plan.symbol,
                    "bias": plan.bias,
                    "thesis": plan.thesis,
                    "entry_zone_low": plan.entry_zone_low,
                    "entry_zone_high": plan.entry_zone_high,
                    "stop_loss": plan.stop_loss,
                    "stop_reasoning": plan.stop_reasoning,
                    "target_1": plan.target_1,
                    "target_2": plan.target_2,
                    "target_3": plan.target_3,
                    "target_reasoning": plan.target_reasoning,
                    "risk_reward": plan.risk_reward,
                    "key_supports": plan.key_supports,
                    "key_resistances": plan.key_resistances,
                    "invalidation_criteria": plan.invalidation_criteria,
                    "technical_summary": plan.technical_summary,
                    "status": "draft",
                    "created_at": plan.created_at,
                    "last_evaluation": plan.last_evaluation,
                    "evaluation_notes": plan.evaluation_notes,
                    "trade_style": plan.trade_style,
                    "trade_style_reasoning": plan.trade_style_reasoning,
                    "holding_period": plan.holding_period,
                    "confidence": plan.confidence,
                    "news_summary": plan.news_summary,
                    "reddit_sentiment": plan.reddit_sentiment,
                    "reddit_buzz": plan.reddit_buzz,
                }

                await session_store.set_draft_plan(session.id, plan_dict)

                # Add system message about draft plan
                await session_store.add_message(
                    session.id,
                    role="assistant",
                    content=f"I've analyzed {symbol} and created a draft trading plan. Review the plan above and let me know if you'd like to adjust anything or have questions about my analysis.",
                    message_type="info"
                )

            except Exception as e:
                logger.error(f"Error generating plan for session {session.id}: {e}")
                await session_store.add_message(
                    session.id,
                    role="system",
                    content=f"Error generating plan: {str(e)}",
                    message_type="info"
                )

        # Run in background
        import asyncio
        asyncio.create_task(generate_plan_for_session())

        return PlanSessionResponse(
            session_id=session.id,
            status=session.status,
            symbol=symbol,
            draft_plan=None,
            messages=[],
            revision_count=0,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    except Exception as e:
        logger.error(f"Error starting plan session for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start planning session: {str(e)}"
        )


@app.get(
    "/plan/{symbol}/session/{session_id}",
    response_model=PlanSessionResponse,
    summary="Get plan session state",
    description="Get the current state of a planning session.",
)
async def get_plan_session(
    symbol: str,
    session_id: str,
    user: User = Depends(get_current_user)
):
    """Get a planning session's current state."""
    from app.storage.plan_session_store import get_plan_session_store

    try:
        session_store = get_plan_session_store()
        session = await session_store.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        if session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )

        draft_plan = None
        if session.draft_plan_data:
            draft_plan = PlanResponse(**session.draft_plan_data)

        return PlanSessionResponse(
            session_id=session.id,
            status=session.status,
            symbol=session.symbol,
            draft_plan=draft_plan,
            messages=[m.to_dict() for m in session.messages],
            revision_count=session.revision_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting plan session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get session: {str(e)}"
        )


@app.post(
    "/plan/{symbol}/session/{session_id}/feedback",
    response_model=PlanFeedbackResponse,
    summary="Submit feedback on draft plan",
    description="Ask a question or request an adjustment to the draft plan.",
)
async def submit_plan_feedback(
    symbol: str,
    session_id: str,
    request: PlanFeedbackRequest,
    user: User = Depends(get_current_user)
):
    """Submit feedback (question or adjustment request) on a draft plan."""
    from app.storage.plan_session_store import get_plan_session_store
    from app.agent.planning_agent import StockPlanningAgent

    try:
        session_store = get_plan_session_store()
        session = await session_store.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        if session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )

        if session.status not in ("draft", "refining"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session is not in a state that accepts feedback"
            )

        # Add user's message
        await session_store.add_message(
            session_id,
            role="user",
            content=request.content,
            message_type=request.feedback_type
        )

        # Process with AI agent
        agent = StockPlanningAgent(symbol.upper(), user.id)

        if request.feedback_type == "question":
            # Handle question about the plan
            ai_response = await agent.handle_user_question(
                session.draft_plan_data,
                request.content,
                [m.to_dict() for m in session.messages]
            )

            await session_store.add_message(
                session_id,
                role="assistant",
                content=ai_response["response"],
                message_type="answer",
                options=ai_response.get("options", [])
            )

            return PlanFeedbackResponse(
                ai_response=ai_response["response"],
                updated_plan=None,
                options=ai_response.get("options", []),
                session_status=session.status
            )

        elif request.feedback_type == "adjust":
            # Handle adjustment request
            await session_store.increment_revision(session_id)

            ai_response = await agent.handle_adjustment_request(
                session.draft_plan_data,
                request.content,
                [m.to_dict() for m in session.messages]
            )

            await session_store.add_message(
                session_id,
                role="assistant",
                content=ai_response["response"],
                message_type="adjustment_response",
                options=ai_response.get("options", [])
            )

            # If AI suggests updated plan, update the draft
            updated_plan = None
            if ai_response.get("updated_plan"):
                await session_store.set_draft_plan(session_id, ai_response["updated_plan"])
                updated_plan = PlanResponse(**ai_response["updated_plan"])

            return PlanFeedbackResponse(
                ai_response=ai_response["response"],
                updated_plan=updated_plan,
                options=ai_response.get("options", []),
                session_status="refining"
            )

        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid feedback type. Use 'question' or 'adjust'."
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing feedback for session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process feedback: {str(e)}"
        )


@app.post(
    "/plan/{symbol}/session/{session_id}/approve",
    response_model=PlanResponse,
    summary="Approve draft plan",
    description="Approve the draft plan and make it the active trading plan.",
)
async def approve_plan_session(
    symbol: str,
    session_id: str,
    user: User = Depends(get_current_user)
):
    """Approve the draft plan and finalize it."""
    from app.storage.plan_session_store import get_plan_session_store
    from app.storage.plan_store import get_plan_store, TradingPlan
    import uuid

    try:
        session_store = get_plan_session_store()
        session = await session_store.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        if session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )

        if not session.draft_plan_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No draft plan to approve"
            )

        # Create the final plan from draft
        plan_store = get_plan_store()
        plan_id = str(uuid.uuid4())

        final_plan = TradingPlan(
            id=plan_id,
            user_id=user.id,
            symbol=symbol.upper(),
            status="active",
            bias=session.draft_plan_data.get("bias", ""),
            thesis=session.draft_plan_data.get("thesis", ""),
            entry_zone_low=session.draft_plan_data.get("entry_zone_low"),
            entry_zone_high=session.draft_plan_data.get("entry_zone_high"),
            stop_loss=session.draft_plan_data.get("stop_loss"),
            stop_reasoning=session.draft_plan_data.get("stop_reasoning", ""),
            target_1=session.draft_plan_data.get("target_1"),
            target_2=session.draft_plan_data.get("target_2"),
            target_3=session.draft_plan_data.get("target_3"),
            target_reasoning=session.draft_plan_data.get("target_reasoning", ""),
            risk_reward=session.draft_plan_data.get("risk_reward"),
            key_supports=session.draft_plan_data.get("key_supports", []),
            key_resistances=session.draft_plan_data.get("key_resistances", []),
            invalidation_criteria=session.draft_plan_data.get("invalidation_criteria", ""),
            trade_style=session.draft_plan_data.get("trade_style", ""),
            trade_style_reasoning=session.draft_plan_data.get("trade_style_reasoning", ""),
            holding_period=session.draft_plan_data.get("holding_period", ""),
            confidence=session.draft_plan_data.get("confidence", 0),
            technical_summary=session.draft_plan_data.get("technical_summary", ""),
            news_summary=session.draft_plan_data.get("news_summary", ""),
            reddit_sentiment=session.draft_plan_data.get("reddit_sentiment", ""),
            reddit_buzz=session.draft_plan_data.get("reddit_buzz", ""),
        )

        await plan_store.save_plan(final_plan)

        # Mark session as approved
        await session_store.approve_session(session_id, plan_id)

        # Add approval message
        await session_store.add_message(
            session_id,
            role="system",
            content="Plan approved and now active. I'll monitor the stock and alert you when price approaches key levels.",
            message_type="approval"
        )

        return PlanResponse(
            symbol=final_plan.symbol,
            bias=final_plan.bias,
            thesis=final_plan.thesis,
            entry_zone_low=final_plan.entry_zone_low,
            entry_zone_high=final_plan.entry_zone_high,
            stop_loss=final_plan.stop_loss,
            stop_reasoning=final_plan.stop_reasoning,
            target_1=final_plan.target_1,
            target_2=final_plan.target_2,
            target_3=final_plan.target_3,
            target_reasoning=final_plan.target_reasoning,
            risk_reward=final_plan.risk_reward,
            key_supports=final_plan.key_supports,
            key_resistances=final_plan.key_resistances,
            invalidation_criteria=final_plan.invalidation_criteria,
            technical_summary=final_plan.technical_summary,
            status=final_plan.status,
            created_at=final_plan.created_at,
            last_evaluation=final_plan.last_evaluation,
            evaluation_notes=final_plan.evaluation_notes,
            trade_style=final_plan.trade_style,
            trade_style_reasoning=final_plan.trade_style_reasoning,
            holding_period=final_plan.holding_period,
            confidence=final_plan.confidence,
            news_summary=final_plan.news_summary,
            reddit_sentiment=final_plan.reddit_sentiment,
            reddit_buzz=final_plan.reddit_buzz,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error approving plan session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to approve plan: {str(e)}"
        )


@app.get(
    "/plan/{symbol}/session/{session_id}/history",
    summary="Get session conversation history",
    description="Get the full conversation history for a planning session.",
)
async def get_session_history(
    symbol: str,
    session_id: str,
    user: User = Depends(get_current_user)
):
    """Get the conversation history for a planning session."""
    from app.storage.plan_session_store import get_plan_session_store

    try:
        session_store = get_plan_session_store()
        session = await session_store.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        if session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )

        return {
            "session_id": session.id,
            "symbol": session.symbol,
            "status": session.status,
            "messages": [m.to_dict() for m in session.messages],
            "revision_count": session.revision_count,
            "created_at": session.created_at,
            "approved_at": session.approved_at,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session history {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get history: {str(e)}"
        )


@app.post(
    "/plan/{symbol}/session/from-existing",
    response_model=PlanSessionResponse,
    summary="Start session from existing plan",
    description="Start a new planning session using an existing approved plan as the draft. Allows modifying existing plans.",
)
async def start_session_from_existing(
    symbol: str,
    user: User = Depends(get_current_user)
):
    """Start a planning session from an existing approved plan."""
    from app.storage.plan_session_store import get_plan_session_store
    from app.storage.plan_store import get_plan_store

    user_id = user.id
    symbol = symbol.upper()

    try:
        plan_store = get_plan_store()
        session_store = get_plan_session_store()

        # Get the existing plan
        existing_plan = await plan_store.get_plan(user_id, symbol)
        if not existing_plan:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No existing plan found for {symbol}"
            )

        # Check for existing active session
        existing_session = await session_store.get_active_session(user_id, symbol)
        if existing_session and existing_session.status in ("draft", "refining"):
            # Return existing session
            draft_plan = None
            if existing_session.draft_plan_data:
                draft_plan = PlanResponse(**existing_session.draft_plan_data)

            return PlanSessionResponse(
                session_id=existing_session.id,
                status=existing_session.status,
                symbol=symbol,
                draft_plan=draft_plan,
                messages=[m.to_dict() for m in existing_session.messages],
                revision_count=existing_session.revision_count,
                created_at=existing_session.created_at,
                updated_at=existing_session.updated_at,
            )

        # Create new session with existing plan as draft
        session = await session_store.create_session(user_id, symbol)

        # Convert existing plan to draft format
        plan_dict = {
            "symbol": existing_plan.symbol,
            "bias": existing_plan.bias,
            "thesis": existing_plan.thesis,
            "entry_zone_low": existing_plan.entry_zone_low,
            "entry_zone_high": existing_plan.entry_zone_high,
            "stop_loss": existing_plan.stop_loss,
            "stop_reasoning": existing_plan.stop_reasoning,
            "target_1": existing_plan.target_1,
            "target_2": existing_plan.target_2,
            "target_3": existing_plan.target_3,
            "target_reasoning": existing_plan.target_reasoning,
            "risk_reward": existing_plan.risk_reward,
            "key_supports": existing_plan.key_supports,
            "key_resistances": existing_plan.key_resistances,
            "invalidation_criteria": existing_plan.invalidation_criteria,
            "technical_summary": existing_plan.technical_summary,
            "status": "draft",
            "created_at": existing_plan.created_at,
            "last_evaluation": existing_plan.last_evaluation,
            "evaluation_notes": existing_plan.evaluation_notes,
            "trade_style": existing_plan.trade_style,
            "trade_style_reasoning": existing_plan.trade_style_reasoning,
            "holding_period": existing_plan.holding_period,
            "confidence": existing_plan.confidence,
            "news_summary": existing_plan.news_summary,
            "reddit_sentiment": existing_plan.reddit_sentiment,
            "reddit_buzz": existing_plan.reddit_buzz,
        }

        # Set the draft plan immediately (no generation needed)
        await session_store.set_draft_plan(session.id, plan_dict)

        # Add info message
        await session_store.add_message(
            session.id,
            role="assistant",
            content=f"I've loaded your existing {symbol} plan for editing. What would you like to adjust? You can ask questions about the plan or request specific changes to levels, targets, or stop loss.",
            message_type="info"
        )

        # Refresh session to get updated data
        session = await session_store.get_session(session.id)

        return PlanSessionResponse(
            session_id=session.id,
            status=session.status,
            symbol=symbol,
            draft_plan=PlanResponse(**plan_dict),
            messages=[m.to_dict() for m in session.messages],
            revision_count=session.revision_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting session from existing plan for {symbol}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start session: {str(e)}"
        )


@app.post(
    "/plan/{symbol}/session/{session_id}/reopen",
    response_model=PlanSessionResponse,
    summary="Reopen approved session",
    description="Reopen an approved planning session to continue making adjustments.",
)
async def reopen_plan_session(
    symbol: str,
    session_id: str,
    user: User = Depends(get_current_user)
):
    """Reopen an approved session to continue adjustments."""
    from app.storage.plan_session_store import get_plan_session_store
    from app.storage.plan_store import get_plan_store

    try:
        session_store = get_plan_session_store()
        session = await session_store.get_session(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )

        if session.user_id != user.id:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized to access this session"
            )

        if session.status != "approved":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Session is not approved (status: {session.status}). Only approved sessions can be reopened."
            )

        # If we have an approved_plan_id, load the current plan data
        if session.approved_plan_id:
            plan_store = get_plan_store()
            current_plan = await plan_store.get_plan(user.id, symbol.upper())

            if current_plan:
                # Update draft with current plan data
                plan_dict = {
                    "symbol": current_plan.symbol,
                    "bias": current_plan.bias,
                    "thesis": current_plan.thesis,
                    "entry_zone_low": current_plan.entry_zone_low,
                    "entry_zone_high": current_plan.entry_zone_high,
                    "stop_loss": current_plan.stop_loss,
                    "stop_reasoning": current_plan.stop_reasoning,
                    "target_1": current_plan.target_1,
                    "target_2": current_plan.target_2,
                    "target_3": current_plan.target_3,
                    "target_reasoning": current_plan.target_reasoning,
                    "risk_reward": current_plan.risk_reward,
                    "key_supports": current_plan.key_supports,
                    "key_resistances": current_plan.key_resistances,
                    "invalidation_criteria": current_plan.invalidation_criteria,
                    "technical_summary": current_plan.technical_summary,
                    "status": "draft",
                    "created_at": current_plan.created_at,
                    "last_evaluation": current_plan.last_evaluation,
                    "evaluation_notes": current_plan.evaluation_notes,
                    "trade_style": current_plan.trade_style,
                    "trade_style_reasoning": current_plan.trade_style_reasoning,
                    "holding_period": current_plan.holding_period,
                    "confidence": current_plan.confidence,
                    "news_summary": current_plan.news_summary,
                    "reddit_sentiment": current_plan.reddit_sentiment,
                    "reddit_buzz": current_plan.reddit_buzz,
                }
                session.draft_plan_data = plan_dict

        # Reopen the session
        session.status = "refining"
        session.approved_at = None  # Clear approved timestamp
        await session_store.update_session(session)

        # Add message about reopening
        await session_store.add_message(
            session.id,
            role="assistant",
            content="Session reopened. You can continue adjusting the plan. What would you like to change?",
            message_type="info"
        )

        # Refresh session
        session = await session_store.get_session(session.id)

        draft_plan = None
        if session.draft_plan_data:
            draft_plan = PlanResponse(**session.draft_plan_data)

        return PlanSessionResponse(
            session_id=session.id,
            status=session.status,
            symbol=symbol.upper(),
            draft_plan=draft_plan,
            messages=[m.to_dict() for m in session.messages],
            revision_count=session.revision_count,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reopening session {session_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reopen session: {str(e)}"
        )


@app.post(
    "/chat",
    summary="Portfolio chat with AI agent",
    description="""
    Chat with the Portfolio Agent about your watchlist and positions.

    The Portfolio Agent can:
    - Summarize your entire portfolio with current prices and P&L
    - Analyze individual stocks (delegates to specialized stock subagents)
    - Provide market context and direction
    - Track positions across all watchlist stocks

    Conversation history is persisted on the server.

    Example questions:
    - "How is my portfolio doing?"
    - "Summarize AAPL for me"
    - "Which of my stocks are near support?"
    - "What's the market direction today?"
    """,
)
async def chat_portfolio(request: ChatRequest, user: User = Depends(get_current_user)):
    """Chat with the portfolio agent about your watchlist and positions."""
    from app.agent.portfolio_agent import get_portfolio_agent

    user_id = user.id
    try:
        agent = get_portfolio_agent(user_id)
        response_text = await agent.chat(request.message)

        # Get portfolio summary for context
        summary = await agent.get_summary()

        return {
            "response": response_text,
            "portfolio_summary": summary,
            "conversation_key": "portfolio",
        }

    except Exception as e:
        logger.error(f"Portfolio chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat failed: {str(e)}"
        )


@app.get(
    "/chat/history",
    summary="Get chat history",
    description="Get persisted chat history for the portfolio or a specific stock.",
)
async def get_chat_history(
    symbol: Optional[str] = None,
    user_id: str = "default",
    limit: int = 50,
):
    """Get chat history from server.

    Args:
        symbol: Stock symbol for stock-specific chat, or None for portfolio chat
        user_id: User identifier
        limit: Maximum number of messages to return

    Returns:
        Chat history with messages
    """
    from app.storage.conversation_store import get_conversation_store

    try:
        conversation_store = get_conversation_store()
        key = symbol.upper() if symbol else "portfolio"

        conversation = await conversation_store.get_conversation(user_id, key)

        return {
            "key": key,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                }
                for msg in conversation.messages[-limit:]
            ],
            "count": len(conversation.messages),
        }

    except Exception as e:
        logger.error(f"Get chat history error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get chat history: {str(e)}"
        )


@app.delete(
    "/chat/history",
    summary="Clear chat history",
    description="Clear chat history for the portfolio or a specific stock.",
)
async def clear_chat_history(
    symbol: Optional[str] = None,
    user_id: str = "default",
):
    """Clear chat history on server.

    Args:
        symbol: Stock symbol for stock-specific chat, or None for portfolio chat
        user_id: User identifier

    Returns:
        Success status
    """
    from app.storage.conversation_store import get_conversation_store

    try:
        conversation_store = get_conversation_store()
        key = symbol.upper() if symbol else "portfolio"

        cleared = await conversation_store.clear_conversation(user_id, key)

        return {"success": cleared, "key": key}

    except Exception as e:
        logger.error(f"Clear chat history error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear chat history: {str(e)}"
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
