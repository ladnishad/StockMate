"""Finnhub API client for fundamental and news data.

This module provides async functions to fetch data from the Finnhub API:
- Basic financials (valuation, growth, health metrics)
- Earnings calendar (upcoming earnings dates)
- Earnings history (historical surprises)
- Company news with sentiment analysis
"""

import httpx
import logging
import re
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta

from app.config import get_settings

logger = logging.getLogger(__name__)

FINNHUB_BASE_URL = "https://finnhub.io/api/v1"


def _get_finnhub_headers() -> Dict[str, str]:
    """Get headers with Finnhub API token."""
    settings = get_settings()
    if not settings.finnhub_api_key:
        raise ValueError("FINNHUB_API_KEY not configured in environment")
    return {"X-Finnhub-Token": settings.finnhub_api_key}


async def fetch_basic_financials(symbol: str) -> Dict[str, Any]:
    """Fetch comprehensive financial metrics from Finnhub.

    Endpoint: /stock/metric?symbol=X&metric=all

    Returns metrics including:
    - Valuation: P/E, P/B, P/S, PEG
    - Growth: EPS growth, revenue growth
    - Profitability: margins, ROE, ROA
    - Health: debt ratios, current ratio
    - Dividends: yield, payout ratio

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')

    Returns:
        Dictionary with financial metrics
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{FINNHUB_BASE_URL}/stock/metric",
                params={"symbol": symbol.upper(), "metric": "all"},
                headers=_get_finnhub_headers(),
            )
            response.raise_for_status()
            data = response.json()

            # Extract from 'metric' object
            metrics = data.get("metric", {})

            return {
                # Valuation
                "market_cap": metrics.get("marketCapitalization"),  # in millions
                "pe_ratio": metrics.get("peBasicExclExtraTTM") or metrics.get("peTTM"),
                "pe_forward": metrics.get("peNormalizedAnnual"),
                "pb_ratio": metrics.get("pbQuarterly") or metrics.get("pbAnnual"),
                "ps_ratio": metrics.get("psQuarterly") or metrics.get("psTTM"),
                "peg_ratio": metrics.get("pegRatio"),
                # Per-share
                "eps": metrics.get("epsTTM") or metrics.get("epsBasicExclExtraItemsTTM"),
                "eps_growth_qoq": metrics.get("epsGrowthQuarterlyYoy"),
                "eps_growth_yoy": metrics.get("epsGrowthTTMYoy"),
                "revenue_per_share": metrics.get("revenuePerShareTTM"),
                "book_value_per_share": metrics.get("bookValuePerShareQuarterly"),
                # Growth
                "revenue_growth_qoq": metrics.get("revenueGrowthQuarterlyYoy"),
                "revenue_growth_yoy": metrics.get("revenueGrowthTTMYoy"),
                "revenue_growth_3y": metrics.get("revenueGrowth3Y"),
                # Profitability
                "gross_margin": metrics.get("grossMarginTTM"),
                "operating_margin": metrics.get("operatingMarginTTM"),
                "net_margin": metrics.get("netProfitMarginTTM"),
                "roe": metrics.get("roeTTM"),
                "roa": metrics.get("roaTTM"),
                "roic": metrics.get("roicTTM"),
                # Health
                "debt_to_equity": metrics.get("totalDebt/totalEquityQuarterly"),
                "current_ratio": metrics.get("currentRatioQuarterly"),
                "quick_ratio": metrics.get("quickRatioQuarterly"),
                "interest_coverage": metrics.get("interestCoverageQuarterly"),
                # Dividends
                "dividend_yield": metrics.get("dividendYieldIndicatedAnnual"),
                "payout_ratio": metrics.get("payoutRatioTTM"),
                # Risk
                "beta": metrics.get("beta"),
                "52_week_high": metrics.get("52WeekHigh"),
                "52_week_low": metrics.get("52WeekLow"),
                "52_week_price_return": metrics.get("52WeekPriceReturnDaily"),
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"Finnhub API error for {symbol}: {e.response.status_code}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching Finnhub financials for {symbol}: {e}")
        return {}


async def fetch_earnings_calendar(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Fetch upcoming earnings dates for a symbol.

    Endpoint: /calendar/earnings?symbol=X

    Args:
        symbol: Stock ticker symbol
        from_date: Start date YYYY-MM-DD (default: today)
        to_date: End date YYYY-MM-DD (default: 90 days from now)

    Returns:
        Dictionary with next_earnings info or None if not found
    """
    try:
        if not from_date:
            from_date = datetime.now().strftime("%Y-%m-%d")
        if not to_date:
            to_date = (datetime.now() + timedelta(days=90)).strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{FINNHUB_BASE_URL}/calendar/earnings",
                params={
                    "symbol": symbol.upper(),
                    "from": from_date,
                    "to": to_date,
                },
                headers=_get_finnhub_headers(),
            )
            response.raise_for_status()
            data = response.json()

            earnings_list = data.get("earningsCalendar", [])

            if not earnings_list:
                return {"next_earnings": None}

            # Find the next upcoming earnings
            today = datetime.now().date()
            for earning in sorted(earnings_list, key=lambda x: x.get("date", "")):
                earn_date_str = earning.get("date")
                if earn_date_str:
                    try:
                        earn_date = datetime.strptime(earn_date_str, "%Y-%m-%d").date()
                        days_until = (earn_date - today).days

                        # Include yesterday for after-hours announcements
                        if days_until >= -1:
                            return {
                                "next_earnings": {
                                    "date": earn_date_str,
                                    "days_until": days_until,
                                    "eps_estimate": earning.get("epsEstimate"),
                                    "revenue_estimate": earning.get("revenueEstimate"),
                                    "hour": earning.get("hour"),  # bmo, amc, dmh
                                }
                            }
                    except ValueError:
                        continue

            return {"next_earnings": None}
    except httpx.HTTPStatusError as e:
        logger.error(f"Finnhub earnings calendar error for {symbol}: {e.response.status_code}")
        return {"next_earnings": None}
    except Exception as e:
        logger.error(f"Error fetching earnings calendar for {symbol}: {e}")
        return {"next_earnings": None}


async def fetch_earnings_history(symbol: str, limit: int = 4) -> Dict[str, Any]:
    """Fetch historical earnings surprises.

    Endpoint: /stock/earnings?symbol=X

    Args:
        symbol: Stock ticker symbol
        limit: Number of recent quarters to analyze (default: 4)

    Returns:
        Dictionary with:
        - last_earnings: Most recent earnings result
        - earnings_beat_rate: Percentage of quarters that beat estimates
        - avg_earnings_surprise: Average surprise percentage
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                f"{FINNHUB_BASE_URL}/stock/earnings",
                params={"symbol": symbol.upper()},
                headers=_get_finnhub_headers(),
            )
            response.raise_for_status()
            earnings = response.json()

            if not earnings:
                return {}

            # Take last N quarters
            recent = earnings[:limit] if len(earnings) >= limit else earnings

            # Calculate beat rate and average surprise
            beats = 0
            surprises = []
            last_earning = None

            for e in recent:
                actual = e.get("actual")
                estimate = e.get("estimate")
                surprise_pct = e.get("surprisePercent")

                if actual is not None and estimate is not None:
                    if actual > estimate:
                        beats += 1
                    if surprise_pct is not None:
                        surprises.append(surprise_pct)

                if last_earning is None and actual is not None:
                    last_earning = {
                        "period": e.get("period", "unknown"),
                        "actual": actual,
                        "estimate": estimate,
                        "surprise_percent": surprise_pct,
                    }

            return {
                "last_earnings": last_earning,
                "earnings_beat_rate": (beats / len(recent) * 100) if recent else None,
                "avg_earnings_surprise": (
                    sum(surprises) / len(surprises) if surprises else None
                ),
            }
    except httpx.HTTPStatusError as e:
        logger.error(f"Finnhub earnings history error for {symbol}: {e.response.status_code}")
        return {}
    except Exception as e:
        logger.error(f"Error fetching earnings history for {symbol}: {e}")
        return {}


# ============================================================================
# News API Functions
# ============================================================================

# Sentiment keywords for basic content analysis
BULLISH_KEYWORDS = [
    "beat", "beats", "exceeded", "exceeds", "surpass", "surpassed",
    "upgrade", "upgraded", "outperform", "buy", "bullish", "rally",
    "surge", "surges", "soar", "soars", "gain", "gains", "profit",
    "growth", "growing", "breakthrough", "record", "high", "strong",
    "positive", "optimistic", "confident", "accelerate", "momentum",
    "expansion", "innovative", "success", "successful", "win", "winning",
    "boost", "boosted", "raises", "raised", "upside", "opportunity",
]

BEARISH_KEYWORDS = [
    "miss", "missed", "misses", "below", "under", "downgrade", "downgraded",
    "sell", "bearish", "decline", "declines", "fall", "falls", "drop",
    "drops", "plunge", "plunges", "loss", "losses", "weak", "weakness",
    "concern", "concerns", "risk", "risks", "warning", "warns", "warned",
    "cut", "cuts", "layoff", "layoffs", "restructure", "struggling",
    "disappointing", "disappointed", "negative", "pessimistic", "fear",
    "crash", "slump", "downturn", "recession", "inflation", "debt",
    "lawsuit", "investigation", "probe", "scandal", "fraud", "downside",
]


def _calculate_sentiment(text: str) -> float:
    """Calculate sentiment score from text using keyword analysis.

    Args:
        text: Article text (headline + summary)

    Returns:
        Sentiment score from -1.0 (very bearish) to 1.0 (very bullish)
    """
    if not text:
        return 0.0

    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    bullish_count = sum(1 for word in words if word in BULLISH_KEYWORDS)
    bearish_count = sum(1 for word in words if word in BEARISH_KEYWORDS)

    total = bullish_count + bearish_count
    if total == 0:
        return 0.0

    # Score from -1 to 1
    score = (bullish_count - bearish_count) / total
    return round(score, 3)


def _get_sentiment_label(score: float) -> str:
    """Convert sentiment score to label."""
    if score > 0.2:
        return "bullish"
    elif score < -0.2:
        return "bearish"
    return "neutral"


async def fetch_company_news(
    symbol: str,
    days_back: int = 7,
    limit: int = 20,
) -> Dict[str, Any]:
    """Fetch company news from Finnhub.

    Endpoint: /company-news?symbol=X&from=YYYY-MM-DD&to=YYYY-MM-DD

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL')
        days_back: Number of days to look back (default: 7)
        limit: Maximum number of articles to return (default: 20)

    Returns:
        Dictionary containing:
        - articles: List of article dicts with headline, summary, url, etc.
        - overall_sentiment: Aggregated sentiment label
        - sentiment_score: Numeric sentiment (-1 to 1)
        - article_count: Number of articles found
        - has_breaking_news: Whether any articles are very recent (<2 hours)
    """
    try:
        to_date = datetime.now().strftime("%Y-%m-%d")
        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(
                f"{FINNHUB_BASE_URL}/company-news",
                params={
                    "symbol": symbol.upper(),
                    "from": from_date,
                    "to": to_date,
                },
                headers=_get_finnhub_headers(),
            )
            response.raise_for_status()
            data = response.json()

            if not data:
                return {
                    "articles": [],
                    "overall_sentiment": "neutral",
                    "sentiment_score": 0.0,
                    "article_count": 0,
                    "key_themes": [],
                    "has_breaking_news": False,
                }

            # Limit articles
            data = data[:limit]

            # Process articles
            articles = []
            sentiment_scores = []
            has_breaking = False
            now = datetime.now()

            for article in data:
                headline = article.get("headline", "")
                summary = article.get("summary", "")
                timestamp = article.get("datetime", 0)

                # Calculate sentiment for this article
                combined_text = f"{headline} {summary}"
                sentiment = _calculate_sentiment(combined_text)
                sentiment_scores.append(sentiment)

                # Check if breaking news (< 2 hours old)
                if timestamp:
                    try:
                        pub_date = datetime.fromtimestamp(timestamp)
                        hours_old = (now - pub_date).total_seconds() / 3600
                        if hours_old < 2:
                            has_breaking = True
                    except (ValueError, TypeError, OSError):
                        pass

                articles.append({
                    "title": headline,
                    "description": summary[:500] if summary else "",
                    "url": article.get("url", ""),
                    "source": article.get("source", ""),
                    "published_date": datetime.fromtimestamp(timestamp).isoformat() if timestamp else "",
                    "tickers": [symbol.upper()],
                    "tags": [article.get("category", "")] if article.get("category") else [],
                    "sentiment_score": sentiment,
                    "sentiment_label": _get_sentiment_label(sentiment),
                })

            # Calculate overall sentiment
            avg_sentiment = (
                sum(sentiment_scores) / len(sentiment_scores)
                if sentiment_scores
                else 0.0
            )

            # Extract categories as themes
            categories = [a.get("category", "") for a in data if a.get("category")]
            unique_categories = list(dict.fromkeys(categories))[:5]

            return {
                "articles": articles,
                "overall_sentiment": _get_sentiment_label(avg_sentiment),
                "sentiment_score": round(avg_sentiment, 3),
                "article_count": len(articles),
                "key_themes": unique_categories,
                "has_breaking_news": has_breaking,
            }

    except httpx.HTTPStatusError as e:
        logger.error(f"Finnhub news API error for {symbol}: {e.response.status_code}")
        return {
            "articles": [],
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "article_count": 0,
            "key_themes": [],
            "has_breaking_news": False,
            "error": f"API error: {e.response.status_code}",
        }
    except ValueError as e:
        logger.warning(f"Finnhub API key not configured: {e}")
        return {
            "articles": [],
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "article_count": 0,
            "key_themes": [],
            "has_breaking_news": False,
            "error": "FINNHUB_API_KEY not configured",
        }
    except Exception as e:
        logger.error(f"Error fetching Finnhub news for {symbol}: {e}")
        return {
            "articles": [],
            "overall_sentiment": "neutral",
            "sentiment_score": 0.0,
            "article_count": 0,
            "key_themes": [],
            "has_breaking_news": False,
            "error": str(e),
        }


async def fetch_news_for_trade_style(
    symbol: str,
    trade_style: str,
) -> Dict[str, Any]:
    """Fetch news with parameters optimized for trade style.

    Args:
        symbol: Stock ticker symbol
        trade_style: One of 'day', 'swing', 'position'

    Returns:
        News data with appropriate time window
    """
    # Different time windows based on trade style
    style_config = {
        "day": {"days_back": 2, "limit": 10},      # Recent news only
        "swing": {"days_back": 7, "limit": 15},    # Week of news
        "position": {"days_back": 30, "limit": 25},  # Month of news
    }

    config = style_config.get(trade_style, style_config["swing"])
    return await fetch_company_news(symbol, **config)
