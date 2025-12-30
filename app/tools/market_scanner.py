"""Market scanner tools for top-down analysis.

This module provides market-wide and sector analysis capabilities for professional
top-down stock selection:

1. Market Overview: S&P 500, Nasdaq, Dow Jones health
2. Sector Performance: 11 SPDR sectors ranked by performance
3. Sector Leaders: Top stocks within strong sectors
4. Market Breadth: Advance/decline and market health metrics

Top-Down Trading Workflow:
1. Check market_overview() - Is the market bullish or bearish?
2. Check sector_performance() - Which sectors are leading?
3. Use find_sector_leaders() - Which stocks are strongest in leading sectors?
4. Run full analysis on selected stocks

This approach aligns your trades with the broader market trend.
"""

from typing import List, Dict, Literal, Optional
from datetime import datetime, timedelta
import logging

from app.tools.market_data import fetch_price_bars, fetch_snapshots
from app.tools.indicators import calculate_ema, calculate_rsi, analyze_volume
from app.tools.analysis import run_analysis

logger = logging.getLogger(__name__)

# Market indices (using ETFs as proxies)
MARKET_INDICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000 (Small Caps)",
}

# SPDR Sector ETFs (11 standard sectors)
SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLC": "Communication Services",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLE": "Energy",
    "XLU": "Utilities",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLRE": "Real Estate",
}

# Popular stocks by sector (for screening)
SECTOR_STOCKS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CSCO", "ADBE", "CRM", "AMD", "INTC"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "XLV": ["UNH", "JNJ", "LLY", "ABBV", "MRK", "TMO", "ABT", "DHR", "PFE", "BMY"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "T", "VZ", "TMUS", "EA"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "SBUX", "TJX", "LOW", "BKNG", "CMG"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "CL", "MDLZ", "KMB"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "ES"],
    "XLI": ["CAT", "GE", "UNP", "RTX", "HON", "UPS", "BA", "LMT", "DE", "MMM"],
    "XLB": ["LIN", "APD", "ECL", "SHW", "DD", "NEM", "FCX", "VMC", "NUE", "MLM"],
    "XLRE": ["AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
}


def get_market_overview(days_back: int = 30) -> Dict:
    """Get overview of major market indices health.

    Analyzes the S&P 500, Nasdaq, Dow Jones, and Russell 2000 to determine
    overall market direction and strength. This is the first step in top-down
    analysis.

    Args:
        days_back: Number of days of historical data (default: 30)

    Returns:
        Dictionary containing:
        - indices: List of index data (symbol, name, price, change %, signal)
        - market_signal: Overall market sentiment ("bullish", "bearish", "neutral")
        - bullish_count: Number of bullish indices
        - bearish_count: Number of bearish indices
        - summary: Text summary of market health

    Example:
        >>> overview = get_market_overview()
        >>> print(overview['market_signal'])  # "bullish"
        >>> print(overview['summary'])  # "3/4 indices bullish - strong market"
        >>> for index in overview['indices']:
        >>>     print(f"{index['name']}: {index['signal']} ({index['change_pct']}%)")
    """
    logger.info(f"Analyzing market overview ({len(MARKET_INDICES)} indices)")

    indices_data = []
    bullish_count = 0
    bearish_count = 0

    for symbol, name in MARKET_INDICES.items():
        try:
            # Fetch data
            bars = fetch_price_bars(symbol, timeframe="1d", days_back=days_back)

            if not bars or len(bars) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            current_price = bars[-1].close
            open_price = bars[0].open
            change_pct = ((current_price - open_price) / open_price) * 100

            # Calculate indicators
            ema_20 = calculate_ema(bars, period=20)
            ema_50 = calculate_ema(bars, period=50) if len(bars) >= 50 else None
            rsi = calculate_rsi(bars, period=14)
            volume = analyze_volume(bars)

            # Determine signal
            signal_factors = 0
            max_factors = 0

            # Factor 1: Price above EMA 20
            max_factors += 1
            if ema_20.signal == "bullish":
                signal_factors += 1

            # Factor 2: Price above EMA 50
            if ema_50:
                max_factors += 1
                if ema_50.signal == "bullish":
                    signal_factors += 1

            # Factor 3: RSI bullish (40-70)
            max_factors += 1
            if 40 <= rsi.value <= 70:
                signal_factors += 1

            # Factor 4: Volume confirmation
            max_factors += 1
            if volume.signal == "bullish":
                signal_factors += 1

            # Determine signal
            signal_pct = (signal_factors / max_factors) * 100
            if signal_pct >= 60:
                signal = "bullish"
                bullish_count += 1
            elif signal_pct <= 40:
                signal = "bearish"
                bearish_count += 1
            else:
                signal = "neutral"

            indices_data.append({
                "symbol": symbol,
                "name": name,
                "price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "signal": signal,
                "signal_strength": round(signal_pct, 0),
                "rsi": round(rsi.value, 1),
                "above_ema20": ema_20.signal == "bullish",
                "above_ema50": ema_50.signal == "bullish" if ema_50 else None,
            })

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue

    # Determine overall market signal
    total_indices = len(indices_data)
    if bullish_count >= total_indices * 0.6:
        market_signal = "bullish"
        summary = f"{bullish_count}/{total_indices} indices bullish - strong market"
    elif bearish_count >= total_indices * 0.6:
        market_signal = "bearish"
        summary = f"{bearish_count}/{total_indices} indices bearish - weak market"
    else:
        market_signal = "neutral"
        summary = f"Mixed signals - {bullish_count} bullish, {bearish_count} bearish"

    logger.info(f"Market Overview: {market_signal} - {summary}")

    return {
        "indices": indices_data,
        "market_signal": market_signal,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": total_indices - bullish_count - bearish_count,
        "total_indices": total_indices,
        "summary": summary,
        "timestamp": datetime.utcnow(),
    }


def get_sector_performance(
    days_back: int = 30,
    sort_by: Literal["performance", "strength", "volume"] = "performance"
) -> Dict:
    """Analyze performance of all 11 SPDR sectors.

    Ranks sectors by performance, strength, and volume to identify sector rotation
    and leadership. This is the second step in top-down analysis.

    Args:
        days_back: Number of days to analyze (default: 30)
        sort_by: Sort criterion - "performance" (% change), "strength" (signal),
                 or "volume" (relative volume)

    Returns:
        Dictionary containing:
        - sectors: List of sector data sorted by criterion
        - leading_sectors: Top 3 sectors
        - lagging_sectors: Bottom 3 sectors
        - rotation_signal: Sector rotation insight

    Example:
        >>> sectors = get_sector_performance(days_back=30)
        >>> print(f"Leading: {sectors['leading_sectors']}")  # ["Technology", "Healthcare"]
        >>> for sector in sectors['sectors'][:3]:
        >>>     print(f"{sector['name']}: {sector['change_pct']}% ({sector['signal']})")
    """
    logger.info(f"Analyzing {len(SECTOR_ETFS)} sectors (last {days_back} days)")

    sectors_data = []

    for symbol, name in SECTOR_ETFS.items():
        try:
            # Fetch data
            bars = fetch_price_bars(symbol, timeframe="1d", days_back=days_back)

            if not bars or len(bars) < 20:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            current_price = bars[-1].close
            start_price = bars[0].open
            change_pct = ((current_price - start_price) / start_price) * 100

            # Calculate indicators
            ema_20 = calculate_ema(bars, period=20)
            rsi = calculate_rsi(bars, period=14)
            volume = analyze_volume(bars)

            # Determine signal strength (0-100)
            signal_factors = 0
            if ema_20.signal == "bullish":
                signal_factors += 33
            if 40 <= rsi.value <= 70:
                signal_factors += 33
            if volume.signal == "bullish":
                signal_factors += 34

            # Classify signal
            if signal_factors >= 67:
                signal = "bullish"
            elif signal_factors <= 33:
                signal = "bearish"
            else:
                signal = "neutral"

            sectors_data.append({
                "symbol": symbol,
                "name": name,
                "price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "signal": signal,
                "signal_strength": signal_factors,
                "rsi": round(rsi.value, 1),
                "relative_volume": round(volume.metadata.get("relative_volume", 1.0), 2),
                "above_ema20": ema_20.signal == "bullish",
            })

        except Exception as e:
            logger.error(f"Error analyzing sector {symbol}: {e}")
            continue

    # Sort by chosen criterion
    if sort_by == "performance":
        sectors_data.sort(key=lambda x: x["change_pct"], reverse=True)
    elif sort_by == "strength":
        sectors_data.sort(key=lambda x: x["signal_strength"], reverse=True)
    elif sort_by == "volume":
        sectors_data.sort(key=lambda x: x["relative_volume"], reverse=True)

    # Identify leaders and laggards
    leading_sectors = [s["name"] for s in sectors_data[:3]]
    lagging_sectors = [s["name"] for s in sectors_data[-3:]]

    # Analyze rotation
    bullish_sectors = [s for s in sectors_data if s["signal"] == "bullish"]
    defensive_sectors = ["Utilities", "Consumer Staples", "Healthcare"]
    cyclical_sectors = ["Technology", "Consumer Discretionary", "Financials", "Industrials"]

    defensive_bullish = sum(1 for s in bullish_sectors if s["name"] in defensive_sectors)
    cyclical_bullish = sum(1 for s in bullish_sectors if s["name"] in cyclical_sectors)

    if cyclical_bullish >= 3:
        rotation_signal = "risk_on"  # Cyclicals leading = bullish market
    elif defensive_bullish >= 2:
        rotation_signal = "risk_off"  # Defensives leading = bearish market
    else:
        rotation_signal = "neutral"

    logger.info(
        f"Sector Analysis: Leaders={leading_sectors}, Rotation={rotation_signal}"
    )

    return {
        "sectors": sectors_data,
        "leading_sectors": leading_sectors,
        "lagging_sectors": lagging_sectors,
        "rotation_signal": rotation_signal,
        "bullish_sectors_count": len(bullish_sectors),
        "total_sectors": len(sectors_data),
        "sort_by": sort_by,
        "timestamp": datetime.utcnow(),
    }


def find_sector_leaders(
    sector_symbol: str,
    min_score: int = 65,
    max_results: int = 5,
) -> Dict:
    """Find top stocks within a specific sector.

    Uses the full StockMate analysis to identify the strongest stocks within
    a given sector. This is the third step in top-down analysis.

    Args:
        sector_symbol: Sector ETF symbol (e.g., "XLK" for Technology)
        min_score: Minimum analysis score to include (default: 65 = BUY threshold)
        max_results: Maximum number of stocks to return (default: 5)

    Returns:
        Dictionary containing:
        - sector_name: Sector name
        - sector_etf: Sector ETF symbol
        - leaders: List of top stocks with full analysis
        - stocks_analyzed: Total stocks analyzed
        - stocks_above_threshold: Stocks meeting min_score

    Example:
        >>> leaders = find_sector_leaders("XLK", min_score=70)
        >>> print(f"Top tech stocks: {leaders['sector_name']}")
        >>> for stock in leaders['leaders']:
        >>>     print(f"{stock['symbol']}: {stock['score']}% - {stock['recommendation']}")
    """
    if sector_symbol not in SECTOR_ETFS:
        raise ValueError(
            f"Invalid sector symbol: {sector_symbol}. "
            f"Valid sectors: {list(SECTOR_ETFS.keys())}"
        )

    sector_name = SECTOR_ETFS[sector_symbol]
    stocks = SECTOR_STOCKS.get(sector_symbol, [])

    logger.info(
        f"Analyzing {len(stocks)} stocks in {sector_name} sector "
        f"(min score: {min_score}%)"
    )

    analyzed_stocks = []

    for symbol in stocks:
        try:
            # Run full StockMate analysis
            analysis = run_analysis(symbol, account_size=10000)

            # Extract reasons from reasoning string (pipe-separated)
            reasons = []
            if analysis.reasoning:
                reasons = [r.strip() for r in analysis.reasoning.split(" | ")][:3]

            # Get current price from analysis (now includes current_price field)
            current_price = analysis.current_price or 0.0

            analyzed_stocks.append({
                "symbol": symbol,
                "score": analysis.confidence,
                "recommendation": analysis.recommendation,
                "current_price": current_price,
                "reasons": reasons,
                "trade_plan": {
                    "entry": analysis.trade_plan.entry_price if analysis.trade_plan else None,
                    "stop": analysis.trade_plan.stop_loss if analysis.trade_plan else None,
                    "target": analysis.trade_plan.target_1 if analysis.trade_plan else None,
                } if analysis.trade_plan else None,
            })

        except Exception as e:
            logger.warning(f"Could not analyze {symbol}: {e}")
            continue

    # Filter by min score and sort
    leaders = [s for s in analyzed_stocks if s["score"] >= min_score]
    leaders.sort(key=lambda x: x["score"], reverse=True)

    # Limit results
    leaders = leaders[:max_results]

    logger.info(
        f"Found {len(leaders)} leaders in {sector_name} "
        f"({len(analyzed_stocks)} analyzed)"
    )

    return {
        "sector_name": sector_name,
        "sector_etf": sector_symbol,
        "leaders": leaders,
        "stocks_analyzed": len(analyzed_stocks),
        "stocks_above_threshold": len([s for s in analyzed_stocks if s["score"] >= min_score]),
        "average_score": round(sum(s["score"] for s in analyzed_stocks) / len(analyzed_stocks), 1) if analyzed_stocks else 0,
        "timestamp": datetime.utcnow(),
    }


def run_market_scan(
    min_sector_change: float = 0.0,
    min_stock_score: int = 65,
    top_sectors: int = 3,
    stocks_per_sector: int = 3,
) -> Dict:
    """Complete top-down market scan: Market → Sectors → Stocks.

    This is the comprehensive workflow that professional traders use:
    1. Check market health (bullish/bearish?)
    2. Find leading sectors
    3. Find best stocks in leading sectors

    Args:
        min_sector_change: Minimum sector performance % (default: 0.0)
        min_stock_score: Minimum stock analysis score (default: 65)
        top_sectors: Number of top sectors to scan (default: 3)
        stocks_per_sector: Stocks to return per sector (default: 3)

    Returns:
        Complete market scan with market overview, sector rankings, and top stocks

    Example:
        >>> scan = run_market_scan(min_stock_score=70, top_sectors=2)
        >>> print(f"Market: {scan['market']['market_signal']}")
        >>> print(f"Leading sectors: {scan['sectors']['leading_sectors']}")
        >>> for sector_stocks in scan['top_stocks']:
        >>>     print(f"\n{sector_stocks['sector_name']}:")
        >>>     for stock in sector_stocks['leaders']:
        >>>         print(f"  {stock['symbol']}: {stock['score']}%")
    """
    logger.info("Running complete top-down market scan")

    # Step 1: Market overview
    market = get_market_overview(days_back=30)

    # Step 2: Sector performance
    sectors = get_sector_performance(days_back=30, sort_by="performance")

    # Step 3: Find leaders in top sectors
    top_stocks = []
    leading_sectors = [
        s for s in sectors["sectors"][:top_sectors]
        if s["change_pct"] >= min_sector_change
    ]

    for sector in leading_sectors:
        sector_leaders = find_sector_leaders(
            sector["symbol"],
            min_score=min_stock_score,
            max_results=stocks_per_sector,
        )
        if sector_leaders["leaders"]:
            top_stocks.append(sector_leaders)

    logger.info(
        f"Market scan complete: {market['market_signal']} market, "
        f"{len(top_stocks)} sectors with leaders"
    )

    return {
        "market": market,
        "sectors": sectors,
        "top_stocks": top_stocks,
        "scan_parameters": {
            "min_sector_change": min_sector_change,
            "min_stock_score": min_stock_score,
            "top_sectors": top_sectors,
            "stocks_per_sector": stocks_per_sector,
        },
        "timestamp": datetime.utcnow(),
    }


def get_quick_market_status() -> Dict:
    """Get quick market status using real-time snapshots.

    This is a fast version of market overview that uses the batch snapshots API
    to get real-time prices for all market indices in a single API call.
    It provides instant price data but without full indicator analysis.

    With AlgoTrader Plus, this provides real-time data from the SIP feed.

    Returns:
        Dictionary containing:
        - indices: Real-time price data for major indices
        - market_direction: Quick assessment (up/down/mixed)
        - timestamp: Current timestamp

    Example:
        >>> status = get_quick_market_status()
        >>> print(f"Market: {status['market_direction']}")
        >>> for idx in status['indices']:
        >>>     print(f"{idx['name']}: ${idx['price']:.2f} ({idx['change_pct']:+.2f}%)")
    """
    logger.info("Fetching quick market status via snapshots")

    symbols = list(MARKET_INDICES.keys())

    try:
        snapshots = fetch_snapshots(symbols)
    except Exception as e:
        logger.error(f"Error fetching snapshots: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow()}

    indices_data = []
    up_count = 0
    down_count = 0

    for symbol, name in MARKET_INDICES.items():
        snapshot = snapshots.get(symbol)
        if not snapshot or not snapshot.get("daily_bar") or not snapshot.get("prev_daily_bar"):
            continue

        daily = snapshot["daily_bar"]
        prev = snapshot["prev_daily_bar"]

        current_price = daily["close"]
        prev_close = prev["close"]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        if change_pct > 0:
            up_count += 1
        elif change_pct < 0:
            down_count += 1

        # Get real-time quote data
        quote = snapshot.get("latest_quote")
        trade = snapshot.get("latest_trade")

        indices_data.append({
            "symbol": symbol,
            "name": name,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(daily["high"], 2),
            "day_low": round(daily["low"], 2),
            "day_volume": daily["volume"],
            "bid": quote["bid_price"] if quote else None,
            "ask": quote["ask_price"] if quote else None,
            "last_trade_price": trade["price"] if trade else None,
        })

    # Determine market direction
    if up_count > down_count:
        market_direction = "up"
    elif down_count > up_count:
        market_direction = "down"
    else:
        market_direction = "mixed"

    avg_change = sum(idx["change_pct"] for idx in indices_data) / len(indices_data) if indices_data else 0

    logger.info(f"Quick market status: {market_direction} (avg: {avg_change:+.2f}%)")

    # Build market status (lazy import to avoid circular dependency)
    from app.services.scheduler import is_market_open, get_next_market_open, get_market_close_time
    market_is_open = is_market_open()
    if market_is_open:
        close_time = get_market_close_time()
        next_event = close_time.isoformat() if close_time else None
        next_event_type = "close"
    else:
        next_event = get_next_market_open().isoformat()
        next_event_type = "open"

    return {
        "indices": indices_data,
        "market_direction": market_direction,
        "up_count": up_count,
        "down_count": down_count,
        "average_change_pct": round(avg_change, 2),
        "timestamp": datetime.utcnow(),
        "market_status": {
            "is_open": market_is_open,
            "next_event": next_event,
            "next_event_type": next_event_type,
        },
    }


def get_quick_sector_status() -> Dict:
    """Get quick sector status using real-time snapshots.

    This is a fast version of sector performance that uses the batch snapshots API
    to get real-time prices for all sector ETFs in a single API call.

    With AlgoTrader Plus, this provides real-time data from the SIP feed.

    Returns:
        Dictionary containing:
        - sectors: Real-time price data for all sectors
        - leading: Top 3 sectors by daily change
        - lagging: Bottom 3 sectors by daily change
        - timestamp: Current timestamp

    Example:
        >>> status = get_quick_sector_status()
        >>> print(f"Leading: {status['leading']}")
        >>> for sector in status['sectors'][:5]:
        >>>     print(f"{sector['name']}: {sector['change_pct']:+.2f}%")
    """
    logger.info("Fetching quick sector status via snapshots")

    symbols = list(SECTOR_ETFS.keys())

    try:
        snapshots = fetch_snapshots(symbols)
    except Exception as e:
        logger.error(f"Error fetching snapshots: {e}")
        return {"error": str(e), "timestamp": datetime.utcnow()}

    sectors_data = []

    for symbol, name in SECTOR_ETFS.items():
        snapshot = snapshots.get(symbol)
        if not snapshot or not snapshot.get("daily_bar") or not snapshot.get("prev_daily_bar"):
            continue

        daily = snapshot["daily_bar"]
        prev = snapshot["prev_daily_bar"]

        current_price = daily["close"]
        prev_close = prev["close"]
        change = current_price - prev_close
        change_pct = (change / prev_close) * 100

        quote = snapshot.get("latest_quote")

        sectors_data.append({
            "symbol": symbol,
            "name": name,
            "price": round(current_price, 2),
            "change": round(change, 2),
            "change_pct": round(change_pct, 2),
            "day_high": round(daily["high"], 2),
            "day_low": round(daily["low"], 2),
            "day_volume": daily["volume"],
            "bid": quote["bid_price"] if quote else None,
            "ask": quote["ask_price"] if quote else None,
        })

    # Sort by change percentage
    sectors_data.sort(key=lambda x: x["change_pct"], reverse=True)

    leading = [s["name"] for s in sectors_data[:3]] if sectors_data else []
    lagging = [s["name"] for s in sectors_data[-3:]] if sectors_data else []

    logger.info(f"Quick sector status: Leading={leading}")

    return {
        "sectors": sectors_data,
        "leading": leading,
        "lagging": lagging,
        "total_sectors": len(sectors_data),
        "timestamp": datetime.utcnow(),
    }
