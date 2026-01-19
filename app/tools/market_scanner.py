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
from app.tools.indicators import (
    calculate_ema,
    calculate_rsi,
    analyze_volume,
    calculate_macd,
    calculate_atr,
    calculate_adx,
    calculate_stochastic,
    detect_divergences,
)
from app.tools.analysis import run_analysis
from app.models.data import Indicator

logger = logging.getLogger(__name__)

# Market indices (using ETFs as proxies)
# Core equity indices + context indicators for comprehensive market analysis
MARKET_INDICES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "DIA": "Dow Jones",
    "IWM": "Russell 2000 (Small Caps)",
    "VXX": "VIX Volatility",     # Volatility regime detection
    "TLT": "20+ Year Treasury",  # Rate sensitivity / risk-off context
}

# Core equity indices (used for bullish/bearish counting)
CORE_INDICES = ["SPY", "QQQ", "DIA", "IWM"]

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


# =============================================================================
# HELPER FUNCTIONS FOR ENHANCED MARKET OVERVIEW
# =============================================================================


def _determine_volatility_regime(vxx_data: Optional[Dict]) -> Dict:
    """Determine volatility regime based on VXX analysis.

    Args:
        vxx_data: Analysis data for VXX containing rsi, change_pct

    Returns:
        Dictionary with volatility regime and related metrics
    """
    if not vxx_data:
        return {"regime": "unknown", "vxx_rsi": None, "interpretation": "VXX data unavailable"}

    rsi = vxx_data.get("rsi", 50)
    change_pct = vxx_data.get("change_pct", 0)

    if rsi > 70 or change_pct > 10:
        regime = "high_fear"
        interpretation = "Elevated volatility - defensive positioning recommended"
    elif rsi > 55 or change_pct > 5:
        regime = "elevated"
        interpretation = "Above-average volatility - use tighter stops"
    elif rsi < 30 or change_pct < -10:
        regime = "complacent"
        interpretation = "Low volatility - potential for volatility expansion"
    else:
        regime = "normal"
        interpretation = "Normal volatility conditions"

    return {
        "regime": regime,
        "vxx_rsi": round(rsi, 1) if rsi else None,
        "vxx_change_pct": round(change_pct, 2) if change_pct else None,
        "interpretation": interpretation,
    }


def _analyze_timeframe(
    symbol: str,
    timeframe: str,
    days_back: int,
) -> Optional[Dict]:
    """Analyze a single timeframe for an index with enhanced indicators.

    Args:
        symbol: Index symbol
        timeframe: "1d", "1h", or "15m"
        days_back: Number of days of data to fetch

    Returns:
        Dictionary with timeframe analysis or None on error
    """
    try:
        bars = fetch_price_bars(symbol, timeframe=timeframe, days_back=days_back)

        if not bars or len(bars) < 15:
            return None

        current_price = bars[-1].close

        # Core indicators
        ema_20 = calculate_ema(bars, period=20)
        rsi = calculate_rsi(bars, period=14)

        # Enhanced indicators (with safety checks for bar count)
        macd = None
        if len(bars) >= 35:  # Need 26 + 9 bars for MACD
            macd = calculate_macd(bars)

        adx = None
        if len(bars) >= 28:  # Need 14 * 2 bars for ADX
            adx = calculate_adx(bars)

        stochastic = None
        if len(bars) >= 20:  # Need 14 + 3 + 3 bars for Stochastic
            stochastic = calculate_stochastic(bars)

        # Calculate signal score (0-100)
        signal_score = 0
        max_score = 0

        # EMA signal (20 points)
        max_score += 20
        if ema_20.signal == "bullish":
            signal_score += 20
        elif ema_20.signal == "neutral":
            signal_score += 10

        # RSI signal (20 points)
        max_score += 20
        if 40 <= rsi.value <= 60:
            signal_score += 20  # Healthy momentum zone
        elif 30 <= rsi.value <= 70:
            signal_score += 15  # Acceptable zone
        elif rsi.value < 30:
            signal_score += 10  # Oversold (potential bounce)
        # Overbought gets 0

        # MACD signal (20 points)
        if macd:
            max_score += 20
            if macd.signal == "bullish":
                signal_score += 20
            elif macd.signal == "neutral":
                signal_score += 10

        # ADX trend strength (20 points)
        if adx:
            max_score += 20
            if adx.value >= 25 and adx.metadata.get("trend_direction") == "bullish":
                signal_score += 20  # Strong bullish trend
            elif adx.value >= 25 and adx.metadata.get("trend_direction") == "bearish":
                signal_score += 0   # Strong bearish trend
            elif adx.value >= 20:
                signal_score += 10  # Emerging trend
            else:
                signal_score += 5   # Weak/no trend

        # Stochastic signal (20 points)
        if stochastic:
            max_score += 20
            if stochastic.signal == "bullish" and not stochastic.metadata.get("is_overbought"):
                signal_score += 20
            elif stochastic.signal == "neutral":
                signal_score += 10
            elif stochastic.metadata.get("is_oversold"):
                signal_score += 15  # Potential reversal

        # Normalize to percentage
        signal_pct = (signal_score / max_score) * 100 if max_score > 0 else 50

        # Determine signal
        if signal_pct >= 65:
            signal = "bullish"
        elif signal_pct <= 35:
            signal = "bearish"
        else:
            signal = "neutral"

        return {
            "timeframe": timeframe,
            "signal": signal,
            "signal_strength": round(signal_pct, 0),
            "price": round(current_price, 2),
            "rsi": round(rsi.value, 1),
            "ema_20_signal": ema_20.signal,
            "macd_signal": macd.signal if macd else None,
            "macd_histogram": macd.metadata.get("histogram") if macd else None,
            "adx_value": round(adx.value, 1) if adx else None,
            "trend_strength": adx.metadata.get("trend_strength") if adx else None,
            "stochastic_k": round(stochastic.metadata.get("percent_k", 0), 1) if stochastic else None,
            "stochastic_signal": stochastic.signal if stochastic else None,
        }

    except Exception as e:
        logger.warning(f"Error analyzing {symbol} on {timeframe}: {e}")
        return None


def _calculate_timeframe_confluence(
    daily: Optional[Dict],
    hourly: Optional[Dict],
    m15: Optional[Dict],
) -> Dict:
    """Calculate confluence score across timeframes.

    Args:
        daily: Daily timeframe analysis
        hourly: Hourly timeframe analysis
        m15: 15-minute timeframe analysis

    Returns:
        Dictionary with confluence metrics
    """
    available_tfs = []
    signals = []

    if daily:
        available_tfs.append("1d")
        signals.append(daily.get("signal"))
    if hourly:
        available_tfs.append("1h")
        signals.append(hourly.get("signal"))
    if m15:
        available_tfs.append("15m")
        signals.append(m15.get("signal"))

    if not signals:
        return {
            "alignment": "unknown",
            "confluence_score": 0,
            "aligned_timeframes": [],
            "conflicting_timeframes": [],
        }

    # Count alignment
    bullish_count = signals.count("bullish")
    bearish_count = signals.count("bearish")
    total = len(signals)

    # Determine alignment
    if bullish_count == total:
        alignment = "full_bullish"
        confluence_score = 100
    elif bearish_count == total:
        alignment = "full_bearish"
        confluence_score = 100
    elif bullish_count > bearish_count:
        alignment = "partial_bullish"
        confluence_score = (bullish_count / total) * 100
    elif bearish_count > bullish_count:
        alignment = "partial_bearish"
        confluence_score = (bearish_count / total) * 100
    else:
        alignment = "mixed"
        confluence_score = 50

    # Identify aligned and conflicting timeframes
    aligned = []
    conflicting = []
    majority_signal = "bullish" if bullish_count >= bearish_count else "bearish"

    for i, tf in enumerate(available_tfs):
        if signals[i] == majority_signal:
            aligned.append(tf)
        elif signals[i] != "neutral":
            conflicting.append(tf)

    return {
        "alignment": alignment,
        "confluence_score": round(confluence_score, 0),
        "aligned_timeframes": aligned,
        "conflicting_timeframes": conflicting,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "neutral_count": signals.count("neutral"),
    }


def _calculate_momentum_score(
    macd: Optional[Indicator],
    stochastic: Optional[Indicator],
    rsi: Indicator,
) -> Dict:
    """Calculate combined momentum score from MACD and Stochastic.

    Args:
        macd: MACD indicator result
        stochastic: Stochastic indicator result
        rsi: RSI indicator result

    Returns:
        Dictionary with momentum metrics
    """
    score = 0
    max_score = 0

    # MACD component (40 points)
    if macd:
        max_score += 40
        histogram = macd.metadata.get("histogram", 0)
        if macd.signal == "bullish":
            score += 30
            if histogram and histogram > 0:
                score += 10  # Positive histogram
        elif macd.signal == "neutral":
            score += 20
        else:
            if histogram and histogram < 0:
                score += 0  # Negative histogram in bearish
            else:
                score += 10  # Improving

    # Stochastic component (40 points)
    if stochastic:
        max_score += 40
        if stochastic.signal == "bullish":
            score += 30
            if stochastic.metadata.get("bullish_crossover"):
                score += 10  # Fresh crossover
        elif stochastic.signal == "neutral":
            score += 20
        else:
            if stochastic.metadata.get("is_overbought"):
                score += 5  # Potential reversal warning

    # RSI component (20 points)
    max_score += 20
    if 50 <= rsi.value <= 60:
        score += 20  # Bullish momentum
    elif 40 <= rsi.value < 50:
        score += 15  # Neutral-bullish
    elif rsi.value < 30:
        score += 10  # Oversold (potential bounce)
    elif rsi.value > 70:
        score += 5   # Overbought warning
    else:
        score += 10

    momentum_pct = (score / max_score) * 100 if max_score > 0 else 50

    # Determine momentum label
    if momentum_pct >= 70:
        label = "strong_bullish"
    elif momentum_pct >= 55:
        label = "bullish"
    elif momentum_pct >= 45:
        label = "neutral"
    elif momentum_pct >= 30:
        label = "bearish"
    else:
        label = "strong_bearish"

    return {
        "score": round(momentum_pct, 0),
        "label": label,
        "macd_contribution": macd.signal if macd else None,
        "stochastic_contribution": stochastic.signal if stochastic else None,
        "rsi_contribution": round(rsi.value, 1),
    }


def _detect_warnings(
    divergence: Optional[Indicator],
    rsi: Indicator,
    stochastic: Optional[Indicator],
    atr: Optional[Indicator],
) -> List[Dict]:
    """Detect potential warning conditions.

    Args:
        divergence: Divergence detection result
        rsi: RSI indicator result
        stochastic: Stochastic indicator result
        atr: ATR indicator result

    Returns:
        List of warning dictionaries
    """
    warnings = []

    # Divergence warnings
    if divergence and divergence.value and divergence.value > 0:
        div_types = divergence.metadata.get("divergence_types", [])
        for div_type in div_types:
            warnings.append({
                "type": "divergence",
                "severity": "high" if "regular" in str(div_type) else "medium",
                "message": f"{str(div_type).replace('_', ' ').title()} divergence detected",
                "action": "Consider reducing position size or tightening stops",
            })

    # Overbought/Oversold warnings
    if rsi.value > 75:
        warnings.append({
            "type": "overbought",
            "severity": "medium",
            "message": f"RSI overbought at {rsi.value:.1f}",
            "action": "Watch for reversal signals before new longs",
        })
    elif rsi.value < 25:
        warnings.append({
            "type": "oversold",
            "severity": "medium",
            "message": f"RSI oversold at {rsi.value:.1f}",
            "action": "Watch for reversal signals before new shorts",
        })

    # Stochastic extreme warnings
    if stochastic:
        if stochastic.metadata.get("is_overbought") and stochastic.metadata.get("bearish_crossover"):
            warnings.append({
                "type": "stochastic_reversal",
                "severity": "high",
                "message": "Stochastic bearish crossover in overbought zone",
                "action": "Potential short-term top - consider taking profits",
            })
        elif stochastic.metadata.get("is_oversold") and stochastic.metadata.get("bullish_crossover"):
            warnings.append({
                "type": "stochastic_reversal",
                "severity": "medium",
                "message": "Stochastic bullish crossover in oversold zone",
                "action": "Potential short-term bottom - watch for confirmation",
            })

    # High volatility warning
    if atr and atr.metadata.get("volatility") == "high":
        warnings.append({
            "type": "high_volatility",
            "severity": "medium",
            "message": f"ATR indicates high volatility ({atr.metadata.get('atr_percentage', 0):.1f}%)",
            "action": "Adjust position sizing and use wider stops",
        })

    return warnings


def get_market_overview(days_back: int = 90) -> Dict:
    """Get comprehensive overview of major market indices health.

    Analyzes the S&P 500, Nasdaq, Dow Jones, Russell 2000, VIX volatility (VXX),
    and Treasury bonds (TLT) to determine overall market direction, strength,
    and risk conditions.

    Enhanced Features:
    - Multi-timeframe analysis (daily, hourly, 15-minute)
    - Volatility regime detection via VXX
    - Advanced momentum scoring (MACD + Stochastic)
    - Trend strength assessment (ADX)
    - Divergence warnings

    Args:
        days_back: Number of days of historical data (default: 90)

    Returns:
        Dictionary containing:
        - indices: List of index data with enhanced signals
        - market_signal: Overall market sentiment ("bullish", "bearish", "neutral")
        - volatility_regime: Current volatility state (high_fear, elevated, normal, complacent)
        - momentum_score: Combined momentum assessment (0-100)
        - trend_strength: Market trend strength from ADX
        - timeframe_alignment: Multi-TF confluence data
        - risk_context: Risk-on/risk-off based on TLT behavior
        - warnings: List of active warnings (divergences, overbought, etc.)
        - summary: Text summary of market health

    Example:
        >>> overview = get_market_overview()
        >>> print(overview['market_signal'])  # "bullish"
        >>> print(overview['volatility_regime']['regime'])  # "normal"
        >>> print(overview['timeframe_alignment']['daily_consensus'])  # "bullish"
        >>> for warning in overview['warnings']:
        >>>     print(f"WARNING: {warning['message']}")
    """
    logger.info(f"Analyzing enhanced market overview ({len(MARKET_INDICES)} indices)")

    indices_data = []
    all_warnings = []

    # Track VXX and TLT separately for context
    vxx_data = None
    tlt_data = None

    # Aggregate momentum and trend data for core indices
    momentum_scores = []
    trend_strengths = []

    # Multi-timeframe data for core indices
    mtf_analyses = {}

    for symbol, name in MARKET_INDICES.items():
        try:
            # === DAILY ANALYSIS (Primary) ===
            daily_analysis = _analyze_timeframe(symbol, "1d", days_back)

            if not daily_analysis:
                logger.warning(f"Insufficient data for {symbol}")
                continue

            # === MULTI-TIMEFRAME (for core indices only) ===
            hourly_analysis = None
            m15_analysis = None

            # Only run MTF for core indices (not VXX/TLT to save API calls)
            if symbol in CORE_INDICES:
                hourly_analysis = _analyze_timeframe(symbol, "1h", 7)  # 7 days hourly
                m15_analysis = _analyze_timeframe(symbol, "15m", 5)    # 5 days 15m (handles weekends)

                mtf_analyses[symbol] = {
                    "daily": daily_analysis,
                    "hourly": hourly_analysis,
                    "m15": m15_analysis,
                    "confluence": _calculate_timeframe_confluence(
                        daily_analysis, hourly_analysis, m15_analysis
                    ),
                }

            # === ENHANCED INDICATORS (from daily data) ===
            bars = fetch_price_bars(symbol, timeframe="1d", days_back=days_back)

            if not bars or len(bars) < 20:
                continue

            # Core indicators
            ema_20 = calculate_ema(bars, period=20)
            ema_50 = calculate_ema(bars, period=50) if len(bars) >= 50 else None
            rsi = calculate_rsi(bars, period=14)
            volume = analyze_volume(bars)

            # Enhanced indicators
            macd = calculate_macd(bars) if len(bars) >= 35 else None
            atr = calculate_atr(bars) if len(bars) >= 15 else None
            adx = calculate_adx(bars) if len(bars) >= 28 else None
            stochastic = calculate_stochastic(bars) if len(bars) >= 20 else None
            divergence = detect_divergences(bars) if len(bars) >= 50 else None

            # Calculate DAILY change (today vs yesterday, not period change)
            current_price = bars[-1].close
            if len(bars) >= 2:
                prev_close = bars[-2].close  # Yesterday's close
                change_pct = ((current_price - prev_close) / prev_close) * 100
            else:
                # Fallback to intraday change if only 1 bar
                change_pct = ((current_price - bars[-1].open) / bars[-1].open) * 100

            # === MOMENTUM SCORE ===
            momentum = _calculate_momentum_score(macd, stochastic, rsi)

            # === TREND STRENGTH ===
            trend_data = None
            if adx:
                trend_data = {
                    "adx": round(adx.value, 1),
                    "strength": adx.metadata.get("trend_strength"),
                    "direction": adx.metadata.get("trend_direction"),
                    "is_trending": adx.metadata.get("is_trending", False),
                }

            # === DETECT WARNINGS ===
            index_warnings = _detect_warnings(divergence, rsi, stochastic, atr)
            for w in index_warnings:
                w["symbol"] = symbol
            all_warnings.extend(index_warnings)

            # === USE ENHANCED SIGNAL FROM TIMEFRAME ANALYSIS ===
            signal = daily_analysis["signal"]
            signal_strength = daily_analysis["signal_strength"]

            # === STORE CONTEXT DATA ===
            if symbol == "VXX":
                vxx_data = {
                    "rsi": rsi.value,
                    "change_pct": change_pct,
                    "signal": signal,
                    "atr_pct": atr.metadata.get("atr_percentage") if atr else None,
                }
            elif symbol == "TLT":
                tlt_data = {
                    "change_pct": change_pct,
                    "signal": signal,
                }

            # Track momentum and trend for core indices only
            if symbol in CORE_INDICES:
                momentum_scores.append(momentum["score"])
                if adx:
                    trend_strengths.append(adx.value)

            # === BUILD INDEX DATA ===
            # Determine if price is above EMAs (direct comparison, not signal threshold)
            above_ema20 = current_price > ema_20.value
            above_ema50 = current_price > ema_50.value if ema_50 else None

            index_entry = {
                "symbol": symbol,
                "name": name,
                "price": round(current_price, 2),
                "change_pct": round(change_pct, 2),
                "signal": signal,
                "signal_strength": signal_strength,
                # Core indicators
                "rsi": round(rsi.value, 1),
                "above_ema20": above_ema20,
                "above_ema50": above_ema50,
                # Enhanced indicators
                "macd_signal": macd.signal if macd else None,
                "macd_histogram": round(macd.metadata.get("histogram", 0), 4) if macd and macd.metadata.get("histogram") else None,
                "adx": trend_data,
                "stochastic_k": round(stochastic.metadata.get("percent_k", 0), 1) if stochastic else None,
                "momentum": momentum,
                # Volatility context
                "atr_pct": round(atr.metadata.get("atr_percentage", 0), 2) if atr and atr.metadata.get("atr_percentage") else None,
                "volatility": atr.metadata.get("volatility") if atr else None,
                # Multi-timeframe (for core indices)
                "timeframes": mtf_analyses.get(symbol),
                # Warnings for this index
                "warnings": [w for w in index_warnings],
            }

            indices_data.append(index_entry)

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            continue

    # === OVERALL MARKET SIGNAL (based on core indices only) ===
    core_indices = [i for i in indices_data if i["symbol"] in CORE_INDICES]
    core_bullish = sum(1 for i in core_indices if i["signal"] == "bullish")
    core_bearish = sum(1 for i in core_indices if i["signal"] == "bearish")
    total_core = len(core_indices)

    if total_core > 0:
        if core_bullish >= total_core * 0.6:
            market_signal = "bullish"
            summary = f"{core_bullish}/{total_core} core indices bullish - strong market"
        elif core_bearish >= total_core * 0.6:
            market_signal = "bearish"
            summary = f"{core_bearish}/{total_core} core indices bearish - weak market"
        else:
            market_signal = "neutral"
            summary = f"Mixed signals - {core_bullish} bullish, {core_bearish} bearish"
    else:
        market_signal = "unknown"
        summary = "Insufficient data for market assessment"

    # === VOLATILITY REGIME ===
    volatility_regime = _determine_volatility_regime(vxx_data)

    # === AGGREGATE MOMENTUM ===
    avg_momentum = sum(momentum_scores) / len(momentum_scores) if momentum_scores else 50
    overall_momentum = {
        "score": round(avg_momentum, 0),
        "label": "bullish" if avg_momentum >= 55 else "bearish" if avg_momentum <= 45 else "neutral",
    }

    # === AGGREGATE TREND STRENGTH ===
    avg_trend = sum(trend_strengths) / len(trend_strengths) if trend_strengths else 0
    overall_trend = {
        "average_adx": round(avg_trend, 1),
        "is_trending": avg_trend >= 25,
        "strength": "strong" if avg_trend >= 40 else "moderate" if avg_trend >= 25 else "weak",
    }

    # === AGGREGATE TIMEFRAME ALIGNMENT ===
    all_daily_signals = [mtf["daily"]["signal"] for mtf in mtf_analyses.values() if mtf.get("daily")]
    all_hourly_signals = [mtf["hourly"]["signal"] for mtf in mtf_analyses.values() if mtf.get("hourly")]
    all_m15_signals = [mtf["m15"]["signal"] for mtf in mtf_analyses.values() if mtf.get("m15")]

    def majority_signal(signals):
        if not signals:
            return "unknown"
        bull = signals.count("bullish")
        bear = signals.count("bearish")
        neut = signals.count("neutral")
        total = len(signals)
        # Need majority (>50%) to call a consensus, otherwise neutral
        if bull > total / 2:
            return "bullish"
        elif bear > total / 2:
            return "bearish"
        return "neutral"

    daily_consensus = majority_signal(all_daily_signals)
    hourly_consensus = majority_signal(all_hourly_signals)
    m15_consensus = majority_signal(all_m15_signals)

    timeframe_alignment = {
        "daily_consensus": daily_consensus,
        "hourly_consensus": hourly_consensus,
        "m15_consensus": m15_consensus,
        "full_alignment": (
            daily_consensus == hourly_consensus == m15_consensus
            and daily_consensus != "unknown"
        ) if all_hourly_signals and all_m15_signals else False,
        "per_index": {sym: data["confluence"] for sym, data in mtf_analyses.items()},
    }

    # === RISK-OFF DETECTION (TLT) ===
    risk_context = "neutral"
    if tlt_data:
        # If TLT is rising while equities are falling, it's risk-off
        if tlt_data["change_pct"] > 1 and market_signal == "bearish":
            risk_context = "risk_off"
            summary += " | TLT rising suggests flight to safety"
        elif tlt_data["change_pct"] < -1 and market_signal == "bullish":
            risk_context = "risk_on"

    logger.info(f"Enhanced Market Overview: {market_signal} - {summary}")

    return {
        "indices": indices_data,
        "market_signal": market_signal,
        "bullish_count": core_bullish,
        "bearish_count": core_bearish,
        "neutral_count": total_core - core_bullish - core_bearish,
        "total_indices": total_core,  # Count of core indices (excludes VXX, TLT)
        # NEW: Enhanced fields
        "volatility_regime": volatility_regime,
        "momentum_score": overall_momentum,
        "trend_strength": overall_trend,
        "timeframe_alignment": timeframe_alignment,
        "risk_context": risk_context,
        "warnings": all_warnings,
        # Summary
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

        # Only count core equity indices for direction (exclude VXX, TLT)
        if symbol in CORE_INDICES:
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
