"""Market Scanner Service for automated stock discovery.

This service provides scheduled and on-demand scanning across three trading styles:
- Day Trading: Intraday patterns (gaps, momentum, reversals)
- Swing Trading: Multi-day patterns (pullbacks, breakouts, continuations)
- Position Trading: Weekly/monthly patterns (trend reversals, base breakouts)

The scanner detects patterns, calculates confidence scores, and generates
human-readable descriptions for each setup found.
"""

import asyncio
import logging
import uuid
from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Set, Tuple
from zoneinfo import ZoneInfo

from app.config import get_settings
from app.models.scanner import (
    ScannerResult,
    ScannerResponse,
    AllScannersResponse,
    ScannerStatusResponse,
    ScanSession,
    ScanCandidate,
    ConfidenceScoreComponents,
    TradingStyle,
    PatternType,
    ConfidenceGrade,
    score_to_grade,
    VOLUME_THRESHOLDS,
    VOLUME_CONFIDENCE_MULTIPLIER,
    MIN_PRICE,
    EXPIRATION_HOURS,
)
from app.models.data import PriceBar
from app.storage.scanner_store import get_scanner_store
from app.services.scheduler import ET, is_market_open, get_next_market_open

logger = logging.getLogger(__name__)

# Scan schedule (ET timezone)
SCAN_SCHEDULE = [
    {"time": time(8, 0), "name": "pre_market", "focus": "Gap setups using extended hours data"},
    {"time": time(9, 30), "name": "market_open", "focus": "Opening momentum, gap continuations"},
    {"time": time(12, 0), "name": "mid_day", "focus": "Consolidation breaks, trend continuations"},
    {"time": time(15, 30), "name": "power_hour", "focus": "Late-day momentum, EOD setups"},
    {"time": time(16, 30), "name": "after_close", "focus": "Next-day position/swing setups"},
]

# Maximum stocks to scan per batch (Alpaca limit)
BATCH_SIZE = 500

# Minimum score to include in results
MIN_SCORE_THRESHOLD = 45


class MarketScannerService:
    """Service for automated market scanning across trading styles."""

    def __init__(self):
        """Initialize the scanner service."""
        self._store = get_scanner_store()
        self._is_scanning = False
        self._current_session: Optional[ScanSession] = None
        self._scheduled_task: Optional[asyncio.Task] = None
        self._running = False

    # =========================================================================
    # Main scanning methods
    # =========================================================================

    async def run_scan(
        self,
        scan_name: str = "on_demand",
        styles: Optional[List[TradingStyle]] = None,
    ) -> AllScannersResponse:
        """Run a complete scan for all or specified trading styles.

        Args:
            scan_name: Name of the scan (e.g., "pre_market", "on_demand")
            styles: Optional list of styles to scan. If None, scans all styles.

        Returns:
            AllScannersResponse with results for all scanned styles
        """
        if self._is_scanning:
            logger.warning("Scan already in progress, returning cached results")
            return await self._get_cached_response()

        self._is_scanning = True
        self._current_session = ScanSession(
            session_id=str(uuid.uuid4()),
            scan_name=scan_name,
        )

        logger.info(f"Starting {scan_name} scan")

        try:
            styles_to_scan = styles or list(TradingStyle)

            # Fetch stock universe
            stock_universe = await self._fetch_stock_universe()
            self._current_session.stocks_scanned = len(stock_universe)

            # Scan each style
            for style in styles_to_scan:
                try:
                    results = await self._scan_style(style, stock_universe)
                    # Handle both sync and async stores
                    if asyncio.iscoroutinefunction(self._store.store_results):
                        await self._store.store_results(
                            style=style,
                            results=results,
                            scan_name=scan_name,
                            total_scanned=len(stock_universe),
                        )
                    else:
                        self._store.store_results(
                            style=style,
                            results=results,
                            scan_name=scan_name,
                            total_scanned=len(stock_universe),
                        )
                    self._current_session.results_found[style.value] = len(results)
                except Exception as e:
                    logger.error(f"Error scanning {style.value}: {e}")
                    self._current_session.errors.append(f"{style.value}: {str(e)}")

            self._current_session.mark_complete()

            # Update next scheduled scan
            next_scan = self._get_next_scheduled_scan_time()
            if next_scan:
                if asyncio.iscoroutinefunction(self._store.set_next_scheduled_scan):
                    await self._store.set_next_scheduled_scan(next_scan)
                else:
                    self._store.set_next_scheduled_scan(next_scan)

            logger.info(
                f"Scan complete: {self._current_session.results_found} "
                f"({self._current_session.stocks_scanned} stocks scanned)"
            )

        finally:
            self._is_scanning = False

        return await self._get_cached_response()

    async def _scan_style(
        self,
        style: TradingStyle,
        stock_universe: List[Dict],
    ) -> List[ScannerResult]:
        """Scan for patterns in a specific trading style.

        Args:
            style: Trading style to scan for
            stock_universe: List of stock data dicts with snapshots and bars

        Returns:
            List of scanner results for this style
        """
        logger.info(f"Scanning for {style.value} trading patterns")

        results = []
        volume_threshold = VOLUME_THRESHOLDS.get(style, 100_000)

        for stock_data in stock_universe:
            try:
                # Filter by volume threshold
                avg_volume = stock_data.get("avg_volume", 0)
                if avg_volume < volume_threshold:
                    continue

                # Detect patterns for this style
                patterns = await self._detect_patterns(style, stock_data)

                if not patterns:
                    continue

                # Calculate confidence score for best pattern
                for pattern_type, pattern_data in patterns:
                    score_components = self._calculate_confidence_score(
                        style, stock_data, pattern_type, pattern_data
                    )
                    total_score = score_components.calculate_total()

                    if total_score < MIN_SCORE_THRESHOLD:
                        continue

                    # Generate description
                    description = self._generate_description(
                        style, pattern_type, stock_data, pattern_data
                    )

                    # Build key levels
                    key_levels = self._extract_key_levels(pattern_data, stock_data)

                    # Check for warnings
                    warnings = self._check_warnings(stock_data)

                    # Calculate expiration
                    expiration_hours = EXPIRATION_HOURS.get(style, 24)
                    expires_at = datetime.utcnow() + timedelta(hours=expiration_hours)

                    result = ScannerResult(
                        symbol=stock_data["symbol"],
                        style=style,
                        confidence_grade=score_to_grade(total_score),
                        confidence_score=round(total_score, 1),
                        current_price=stock_data.get("current_price", 0),
                        description=description,
                        pattern_type=pattern_type,
                        key_levels=key_levels,
                        detected_at=datetime.utcnow(),
                        warnings=warnings,
                        volume_multiple=pattern_data.get("volume_multiple"),
                        gap_pct=pattern_data.get("gap_pct"),
                        fib_level=pattern_data.get("fib_level"),
                        rsi_value=pattern_data.get("rsi_value"),
                        vwap=pattern_data.get("vwap"),
                        expires_at=expires_at,
                    )
                    results.append(result)

            except Exception as e:
                logger.debug(f"Error processing {stock_data.get('symbol', 'unknown')}: {e}")
                continue

        # Sort by confidence and take top 10
        results.sort(key=lambda r: r.confidence_score, reverse=True)
        return results[:10]

    # =========================================================================
    # Stock universe fetching
    # =========================================================================

    async def _fetch_stock_universe(self) -> List[Dict]:
        """Fetch the stock universe with price data.

        Returns:
            List of stock data dictionaries with snapshots and bars
        """
        from app.tools.market_data import fetch_snapshots, fetch_price_bars

        logger.info("Fetching stock universe")

        # Get active stocks from Alpaca
        symbols = await self._get_active_symbols()
        logger.info(f"Found {len(symbols)} active symbols")

        # Fetch snapshots in batches
        all_snapshots = {}
        for i in range(0, len(symbols), BATCH_SIZE):
            batch = symbols[i:i + BATCH_SIZE]
            try:
                snapshots = fetch_snapshots(batch)
                all_snapshots.update(snapshots)
            except Exception as e:
                logger.warning(f"Error fetching snapshot batch: {e}")

        logger.info(f"Fetched snapshots for {len(all_snapshots)} symbols")

        # Build stock data list
        stock_universe = []

        for symbol, snapshot in all_snapshots.items():
            try:
                if not snapshot:
                    continue

                daily_bar = snapshot.get("daily_bar")
                prev_bar = snapshot.get("prev_daily_bar")
                quote = snapshot.get("latest_quote")

                if not daily_bar or not prev_bar:
                    continue

                current_price = daily_bar.get("close", 0)
                prev_close = prev_bar.get("close", 0)

                # Apply minimum price filter
                if current_price < MIN_PRICE:
                    continue

                # Calculate change
                change_pct = ((current_price - prev_close) / prev_close * 100) if prev_close > 0 else 0

                # Get volume data
                current_volume = daily_bar.get("volume", 0)

                # Calculate true 20-day average volume
                try:
                    bars_20d = fetch_price_bars(symbol, timeframe="1d", days_back=20)
                    if bars_20d and len(bars_20d) >= 10:
                        avg_volume = sum(b.volume for b in bars_20d) / len(bars_20d)
                    else:
                        avg_volume = current_volume  # Fallback to current volume
                except Exception:
                    avg_volume = current_volume  # Fallback on error

                stock_data = {
                    "symbol": symbol,
                    "current_price": current_price,
                    "prev_close": prev_close,
                    "change_pct": change_pct,
                    "open_price": daily_bar.get("open", current_price),
                    "high_price": daily_bar.get("high", current_price),
                    "low_price": daily_bar.get("low", current_price),
                    "current_volume": current_volume,
                    "avg_volume": avg_volume,  # True 20-day average volume
                    "bid": quote.get("bid_price") if quote else None,
                    "ask": quote.get("ask_price") if quote else None,
                    "snapshot": snapshot,
                }

                stock_universe.append(stock_data)

            except Exception as e:
                logger.debug(f"Error processing {symbol}: {e}")
                continue

        logger.info(f"Stock universe: {len(stock_universe)} stocks after filtering")
        return stock_universe

    async def _get_active_symbols(self) -> List[str]:
        """Get list of active tradeable symbols from Alpaca.

        Returns:
            List of stock ticker symbols
        """
        from app.config import get_settings
        from alpaca.trading.client import TradingClient

        settings = get_settings()

        try:
            client = TradingClient(
                api_key=settings.alpaca_api_key,
                secret_key=settings.alpaca_secret_key,
                paper=settings.alpaca_paper,
            )

            # Get all active assets
            assets = client.get_all_assets()

            # Filter to tradeable US stocks only (no ETFs)
            symbols = [
                asset.symbol
                for asset in assets
                if asset.tradable
                and asset.status == "active"
                and asset.asset_class == "us_equity"
                and asset.exchange in ("NYSE", "NASDAQ", "AMEX", "ARCA", "BATS")
                and not asset.symbol.endswith("W")  # No warrants
                and "." not in asset.symbol  # No special classes
            ]

            return symbols

        except Exception as e:
            logger.error(f"Error fetching assets from Alpaca: {e}")
            # Fallback to sector stocks if API fails
            from app.tools.market_scanner import SECTOR_STOCKS
            fallback = []
            for stocks in SECTOR_STOCKS.values():
                fallback.extend(stocks)
            return list(set(fallback))

    # =========================================================================
    # Pattern detection
    # =========================================================================

    async def _detect_patterns(
        self,
        style: TradingStyle,
        stock_data: Dict,
    ) -> List[Tuple[PatternType, Dict]]:
        """Detect patterns for a trading style.

        Args:
            style: Trading style
            stock_data: Stock data dictionary

        Returns:
            List of (pattern_type, pattern_data) tuples
        """
        patterns = []

        if style == TradingStyle.DAY:
            patterns.extend(await self._detect_day_patterns(stock_data))
        elif style == TradingStyle.SWING:
            patterns.extend(await self._detect_swing_patterns(stock_data))
        elif style == TradingStyle.POSITION:
            patterns.extend(await self._detect_position_patterns(stock_data))

        return patterns

    async def _detect_day_patterns(self, stock_data: Dict) -> List[Tuple[PatternType, Dict]]:
        """Detect day trading patterns.

        Day trading patterns focus on:
        - Gap plays (gap-and-go, gap fills, gap downs)
        - Momentum breakouts (VWAP reclaims, resistance breaks, momentum surges)
        - Reversal patterns (oversold bounces, panic dip buys, failed breakdowns)
        - Support patterns (support bounce)
        """
        patterns = []
        symbol = stock_data["symbol"]

        current_price = stock_data.get("current_price", 0)
        open_price = stock_data.get("open_price", 0)
        prev_close = stock_data.get("prev_close", 0)
        high_price = stock_data.get("high_price", 0)
        low_price = stock_data.get("low_price", 0)
        current_volume = stock_data.get("current_volume", 0)
        avg_volume = stock_data.get("avg_volume", 1)

        volume_multiple = current_volume / avg_volume if avg_volume > 0 else 1

        # Gap detection
        if prev_close > 0:
            gap_pct = ((open_price - prev_close) / prev_close) * 100

            # Gap up > 2%
            if gap_pct >= 2:
                pattern_data = {
                    "gap_pct": round(gap_pct, 1),
                    "volume_multiple": round(volume_multiple, 1),
                    "gap_direction": "up",
                    "support": prev_close,
                    "resistance": high_price,
                }

                # Gap and go (price above open)
                if current_price > open_price:
                    patterns.append((PatternType.GAP_UP, pattern_data))

                # Gap fill in progress (price below open, above prev close)
                elif current_price < open_price and current_price > prev_close:
                    pattern_data["gap_fill_level"] = prev_close
                    patterns.append((PatternType.GAP_FILL, pattern_data))

            # Gap down > 2%
            elif gap_pct <= -2:
                pattern_data = {
                    "gap_pct": round(gap_pct, 1),
                    "volume_multiple": round(volume_multiple, 1),
                    "gap_direction": "down",
                    "support": low_price,
                    "resistance": prev_close,
                }

                # GAP_DOWN pattern - gap down and price still below open
                if current_price < open_price:
                    patterns.append((PatternType.GAP_DOWN, pattern_data))

                # Gap fill bounce (price recovering)
                if current_price > open_price:
                    pattern_data["gap_fill_level"] = prev_close
                    patterns.append((PatternType.GAP_FILL, pattern_data))

        # Momentum breakout (new high of day with volume)
        if volume_multiple >= 2 and current_price >= high_price * 0.99:
            patterns.append((PatternType.NEW_HOD, {
                "volume_multiple": round(volume_multiple, 1),
                "high_price": high_price,
            }))

        # MOMENTUM_SURGE: Price up > 5% AND volume > 3x average
        if prev_close > 0:
            price_change_pct = ((current_price - prev_close) / prev_close) * 100
            if price_change_pct > 5 and volume_multiple > 3:
                patterns.append((PatternType.MOMENTUM_SURGE, {
                    "price_change_pct": round(price_change_pct, 1),
                    "volume_multiple": round(volume_multiple, 1),
                    "support": low_price,
                    "resistance": high_price,
                }))

        # Try to get RSI, VWAP, and MACD for reversal/momentum patterns
        try:
            from app.tools.market_data import fetch_price_bars
            from app.tools.indicators import calculate_rsi, calculate_vwap, calculate_macd
            from app.tools.analysis import find_structural_pivots

            # Fetch intraday bars for indicator calculations
            bars = fetch_price_bars(symbol, timeframe="15m", days_back=2)

            if bars and len(bars) >= 35:  # Need at least 35 bars for MACD (26+9)
                rsi = calculate_rsi(bars, period=14)
                vwap = calculate_vwap(bars)
                macd = calculate_macd(bars)

                # Store MACD data in stock_data for confidence scoring
                stock_data["macd_signal"] = macd.signal
                stock_data["macd_histogram"] = macd.metadata.get("histogram", 0)

                # Oversold bounce (RSI < 30, price bouncing)
                if rsi.value < 30 and current_price > low_price * 1.01:
                    patterns.append((PatternType.OVERSOLD_BOUNCE, {
                        "rsi_value": round(rsi.value, 1),
                        "vwap": vwap.value,
                        "volume_multiple": round(volume_multiple, 1),
                        "macd_signal": macd.signal,
                    }))

                # PANIC_DIP_BUY: RSI < 20 AND price > 102% of intraday low AND volume > 2x
                if rsi.value < 20 and current_price > low_price * 1.02 and volume_multiple > 2:
                    patterns.append((PatternType.PANIC_DIP_BUY, {
                        "rsi_value": round(rsi.value, 1),
                        "vwap": vwap.value,
                        "volume_multiple": round(volume_multiple, 1),
                        "support": low_price,
                        "resistance": high_price,
                    }))

                # VWAP reclaim
                if vwap.value > 0:
                    # Price crossed above VWAP
                    if current_price > vwap.value and low_price < vwap.value:
                        patterns.append((PatternType.VWAP_RECLAIM, {
                            "vwap": round(vwap.value, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                            "macd_signal": macd.signal,
                        }))

                # SUPPORT_BOUNCE: Price touches support level + bounces with volume
                try:
                    pivots = find_structural_pivots(bars, lookback=20)
                    supports = [p.price for p in pivots if p.type == "support"]

                    if supports:
                        nearest_support = min(supports, key=lambda s: abs(current_price - s))
                        # Price touched support (within 1%) and bounced
                        if low_price <= nearest_support * 1.01 and current_price > nearest_support * 1.01:
                            if volume_multiple >= 1.5:
                                patterns.append((PatternType.SUPPORT_BOUNCE, {
                                    "support": round(nearest_support, 2),
                                    "volume_multiple": round(volume_multiple, 1),
                                    "rsi_value": round(rsi.value, 1),
                                    "resistance": high_price,
                                }))
                except Exception:
                    pass

                # FAILED_BREAKDOWN: Broke below support then reclaimed within same day
                try:
                    if supports:
                        for support in supports:
                            # Price broke below support at some point (low < support)
                            # but current price is back above support
                            if low_price < support * 0.99 and current_price > support * 1.01:
                                patterns.append((PatternType.FAILED_BREAKDOWN, {
                                    "support": round(support, 2),
                                    "volume_multiple": round(volume_multiple, 1),
                                    "rsi_value": round(rsi.value, 1),
                                    "resistance": high_price,
                                }))
                                break  # Only detect once
                except Exception:
                    pass

            elif bars and len(bars) >= 14:
                # Fallback without MACD if not enough bars
                rsi = calculate_rsi(bars, period=14)
                vwap = calculate_vwap(bars)

                # Oversold bounce (RSI < 30, price bouncing)
                if rsi.value < 30 and current_price > low_price * 1.01:
                    patterns.append((PatternType.OVERSOLD_BOUNCE, {
                        "rsi_value": round(rsi.value, 1),
                        "vwap": vwap.value,
                        "volume_multiple": round(volume_multiple, 1),
                    }))

                # VWAP reclaim
                if vwap.value > 0:
                    if current_price > vwap.value and low_price < vwap.value:
                        patterns.append((PatternType.VWAP_RECLAIM, {
                            "vwap": round(vwap.value, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                        }))

        except Exception as e:
            logger.debug(f"Could not fetch indicators for {symbol}: {e}")

        return patterns

    async def _detect_swing_patterns(self, stock_data: Dict) -> List[Tuple[PatternType, Dict]]:
        """Detect swing trading patterns.

        Swing trading patterns focus on:
        - Trend continuations (bull flag, bear flag, trend continuation)
        - Pullbacks to key levels (pullback to support, fib retracement)
        - Breakout setups with consolidation (range breakout, channel breakout)
        - Structural patterns (higher low, lower high)
        """
        patterns = []
        symbol = stock_data["symbol"]

        try:
            from app.tools.market_data import fetch_price_bars
            from app.tools.indicators import calculate_ema, calculate_rsi, analyze_volume, calculate_macd
            from app.tools.analysis import calculate_fibonacci_levels, find_structural_pivots

            # Fetch daily bars for swing analysis
            bars = fetch_price_bars(symbol, timeframe="1d", days_back=60)

            if not bars or len(bars) < 30:
                return patterns

            current_price = stock_data.get("current_price", bars[-1].close)
            volume_result = analyze_volume(bars)
            volume_multiple = volume_result.metadata.get("relative_volume", 1.0)

            # Calculate EMAs
            ema_9 = calculate_ema(bars, period=9)
            ema_21 = calculate_ema(bars, period=21)
            ema_50 = calculate_ema(bars, period=50) if len(bars) >= 50 else None
            rsi = calculate_rsi(bars, period=14)

            # Calculate MACD for trend confirmation
            macd = None
            if len(bars) >= 35:
                macd = calculate_macd(bars)
                # Store MACD data in stock_data for confidence scoring
                stock_data["macd_signal"] = macd.signal
                stock_data["macd_histogram"] = macd.metadata.get("histogram", 0)

            # Calculate Fibonacci levels
            try:
                fib = calculate_fibonacci_levels(bars, trade_type="swing")
                fib_metadata = fib.metadata
                nearest_level = fib_metadata.get("nearest_level")
                at_entry_level = fib_metadata.get("at_entry_level", False)
            except Exception:
                fib_metadata = {}
                nearest_level = None
                at_entry_level = False

            # Get support/resistance
            try:
                pivots = find_structural_pivots(bars, lookback=20)
                supports = [p.price for p in pivots if p.type == "support"]
                resistances = [p.price for p in pivots if p.type == "resistance"]
            except Exception:
                supports = []
                resistances = []

            # Bull flag pattern (consolidating near highs)
            if ema_9.signal == "bullish" and ema_21.signal == "bullish":
                # Check for consolidation (low volatility recently)
                recent_range = max(b.high for b in bars[-5:]) - min(b.low for b in bars[-5:])
                avg_range = sum(b.high - b.low for b in bars[-20:]) / 20

                if recent_range < avg_range * 0.7:  # Consolidation
                    consolidation_days = 5
                    resistance = max(b.high for b in bars[-10:])
                    patterns.append((PatternType.BULL_FLAG, {
                        "days": consolidation_days,
                        "resistance": round(resistance, 2),
                        "volume_multiple": round(volume_multiple, 1),
                        "rsi_value": round(rsi.value, 1),
                        "macd_signal": macd.signal if macd else None,
                    }))

            # BEAR_FLAG: EMA 9 bearish + EMA 21 bearish + consolidation near lows
            if ema_9.signal == "bearish" and ema_21.signal == "bearish":
                recent_range = max(b.high for b in bars[-5:]) - min(b.low for b in bars[-5:])
                avg_range = sum(b.high - b.low for b in bars[-20:]) / 20

                if recent_range < avg_range * 0.7:  # Consolidation
                    consolidation_days = 5
                    support = min(b.low for b in bars[-10:])
                    patterns.append((PatternType.BEAR_FLAG, {
                        "days": consolidation_days,
                        "support": round(support, 2),
                        "volume_multiple": round(volume_multiple, 1),
                        "rsi_value": round(rsi.value, 1),
                        "macd_signal": macd.signal if macd else None,
                    }))

            # Pullback to Fibonacci level
            if at_entry_level and nearest_level:
                # Price at a Fib level and trend is up
                if ema_21.signal == "bullish":
                    fib_pct = float(nearest_level) * 100
                    fib_price = fib_metadata.get("retracement_levels", {}).get(nearest_level, current_price)
                    patterns.append((PatternType.FIB_RETRACEMENT, {
                        "fib_level": round(fib_pct, 1),
                        "price": round(fib_price, 2) if isinstance(fib_price, (int, float)) else current_price,
                        "volume_multiple": round(volume_multiple, 1),
                        "rsi_value": round(rsi.value, 1),
                        "macd_signal": macd.signal if macd else None,
                    }))

            # PULLBACK_TO_SUPPORT: Price near support + RSI 40-50 + uptrend intact
            if supports and ema_21.signal == "bullish":
                nearest_support = min(supports, key=lambda s: abs(current_price - s))
                # Price within 3% of support
                if current_price <= nearest_support * 1.03 and current_price >= nearest_support * 0.98:
                    # RSI in healthy pullback zone (40-50)
                    if 40 <= rsi.value <= 55:
                        patterns.append((PatternType.PULLBACK_TO_SUPPORT, {
                            "support": round(nearest_support, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                            "resistance": max(resistances) if resistances else current_price * 1.05,
                        }))

            # Higher low confirmed
            if len(bars) >= 30:
                # Find recent lows
                recent_lows = [(i, bars[i].low) for i in range(-20, -5) if bars[i].low < bars[i-1].low and bars[i].low < bars[i+1].low]
                if len(recent_lows) >= 2:
                    if recent_lows[-1][1] > recent_lows[-2][1]:  # Higher low
                        support = recent_lows[-1][1]
                        patterns.append((PatternType.HIGHER_LOW, {
                            "support": round(support, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                        }))

            # LOWER_HIGH: Two recent highs where latest < previous
            if len(bars) >= 30:
                # Find recent highs
                recent_highs = [(i, bars[i].high) for i in range(-20, -5) if bars[i].high > bars[i-1].high and bars[i].high > bars[i+1].high]
                if len(recent_highs) >= 2:
                    if recent_highs[-1][1] < recent_highs[-2][1]:  # Lower high
                        resistance = recent_highs[-1][1]
                        patterns.append((PatternType.LOWER_HIGH, {
                            "resistance": round(resistance, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                            "support": min(supports) if supports else current_price * 0.95,
                        }))

            # TREND_CONTINUATION: Price > EMA 21 + EMA 21 > EMA 50 + RSI 50-70
            if ema_50 and current_price > ema_21.value and ema_21.value > ema_50.value:
                if 50 <= rsi.value <= 70:
                    patterns.append((PatternType.TREND_CONTINUATION, {
                        "ema_21": round(ema_21.value, 2),
                        "ema_50": round(ema_50.value, 2),
                        "volume_multiple": round(volume_multiple, 1),
                        "rsi_value": round(rsi.value, 1),
                        "macd_signal": macd.signal if macd else None,
                        "resistance": max(resistances) if resistances else current_price * 1.05,
                    }))

            # Range breakout
            if resistances and current_price > max(resistances) * 0.98:
                resistance = max(resistances)
                # Check how long price was below this level
                days_below = sum(1 for b in bars[-20:] if b.high < resistance)
                if days_below >= 10:
                    patterns.append((PatternType.RANGE_BREAKOUT, {
                        "days": days_below,
                        "resistance": round(resistance, 2),
                        "volume_multiple": round(volume_multiple, 1),
                        "rsi_value": round(rsi.value, 1),
                    }))

            # CHANNEL_BREAKOUT: Price breaks above upper channel line
            # Detect channel using linear regression on highs
            if len(bars) >= 20:
                try:
                    import numpy as np
                    highs_20 = [b.high for b in bars[-20:]]
                    lows_20 = [b.low for b in bars[-20:]]

                    # Calculate upper channel using recent highs
                    x = np.arange(len(highs_20))
                    high_slope, high_intercept = np.polyfit(x, highs_20, 1)

                    # Projected upper channel at current bar
                    upper_channel = high_intercept + high_slope * (len(highs_20) - 1)

                    # Channel breakout if current price is above upper channel
                    if current_price > upper_channel * 1.01 and volume_multiple >= 1.5:
                        patterns.append((PatternType.CHANNEL_BREAKOUT, {
                            "resistance": round(upper_channel, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                        }))
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error detecting swing patterns for {symbol}: {e}")

        return patterns

    async def _detect_position_patterns(self, stock_data: Dict) -> List[Tuple[PatternType, Dict]]:
        """Detect position trading patterns.

        Position trading patterns focus on:
        - Long-term trend reversals (golden cross, death cross)
        - Major support/resistance levels
        - Base breakouts (weeks/months of accumulation)
        - Trendline bounces
        """
        patterns = []
        symbol = stock_data["symbol"]

        try:
            from app.tools.market_data import fetch_price_bars
            from app.tools.indicators import calculate_ema, calculate_rsi, analyze_volume
            import numpy as np

            # Fetch daily bars for position analysis (longer lookback)
            bars = fetch_price_bars(symbol, timeframe="1d", days_back=120)

            if not bars or len(bars) < 60:
                return patterns

            current_price = stock_data.get("current_price", bars[-1].close)
            low_price = stock_data.get("low_price", bars[-1].low)
            volume_result = analyze_volume(bars)
            volume_multiple = volume_result.metadata.get("relative_volume", 1.0)

            # Calculate long-term EMAs
            ema_50 = calculate_ema(bars, period=50)
            ema_200 = calculate_ema(bars[-200:], period=200) if len(bars) >= 200 else None
            rsi = calculate_rsi(bars, period=14)

            # Golden cross (50 EMA crossing above 200 EMA)
            if ema_200 and len(bars) >= 200:
                ema_50_prev = calculate_ema(bars[:-1], period=50)
                ema_200_prev = calculate_ema(bars[-201:-1], period=200) if len(bars) >= 201 else None

                if ema_200_prev:
                    # Check for golden crossover
                    if ema_50_prev.value < ema_200_prev.value and ema_50.value > ema_200.value:
                        patterns.append((PatternType.GOLDEN_CROSS, {
                            "ema_50": round(ema_50.value, 2),
                            "ema_200": round(ema_200.value, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                        }))

                    # DEATH_CROSS: EMA 50 prev > EMA 200 prev AND EMA 50 now < EMA 200 now
                    if ema_50_prev.value > ema_200_prev.value and ema_50.value < ema_200.value:
                        patterns.append((PatternType.DEATH_CROSS, {
                            "ema_50": round(ema_50.value, 2),
                            "ema_200": round(ema_200.value, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                        }))

            # Weekly breakout (price above N-week high)
            weeks = 12
            bars_for_weeks = min(weeks * 5, len(bars))  # ~5 trading days per week
            period_high = max(b.high for b in bars[-bars_for_weeks:-1])

            if current_price > period_high:
                patterns.append((PatternType.WEEKLY_BREAKOUT, {
                    "weeks": weeks,
                    "level": round(period_high, 2),
                    "volume_multiple": round(volume_multiple, 1),
                    "rsi_value": round(rsi.value, 1),
                }))

            # Major support test (price near long-term low)
            period_low = min(b.low for b in bars[-bars_for_weeks:])
            if current_price < period_low * 1.05:  # Within 5% of lows
                patterns.append((PatternType.MAJOR_SUPPORT_TEST, {
                    "level": round(period_low, 2),
                    "volume_multiple": round(volume_multiple, 1),
                    "rsi_value": round(rsi.value, 1),
                }))

            # Base breakout (prolonged consolidation ending)
            # Check for low volatility over extended period followed by expansion
            months = 3
            base_bars = min(months * 20, len(bars))  # ~20 trading days per month

            # Calculate average range over base period
            base_ranges = [(b.high - b.low) / b.close for b in bars[-base_bars:-5]]
            recent_ranges = [(b.high - b.low) / b.close for b in bars[-5:]]

            if base_ranges and recent_ranges:
                avg_base_range = sum(base_ranges) / len(base_ranges)
                avg_recent_range = sum(recent_ranges) / len(recent_ranges)

                # Consolidation then expansion
                if avg_base_range < 0.03 and avg_recent_range > avg_base_range * 1.5:
                    base_high = max(b.high for b in bars[-base_bars:-5])
                    if current_price > base_high:
                        patterns.append((PatternType.BASE_BREAKOUT, {
                            "months": months,
                            "level": round(base_high, 2),
                            "volume_multiple": round(volume_multiple, 1),
                            "rsi_value": round(rsi.value, 1),
                        }))

            # TRENDLINE_BOUNCE: Price touches rising trendline + RSI not oversold
            # Detect rising trendline using linear regression on lows
            if len(bars) >= 60:
                try:
                    # Get lows for trendline calculation
                    lows_60 = [b.low for b in bars[-60:]]
                    x = np.arange(len(lows_60))

                    # Fit linear regression to lows
                    low_slope, low_intercept = np.polyfit(x, lows_60, 1)

                    # Only consider rising trendline (positive slope)
                    if low_slope > 0:
                        # Calculate trendline value at current bar
                        trendline_value = low_intercept + low_slope * (len(lows_60) - 1)

                        # Price touched trendline (within 2%) and bounced
                        # Also check that RSI is not oversold (> 30)
                        if low_price <= trendline_value * 1.02 and current_price > trendline_value:
                            if rsi.value > 30:
                                patterns.append((PatternType.TRENDLINE_BOUNCE, {
                                    "trendline_value": round(trendline_value, 2),
                                    "volume_multiple": round(volume_multiple, 1),
                                    "rsi_value": round(rsi.value, 1),
                                    "support": round(trendline_value, 2),
                                }))
                except Exception:
                    pass

        except Exception as e:
            logger.debug(f"Error detecting position patterns for {symbol}: {e}")

        return patterns

    # =========================================================================
    # Confidence scoring
    # =========================================================================

    def _calculate_confidence_score(
        self,
        style: TradingStyle,
        stock_data: Dict,
        pattern_type: PatternType,
        pattern_data: Dict,
    ) -> ConfidenceScoreComponents:
        """Calculate confidence score for a detected pattern.

        Score components (weights):
        - Pattern clarity: 25%
        - Volume confirmation: 25%
        - Indicator alignment: 20%
        - Risk/reward ratio: 15%
        - Fib level proximity: 15%

        Args:
            style: Trading style
            stock_data: Stock data dictionary
            pattern_type: Type of pattern detected
            pattern_data: Pattern-specific data

        Returns:
            ConfidenceScoreComponents with individual scores
        """
        components = ConfidenceScoreComponents()

        # Pattern clarity (0-100)
        # Higher for cleaner patterns
        pattern_scores = {
            # Day trading patterns
            PatternType.GAP_UP: 75,
            PatternType.GAP_DOWN: 75,
            PatternType.GAP_FILL: 65,
            PatternType.RESISTANCE_BREAKOUT: 80,
            PatternType.VWAP_RECLAIM: 70,
            PatternType.OVERSOLD_BOUNCE: 65,
            PatternType.NEW_HOD: 70,
            PatternType.PANIC_DIP_BUY: 70,
            PatternType.FAILED_BREAKDOWN: 75,
            PatternType.SUPPORT_BOUNCE: 70,
            PatternType.MOMENTUM_SURGE: 75,
            # Swing trading patterns
            PatternType.BULL_FLAG: 80,
            PatternType.BEAR_FLAG: 80,
            PatternType.FIB_RETRACEMENT: 85,
            PatternType.HIGHER_LOW: 75,
            PatternType.LOWER_HIGH: 75,
            PatternType.RANGE_BREAKOUT: 75,
            PatternType.PULLBACK_TO_SUPPORT: 70,
            PatternType.TREND_CONTINUATION: 75,
            PatternType.CHANNEL_BREAKOUT: 80,
            # Position trading patterns
            PatternType.GOLDEN_CROSS: 85,
            PatternType.DEATH_CROSS: 85,
            PatternType.WEEKLY_BREAKOUT: 80,
            PatternType.BASE_BREAKOUT: 85,
            PatternType.MAJOR_SUPPORT_TEST: 70,
            PatternType.TRENDLINE_BOUNCE: 75,
        }
        components.pattern_clarity = pattern_scores.get(pattern_type, 60)

        # Volume confirmation (0-100)
        volume_multiple = pattern_data.get("volume_multiple", 1.0)
        required_multiple = VOLUME_CONFIDENCE_MULTIPLIER.get(style, 1.5)

        if volume_multiple >= required_multiple * 1.5:
            components.volume_confirmation = 100
        elif volume_multiple >= required_multiple:
            components.volume_confirmation = 80
        elif volume_multiple >= 1.0:
            components.volume_confirmation = 60
        else:
            components.volume_confirmation = 40

        # Indicator alignment (0-100)
        # Now includes RSI and MACD signal alignment
        rsi_value = pattern_data.get("rsi_value")
        macd_signal = pattern_data.get("macd_signal") or stock_data.get("macd_signal")

        indicator_score = 60  # Default

        # RSI contribution
        if rsi_value:
            # RSI in healthy range (40-60) is optimal
            if 40 <= rsi_value <= 60:
                indicator_score = 75
            elif 30 <= rsi_value <= 70:
                indicator_score = 65
            else:
                indicator_score = 50

        # MACD signal alignment bonus
        if macd_signal:
            # Determine expected MACD direction based on pattern type
            bullish_patterns = {
                PatternType.GAP_UP, PatternType.OVERSOLD_BOUNCE, PatternType.NEW_HOD,
                PatternType.PANIC_DIP_BUY, PatternType.FAILED_BREAKDOWN, PatternType.SUPPORT_BOUNCE,
                PatternType.MOMENTUM_SURGE, PatternType.BULL_FLAG, PatternType.FIB_RETRACEMENT,
                PatternType.HIGHER_LOW, PatternType.RANGE_BREAKOUT, PatternType.PULLBACK_TO_SUPPORT,
                PatternType.TREND_CONTINUATION, PatternType.CHANNEL_BREAKOUT, PatternType.GOLDEN_CROSS,
                PatternType.WEEKLY_BREAKOUT, PatternType.BASE_BREAKOUT, PatternType.TRENDLINE_BOUNCE,
                PatternType.VWAP_RECLAIM,
            }
            bearish_patterns = {
                PatternType.GAP_DOWN, PatternType.BEAR_FLAG, PatternType.LOWER_HIGH,
                PatternType.DEATH_CROSS,
            }

            if pattern_type in bullish_patterns and macd_signal == "bullish":
                indicator_score = min(100, indicator_score + 15)  # Bonus for alignment
            elif pattern_type in bearish_patterns and macd_signal == "bearish":
                indicator_score = min(100, indicator_score + 15)  # Bonus for alignment
            elif pattern_type in bullish_patterns and macd_signal == "bearish":
                indicator_score = max(40, indicator_score - 10)  # Penalty for misalignment
            elif pattern_type in bearish_patterns and macd_signal == "bullish":
                indicator_score = max(40, indicator_score - 10)  # Penalty for misalignment

        components.indicator_alignment = indicator_score

        # Risk/reward ratio (0-100)
        # Based on key levels if available
        key_levels = self._extract_key_levels(pattern_data, stock_data)
        if "support" in key_levels and "resistance" in key_levels:
            current = stock_data.get("current_price", 0)
            support = key_levels["support"]
            resistance = key_levels["resistance"]

            if current > 0 and support > 0 and resistance > current:
                risk = current - support
                reward = resistance - current

                if risk > 0:
                    rr_ratio = reward / risk
                    if rr_ratio >= 3:
                        components.risk_reward_ratio = 100
                    elif rr_ratio >= 2:
                        components.risk_reward_ratio = 80
                    elif rr_ratio >= 1.5:
                        components.risk_reward_ratio = 60
                    else:
                        components.risk_reward_ratio = 40
                else:
                    components.risk_reward_ratio = 50
            else:
                components.risk_reward_ratio = 50
        else:
            components.risk_reward_ratio = 50

        # Fib level proximity (0-100)
        fib_level = pattern_data.get("fib_level")
        if fib_level:
            # Key Fib levels: 38.2%, 50%, 61.8%
            key_fibs = [38.2, 50.0, 61.8]
            min_distance = min(abs(fib_level - kf) for kf in key_fibs)

            if min_distance < 2:  # Very close to key level
                components.fib_level_proximity = 100
            elif min_distance < 5:
                components.fib_level_proximity = 80
            elif min_distance < 10:
                components.fib_level_proximity = 60
            else:
                components.fib_level_proximity = 40
        else:
            components.fib_level_proximity = 50  # Default

        return components

    # =========================================================================
    # Description generation
    # =========================================================================

    def _generate_description(
        self,
        style: TradingStyle,
        pattern_type: PatternType,
        stock_data: Dict,
        pattern_data: Dict,
    ) -> str:
        """Generate a human-readable description for a pattern.

        Uses template-based descriptions per the specification.

        Args:
            style: Trading style
            pattern_type: Type of pattern
            stock_data: Stock data
            pattern_data: Pattern-specific data

        Returns:
            Template-based description string
        """
        # Extract common values
        volume_multiple = pattern_data.get("volume_multiple", 1.0)
        gap_pct = pattern_data.get("gap_pct")
        fib_level = pattern_data.get("fib_level")
        rsi_value = pattern_data.get("rsi_value")
        vwap = pattern_data.get("vwap")
        resistance = pattern_data.get("resistance")
        support = pattern_data.get("support")
        days = pattern_data.get("days")
        weeks = pattern_data.get("weeks")
        months = pattern_data.get("months")
        level = pattern_data.get("level")

        # Day trade templates
        if style == TradingStyle.DAY:
            if pattern_type == PatternType.GAP_UP:
                return f"Gap up {gap_pct}%: {volume_multiple}x volume, watching ${support:.2f} support"
            elif pattern_type == PatternType.GAP_DOWN:
                return f"Gap down {abs(gap_pct)}%: Watching ${resistance:.2f} for fill"
            elif pattern_type == PatternType.GAP_FILL:
                fill_level = pattern_data.get("gap_fill_level", support)
                return f"Gap fill in progress: Target ${fill_level:.2f}"
            elif pattern_type == PatternType.RESISTANCE_BREAKOUT:
                return f"Breaking above ${resistance:.2f} with {volume_multiple}x volume surge"
            elif pattern_type == PatternType.VWAP_RECLAIM:
                return f"VWAP reclaim at ${vwap:.2f}: {volume_multiple}x volume"
            elif pattern_type == PatternType.OVERSOLD_BOUNCE:
                return f"Oversold bounce: RSI at {rsi_value:.0f}, reclaiming VWAP at ${vwap:.2f}"
            elif pattern_type == PatternType.NEW_HOD:
                return f"Momentum continuation: New HOD with {volume_multiple}x volume"
            elif pattern_type == PatternType.PANIC_DIP_BUY:
                return f"Panic dip buy: RSI at {rsi_value:.0f}, bouncing with {volume_multiple}x volume"
            elif pattern_type == PatternType.FAILED_BREAKDOWN:
                return f"Failed breakdown: Reclaimed ${support:.2f} support with strength"
            elif pattern_type == PatternType.SUPPORT_BOUNCE:
                return f"Support bounce: Touching ${support:.2f} with {volume_multiple}x volume"
            elif pattern_type == PatternType.MOMENTUM_SURGE:
                price_change = pattern_data.get("price_change_pct", 5)
                return f"Momentum surge: Up {price_change:.1f}% with {volume_multiple}x volume"

        # Swing trade templates
        elif style == TradingStyle.SWING:
            if pattern_type == PatternType.BULL_FLAG:
                return f"Bull flag forming: {days} days of consolidation near ${resistance:.2f}"
            elif pattern_type == PatternType.BEAR_FLAG:
                return f"Bear flag forming: {days} days of consolidation near ${support:.2f}"
            elif pattern_type == PatternType.FIB_RETRACEMENT:
                price = pattern_data.get("price", stock_data.get("current_price", 0))
                return f"Pullback to {fib_level:.1f}% Fib (${price:.2f}): Trend intact"
            elif pattern_type == PatternType.RANGE_BREAKOUT:
                return f"Breaking out of {days}-day range with volume"
            elif pattern_type == PatternType.HIGHER_LOW:
                return f"Higher low confirmed at ${support:.2f}: Uptrend resuming"
            elif pattern_type == PatternType.LOWER_HIGH:
                return f"Lower high confirmed at ${resistance:.2f}: Downtrend continuing"
            elif pattern_type == PatternType.CHANNEL_BREAKOUT:
                return f"Channel breakout: Above ${resistance:.2f} trendline"
            elif pattern_type == PatternType.PULLBACK_TO_SUPPORT:
                return f"Pullback to support at ${support:.2f}: RSI at {rsi_value:.0f}"
            elif pattern_type == PatternType.TREND_CONTINUATION:
                ema_21 = pattern_data.get("ema_21", 0)
                return f"Trend continuation: Price above 21 EMA (${ema_21:.2f})"

        # Position trade templates
        elif style == TradingStyle.POSITION:
            if pattern_type == PatternType.WEEKLY_BREAKOUT:
                return f"Weekly breakout: Above {weeks}-week resistance at ${level:.2f}"
            elif pattern_type == PatternType.MAJOR_SUPPORT_TEST:
                return f"Testing major support at ${level:.2f}: Potential reversal zone"
            elif pattern_type == PatternType.BASE_BREAKOUT:
                return f"Base breakout: {months} months of accumulation complete"
            elif pattern_type == PatternType.GOLDEN_CROSS:
                return f"Golden cross forming: 50 EMA crossing above 200 EMA"
            elif pattern_type == PatternType.DEATH_CROSS:
                return f"Death cross forming: 50 EMA crossing below 200 EMA"
            elif pattern_type == PatternType.TRENDLINE_BOUNCE:
                trendline = pattern_data.get("trendline_value", support)
                return f"Long-term uptrend: Bouncing off rising trendline at ${trendline:.2f}"

        # Default description
        return f"{pattern_type.value.replace('_', ' ').title()}: Setup detected"

    # =========================================================================
    # Helper methods
    # =========================================================================

    def _extract_key_levels(self, pattern_data: Dict, stock_data: Dict) -> Dict[str, float]:
        """Extract key price levels from pattern data.

        Args:
            pattern_data: Pattern-specific data
            stock_data: Stock data

        Returns:
            Dictionary of key levels (support, resistance, etc.)
        """
        levels = {}

        # Direct level mappings
        for key in ["support", "resistance", "entry", "stop", "target", "level", "vwap"]:
            if key in pattern_data and pattern_data[key]:
                levels[key] = round(pattern_data[key], 2)

        # Gap-specific levels
        if "gap_fill_level" in pattern_data:
            levels["target"] = round(pattern_data["gap_fill_level"], 2)

        # Ensure we have at least basic levels from stock data
        if "support" not in levels and stock_data.get("low_price"):
            levels["support"] = round(stock_data["low_price"], 2)
        if "resistance" not in levels and stock_data.get("high_price"):
            levels["resistance"] = round(stock_data["high_price"], 2)

        return levels

    def _check_warnings(self, stock_data: Dict) -> List[str]:
        """Check for warning conditions.

        Args:
            stock_data: Stock data

        Returns:
            List of warning messages
        """
        warnings = []

        # TODO: Implement earnings calendar check
        # Would check if earnings are within N days and add:
        # warnings.append(f"Earnings in {days} days")

        # TODO: Implement halted stock check
        # Would check trading status and add:
        # warnings.append("Halted")

        # TODO: Implement trading restriction check
        # warnings.append("Trading Restricted")

        return warnings

    async def _get_cached_response(self) -> AllScannersResponse:
        """Get cached response from store.

        Returns:
            AllScannersResponse with current cached results
        """
        # Handle both sync and async stores
        if asyncio.iscoroutinefunction(self._store.get_response):
            day_response = await self._store.get_response(TradingStyle.DAY)
            swing_response = await self._store.get_response(TradingStyle.SWING)
            position_response = await self._store.get_response(TradingStyle.POSITION)
        else:
            day_response = self._store.get_response(TradingStyle.DAY)
            swing_response = self._store.get_response(TradingStyle.SWING)
            position_response = self._store.get_response(TradingStyle.POSITION)

        return AllScannersResponse(
            day=day_response,
            swing=swing_response,
            position=position_response,
            scan_time=datetime.utcnow(),
            next_scheduled_scan=self._get_next_scheduled_scan_time(),
        )

    # =========================================================================
    # Scheduling
    # =========================================================================

    def _get_next_scheduled_scan_time(self) -> Optional[datetime]:
        """Get the next scheduled scan time.

        Returns:
            datetime of next scheduled scan, or None if no scans scheduled
        """
        now = datetime.now(ET)
        today_date = now.date()

        # Check if market is open today
        today_str = today_date.strftime("%Y-%m-%d")
        from app.services.scheduler import MARKET_HOLIDAYS

        if today_str in MARKET_HOLIDAYS or now.weekday() >= 5:
            # Market closed, find next trading day
            next_open = get_next_market_open()
            # Return first scan of next trading day
            return datetime.combine(next_open.date(), SCAN_SCHEDULE[0]["time"], tzinfo=ET)

        # Find next scan today
        for scan in SCAN_SCHEDULE:
            scan_time = datetime.combine(today_date, scan["time"], tzinfo=ET)
            if scan_time > now:
                return scan_time

        # All scans done today, return first scan of next trading day
        next_day = today_date + timedelta(days=1)
        while next_day.weekday() >= 5 or next_day.strftime("%Y-%m-%d") in MARKET_HOLIDAYS:
            next_day += timedelta(days=1)

        return datetime.combine(next_day, SCAN_SCHEDULE[0]["time"], tzinfo=ET)

    async def start_scheduled_scanning(self) -> None:
        """Start the scheduled scanning task."""
        if self._running:
            logger.warning("Scheduled scanning already running")
            return

        self._running = True
        self._scheduled_task = asyncio.create_task(self._scheduling_loop())
        logger.info("Market scanner scheduled scanning started")

    async def stop_scheduled_scanning(self) -> None:
        """Stop the scheduled scanning task."""
        if not self._running:
            return

        self._running = False
        if self._scheduled_task:
            self._scheduled_task.cancel()
            try:
                await self._scheduled_task
            except asyncio.CancelledError:
                pass
            self._scheduled_task = None

        logger.info("Market scanner scheduled scanning stopped")

    async def _scheduling_loop(self) -> None:
        """Main scheduling loop that runs scans at scheduled times."""
        logger.info("Scanner scheduling loop started")

        while self._running:
            now = datetime.now(ET)

            # Find the next scan
            next_scan_time = self._get_next_scheduled_scan_time()
            if not next_scan_time:
                await asyncio.sleep(60)
                continue

            # Calculate wait time
            wait_seconds = (next_scan_time - now).total_seconds()

            if wait_seconds > 0:
                # Wait until next scan time (check every minute in case of stop)
                wait_time = min(wait_seconds, 60)
                await asyncio.sleep(wait_time)
                continue

            # Time to run a scan
            # Find which scan this is
            scan_name = "on_demand"
            for scan in SCAN_SCHEDULE:
                scan_time = datetime.combine(now.date(), scan["time"], tzinfo=ET)
                if abs((scan_time - now).total_seconds()) < 60:  # Within 1 minute
                    scan_name = scan["name"]
                    break

            logger.info(f"Running scheduled {scan_name} scan")

            try:
                await self.run_scan(scan_name=scan_name)
            except Exception as e:
                logger.error(f"Scheduled scan error: {e}")

            # Wait a bit before checking for next scan
            await asyncio.sleep(60)

    async def get_status(self) -> ScannerStatusResponse:
        """Get current scanner status.

        Returns:
            ScannerStatusResponse with scan metadata
        """
        # Handle both sync (ScannerStore) and async (DatabaseScannerStore) stores
        store_status_result = self._store.get_status()

        # Check if it's a coroutine (async store)
        if asyncio.iscoroutine(store_status_result):
            store_status = await store_status_result
        else:
            store_status = store_status_result

        return ScannerStatusResponse(
            last_scan_time=store_status.get("last_scan_time"),
            next_scheduled_scan=self._get_next_scheduled_scan_time(),
            current_scan_name=store_status.get("current_scan_name"),
            is_scanning=self._is_scanning,
            total_results=store_status.get("total_results", {"day": 0, "swing": 0, "position": 0}),
        )


# Singleton instance
_scanner_service: Optional[MarketScannerService] = None


def get_scanner_service() -> MarketScannerService:
    """Get the singleton scanner service instance."""
    global _scanner_service
    if _scanner_service is None:
        _scanner_service = MarketScannerService()
    return _scanner_service
