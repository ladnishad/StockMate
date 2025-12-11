"""Real-time WebSocket streaming using AlgoTrader Plus SIP feed.

This module provides real-time market data streaming capabilities using
Alpaca's WebSocket API. With AlgoTrader Plus subscription, data comes from
the SIP feed (all US exchanges) with no delay.

Available streams:
- Quotes: Real-time bid/ask updates
- Trades: Real-time trade executions
- Bars: Real-time minute bar updates
- Daily Bars: Daily bar updates throughout the trading day

Usage:
    from app.tools.streaming import get_streamer

    streamer = get_streamer()

    async def quote_handler(quote):
        print(f"{quote.symbol}: {quote.bid_price} / {quote.ask_price}")

    streamer.subscribe_quotes(quote_handler, "AAPL", "MSFT")
    streamer.run()  # Blocking - runs the event loop
"""

import logging
import asyncio
from typing import Callable, List, Optional, Set, Dict, Any
from threading import Thread

from alpaca.data.live import StockDataStream
from alpaca.data.enums import DataFeed

from app.config import get_settings

logger = logging.getLogger(__name__)


class StockStreamer:
    """Manages real-time stock data streaming from Alpaca.

    This class wraps the alpaca-py StockDataStream to provide a simplified
    interface for subscribing to real-time market data.

    With AlgoTrader Plus subscription, data comes from the SIP feed which
    includes all US exchanges with no delay.

    Attributes:
        stream: The underlying Alpaca StockDataStream
        subscribed_quotes: Set of symbols subscribed to quote updates
        subscribed_trades: Set of symbols subscribed to trade updates
        subscribed_bars: Set of symbols subscribed to bar updates
    """

    def __init__(self):
        """Initialize the stock streamer with Alpaca credentials."""
        settings = get_settings()

        # Determine feed based on config
        feed = DataFeed.SIP if settings.alpaca_data_feed == "sip" else DataFeed.IEX

        self.stream = StockDataStream(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            feed=feed,
        )

        self._running = False
        self._thread: Optional[Thread] = None

        # Track subscriptions
        self.subscribed_quotes: Set[str] = set()
        self.subscribed_trades: Set[str] = set()
        self.subscribed_bars: Set[str] = set()
        self.subscribed_daily_bars: Set[str] = set()

        # Store handlers
        self._quote_handlers: Dict[str, List[Callable]] = {}
        self._trade_handlers: Dict[str, List[Callable]] = {}
        self._bar_handlers: Dict[str, List[Callable]] = {}
        self._daily_bar_handlers: Dict[str, List[Callable]] = {}

        logger.info(f"StockStreamer initialized with {feed.value} feed")

    def subscribe_quotes(
        self,
        handler: Callable,
        *symbols: str,
    ) -> None:
        """Subscribe to real-time quote updates for symbols.

        Args:
            handler: Async function to call with each quote update.
                    Receives an alpaca Quote object with bid_price, ask_price,
                    bid_size, ask_size, timestamp, etc.
            *symbols: One or more stock symbols, or "*" for all symbols
        """
        self.stream.subscribe_quotes(handler, *symbols)
        self.subscribed_quotes.update(symbols)
        logger.info(f"Subscribed to quotes for: {symbols}")

    def subscribe_trades(
        self,
        handler: Callable,
        *symbols: str,
    ) -> None:
        """Subscribe to real-time trade updates for symbols.

        Args:
            handler: Async function to call with each trade update.
                    Receives an alpaca Trade object with price, size,
                    exchange, timestamp, conditions, etc.
            *symbols: One or more stock symbols, or "*" for all symbols
        """
        self.stream.subscribe_trades(handler, *symbols)
        self.subscribed_trades.update(symbols)
        logger.info(f"Subscribed to trades for: {symbols}")

    def subscribe_bars(
        self,
        handler: Callable,
        *symbols: str,
    ) -> None:
        """Subscribe to real-time minute bar updates for symbols.

        Bars are emitted at the end of each minute with OHLCV data.

        Args:
            handler: Async function to call with each bar update.
                    Receives an alpaca Bar object with open, high, low,
                    close, volume, timestamp, vwap, etc.
            *symbols: One or more stock symbols, or "*" for all symbols
        """
        self.stream.subscribe_bars(handler, *symbols)
        self.subscribed_bars.update(symbols)
        logger.info(f"Subscribed to bars for: {symbols}")

    def subscribe_daily_bars(
        self,
        handler: Callable,
        *symbols: str,
    ) -> None:
        """Subscribe to daily bar updates for symbols.

        Daily bars are updated throughout the trading day.

        Args:
            handler: Async function to call with each daily bar update.
            *symbols: One or more stock symbols, or "*" for all symbols
        """
        self.stream.subscribe_daily_bars(handler, *symbols)
        self.subscribed_daily_bars.update(symbols)
        logger.info(f"Subscribed to daily bars for: {symbols}")

    def unsubscribe_quotes(self, *symbols: str) -> None:
        """Unsubscribe from quote updates for symbols."""
        self.stream.unsubscribe_quotes(*symbols)
        self.subscribed_quotes.difference_update(symbols)
        logger.info(f"Unsubscribed from quotes for: {symbols}")

    def unsubscribe_trades(self, *symbols: str) -> None:
        """Unsubscribe from trade updates for symbols."""
        self.stream.unsubscribe_trades(*symbols)
        self.subscribed_trades.difference_update(symbols)
        logger.info(f"Unsubscribed from trades for: {symbols}")

    def unsubscribe_bars(self, *symbols: str) -> None:
        """Unsubscribe from bar updates for symbols."""
        self.stream.unsubscribe_bars(*symbols)
        self.subscribed_bars.difference_update(symbols)
        logger.info(f"Unsubscribed from bars for: {symbols}")

    def run(self) -> None:
        """Start the streaming connection (blocking).

        This method blocks and runs the WebSocket event loop.
        Use run_in_background() for non-blocking operation.
        """
        if self._running:
            logger.warning("Streamer is already running")
            return

        logger.info("Starting stock data stream...")
        self._running = True

        try:
            self.stream.run()
        except Exception as e:
            logger.error(f"Stream error: {e}")
            self._running = False
            raise

    def run_in_background(self) -> None:
        """Start the streaming connection in a background thread.

        This allows the stream to run without blocking the main thread.
        """
        if self._running:
            logger.warning("Streamer is already running")
            return

        def _run():
            try:
                self._running = True
                self.stream.run()
            except Exception as e:
                logger.error(f"Background stream error: {e}")
            finally:
                self._running = False

        self._thread = Thread(target=_run, daemon=True)
        self._thread.start()
        logger.info("Stock data stream started in background")

    def stop(self) -> None:
        """Stop the streaming connection."""
        if not self._running:
            logger.warning("Streamer is not running")
            return

        logger.info("Stopping stock data stream...")
        self.stream.stop()
        self._running = False

        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None

    @property
    def is_running(self) -> bool:
        """Check if the streamer is currently running."""
        return self._running

    def get_subscription_status(self) -> Dict[str, Any]:
        """Get current subscription status.

        Returns:
            Dictionary with subscription details
        """
        return {
            "running": self._running,
            "quotes": list(self.subscribed_quotes),
            "trades": list(self.subscribed_trades),
            "bars": list(self.subscribed_bars),
            "daily_bars": list(self.subscribed_daily_bars),
        }


# Singleton instance
_streamer: Optional[StockStreamer] = None


def get_streamer() -> StockStreamer:
    """Get or create the singleton StockStreamer instance.

    Returns:
        The shared StockStreamer instance

    Example:
        streamer = get_streamer()
        streamer.subscribe_quotes(my_handler, "AAPL")
        streamer.run_in_background()
    """
    global _streamer
    if _streamer is None:
        _streamer = StockStreamer()
    return _streamer


def reset_streamer() -> None:
    """Reset the singleton streamer instance.

    Useful for testing or when you need to recreate the connection.
    """
    global _streamer
    if _streamer is not None:
        if _streamer.is_running:
            _streamer.stop()
        _streamer = None
    logger.info("Streamer instance reset")
