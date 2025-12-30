"""Generate candlestick charts for visual analysis.

This module creates professional candlestick charts with technical indicators
that can be sent to Claude Vision for visual pattern recognition.
"""

import base64
import logging
from io import BytesIO
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd
import numpy as np

from app.models.data import PriceBar

logger = logging.getLogger(__name__)


def generate_chart_image(
    symbol: str,
    bars: List[PriceBar],
    indicators: Optional[Dict[str, Any]] = None,
    lookback: int = 60,
    show_volume: bool = True,
    show_rsi: bool = True,
) -> str:
    """Generate candlestick chart with indicators.

    Creates a professional trading chart with:
    - Candlestick price action
    - EMA lines (9, 21, 50 period)
    - Volume bars
    - RSI indicator subplot

    Args:
        symbol: Stock ticker symbol
        bars: List of PriceBar objects with OHLCV data
        indicators: Dictionary containing indicator values:
            - ema_9: List of 9-period EMA values
            - ema_21: List of 21-period EMA values
            - ema_50: List of 50-period EMA values
            - rsi: List of RSI values
        lookback: Number of bars to show (default 60)
        show_volume: Whether to show volume bars (default True)
        show_rsi: Whether to show RSI subplot (default True)

    Returns:
        Base64-encoded PNG image string

    Raises:
        ValueError: If insufficient data or invalid bars
    """
    if not bars:
        raise ValueError("No price bars provided")

    if len(bars) < 10:
        raise ValueError(f"Insufficient data: need at least 10 bars, got {len(bars)}")

    # Limit to lookback period
    bars_to_plot = bars[-lookback:] if len(bars) > lookback else bars

    logger.info(f"Generating chart for {symbol} with {len(bars_to_plot)} bars")

    try:
        # Convert bars to DataFrame
        df = pd.DataFrame([{
            'Date': bar.timestamp,
            'Open': bar.open,
            'High': bar.high,
            'Low': bar.low,
            'Close': bar.close,
            'Volume': bar.volume
        } for bar in bars_to_plot])

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

        # Create additional plots for indicators
        add_plots = []

        if indicators:
            # EMA lines
            if 'ema_9' in indicators and indicators['ema_9']:
                ema9_data = indicators['ema_9'][-len(df):]
                if len(ema9_data) == len(df):
                    ema9 = pd.Series(ema9_data, index=df.index)
                    add_plots.append(mpf.make_addplot(
                        ema9, color='#2196F3', width=1.2, label='EMA 9'
                    ))

            if 'ema_21' in indicators and indicators['ema_21']:
                ema21_data = indicators['ema_21'][-len(df):]
                if len(ema21_data) == len(df):
                    ema21 = pd.Series(ema21_data, index=df.index)
                    add_plots.append(mpf.make_addplot(
                        ema21, color='#FF9800', width=1.2, label='EMA 21'
                    ))

            if 'ema_50' in indicators and indicators['ema_50']:
                ema50_data = indicators['ema_50'][-len(df):]
                if len(ema50_data) == len(df):
                    ema50 = pd.Series(ema50_data, index=df.index)
                    add_plots.append(mpf.make_addplot(
                        ema50, color='#F44336', width=1.5, label='EMA 50'
                    ))

            # RSI subplot
            if show_rsi and 'rsi' in indicators and indicators['rsi']:
                rsi_data = indicators['rsi'][-len(df):]
                if len(rsi_data) == len(df):
                    rsi = pd.Series(rsi_data, index=df.index)
                    add_plots.append(mpf.make_addplot(
                        rsi, panel=2, color='#9C27B0', ylabel='RSI', ylim=(0, 100)
                    ))
                    # Add RSI overbought/oversold lines
                    add_plots.append(mpf.make_addplot(
                        pd.Series([70] * len(df), index=df.index),
                        panel=2, color='#F44336', linestyle='--', width=0.8
                    ))
                    add_plots.append(mpf.make_addplot(
                        pd.Series([30] * len(df), index=df.index),
                        panel=2, color='#4CAF50', linestyle='--', width=0.8
                    ))

        # Create chart style
        mc = mpf.make_marketcolors(
            up='#26A69A',      # Green for up candles
            down='#EF5350',    # Red for down candles
            edge='inherit',
            wick='inherit',
            volume={'up': '#26A69A', 'down': '#EF5350'},
        )

        style = mpf.make_mpf_style(
            base_mpf_style='yahoo',
            marketcolors=mc,
            gridstyle='-',
            gridcolor='#E0E0E0',
            y_on_right=True,
            facecolor='white',
            figcolor='white',
        )

        # Determine panel ratios based on what's shown
        if show_rsi and indicators and 'rsi' in indicators:
            panel_ratios = (5, 1.5, 2)  # Price, Volume, RSI
        elif show_volume:
            panel_ratios = (5, 1.5)  # Price, Volume
        else:
            panel_ratios = (1,)  # Price only

        # Generate chart to buffer
        # Note: Per Anthropic Vision docs, optimal image size is within 1568x1568 px
        # Using figsize=(10.5, 7) at dpi=100 gives ~1050x700 px, well within limits
        # This avoids resize latency while maintaining good visual quality
        buf = BytesIO()

        fig, axes = mpf.plot(
            df,
            type='candle',
            style=style,
            volume=show_volume,
            addplot=add_plots if add_plots else None,
            title=f'\n{symbol} - Daily Chart',
            ylabel='Price ($)',
            ylabel_lower='Volume',
            savefig=dict(fname=buf, dpi=100, bbox_inches='tight', pad_inches=0.2),
            panel_ratios=panel_ratios,
            figsize=(10.5, 7),
            returnfig=True,
        )

        # Add legend for EMA lines
        if add_plots:
            axes[0].legend(['EMA 9', 'EMA 21', 'EMA 50'], loc='upper left', fontsize=8)

        plt.close(fig)

        # Convert to base64
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()

        logger.info(f"Chart generated successfully for {symbol}")
        return image_base64

    except Exception as e:
        logger.error(f"Failed to generate chart for {symbol}: {e}")
        raise


def generate_simple_chart(
    symbol: str,
    bars: List[PriceBar],
    lookback: int = 60,
) -> str:
    """Generate a simple candlestick chart without indicators.

    Useful for quick visual checks or when indicator data isn't available.

    Args:
        symbol: Stock ticker symbol
        bars: List of PriceBar objects
        lookback: Number of bars to show

    Returns:
        Base64-encoded PNG image string
    """
    return generate_chart_image(
        symbol=symbol,
        bars=bars,
        indicators=None,
        lookback=lookback,
        show_volume=True,
        show_rsi=False,
    )
