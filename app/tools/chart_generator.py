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
    fibonacci_levels: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate candlestick chart with indicators.

    Creates a professional trading chart with:
    - Candlestick price action
    - EMA lines (9, 21, 50 period)
    - Volume bars
    - RSI indicator subplot
    - Fibonacci retracement and extension levels (optional)

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
        fibonacci_levels: Optional dictionary containing Fibonacci levels:
            - swing_high: float - Swing high price
            - swing_low: float - Swing low price
            - retracement: dict - Retracement levels (0.236, 0.382, 0.500, 0.618, 0.786)
            - extension: dict - Extension levels (1.272, 1.618, 2.618)
            - trend: str - "uptrend" or "downtrend"

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

        # Plot Fibonacci levels if provided
        fib_labels = []
        if fibonacci_levels:
            try:
                # Get the price axis (main chart)
                price_ax = axes[0]

                # Get the visible price range for filtering
                y_min, y_max = price_ax.get_ylim()

                trend = fibonacci_levels.get('trend', 'uptrend')

                # Plot retracement levels (dashed lines)
                if 'retracement' in fibonacci_levels:
                    for level_name, price in fibonacci_levels['retracement'].items():
                        if price and y_min <= price <= y_max:
                            # Color logic based on level significance:
                            # In UPTREND: Higher % levels (0.618, 0.786) are near swing low = SUPPORT (green)
                            #             Lower % levels (0.236, 0.382) are near swing high = RESISTANCE (red)
                            # In DOWNTREND: Inverse - lower levels are support, higher are resistance
                            level_value = float(level_name)
                            if trend == 'uptrend':
                                # Deep retracements (>50%) = strong support, shallow (<50%) = resistance zones
                                color = '#4CAF50' if level_value > 0.5 else '#FF5722'
                            else:
                                # In downtrend: deep retracements are resistance, shallow are support
                                color = '#FF5722' if level_value > 0.5 else '#4CAF50'

                            # Draw horizontal line
                            price_ax.axhline(
                                y=price,
                                color=color,
                                linestyle='--',
                                linewidth=1.2,
                                alpha=0.7,
                                zorder=2
                            )

                            # Add label on the right side
                            price_ax.text(
                                1.002, price, f' {level_name} - ${price:.2f}',
                                transform=price_ax.get_yaxis_transform(),
                                fontsize=7,
                                verticalalignment='center',
                                color=color,
                                weight='bold'
                            )
                            fib_labels.append(f'Fib {level_name}')

                # Plot extension levels (dotted lines)
                if 'extension' in fibonacci_levels:
                    for level_name, price in fibonacci_levels['extension'].items():
                        if price and y_min <= price <= y_max:
                            # Extension levels: targets in trend direction
                            level_value = float(level_name)
                            if trend == 'uptrend':
                                color = '#2196F3'  # Blue for upside targets
                            else:
                                color = '#9C27B0'  # Purple for downside targets

                            # Draw horizontal line
                            price_ax.axhline(
                                y=price,
                                color=color,
                                linestyle=':',
                                linewidth=1.5,
                                alpha=0.7,
                                zorder=2
                            )

                            # Add label on the right side
                            price_ax.text(
                                1.002, price, f' {level_name} - ${price:.2f}',
                                transform=price_ax.get_yaxis_transform(),
                                fontsize=7,
                                verticalalignment='center',
                                color=color,
                                weight='bold'
                            )
                            fib_labels.append(f'Fib Ext {level_name}')

                logger.info(f"Plotted {len(fib_labels)} Fibonacci levels for {symbol}")
            except Exception as e:
                logger.warning(f"Failed to plot Fibonacci levels: {e}")

        # Add legend for EMA lines and Fibonacci
        legend_items = []
        if indicators:
            if 'ema_9' in indicators and indicators['ema_9']:
                legend_items.append('EMA 9')
            if 'ema_21' in indicators and indicators['ema_21']:
                legend_items.append('EMA 21')
            if 'ema_50' in indicators and indicators['ema_50']:
                legend_items.append('EMA 50')

        if fib_labels:
            legend_items.append('Fibonacci Levels')

        if legend_items:
            axes[0].legend(legend_items, loc='upper left', fontsize=8)

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
