"""Services module for background processing and monitoring."""

from app.services.scheduler import (
    MarketHoursScheduler,
    get_scheduler,
    is_market_open,
    get_next_market_open,
    get_market_close_time,
    seconds_until_market_open,
    seconds_until_market_close,
    get_market_status,
)
from app.services.alpaca_clock import (
    AlpacaClockService,
    get_clock_service,
    get_market_clock,
    MarketClockData,
)
from app.services.price_monitor import PriceMonitor, TriggerEvent, get_price_monitor
from app.services.push_notification import (
    APNsService,
    get_apns_service,
    send_trading_alert,
)
from app.services.plan_evaluator import PlanEvaluator, get_plan_evaluator

__all__ = [
    # Market Hours Scheduler
    "MarketHoursScheduler",
    "get_scheduler",
    "is_market_open",
    "get_next_market_open",
    "get_market_close_time",
    "seconds_until_market_open",
    "seconds_until_market_close",
    "get_market_status",
    # Alpaca Clock Service
    "AlpacaClockService",
    "get_clock_service",
    "get_market_clock",
    "MarketClockData",
    # Price Monitor
    "PriceMonitor",
    "TriggerEvent",
    "get_price_monitor",
    # Push Notifications
    "APNsService",
    "get_apns_service",
    "send_trading_alert",
    # Plan Evaluator
    "PlanEvaluator",
    "get_plan_evaluator",
]
