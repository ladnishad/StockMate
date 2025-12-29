"""Watchlist service with comprehensive cleanup.

Handles watchlist operations that require coordinated cleanup
across multiple data stores.
"""

import logging
from typing import TypedDict

from app.storage.watchlist_store import get_watchlist_store
from app.storage.plan_store import get_plan_store
from app.storage.position_store import get_position_store
from app.storage.conversation_store import get_conversation_store
from app.storage.alert_history import get_alert_history
from app.storage.database import delete_agent_context

logger = logging.getLogger(__name__)


class CleanupResults(TypedDict):
    """Results from symbol cleanup operation."""

    watchlist: bool
    trading_plan: bool
    position: bool
    conversations: bool
    alerts: int
    agent_context: bool


async def remove_symbol_with_cleanup(user_id: str, symbol: str) -> CleanupResults:
    """Remove symbol from watchlist and clean up all related data.

    This function ensures complete data cleanup when a user removes
    a stock from their watchlist, including:
    - Watchlist entry
    - Trading plans
    - Positions
    - Conversation history
    - Alert history
    - Agent context cache

    Args:
        user_id: User identifier
        symbol: Stock ticker symbol

    Returns:
        CleanupResults dict showing what was deleted
    """
    symbol = symbol.upper()

    results: CleanupResults = {
        "watchlist": False,
        "trading_plan": False,
        "position": False,
        "conversations": False,
        "alerts": 0,
        "agent_context": False,
    }

    # 1. Remove from watchlist (primary operation)
    store = get_watchlist_store()
    # Handle both sync (JSON) and async (Postgres) stores
    remove_result = store.remove_symbol(user_id, symbol)
    if hasattr(remove_result, "__await__"):
        results["watchlist"] = await remove_result
    else:
        results["watchlist"] = remove_result

    # 2. Delete trading plan (if exists)
    try:
        plan_store = get_plan_store()
        results["trading_plan"] = await plan_store.delete_plan(user_id, symbol)
    except Exception as e:
        logger.warning(f"Failed to delete trading plan for {symbol}: {e}")

    # 3. Delete position (if exists)
    try:
        position_store = get_position_store()
        results["position"] = await position_store.delete_position(user_id, symbol)
    except Exception as e:
        logger.warning(f"Failed to delete position for {symbol}: {e}")

    # 4. Clear conversation history
    try:
        conversation_store = get_conversation_store()
        results["conversations"] = await conversation_store.clear_conversation(
            user_id, symbol
        )
    except Exception as e:
        logger.warning(f"Failed to clear conversations for {symbol}: {e}")

    # 5. Delete alerts
    try:
        alert_history = get_alert_history()
        results["alerts"] = await alert_history.delete_alerts_for_symbol(
            user_id, symbol
        )
    except Exception as e:
        logger.warning(f"Failed to delete alerts for {symbol}: {e}")

    # 6. Clear agent context cache
    try:
        results["agent_context"] = await delete_agent_context(symbol)
    except Exception as e:
        logger.warning(f"Failed to delete agent context for {symbol}: {e}")

    logger.info(f"Cleanup for {symbol} (user: {user_id}): {results}")
    return results
