"""Alert history storage.

Tracks sent alerts for:
- Deduplication (avoid sending same alert twice)
- Audit trail (review what was sent)
- User acknowledgment
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Literal
from pydantic import BaseModel

from app.storage.database import get_database

logger = logging.getLogger(__name__)

AlertType = Literal["BUY", "STOP", "SELL"]


class Alert(BaseModel):
    """Alert data model."""

    id: str
    user_id: str
    symbol: str
    alert_type: AlertType
    message: str
    price_at_alert: float
    sent_at: str
    acknowledged: bool = False


class AlertHistory:
    """Manages alert history in SQLite."""

    def __init__(self, cooldown_minutes: int = 15):
        """Initialize alert history.

        Args:
            cooldown_minutes: Minimum minutes between same alert type for same stock
        """
        self.db = get_database()
        self.cooldown_minutes = cooldown_minutes

    async def can_send_alert(
        self,
        user_id: str,
        symbol: str,
        alert_type: AlertType,
    ) -> bool:
        """Check if an alert can be sent (not in cooldown).

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            alert_type: Type of alert (BUY, STOP, SELL)

        Returns:
            True if alert can be sent, False if in cooldown
        """
        symbol = symbol.upper()
        cutoff = (datetime.utcnow() - timedelta(minutes=self.cooldown_minutes)).isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT COUNT(*) as count FROM alerts
                WHERE user_id = ? AND symbol = ? AND alert_type = ? AND sent_at > ?
                """,
                (user_id, symbol, alert_type, cutoff),
            )
            row = await cursor.fetchone()
            count = row["count"] if row else 0

            if count > 0:
                logger.debug(f"Alert cooldown active for {symbol} {alert_type}")
                return False

            return True

    async def record_alert(
        self,
        user_id: str,
        symbol: str,
        alert_type: AlertType,
        message: str,
        price_at_alert: float,
    ) -> Alert:
        """Record a sent alert.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            alert_type: Type of alert (BUY, STOP, SELL)
            message: Full alert message
            price_at_alert: Price when alert was generated

        Returns:
            Created Alert object
        """
        symbol = symbol.upper()
        alert_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO alerts (id, user_id, symbol, alert_type, message, price_at_alert, sent_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (alert_id, user_id, symbol, alert_type, message, price_at_alert, now),
            )
            await conn.commit()

        logger.info(f"Recorded {alert_type} alert for {symbol} @ ${price_at_alert}")

        return Alert(
            id=alert_id,
            user_id=user_id,
            symbol=symbol,
            alert_type=alert_type,
            message=message,
            price_at_alert=price_at_alert,
            sent_at=now,
            acknowledged=False,
        )

    async def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged.

        Args:
            alert_id: Alert identifier

        Returns:
            True if updated, False if not found
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "UPDATE alerts SET acknowledged = 1 WHERE id = ?",
                (alert_id,),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def get_recent_alerts(
        self,
        user_id: str,
        limit: int = 50,
        symbol: Optional[str] = None,
    ) -> List[Alert]:
        """Get recent alerts for a user.

        Args:
            user_id: User identifier
            limit: Maximum alerts to return
            symbol: Optional filter by symbol

        Returns:
            List of Alert objects, most recent first
        """
        async with self.db.connection() as conn:
            if symbol:
                symbol = symbol.upper()
                cursor = await conn.execute(
                    """
                    SELECT * FROM alerts
                    WHERE user_id = ? AND symbol = ?
                    ORDER BY sent_at DESC
                    LIMIT ?
                    """,
                    (user_id, symbol, limit),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM alerts
                    WHERE user_id = ?
                    ORDER BY sent_at DESC
                    LIMIT ?
                    """,
                    (user_id, limit),
                )

            rows = await cursor.fetchall()
            return [self._row_to_alert(row) for row in rows]

    async def get_last_alert(
        self,
        user_id: str,
        symbol: str,
        alert_type: Optional[AlertType] = None,
    ) -> Optional[Alert]:
        """Get the most recent alert for a symbol.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            alert_type: Optional filter by alert type

        Returns:
            Most recent Alert or None
        """
        symbol = symbol.upper()

        async with self.db.connection() as conn:
            if alert_type:
                cursor = await conn.execute(
                    """
                    SELECT * FROM alerts
                    WHERE user_id = ? AND symbol = ? AND alert_type = ?
                    ORDER BY sent_at DESC
                    LIMIT 1
                    """,
                    (user_id, symbol, alert_type),
                )
            else:
                cursor = await conn.execute(
                    """
                    SELECT * FROM alerts
                    WHERE user_id = ? AND symbol = ?
                    ORDER BY sent_at DESC
                    LIMIT 1
                    """,
                    (user_id, symbol),
                )

            row = await cursor.fetchone()
            return self._row_to_alert(row) if row else None

    async def cleanup_old_alerts(self, days: int = 30) -> int:
        """Delete alerts older than specified days.

        Args:
            days: Delete alerts older than this many days

        Returns:
            Number of alerts deleted
        """
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM alerts WHERE sent_at < ?",
                (cutoff,),
            )
            await conn.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} alerts older than {days} days")
            return deleted

    async def delete_alerts_for_symbol(self, user_id: str, symbol: str) -> int:
        """Delete all alerts for a specific symbol.

        Used when removing a stock from watchlist to clean up associated data.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol

        Returns:
            Number of alerts deleted
        """
        symbol = symbol.upper()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM alerts WHERE user_id = ? AND symbol = ?",
                (user_id, symbol),
            )
            await conn.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Deleted {deleted} alerts for {symbol} (user: {user_id})")
            return deleted

    def _row_to_alert(self, row) -> Alert:
        """Convert database row to Alert object."""
        return Alert(
            id=row["id"],
            user_id=row["user_id"],
            symbol=row["symbol"],
            alert_type=row["alert_type"],
            message=row["message"],
            price_at_alert=row["price_at_alert"],
            sent_at=row["sent_at"],
            acknowledged=bool(row["acknowledged"]),
        )


# Singleton instance
_history: Optional[AlertHistory] = None


def get_alert_history(cooldown_minutes: int = 15) -> AlertHistory:
    """Get the singleton alert history instance."""
    global _history
    if _history is None:
        _history = AlertHistory(cooldown_minutes=cooldown_minutes)
    return _history
