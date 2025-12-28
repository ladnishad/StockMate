"""Device token storage for push notifications.

Manages APNs device tokens for iOS push notifications.
Supports multiple devices per user.
"""

import logging
import uuid
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel

from app.storage.database import get_database

logger = logging.getLogger(__name__)


class DeviceToken(BaseModel):
    """Device token data model."""

    id: str
    user_id: str
    device_token: str
    platform: str = "ios"
    created_at: str
    last_used: Optional[str] = None


class DeviceStore:
    """Manages device tokens in SQLite."""

    def __init__(self):
        """Initialize device store."""
        self.db = get_database()

    async def register_device(
        self,
        user_id: str,
        device_token: str,
        platform: str = "ios",
    ) -> DeviceToken:
        """Register a device for push notifications.

        If device token already exists, updates the user_id association.

        Args:
            user_id: User identifier
            device_token: APNs device token
            platform: Platform (ios, android)

        Returns:
            DeviceToken object
        """
        device_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            # Upsert - update if exists, insert if not
            await conn.execute(
                """
                INSERT INTO device_tokens (id, user_id, device_token, platform, created_at, last_used)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(device_token) DO UPDATE SET
                    user_id = excluded.user_id,
                    last_used = excluded.last_used
                """,
                (device_id, user_id, device_token, platform, now, now),
            )
            await conn.commit()

        logger.info(f"Registered device for user {user_id} ({platform})")

        return DeviceToken(
            id=device_id,
            user_id=user_id,
            device_token=device_token,
            platform=platform,
            created_at=now,
            last_used=now,
        )

    async def unregister_device(self, device_token: str) -> bool:
        """Remove a device from push notifications.

        Args:
            device_token: APNs device token

        Returns:
            True if removed, False if not found
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM device_tokens WHERE device_token = ?",
                (device_token,),
            )
            await conn.commit()
            deleted = cursor.rowcount > 0

        if deleted:
            logger.info(f"Unregistered device token")

        return deleted

    async def get_user_devices(self, user_id: str) -> List[DeviceToken]:
        """Get all registered devices for a user.

        Args:
            user_id: User identifier

        Returns:
            List of DeviceToken objects
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM device_tokens WHERE user_id = ? ORDER BY last_used DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()
            return [self._row_to_device(row) for row in rows]

    async def get_device_tokens_for_user(self, user_id: str) -> List[str]:
        """Get just the token strings for a user (for sending notifications).

        Args:
            user_id: User identifier

        Returns:
            List of device token strings
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "SELECT device_token FROM device_tokens WHERE user_id = ?",
                (user_id,),
            )
            rows = await cursor.fetchall()
            return [row["device_token"] for row in rows]

    async def update_last_used(self, device_token: str) -> None:
        """Update the last_used timestamp for a device.

        Args:
            device_token: APNs device token
        """
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            await conn.execute(
                "UPDATE device_tokens SET last_used = ? WHERE device_token = ?",
                (now, device_token),
            )
            await conn.commit()

    async def cleanup_stale_devices(self, days: int = 90) -> int:
        """Remove devices that haven't been used in a while.

        Args:
            days: Remove devices not used in this many days

        Returns:
            Number of devices removed
        """
        from datetime import timedelta

        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM device_tokens WHERE last_used < ?",
                (cutoff,),
            )
            await conn.commit()

            deleted = cursor.rowcount
            if deleted > 0:
                logger.info(f"Cleaned up {deleted} stale device tokens")
            return deleted

    async def get_all_active_users(self) -> List[str]:
        """Get list of all user IDs with registered devices.

        Returns:
            List of unique user IDs
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "SELECT DISTINCT user_id FROM device_tokens",
            )
            rows = await cursor.fetchall()
            return [row["user_id"] for row in rows]

    def _row_to_device(self, row) -> DeviceToken:
        """Convert database row to DeviceToken object."""
        return DeviceToken(
            id=row["id"],
            user_id=row["user_id"],
            device_token=row["device_token"],
            platform=row["platform"],
            created_at=row["created_at"],
            last_used=row["last_used"],
        )


# Singleton instance
_store: Optional[DeviceStore] = None


def get_device_store() -> DeviceStore:
    """Get the singleton device store instance."""
    global _store
    if _store is None:
        _store = DeviceStore()
    return _store
