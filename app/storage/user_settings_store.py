"""User settings storage.

Manages user preferences like AI provider selection.
"""

import logging
from datetime import datetime
from typing import Optional
from pydantic import BaseModel

from app.storage.database import get_database

logger = logging.getLogger(__name__)


class UserSettings(BaseModel):
    """User settings data model."""

    user_id: str
    model_provider: str = "grok"  # Default to Grok for real-time X/Twitter sentiment
    is_admin: bool = False  # Admin flag for viewing usage data
    created_at: str
    updated_at: str


class UserSettingsStore:
    """Manages user settings in the database."""

    def __init__(self):
        """Initialize user settings store."""
        self.db = get_database()

    async def get_settings(self, user_id: str) -> UserSettings:
        """Get settings for a user.

        Returns default settings if none exist.

        Args:
            user_id: User identifier

        Returns:
            UserSettings object
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            row = await cursor.fetchone()

            if row:
                return self._row_to_settings(row)

        # Return default settings if none exist
        # Check if user is admin via environment variable
        from app.config import get_settings
        settings = get_settings()
        is_admin = user_id.lower() in settings.admin_user_list

        now = datetime.utcnow().isoformat()
        return UserSettings(
            user_id=user_id,
            model_provider="grok",  # Default to Grok
            is_admin=is_admin,
            created_at=now,
            updated_at=now,
        )

    async def update_provider(self, user_id: str, provider: str) -> UserSettings:
        """Update the AI provider preference for a user.

        Args:
            user_id: User identifier
            provider: AI provider ("claude" or "grok")

        Returns:
            Updated UserSettings object
        """
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            # Upsert - update if exists, insert if not
            await conn.execute(
                """
                INSERT INTO user_settings (user_id, model_provider, created_at, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    model_provider = excluded.model_provider,
                    updated_at = excluded.updated_at
                """,
                (user_id, provider, now, now),
            )
            await conn.commit()

        logger.info(f"Updated AI provider for user {user_id} to {provider}")

        return UserSettings(
            user_id=user_id,
            model_provider=provider,
            created_at=now,
            updated_at=now,
        )

    async def is_user_admin(self, user_id: str, email: Optional[str] = None) -> bool:
        """Check if a user has admin privileges.

        Admin status is determined by:
        1. is_admin flag in user_settings table
        2. user_id or email in ADMIN_USERS environment variable

        Args:
            user_id: User identifier
            email: User email (optional, for env var check)

        Returns:
            True if user is admin
        """
        # Check environment variable first (highest priority)
        from app.config import get_settings
        settings = get_settings()
        admin_list = settings.admin_user_list

        if user_id.lower() in admin_list:
            return True
        if email and email.lower() in admin_list:
            return True

        # Check database flag
        user_settings = await self.get_settings(user_id)
        return user_settings.is_admin

    async def set_admin_status(self, user_id: str, is_admin: bool) -> UserSettings:
        """Set admin status for a user.

        Args:
            user_id: User identifier
            is_admin: Whether user should be admin

        Returns:
            Updated UserSettings object
        """
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            # Check if settings exist
            cursor = await conn.execute(
                "SELECT * FROM user_settings WHERE user_id = ?",
                (user_id,),
            )
            existing = await cursor.fetchone()

            if existing:
                # Update existing
                await conn.execute(
                    """
                    UPDATE user_settings
                    SET is_admin = ?, updated_at = ?
                    WHERE user_id = ?
                    """,
                    (1 if is_admin else 0, now, user_id),
                )
            else:
                # Insert new
                await conn.execute(
                    """
                    INSERT INTO user_settings (user_id, model_provider, is_admin, created_at, updated_at)
                    VALUES (?, 'grok', ?, ?, ?)
                    """,
                    (user_id, 1 if is_admin else 0, now, now),
                )
            await conn.commit()

        logger.info(f"Set admin status for user {user_id} to {is_admin}")

        return await self.get_settings(user_id)

    def _row_to_settings(self, row) -> UserSettings:
        """Convert database row to UserSettings object."""
        # Handle is_admin field (may not exist in old rows)
        is_admin = False
        if "is_admin" in row.keys():
            is_admin = bool(row["is_admin"])
        else:
            # Fall back to env var check
            from app.config import get_settings
            settings = get_settings()
            is_admin = row["user_id"].lower() in settings.admin_user_list

        return UserSettings(
            user_id=row["user_id"],
            model_provider=row["model_provider"],
            is_admin=is_admin,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Singleton instance
_store: Optional[UserSettingsStore] = None


def get_user_settings_store() -> UserSettingsStore:
    """Get the singleton user settings store instance."""
    global _store
    if _store is None:
        _store = UserSettingsStore()
    return _store
