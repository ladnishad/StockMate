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
        now = datetime.utcnow().isoformat()
        return UserSettings(
            user_id=user_id,
            model_provider="grok",  # Default to Grok
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

    def _row_to_settings(self, row) -> UserSettings:
        """Convert database row to UserSettings object."""
        return UserSettings(
            user_id=row["user_id"],
            model_provider=row["model_provider"],
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
