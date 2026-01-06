"""Subscription service for managing user subscription tiers.

Handles:
- Getting user's current subscription tier
- Checking tier limits (watchlist, multi-model access)
- Admin tier updates
- Tier enforcement for various features
"""

import logging
from datetime import datetime
from typing import Optional

from app.config import get_settings
from app.models.subscription import (
    SubscriptionTier,
    TierInfo,
    TierLimits,
    UserSubscription,
    get_tier_info,
    get_tier_limits,
    TIER_CONFIGS,
)

logger = logging.getLogger(__name__)


class SubscriptionService:
    """Service for managing user subscriptions."""

    def __init__(self):
        self._settings = get_settings()

    async def get_user_tier(self, user_id: str) -> SubscriptionTier:
        """Get the subscription tier for a user.

        Args:
            user_id: The user's ID

        Returns:
            The user's subscription tier (defaults to BASE if not set)
        """
        if self._settings.use_postgres:
            return await self._get_tier_postgres(user_id)
        else:
            return await self._get_tier_sqlite(user_id)

    async def _get_tier_postgres(self, user_id: str) -> SubscriptionTier:
        """Get tier from PostgreSQL."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT subscription_tier FROM user_settings WHERE user_id = $1",
                user_id
            )
            if row and row["subscription_tier"]:
                try:
                    return SubscriptionTier(row["subscription_tier"])
                except ValueError:
                    logger.warning(f"Invalid tier '{row['subscription_tier']}' for user {user_id}, defaulting to BASE")
                    return SubscriptionTier.BASE
            return SubscriptionTier.BASE

    async def _get_tier_sqlite(self, user_id: str) -> SubscriptionTier:
        """Get tier from SQLite (development)."""
        from app.storage.database import get_database

        db = get_database()
        async with db.connection() as conn:
            cursor = await conn.execute(
                "SELECT subscription_tier FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            row = await cursor.fetchone()
            if row and row[0]:
                try:
                    return SubscriptionTier(row[0])
                except ValueError:
                    return SubscriptionTier.BASE
            return SubscriptionTier.BASE

    async def set_user_tier(
        self,
        user_id: str,
        tier: SubscriptionTier,
        admin_id: Optional[str] = None
    ) -> tuple[SubscriptionTier, SubscriptionTier]:
        """Set a user's subscription tier (admin only).

        Args:
            user_id: The user's ID
            tier: The new subscription tier
            admin_id: The admin making the change (for logging)

        Returns:
            Tuple of (old_tier, new_tier)
        """
        old_tier = await self.get_user_tier(user_id)

        if self._settings.use_postgres:
            await self._set_tier_postgres(user_id, tier)
        else:
            await self._set_tier_sqlite(user_id, tier)

        logger.info(
            f"Subscription tier updated: user={user_id}, "
            f"old_tier={old_tier.value}, new_tier={tier.value}, "
            f"admin={admin_id or 'system'}"
        )

        return old_tier, tier

    async def _set_tier_postgres(self, user_id: str, tier: SubscriptionTier) -> None:
        """Set tier in PostgreSQL."""
        from app.storage.postgres import get_connection

        now = datetime.utcnow().isoformat()

        async with get_connection() as conn:
            # Upsert user settings with new tier
            await conn.execute(
                """
                INSERT INTO user_settings (user_id, subscription_tier, model_provider, created_at, updated_at)
                VALUES ($1, $2, 'claude', $3, $3)
                ON CONFLICT (user_id)
                DO UPDATE SET subscription_tier = $2, updated_at = $3
                """,
                user_id, tier.value, now
            )

    async def _set_tier_sqlite(self, user_id: str, tier: SubscriptionTier) -> None:
        """Set tier in SQLite."""
        from app.storage.database import get_database

        db = get_database()
        now = datetime.utcnow().isoformat()

        async with db.connection() as conn:
            # Check if user exists
            cursor = await conn.execute(
                "SELECT user_id FROM user_settings WHERE user_id = ?",
                (user_id,)
            )
            exists = await cursor.fetchone()

            if exists:
                await conn.execute(
                    "UPDATE user_settings SET subscription_tier = ?, updated_at = ? WHERE user_id = ?",
                    (tier.value, now, user_id)
                )
            else:
                await conn.execute(
                    """INSERT INTO user_settings (user_id, subscription_tier, model_provider, created_at, updated_at)
                       VALUES (?, ?, 'claude', ?, ?)""",
                    (user_id, tier.value, now, now)
                )
            await conn.commit()

    async def get_user_subscription(self, user_id: str) -> UserSubscription:
        """Get complete subscription information for a user.

        Args:
            user_id: The user's ID

        Returns:
            UserSubscription with tier info and current usage
        """
        tier = await self.get_user_tier(user_id)
        tier_info = get_tier_info(tier)

        # Get current watchlist count
        watchlist_count = await self._get_watchlist_count(user_id)

        # Calculate remaining
        if tier_info.watchlist_limit == -1:
            watchlist_remaining = -1  # Unlimited
            can_add = True
        else:
            watchlist_remaining = max(0, tier_info.watchlist_limit - watchlist_count)
            can_add = watchlist_count < tier_info.watchlist_limit

        return UserSubscription(
            user_id=user_id,
            tier=tier,
            tier_info=tier_info,
            watchlist_count=watchlist_count,
            watchlist_remaining=watchlist_remaining,
            can_add_to_watchlist=can_add,
        )

    async def _get_watchlist_count(self, user_id: str) -> int:
        """Get the number of items in user's watchlist."""
        if self._settings.use_postgres:
            from app.storage.postgres import get_connection

            async with get_connection() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM watchlist WHERE user_id = $1",
                    user_id
                )
                return count or 0
        else:
            from app.storage.watchlist_store import get_watchlist_store

            store = get_watchlist_store()
            return store.get_symbol_count(user_id)

    async def can_add_to_watchlist(self, user_id: str) -> tuple[bool, int, int]:
        """Check if user can add more items to their watchlist.

        Args:
            user_id: The user's ID

        Returns:
            Tuple of (can_add, current_count, limit)
            limit is -1 for unlimited
        """
        subscription = await self.get_user_subscription(user_id)
        return (
            subscription.can_add_to_watchlist,
            subscription.watchlist_count,
            subscription.tier_info.watchlist_limit,
        )

    async def has_multi_model_access(self, user_id: str) -> bool:
        """Check if user has access to multiple AI models.

        Args:
            user_id: The user's ID

        Returns:
            True if user can use multiple AI models (Claude + Grok)
        """
        tier = await self.get_user_tier(user_id)
        tier_info = get_tier_info(tier)
        return tier_info.multi_model_access

    async def get_available_providers(self, user_id: str) -> list[str]:
        """Get the list of AI providers available to a user based on their tier.

        Args:
            user_id: The user's ID

        Returns:
            List of available provider names
        """
        has_multi = await self.has_multi_model_access(user_id)

        if has_multi:
            # Check which providers are actually configured
            providers = []
            if self._settings.claude_api_key:
                providers.append("claude")
            if self._settings.grok_api_key:
                providers.append("grok")
            return providers if providers else ["claude"]
        else:
            # Base tier only gets Claude
            return ["claude"]

    async def get_default_provider(self, user_id: str) -> str:
        """Get the default AI provider for a user based on their tier.

        Base tier users always get Claude (Grok disabled).
        Higher tiers get their saved preference or default to grok.

        Args:
            user_id: The user's ID

        Returns:
            Default provider name
        """
        has_multi = await self.has_multi_model_access(user_id)

        if not has_multi:
            return "claude"  # Base tier always defaults to Claude

        # For multi-model tiers, check user preference
        if self._settings.use_postgres:
            from app.storage.postgres import get_connection

            async with get_connection() as conn:
                row = await conn.fetchrow(
                    "SELECT model_provider FROM user_settings WHERE user_id = $1",
                    user_id
                )
                if row and row["model_provider"]:
                    return row["model_provider"]
        else:
            from app.storage.database import get_database

            db = get_database()
            async with db.connection() as conn:
                cursor = await conn.execute(
                    "SELECT model_provider FROM user_settings WHERE user_id = ?",
                    (user_id,)
                )
                row = await cursor.fetchone()
                if row and row[0]:
                    return row[0]

        # Default to grok for multi-model tiers
        return "grok" if self._settings.grok_api_key else "claude"


# Singleton instance
_subscription_service: Optional[SubscriptionService] = None


def get_subscription_service() -> SubscriptionService:
    """Get the singleton subscription service instance."""
    global _subscription_service
    if _subscription_service is None:
        _subscription_service = SubscriptionService()
    return _subscription_service
