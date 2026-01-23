"""Subscription tier models and definitions.

Defines subscription tiers with their limits and features:
- Base: Free tier with limited features
- Premium: $20/month with expanded features
- Pro: $50/month with professional features
- Unlimited: $200/month with all features
"""

from enum import Enum
from typing import Optional
from pydantic import BaseModel


class SubscriptionTier(str, Enum):
    """Available subscription tiers."""
    BASE = "base"
    PREMIUM = "premium"
    PRO = "pro"
    UNLIMITED = "unlimited"


class TierLimits(BaseModel):
    """Limits and features for a subscription tier."""
    watchlist_limit: int  # -1 means unlimited
    multi_model_access: bool
    price_per_month: int  # in USD, 0 for free


class TierInfo(BaseModel):
    """Complete information about a subscription tier."""
    tier: SubscriptionTier
    name: str
    description: str
    price_per_month: int
    watchlist_limit: int
    multi_model_access: bool
    features: list[str]


# Tier configurations
TIER_CONFIGS: dict[SubscriptionTier, TierInfo] = {
    SubscriptionTier.BASE: TierInfo(
        tier=SubscriptionTier.BASE,
        name="Base",
        description="Free tier for getting started",
        price_per_month=0,
        watchlist_limit=2,
        multi_model_access=False,
        features=[
            "Add up to 2 stocks to watchlist",
            "Claude AI for analysis",
            "Basic trading plans",
            "Real-time price updates",
        ]
    ),
    SubscriptionTier.PREMIUM: TierInfo(
        tier=SubscriptionTier.PREMIUM,
        name="Premium",
        description="For active traders",
        price_per_month=20,
        watchlist_limit=5,
        multi_model_access=True,
        features=[
            "Add up to 5 stocks to watchlist",
            "Multi-model AI access (Claude + Grok)",
            "Advanced trading plans",
            "Real-time X/Twitter sentiment",
            "Priority support",
        ]
    ),
    SubscriptionTier.PRO: TierInfo(
        tier=SubscriptionTier.PRO,
        name="Pro",
        description="For serious traders",
        price_per_month=50,
        watchlist_limit=20,
        multi_model_access=True,
        features=[
            "Add up to 20 stocks to watchlist",
            "Multi-model AI access (Claude + Grok)",
            "Professional trading plans",
            "Real-time X/Twitter sentiment",
            "Advanced technical analysis",
            "Priority support",
        ]
    ),
    SubscriptionTier.UNLIMITED: TierInfo(
        tier=SubscriptionTier.UNLIMITED,
        name="Unlimited",
        description="For professional traders",
        price_per_month=200,
        watchlist_limit=-1,  # Unlimited
        multi_model_access=True,
        features=[
            "Unlimited stocks in watchlist",
            "Multi-model AI access (Claude + Grok)",
            "Professional trading plans",
            "Real-time X/Twitter sentiment",
            "Advanced technical analysis",
            "Dedicated support",
            "Early access to new features",
        ]
    ),
}


def get_tier_limits(tier: SubscriptionTier) -> TierLimits:
    """Get the limits for a subscription tier."""
    config = TIER_CONFIGS[tier]
    return TierLimits(
        watchlist_limit=config.watchlist_limit,
        multi_model_access=config.multi_model_access,
        price_per_month=config.price_per_month,
    )


def get_tier_info(tier: SubscriptionTier) -> TierInfo:
    """Get complete information about a subscription tier."""
    return TIER_CONFIGS[tier]


def get_all_tiers() -> list[TierInfo]:
    """Get information about all subscription tiers."""
    return list(TIER_CONFIGS.values())


class UserSubscription(BaseModel):
    """User's current subscription status."""
    user_id: str
    tier: SubscriptionTier
    tier_info: TierInfo
    watchlist_count: int
    watchlist_remaining: int  # -1 for unlimited
    can_add_to_watchlist: bool


class SubscriptionUpdateRequest(BaseModel):
    """Request to update a user's subscription tier (admin only)."""
    user_id: str
    tier: SubscriptionTier


class SubscriptionUpdateResponse(BaseModel):
    """Response after updating a user's subscription."""
    success: bool
    user_id: str
    old_tier: SubscriptionTier
    new_tier: SubscriptionTier
    message: str
