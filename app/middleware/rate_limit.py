"""Rate limiting middleware for StockMate.

Uses slowapi to provide per-user rate limiting.
"""

import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request

from app.config import get_settings

logger = logging.getLogger(__name__)


def get_user_identifier(request: Request) -> str:
    """Get rate limit identifier from request.

    Priority:
    1. User ID from JWT token (if authenticated)
    2. IP address (for unauthenticated requests)
    """
    # Try to get user from auth header
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        # Extract user ID from token for rate limiting
        # This is a simplified approach - in production, you'd decode the JWT
        token = auth_header.replace("Bearer ", "")
        # Use first 32 chars of token as identifier (enough for uniqueness)
        return f"user:{token[:32]}"

    # Fall back to IP address
    return get_remote_address(request)


# Create limiter instance
limiter = Limiter(
    key_func=get_user_identifier,
    default_limits=["60/minute"],  # Default rate limit
)


def get_rate_limit_string(per_minute: int) -> str:
    """Create rate limit string for slowapi."""
    return f"{per_minute}/minute"


# Rate limit decorators for different endpoint types
def rate_limit_standard(func):
    """Standard rate limit for most endpoints."""
    settings = get_settings()
    return limiter.limit(get_rate_limit_string(settings.rate_limit_per_minute))(func)


def rate_limit_ai(func):
    """Stricter rate limit for AI endpoints (expensive operations)."""
    settings = get_settings()
    return limiter.limit(get_rate_limit_string(settings.rate_limit_ai_per_minute))(func)


# Export RateLimitExceeded for error handling
__all__ = ["limiter", "RateLimitExceeded", "rate_limit_standard", "rate_limit_ai"]
