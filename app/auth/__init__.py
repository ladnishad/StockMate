"""Authentication module for StockMate.

Provides Supabase-based authentication with JWT token validation.
"""

from app.auth.dependencies import get_current_user, get_optional_user, get_admin_user
from app.auth.models import User, TokenPayload, AuthResponse
from app.auth.router import router as auth_router

__all__ = [
    "get_current_user",
    "get_optional_user",
    "get_admin_user",
    "User",
    "TokenPayload",
    "AuthResponse",
    "auth_router",
]
