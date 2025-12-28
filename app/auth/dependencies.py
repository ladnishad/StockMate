"""Authentication dependencies for FastAPI.

Provides dependency injection for route protection:
- get_current_user: Requires valid JWT, returns User
- get_optional_user: Returns User if authenticated, None otherwise
"""

import logging
from typing import Optional
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError

from app.config import get_settings
from app.auth.models import User, TokenPayload

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme for Swagger UI
security = HTTPBearer(auto_error=False)


def decode_jwt(token: str) -> TokenPayload:
    """Decode and validate a Supabase JWT token.

    Args:
        token: JWT access token from Authorization header

    Returns:
        TokenPayload with user info

    Raises:
        HTTPException: If token is invalid or expired
    """
    settings = get_settings()

    if not settings.supabase_jwt_secret:
        logger.error("SUPABASE_JWT_SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured",
        )

    try:
        # Supabase uses HS256 algorithm
        payload = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",  # Supabase sets this for authenticated users
        )
        return TokenPayload(**payload)

    except ExpiredSignatureError:
        logger.debug("JWT token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError as e:
        logger.debug(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> User:
    """Get the current authenticated user from JWT token.

    This is a FastAPI dependency that validates the JWT token and returns
    the authenticated user. Use this for protected endpoints.

    Args:
        credentials: Bearer token from Authorization header

    Returns:
        User object with id, email, etc.

    Raises:
        HTTPException 401: If no token provided or token is invalid
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token_payload = decode_jwt(credentials.credentials)

    # Build User from JWT claims
    return User(
        id=token_payload.sub,
        email=token_payload.email or "",
        email_verified=True,  # Supabase only allows authenticated users to get tokens
    )


async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[User]:
    """Get current user if authenticated, None otherwise.

    This is a FastAPI dependency for endpoints that work both with and
    without authentication (e.g., public endpoints with optional personalization).

    Args:
        credentials: Optional Bearer token from Authorization header

    Returns:
        User object if authenticated, None if not
    """
    if not credentials:
        return None

    try:
        token_payload = decode_jwt(credentials.credentials)
        return User(
            id=token_payload.sub,
            email=token_payload.email or "",
            email_verified=True,
        )
    except HTTPException:
        # Token invalid or expired - treat as unauthenticated
        return None


def get_user_id(user: User = Depends(get_current_user)) -> str:
    """Convenience dependency to get just the user ID.

    Use this when you only need the user_id for database queries.

    Returns:
        User ID string (Supabase UUID)
    """
    return user.id
