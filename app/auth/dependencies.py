"""Authentication dependencies for FastAPI.

Provides dependency injection for route protection:
- get_current_user: Requires valid JWT, returns User
- get_optional_user: Returns User if authenticated, None otherwise
"""

import logging
from typing import Optional
from datetime import datetime, timezone
import time
import httpx

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError, ExpiredSignatureError
from jose.backends import ECKey

from app.config import get_settings
from app.auth.models import User, TokenPayload

logger = logging.getLogger(__name__)

# HTTP Bearer token scheme for Swagger UI
security = HTTPBearer(auto_error=False)

# JWKS cache
_jwks_cache: dict = {}
_jwks_cache_time: float = 0
JWKS_CACHE_TTL = 3600  # Cache JWKS for 1 hour


def get_jwks(supabase_url: str) -> dict:
    """Fetch JWKS from Supabase for ES256 token verification.

    Args:
        supabase_url: Supabase project URL

    Returns:
        JWKS dict with keys
    """
    global _jwks_cache, _jwks_cache_time

    # Return cached JWKS if still valid
    if _jwks_cache and (time.time() - _jwks_cache_time) < JWKS_CACHE_TTL:
        return _jwks_cache

    jwks_url = f"{supabase_url}/auth/v1/.well-known/jwks.json"

    try:
        response = httpx.get(jwks_url, timeout=10)
        response.raise_for_status()
        _jwks_cache = response.json()
        _jwks_cache_time = time.time()
        logger.debug(f"Fetched JWKS from {jwks_url}")
        return _jwks_cache
    except Exception as e:
        logger.error(f"Failed to fetch JWKS: {e}")
        # Return cached version if available, even if expired
        if _jwks_cache:
            return _jwks_cache
        raise


def get_signing_key(token: str, jwks: dict) -> dict:
    """Get the signing key from JWKS that matches the token's kid.

    Args:
        token: JWT token
        jwks: JWKS dict

    Returns:
        Key dict from JWKS
    """
    # Get the key ID from token header
    unverified_header = jwt.get_unverified_header(token)
    kid = unverified_header.get("kid")

    if not kid:
        raise ValueError("Token header missing 'kid'")

    # Find matching key in JWKS
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return key

    raise ValueError(f"No matching key found for kid: {kid}")


def decode_jwt(token: str) -> TokenPayload:
    """Decode and validate a Supabase JWT token.

    Supports both HS256 (legacy) and ES256 (current) algorithms.

    Args:
        token: JWT access token from Authorization header

    Returns:
        TokenPayload with user info

    Raises:
        HTTPException: If token is invalid or expired
    """
    settings = get_settings()

    if not settings.supabase_url:
        logger.error("SUPABASE_URL not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication not configured",
        )

    try:
        # Check which algorithm the token uses
        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg", "HS256")

        if algorithm == "ES256":
            # ES256: Use JWKS public key
            jwks = get_jwks(settings.supabase_url)
            signing_key = get_signing_key(token, jwks)

            payload = jwt.decode(
                token,
                signing_key,
                algorithms=["ES256"],
                audience="authenticated",
            )
        else:
            # HS256: Use JWT secret (legacy)
            if not settings.supabase_jwt_secret:
                logger.error("SUPABASE_JWT_SECRET not configured for HS256")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication not configured",
                )

            payload = jwt.decode(
                token,
                settings.supabase_jwt_secret,
                algorithms=["HS256"],
                audience="authenticated",
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
    except Exception as e:
        logger.error(f"Unexpected auth error: {e}")
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
    import os

    # Test mode bypass - return a test user
    if os.getenv("BYPASS_AUTH", "").lower() == "true":
        return User(
            id="test-user-00000000-0000-0000-0000-000000000000",
            email="test@stockmate.local",
            email_verified=True,
        )

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
