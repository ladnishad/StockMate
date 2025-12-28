"""Authentication router for StockMate.

Provides endpoints for user authentication via Supabase:
- POST /auth/signup - Register new user
- POST /auth/login - Login and get tokens
- POST /auth/refresh - Refresh access token
- POST /auth/logout - Logout (client-side)
- GET /auth/me - Get current user info
- POST /auth/reset-password - Request password reset
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status, Depends
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from supabase_auth.errors import AuthApiError

from app.config import get_settings
from app.auth.models import (
    SignUpRequest,
    LoginRequest,
    RefreshRequest,
    PasswordResetRequest,
    AuthResponse,
    PasswordResetResponse,
    UserResponse,
    MessageResponse,
    User,
)
from app.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


def get_supabase_client() -> Client:
    """Get Supabase client for auth operations."""
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_service_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase not configured",
        )

    return create_client(
        settings.supabase_url,
        settings.supabase_service_key,
        options=ClientOptions(auto_refresh_token=False),
    )


@router.post(
    "/signup",
    response_model=AuthResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    description="Create a new user account with email and password.",
)
async def signup(request: SignUpRequest):
    """Register a new user with Supabase Auth."""
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )

    try:
        # Use anon key for client-side signup
        supabase = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )

        response = supabase.auth.sign_up({
            "email": request.email,
            "password": request.password,
        })

        if not response.user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create user",
            )

        if not response.session:
            # Email confirmation required
            raise HTTPException(
                status_code=status.HTTP_200_OK,
                detail="Please check your email to confirm your account",
            )

        logger.info(f"New user registered: {response.user.email}")

        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            token_type="bearer",
            expires_in=response.session.expires_in or 3600,
            user=User(
                id=response.user.id,
                email=response.user.email or "",
                email_verified=response.user.email_confirmed_at is not None,
                created_at=response.user.created_at,
            ),
        )

    except AuthApiError as e:
        logger.warning(f"Signup failed: {e.message}")
        if "already registered" in str(e.message).lower():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="An account with this email already exists",
            )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e.message),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Signup error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create account",
        )


@router.post(
    "/login",
    response_model=AuthResponse,
    summary="Login with email and password",
    description="Authenticate user and return JWT tokens.",
)
async def login(request: LoginRequest):
    """Login user with Supabase Auth."""
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )

    try:
        supabase = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )

        response = supabase.auth.sign_in_with_password({
            "email": request.email,
            "password": request.password,
        })

        if not response.user or not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password",
            )

        logger.info(f"User logged in: {response.user.email}")

        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            token_type="bearer",
            expires_in=response.session.expires_in or 3600,
            user=User(
                id=response.user.id,
                email=response.user.email or "",
                email_verified=response.user.email_confirmed_at is not None,
                created_at=response.user.created_at,
            ),
        )

    except AuthApiError as e:
        logger.warning(f"Login failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed",
        )


@router.post(
    "/refresh",
    response_model=AuthResponse,
    summary="Refresh access token",
    description="Get a new access token using a refresh token.",
)
async def refresh_token(request: RefreshRequest):
    """Refresh access token using refresh token."""
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )

    try:
        supabase = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )

        response = supabase.auth.refresh_session(request.refresh_token)

        if not response.user or not response.session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token",
            )

        return AuthResponse(
            access_token=response.session.access_token,
            refresh_token=response.session.refresh_token,
            token_type="bearer",
            expires_in=response.session.expires_in or 3600,
            user=User(
                id=response.user.id,
                email=response.user.email or "",
                email_verified=response.user.email_confirmed_at is not None,
                created_at=response.user.created_at,
            ),
        )

    except AuthApiError as e:
        logger.warning(f"Token refresh failed: {e.message}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to refresh token",
        )


@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout user",
    description="Invalidate the current session. Client should discard tokens.",
)
async def logout(user: User = Depends(get_current_user)):
    """Logout user (client-side token deletion).

    Note: Supabase JWTs are stateless, so logout is primarily client-side.
    This endpoint exists for API consistency and can be extended to
    invalidate refresh tokens if needed.
    """
    logger.info(f"User logged out: {user.email}")
    return MessageResponse(message="Successfully logged out")


@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get information about the currently authenticated user.",
)
async def get_me(user: User = Depends(get_current_user)):
    """Get current authenticated user info."""
    return UserResponse(
        id=user.id,
        email=user.email,
        email_verified=user.email_verified,
        created_at=user.created_at,
    )


@router.post(
    "/reset-password",
    response_model=PasswordResetResponse,
    summary="Request password reset",
    description="Send a password reset email to the user's email address.",
)
async def reset_password(request: PasswordResetRequest):
    """Request password reset email."""
    settings = get_settings()

    if not settings.supabase_url or not settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not configured",
        )

    try:
        supabase = create_client(
            settings.supabase_url,
            settings.supabase_anon_key,
        )

        # Supabase will send reset email if account exists
        # We don't reveal if account exists for security
        supabase.auth.reset_password_email(request.email)

        logger.info(f"Password reset requested for: {request.email}")

        return PasswordResetResponse(
            message="If an account exists with this email, a password reset link has been sent"
        )

    except Exception as e:
        logger.error(f"Password reset error: {e}")
        # Don't reveal if account exists
        return PasswordResetResponse(
            message="If an account exists with this email, a password reset link has been sent"
        )
