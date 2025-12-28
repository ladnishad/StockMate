"""Authentication models for StockMate."""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class User(BaseModel):
    """Authenticated user model."""

    id: str = Field(..., description="Supabase user UUID")
    email: str = Field(..., description="User email address")
    email_verified: bool = Field(default=False, description="Whether email is verified")
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class TokenPayload(BaseModel):
    """JWT token payload from Supabase."""

    sub: str = Field(..., description="Subject (user_id)")
    email: Optional[str] = None
    exp: int = Field(..., description="Expiration timestamp")
    aud: Optional[str] = None
    iat: Optional[int] = None
    iss: Optional[str] = None
    role: Optional[str] = None


class SignUpRequest(BaseModel):
    """Request model for user registration."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="Password (min 8 characters)")


class LoginRequest(BaseModel):
    """Request model for user login."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class AuthResponse(BaseModel):
    """Response model for authentication endpoints."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="Refresh token for getting new access tokens")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiration time in seconds")
    user: User = Field(..., description="Authenticated user info")


class RefreshRequest(BaseModel):
    """Request model for token refresh."""

    refresh_token: str = Field(..., description="Refresh token from previous login")


class PasswordResetRequest(BaseModel):
    """Request model for password reset."""

    email: EmailStr = Field(..., description="Email address to send reset link to")


class PasswordResetResponse(BaseModel):
    """Response model for password reset."""

    message: str = Field(default="Password reset email sent if account exists")


class UserResponse(BaseModel):
    """Response model for user info endpoint."""

    id: str
    email: str
    email_verified: bool
    created_at: Optional[datetime] = None


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
