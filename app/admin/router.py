"""Admin API router for StockMate.

Provides admin-only endpoints for:
- Usage tracking and cost analytics
- User management
- System monitoring
"""

import logging
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.auth import get_admin_user, User
from app.storage.usage_store import get_usage_store
from app.storage.user_settings_store import get_user_settings_store
from app.services.subscription_service import get_subscription_service
from app.models.usage import (
    ModelProvider,
    UsageRecord,
    UsageSummary,
    UserUsageSummary,
    OperationTypeBreakdown,
    UsageByOperationResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class UsageListResponse(BaseModel):
    """Response for usage records list."""
    records: List[UsageRecord]
    total_count: int
    page: int
    page_size: int
    has_more: bool


class UsageSummaryResponse(BaseModel):
    """Response for usage summary."""
    summary: UsageSummary
    period_days: int


class AllUsersSummaryResponse(BaseModel):
    """Response for all users usage summary."""
    users: List[UserUsageSummary]
    total_count: int
    period_start: str
    period_end: str
    grand_total_cost: float


class DailyCostItem(BaseModel):
    """Daily cost breakdown item."""
    date: str
    requests: int
    tokens: int
    cost: float
    claude_cost: float
    grok_cost: float


class DailyCostsResponse(BaseModel):
    """Response for daily costs."""
    daily_costs: List[DailyCostItem]
    total_cost: float
    period_days: int


class AdminStatusResponse(BaseModel):
    """Response for admin status check."""
    is_admin: bool
    user_id: str
    email: str


class SetAdminRequest(BaseModel):
    """Request to set admin status."""
    user_id: str = Field(..., description="User ID to modify")
    is_admin: bool = Field(..., description="Whether to grant admin access")


class SetAdminResponse(BaseModel):
    """Response for set admin status."""
    user_id: str
    is_admin: bool
    message: str


# =============================================================================
# ADMIN STATUS ENDPOINTS
# =============================================================================

@router.get(
    "/status",
    response_model=AdminStatusResponse,
    summary="Check admin status",
    description="Check if the current user has admin privileges.",
)
async def check_admin_status(
    admin: User = Depends(get_admin_user),
) -> AdminStatusResponse:
    """Check if the current user is an admin."""
    return AdminStatusResponse(
        is_admin=True,
        user_id=admin.id,
        email=admin.email,
    )


@router.post(
    "/users/{user_id}/admin",
    response_model=SetAdminResponse,
    summary="Set user admin status",
    description="Grant or revoke admin access for a user (requires admin).",
)
async def set_user_admin_status(
    user_id: str,
    request: SetAdminRequest,
    admin: User = Depends(get_admin_user),
) -> SetAdminResponse:
    """Set admin status for a user."""
    if user_id != request.user_id:
        raise HTTPException(status_code=400, detail="User ID mismatch")

    settings_store = get_user_settings_store()
    await settings_store.set_admin_status(user_id, request.is_admin)

    action = "granted" if request.is_admin else "revoked"
    return SetAdminResponse(
        user_id=user_id,
        is_admin=request.is_admin,
        message=f"Admin access {action} for user {user_id}",
    )


# =============================================================================
# USAGE TRACKING ENDPOINTS
# =============================================================================

@router.get(
    "/usage",
    response_model=UsageListResponse,
    summary="Get all usage records",
    description="Get paginated list of all API usage records (admin only).",
)
async def get_all_usage(
    admin: User = Depends(get_admin_user),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    provider: Optional[str] = Query(None, description="Filter by provider (claude or grok)"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Records per page"),
) -> UsageListResponse:
    """Get all usage records with optional filters."""
    store = get_usage_store()
    offset = (page - 1) * page_size

    # Convert provider string to enum if provided
    provider_enum = None
    if provider:
        try:
            provider_enum = ModelProvider(provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

    records, total_count = await store.get_all_usage(
        start_date=start_date,
        end_date=end_date,
        provider=provider_enum,
        user_id=user_id,
        limit=page_size,
        offset=offset,
    )

    return UsageListResponse(
        records=records,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_more=(offset + len(records)) < total_count,
    )


@router.get(
    "/usage/summary",
    response_model=UsageSummaryResponse,
    summary="Get usage summary",
    description="Get aggregated usage summary for a time period (admin only).",
)
async def get_usage_summary(
    admin: User = Depends(get_admin_user),
    user_id: Optional[str] = Query(None, description="Filter by user ID (None for all)"),
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
) -> UsageSummaryResponse:
    """Get aggregated usage summary."""
    store = get_usage_store()

    end_date = datetime.utcnow().isoformat()
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

    summary = await store.get_usage_summary(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )

    return UsageSummaryResponse(
        summary=summary,
        period_days=days,
    )


@router.get(
    "/usage/users",
    response_model=AllUsersSummaryResponse,
    summary="Get usage by user",
    description="Get usage summary for all users, sorted by cost (admin only).",
)
async def get_usage_by_user(
    admin: User = Depends(get_admin_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
    limit: int = Query(50, ge=1, le=200, description="Max users to return"),
) -> AllUsersSummaryResponse:
    """Get usage breakdown by user."""
    store = get_usage_store()
    settings_store = get_user_settings_store()
    subscription_service = get_subscription_service()

    end_date = datetime.utcnow().isoformat()
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

    # Get users with usage data
    users_with_usage = await store.get_all_users_summary(
        start_date=start_date,
        end_date=end_date,
        limit=limit,
    )

    # Create a dict for easy lookup
    users_dict = {u.user_id: u for u in users_with_usage}

    # Get all users from settings (includes users without usage)
    all_user_ids = await settings_store.get_all_user_ids()

    # Add users without usage data
    for user_id in all_user_ids:
        if user_id not in users_dict:
            # Create empty usage summary for this user
            users_dict[user_id] = UserUsageSummary(
                user_id=user_id,
                total_requests=0,
                total_tokens=0,
                total_cost=0.0,
                claude_cost=0.0,
                grok_cost=0.0,
                plan_generations=0,
                chat_requests=0,
                evaluations=0,
                orchestrator_calls=0,
                subagent_calls=0,
                image_analyses=0,
                plan_generation_cost=0.0,
                chat_cost=0.0,
                evaluation_cost=0.0,
                orchestrator_cost=0.0,
                subagent_cost=0.0,
                image_analysis_cost=0.0,
            )

    # Fetch subscription tier and email for each user, creating new instances with updated data
    updated_users = []
    for user_id, user in users_dict.items():
        # Get subscription tier
        try:
            tier = await subscription_service.get_user_tier(user_id)
            subscription_tier = tier.value
        except Exception as e:
            logger.warning(f"Failed to get subscription tier for user {user_id}: {e}")
            subscription_tier = "base"

        # Get email from settings
        email = None
        try:
            settings = await settings_store.get_settings(user_id)
            if settings and settings.email:
                email = settings.email
        except Exception:
            pass

        # Create new instance with updated fields
        updated_user = UserUsageSummary(
            user_id=user.user_id,
            email=email,
            subscription_tier=subscription_tier,
            total_requests=user.total_requests,
            total_tokens=user.total_tokens,
            total_cost=user.total_cost,
            claude_cost=user.claude_cost,
            grok_cost=user.grok_cost,
            last_request_at=user.last_request_at,
            plan_generations=user.plan_generations,
            chat_requests=user.chat_requests,
            evaluations=user.evaluations,
            orchestrator_calls=user.orchestrator_calls,
            subagent_calls=user.subagent_calls,
            image_analyses=user.image_analyses,
            plan_generation_cost=user.plan_generation_cost,
            chat_cost=user.chat_cost,
            evaluation_cost=user.evaluation_cost,
            orchestrator_cost=user.orchestrator_cost,
            subagent_cost=user.subagent_cost,
            image_analysis_cost=user.image_analysis_cost,
        )
        updated_users.append(updated_user)

    # Sort by cost descending, then by user_id for users with no cost
    users = sorted(updated_users, key=lambda u: (-u.total_cost, u.user_id))[:limit]

    grand_total = sum(u.total_cost for u in users)

    return AllUsersSummaryResponse(
        users=users,
        total_count=len(users),
        period_start=start_date,
        period_end=end_date,
        grand_total_cost=round(grand_total, 6),
    )


@router.get(
    "/usage/daily",
    response_model=DailyCostsResponse,
    summary="Get daily cost breakdown",
    description="Get daily cost breakdown for charting (admin only).",
)
async def get_daily_costs(
    admin: User = Depends(get_admin_user),
    user_id: Optional[str] = Query(None, description="Filter by user ID (None for all)"),
    days: int = Query(30, ge=1, le=90, description="Number of days to include"),
) -> DailyCostsResponse:
    """Get daily cost breakdown."""
    store = get_usage_store()

    daily_data = await store.get_daily_costs(
        user_id=user_id,
        days=days,
    )

    daily_costs = [DailyCostItem(**d) for d in daily_data]
    total_cost = sum(d.cost for d in daily_costs)

    return DailyCostsResponse(
        daily_costs=daily_costs,
        total_cost=round(total_cost, 6),
        period_days=days,
    )


@router.get(
    "/usage/by-operation",
    response_model=UsageByOperationResponse,
    summary="Get usage by operation type",
    description="Get usage breakdown by operation type (plan generation vs evaluation, etc.) - admin only.",
)
async def get_usage_by_operation_type(
    admin: User = Depends(get_admin_user),
    user_id: Optional[str] = Query(None, description="Filter by user ID (None for all)"),
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
) -> UsageByOperationResponse:
    """Get usage breakdown by operation type.

    Shows costs and request counts for:
    - plan_generation: New trading plan creation
    - plan_evaluation: Evaluating existing plans
    - chat: Conversational AI interactions
    - orchestrator: Plan synthesis
    - subagent: Individual trade-style analysis
    - image_analysis: Chart vision analysis
    """
    store = get_usage_store()

    end_date = datetime.utcnow().isoformat()
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

    breakdowns_data = await store.get_usage_by_operation(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
    )

    breakdowns = [OperationTypeBreakdown(**d) for d in breakdowns_data]
    total_cost = sum(b.total_cost for b in breakdowns)

    return UsageByOperationResponse(
        user_id=user_id,
        period_start=start_date,
        period_end=end_date,
        breakdowns=breakdowns,
        total_cost=round(total_cost, 6),
    )


@router.get(
    "/usage/user/{user_id}",
    response_model=UsageListResponse,
    summary="Get user's usage records",
    description="Get paginated usage records for a specific user (admin only).",
)
async def get_user_usage(
    user_id: str,
    admin: User = Depends(get_admin_user),
    start_date: Optional[str] = Query(None, description="Start date (ISO format)"),
    end_date: Optional[str] = Query(None, description="End date (ISO format)"),
    provider: Optional[str] = Query(None, description="Filter by provider"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(50, ge=1, le=200, description="Records per page"),
) -> UsageListResponse:
    """Get usage records for a specific user."""
    store = get_usage_store()
    offset = (page - 1) * page_size

    # Convert provider string to enum if provided
    provider_enum = None
    if provider:
        try:
            provider_enum = ModelProvider(provider.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid provider: {provider}")

    records, total_count = await store.get_user_usage(
        user_id=user_id,
        start_date=start_date,
        end_date=end_date,
        provider=provider_enum,
        limit=page_size,
        offset=offset,
    )

    return UsageListResponse(
        records=records,
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_more=(offset + len(records)) < total_count,
    )


# =============================================================================
# USER'S OWN USAGE ENDPOINT (Non-admin)
# =============================================================================

# Note: This is included in admin router but accessible to all authenticated users
# to see their own usage. Admin check is bypassed for own user_id.

from app.auth import get_current_user


@router.get(
    "/my-usage",
    response_model=UsageSummaryResponse,
    summary="Get my usage summary",
    description="Get usage summary for the current user.",
    tags=["User"],
)
async def get_my_usage(
    user: User = Depends(get_current_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to include"),
) -> UsageSummaryResponse:
    """Get usage summary for the current authenticated user."""
    store = get_usage_store()

    end_date = datetime.utcnow().isoformat()
    start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

    summary = await store.get_usage_summary(
        user_id=user.id,
        start_date=start_date,
        end_date=end_date,
    )

    return UsageSummaryResponse(
        summary=summary,
        period_days=days,
    )
