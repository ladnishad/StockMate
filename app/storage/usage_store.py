"""Usage tracking storage.

Persists AI API usage records for billing and analytics.
"""

import logging
import uuid
from datetime import datetime, timedelta
from typing import Optional, List, Tuple

from app.storage.database import get_database
from app.models.usage import (
    UsageRecord,
    UsageSummary,
    UserUsageSummary,
    ModelProvider,
    OperationType,
    calculate_cost,
)

logger = logging.getLogger(__name__)


class UsageStore:
    """Manages API usage records in the database."""

    def __init__(self):
        """Initialize usage store."""
        self.db = get_database()

    async def log_usage(
        self,
        user_id: str,
        provider: ModelProvider,
        model: str,
        operation_type: OperationType,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int = 0,
        symbol: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> UsageRecord:
        """Log an API usage record.

        Args:
            user_id: User who made the request
            provider: AI provider (claude or grok)
            model: Model ID used
            operation_type: Type of operation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tool_calls: Number of tool calls (web search, etc.)
            symbol: Stock symbol if applicable
            endpoint: API endpoint called

        Returns:
            Created UsageRecord
        """
        record_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        total_tokens = input_tokens + output_tokens

        # Calculate costs
        token_cost, tool_cost = calculate_cost(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tool_calls=tool_calls,
        )
        estimated_cost = token_cost + tool_cost

        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO api_usage (
                    id, user_id, provider, model, operation_type,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost, tool_calls, tool_cost,
                    symbol, endpoint, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record_id, user_id, provider.value, model, operation_type.value,
                    input_tokens, output_tokens, total_tokens,
                    estimated_cost, tool_calls, tool_cost,
                    symbol, endpoint, now,
                ),
            )
            await conn.commit()

        logger.debug(
            f"Logged usage: user={user_id}, provider={provider.value}, "
            f"tokens={total_tokens}, cost=${estimated_cost:.6f}"
        )

        return UsageRecord(
            id=record_id,
            user_id=user_id,
            provider=provider,
            model=model,
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            estimated_cost=estimated_cost,
            tool_calls=tool_calls,
            tool_cost=tool_cost,
            symbol=symbol,
            endpoint=endpoint,
            created_at=now,
        )

    async def get_user_usage(
        self,
        user_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        provider: Optional[ModelProvider] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[UsageRecord], int]:
        """Get usage records for a specific user.

        Args:
            user_id: User ID to query
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            provider: Filter by provider
            limit: Max records to return
            offset: Offset for pagination

        Returns:
            Tuple of (records, total_count)
        """
        conditions = ["user_id = ?"]
        params = [user_id]

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)
        if provider:
            conditions.append("provider = ?")
            params.append(provider.value)

        where_clause = " AND ".join(conditions)

        async with self.db.connection() as conn:
            # Get total count
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM api_usage WHERE {where_clause}",
                params,
            )
            row = await cursor.fetchone()
            total_count = row[0] if row else 0

            # Get records
            cursor = await conn.execute(
                f"""
                SELECT * FROM api_usage
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            )
            rows = await cursor.fetchall()

        records = [self._row_to_record(row) for row in rows]
        return records, total_count

    async def get_all_usage(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        provider: Optional[ModelProvider] = None,
        user_id: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[UsageRecord], int]:
        """Get all usage records (admin only).

        Args:
            start_date: Filter by start date (ISO format)
            end_date: Filter by end date (ISO format)
            provider: Filter by provider
            user_id: Filter by specific user
            limit: Max records to return
            offset: Offset for pagination

        Returns:
            Tuple of (records, total_count)
        """
        conditions = []
        params = []

        if start_date:
            conditions.append("created_at >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("created_at <= ?")
            params.append(end_date)
        if provider:
            conditions.append("provider = ?")
            params.append(provider.value)
        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        async with self.db.connection() as conn:
            # Get total count
            cursor = await conn.execute(
                f"SELECT COUNT(*) FROM api_usage WHERE {where_clause}",
                params,
            )
            row = await cursor.fetchone()
            total_count = row[0] if row else 0

            # Get records
            cursor = await conn.execute(
                f"""
                SELECT * FROM api_usage
                WHERE {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
                """,
                params + [limit, offset],
            )
            rows = await cursor.fetchall()

        records = [self._row_to_record(row) for row in rows]
        return records, total_count

    async def get_usage_summary(
        self,
        user_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> UsageSummary:
        """Get aggregated usage summary.

        Args:
            user_id: Filter by specific user (None for all users)
            start_date: Start of period (ISO format)
            end_date: End of period (ISO format)

        Returns:
            UsageSummary with aggregated stats
        """
        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow().isoformat()
        if not start_date:
            start_dt = datetime.utcnow() - timedelta(days=30)
            start_date = start_dt.isoformat()

        conditions = ["created_at >= ?", "created_at <= ?"]
        params = [start_date, end_date]

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions)

        async with self.db.connection() as conn:
            # Get Claude stats
            cursor = await conn.execute(
                f"""
                SELECT
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(estimated_cost), 0) as cost
                FROM api_usage
                WHERE {where_clause} AND provider = 'claude'
                """,
                params,
            )
            claude_row = await cursor.fetchone()

            # Get Grok stats
            cursor = await conn.execute(
                f"""
                SELECT
                    COUNT(*) as requests,
                    COALESCE(SUM(input_tokens), 0) as input_tokens,
                    COALESCE(SUM(output_tokens), 0) as output_tokens,
                    COALESCE(SUM(estimated_cost), 0) as cost,
                    COALESCE(SUM(tool_calls), 0) as tool_calls,
                    COALESCE(SUM(tool_cost), 0) as tool_cost
                FROM api_usage
                WHERE {where_clause} AND provider = 'grok'
                """,
                params,
            )
            grok_row = await cursor.fetchone()

        return UsageSummary(
            user_id=user_id,
            period_start=start_date,
            period_end=end_date,
            claude_requests=claude_row[0] if claude_row else 0,
            claude_input_tokens=claude_row[1] if claude_row else 0,
            claude_output_tokens=claude_row[2] if claude_row else 0,
            claude_cost=round(claude_row[3] if claude_row else 0, 6),
            grok_requests=grok_row[0] if grok_row else 0,
            grok_input_tokens=grok_row[1] if grok_row else 0,
            grok_output_tokens=grok_row[2] if grok_row else 0,
            grok_cost=round(grok_row[3] if grok_row else 0, 6),
            grok_tool_calls=grok_row[4] if grok_row else 0,
            grok_tool_cost=round(grok_row[5] if grok_row else 0, 6),
            total_requests=(claude_row[0] if claude_row else 0) + (grok_row[0] if grok_row else 0),
            total_tokens=(
                (claude_row[1] if claude_row else 0) + (claude_row[2] if claude_row else 0) +
                (grok_row[1] if grok_row else 0) + (grok_row[2] if grok_row else 0)
            ),
            total_cost=round(
                (claude_row[3] if claude_row else 0) + (grok_row[3] if grok_row else 0),
                6,
            ),
        )

    async def get_all_users_summary(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 50,
    ) -> List[UserUsageSummary]:
        """Get usage summary for all users (admin only).

        Args:
            start_date: Start of period (ISO format)
            end_date: End of period (ISO format)
            limit: Max users to return

        Returns:
            List of UserUsageSummary sorted by total cost descending
        """
        # Default to last 30 days if no dates provided
        if not end_date:
            end_date = datetime.utcnow().isoformat()
        if not start_date:
            start_dt = datetime.utcnow() - timedelta(days=30)
            start_date = start_dt.isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT
                    user_id,
                    COUNT(*) as total_requests,
                    COALESCE(SUM(total_tokens), 0) as total_tokens,
                    COALESCE(SUM(estimated_cost), 0) as total_cost,
                    COALESCE(SUM(CASE WHEN provider = 'claude' THEN estimated_cost ELSE 0 END), 0) as claude_cost,
                    COALESCE(SUM(CASE WHEN provider = 'grok' THEN estimated_cost ELSE 0 END), 0) as grok_cost,
                    MAX(created_at) as last_request_at,
                    COALESCE(SUM(CASE WHEN operation_type = 'plan_generation' THEN 1 ELSE 0 END), 0) as plan_generations,
                    COALESCE(SUM(CASE WHEN operation_type = 'chat' THEN 1 ELSE 0 END), 0) as chat_requests,
                    COALESCE(SUM(CASE WHEN operation_type = 'plan_evaluation' THEN 1 ELSE 0 END), 0) as evaluations
                FROM api_usage
                WHERE created_at >= ? AND created_at <= ?
                GROUP BY user_id
                ORDER BY total_cost DESC
                LIMIT ?
                """,
                (start_date, end_date, limit),
            )
            rows = await cursor.fetchall()

        summaries = []
        for row in rows:
            summaries.append(UserUsageSummary(
                user_id=row[0],
                total_requests=row[1],
                total_tokens=row[2],
                total_cost=round(row[3], 6),
                claude_cost=round(row[4], 6),
                grok_cost=round(row[5], 6),
                last_request_at=row[6],
                plan_generations=row[7],
                chat_requests=row[8],
                evaluations=row[9],
            ))

        return summaries

    async def get_daily_costs(
        self,
        user_id: Optional[str] = None,
        days: int = 30,
    ) -> List[dict]:
        """Get daily cost breakdown for charting.

        Args:
            user_id: Filter by specific user (None for all users)
            days: Number of days to include

        Returns:
            List of daily summaries with date, cost, tokens, requests
        """
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()

        conditions = ["created_at >= ?"]
        params = [start_date]

        if user_id:
            conditions.append("user_id = ?")
            params.append(user_id)

        where_clause = " AND ".join(conditions)

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                f"""
                SELECT
                    DATE(created_at) as date,
                    COUNT(*) as requests,
                    COALESCE(SUM(total_tokens), 0) as tokens,
                    COALESCE(SUM(estimated_cost), 0) as cost,
                    COALESCE(SUM(CASE WHEN provider = 'claude' THEN estimated_cost ELSE 0 END), 0) as claude_cost,
                    COALESCE(SUM(CASE WHEN provider = 'grok' THEN estimated_cost ELSE 0 END), 0) as grok_cost
                FROM api_usage
                WHERE {where_clause}
                GROUP BY DATE(created_at)
                ORDER BY date DESC
                """,
                params,
            )
            rows = await cursor.fetchall()

        return [
            {
                "date": row[0],
                "requests": row[1],
                "tokens": row[2],
                "cost": round(row[3], 6),
                "claude_cost": round(row[4], 6),
                "grok_cost": round(row[5], 6),
            }
            for row in rows
        ]

    def _row_to_record(self, row) -> UsageRecord:
        """Convert database row to UsageRecord."""
        return UsageRecord(
            id=row["id"],
            user_id=row["user_id"],
            provider=ModelProvider(row["provider"]),
            model=row["model"],
            operation_type=OperationType(row["operation_type"]),
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            total_tokens=row["total_tokens"],
            estimated_cost=row["estimated_cost"],
            tool_calls=row["tool_calls"],
            tool_cost=row["tool_cost"],
            symbol=row["symbol"],
            endpoint=row["endpoint"],
            created_at=row["created_at"],
        )


# Singleton instance
_store: Optional[UsageStore] = None


def get_usage_store() -> UsageStore:
    """Get the singleton usage store instance."""
    global _store
    if _store is None:
        _store = UsageStore()
    return _store
