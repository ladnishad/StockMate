"""Trading Plan storage for the planning agent.

Stores AI-generated trading plans that evolve as the stock progresses.
Supports both SQLite (development) and PostgreSQL (production).
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import aiosqlite

from app.config import get_settings
from app.storage.database import get_db_path

logger = logging.getLogger(__name__)


@dataclass
class TradingPlan:
    """AI-generated trading plan for a stock."""

    id: str
    user_id: str
    symbol: str
    status: str = "active"  # active, invalidated, completed, stopped_out

    # Plan details
    bias: str = ""  # bullish, bearish, neutral
    thesis: str = ""  # Why this trade makes sense (can be updated during evaluation)
    original_thesis: str = ""  # Preserved from plan creation, never changes
    entry_zone_low: Optional[float] = None
    entry_zone_high: Optional[float] = None
    stop_loss: Optional[float] = None
    stop_reasoning: str = ""
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    target_reasoning: str = ""
    risk_reward: Optional[float] = None
    position_size_pct: Optional[float] = None  # % of account

    # Key levels to watch
    key_supports: List[float] = field(default_factory=list)
    key_resistances: List[float] = field(default_factory=list)
    invalidation_criteria: str = ""  # What would invalidate this plan

    # Trade style (determined by agent)
    trade_style: str = ""  # day, swing, position
    trade_style_reasoning: str = ""  # Why this style fits the setup
    holding_period: str = ""  # e.g., "1-3 days", "1-2 weeks"
    confidence: int = 0  # 0-100 confidence score

    # Context at plan creation
    price_at_creation: Optional[float] = None
    rsi_at_creation: Optional[float] = None
    market_direction_at_creation: str = ""
    technical_summary: str = ""

    # External sentiment (from web/social search)
    news_summary: str = ""  # Brief summary of recent news/catalysts
    social_sentiment: str = ""  # bullish, bearish, neutral, mixed, none
    social_buzz: str = ""  # Summary of social discussion if found
    sentiment_source: str = ""  # "reddit" or "x" - which platform was searched

    # Tracking
    created_at: str = ""
    updated_at: str = ""
    last_evaluation: str = ""
    evaluation_notes: str = ""  # Latest AI evaluation

    # Validation warnings (if price levels don't match bias)
    validation_warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert lists to JSON strings for storage
        data["key_supports"] = json.dumps(data["key_supports"])
        data["key_resistances"] = json.dumps(data["key_resistances"])
        data["validation_warnings"] = json.dumps(data["validation_warnings"])
        return data

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "TradingPlan":
        """Create from database row."""
        data = dict(row)
        # Parse JSON lists
        data["key_supports"] = json.loads(data.get("key_supports", "[]"))
        data["key_resistances"] = json.loads(data.get("key_resistances", "[]"))
        data["validation_warnings"] = json.loads(data.get("validation_warnings", "[]"))
        # Migrate old field names
        data = _migrate_plan_data(data)
        return cls(**data)


def _migrate_plan_data(data: dict) -> dict:
    """Migrate old field names to new ones for backward compatibility."""
    # Rename reddit_sentiment -> social_sentiment
    if "reddit_sentiment" in data:
        data["social_sentiment"] = data.pop("reddit_sentiment")
    # Rename reddit_buzz -> social_buzz
    if "reddit_buzz" in data:
        data["social_buzz"] = data.pop("reddit_buzz")
    # Set default sentiment_source for old plans
    if "sentiment_source" not in data or not data.get("sentiment_source"):
        # Old plans used Claude with Reddit search
        if data.get("social_sentiment") or data.get("social_buzz"):
            data["sentiment_source"] = "reddit"
    return data


class PlanStore:
    """Async storage for trading plans."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_db_path()
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the trading_plans table exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS trading_plans (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT DEFAULT 'active',

                    bias TEXT,
                    thesis TEXT,
                    original_thesis TEXT DEFAULT '',
                    entry_zone_low REAL,
                    entry_zone_high REAL,
                    stop_loss REAL,
                    stop_reasoning TEXT,
                    target_1 REAL,
                    target_2 REAL,
                    target_3 REAL,
                    target_reasoning TEXT,
                    risk_reward REAL,
                    position_size_pct REAL,

                    key_supports TEXT DEFAULT '[]',
                    key_resistances TEXT DEFAULT '[]',
                    invalidation_criteria TEXT,

                    trade_style TEXT,
                    trade_style_reasoning TEXT,
                    holding_period TEXT,
                    confidence INTEGER DEFAULT 0,

                    price_at_creation REAL,
                    rsi_at_creation REAL,
                    market_direction_at_creation TEXT,
                    technical_summary TEXT,

                    news_summary TEXT DEFAULT '',
                    social_sentiment TEXT DEFAULT '',
                    social_buzz TEXT DEFAULT '',
                    sentiment_source TEXT DEFAULT '',

                    created_at TEXT,
                    updated_at TEXT,
                    last_evaluation TEXT,
                    evaluation_notes TEXT,
                    validation_warnings TEXT DEFAULT '[]',

                    UNIQUE(user_id, symbol)
                )
            """)
            await db.commit()

        self._initialized = True

    async def save_plan(self, plan: TradingPlan) -> TradingPlan:
        """Save or update a trading plan."""
        await self._ensure_table()

        now = datetime.utcnow().isoformat()
        if not plan.created_at:
            plan.created_at = now
        plan.updated_at = now

        data = plan.to_dict()

        async with aiosqlite.connect(self.db_path) as db:
            # Upsert
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])
            updates = ", ".join([f"{k} = excluded.{k}" for k in data.keys() if k != "id"])

            await db.execute(f"""
                INSERT INTO trading_plans ({columns})
                VALUES ({placeholders})
                ON CONFLICT(user_id, symbol) DO UPDATE SET {updates}
            """, list(data.values()))
            await db.commit()

        logger.info(f"Saved trading plan for {plan.symbol} (user: {plan.user_id})")
        return plan

    async def get_plan(self, user_id: str, symbol: str) -> Optional[TradingPlan]:
        """Get trading plan for a user and symbol."""
        await self._ensure_table()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trading_plans WHERE user_id = ? AND symbol = ?",
                (user_id, symbol.upper())
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return TradingPlan.from_row(row)
        return None

    async def get_active_plans(self, user_id: str) -> List[TradingPlan]:
        """Get all active trading plans for a user."""
        await self._ensure_table()

        plans = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trading_plans WHERE user_id = ? AND status = 'active' ORDER BY updated_at DESC",
                (user_id,)
            ) as cursor:
                async for row in cursor:
                    plans.append(TradingPlan.from_row(row))
        return plans

    async def get_all_active_plans(self) -> List[TradingPlan]:
        """Get all active trading plans across all users."""
        await self._ensure_table()

        plans = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM trading_plans WHERE status = 'active' ORDER BY updated_at DESC"
            ) as cursor:
                async for row in cursor:
                    plans.append(TradingPlan.from_row(row))
        return plans

    async def get_plans_due_for_evaluation(self, minutes_threshold: int = 15) -> List[TradingPlan]:
        """Get all active plans across all users that are due for evaluation.

        A plan is due if:
        - It has a last_evaluation timestamp older than minutes_threshold, OR
        - It has no last_evaluation and created_at is older than minutes_threshold
        """
        await self._ensure_table()

        cutoff = (datetime.utcnow() - timedelta(minutes=minutes_threshold)).isoformat()

        plans = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM trading_plans
                   WHERE status = 'active'
                   AND (
                       (last_evaluation IS NULL OR last_evaluation = '')
                       AND created_at < ?
                       OR
                       (last_evaluation IS NOT NULL AND last_evaluation != ''
                        AND last_evaluation < ?)
                   )
                   ORDER BY COALESCE(NULLIF(last_evaluation, ''), created_at) ASC""",
                (cutoff, cutoff)
            ) as cursor:
                async for row in cursor:
                    plans.append(TradingPlan.from_row(row))
        return plans

    async def update_evaluation(
        self,
        user_id: str,
        symbol: str,
        notes: str,
        new_status: Optional[str] = None,
        adjustments: Optional[Dict[str, Any]] = None
    ) -> Optional[TradingPlan]:
        """Update the evaluation notes and optionally adjust plan values.

        Args:
            user_id: User ID
            symbol: Stock symbol
            notes: Evaluation notes from AI
            new_status: New plan status (active, invalidated, etc.)
            adjustments: Dict of field adjustments (stop_loss, target_1, etc.)
        """
        await self._ensure_table()

        now = datetime.utcnow().isoformat()

        # Build dynamic update query
        updates = ["last_evaluation = ?", "evaluation_notes = ?", "updated_at = ?"]
        values = [now, notes, now]

        if new_status:
            updates.append("status = ?")
            values.append(new_status)

        # Apply any adjustments to plan values
        if adjustments:
            allowed_fields = {
                "thesis",  # Can update thesis during evaluation
                "stop_loss", "target_1", "target_2", "target_3",
                "entry_zone_low", "entry_zone_high", "risk_reward",
                "stop_reasoning", "target_reasoning", "invalidation_criteria",
                "key_supports", "key_resistances"
            }
            # Fields that need JSON serialization (lists)
            list_fields = {"key_supports", "key_resistances"}

            for field, value in adjustments.items():
                if field in allowed_fields and value is not None:
                    updates.append(f"{field} = ?")
                    # Serialize list fields to JSON
                    if field in list_fields:
                        values.append(json.dumps(value))
                    else:
                        values.append(value)
                    logger.info(f"Adjusting {field} to {value} for {symbol}")

        values.extend([user_id, symbol.upper()])

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(f"""
                UPDATE trading_plans
                SET {', '.join(updates)}
                WHERE user_id = ? AND symbol = ?
            """, values)
            await db.commit()

        return await self.get_plan(user_id, symbol)

    async def delete_plan(self, user_id: str, symbol: str) -> bool:
        """Delete a trading plan."""
        await self._ensure_table()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM trading_plans WHERE user_id = ? AND symbol = ?",
                (user_id, symbol.upper())
            )
            await db.commit()
            return cursor.rowcount > 0


class DatabasePlanStore:
    """PostgreSQL-backed trading plan storage for production.

    Uses JSONB for flexible plan storage.
    """

    def __init__(self):
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the trading_plans table exists (handled by postgres.py init)."""
        self._initialized = True

    async def save_plan(self, plan: TradingPlan) -> TradingPlan:
        """Save or update a trading plan in Postgres."""
        from app.storage.postgres import get_connection

        now = datetime.utcnow().isoformat()
        if not plan.created_at:
            plan.created_at = now
        plan.updated_at = now

        # Convert to dict for JSONB storage
        plan_data = asdict(plan)

        async with get_connection() as conn:
            # Check if exists
            existing = await conn.fetchval(
                "SELECT id FROM trading_plans WHERE user_id = $1 AND symbol = $2",
                plan.user_id, plan.symbol.upper()
            )

            if existing:
                # Update
                await conn.execute(
                    """UPDATE trading_plans
                       SET plan_data = $1, status = $2, updated_at = $3
                       WHERE user_id = $4 AND symbol = $5""",
                    json.dumps(plan_data), plan.status, now,
                    plan.user_id, plan.symbol.upper()
                )
            else:
                # Insert
                await conn.execute(
                    """INSERT INTO trading_plans (id, user_id, symbol, plan_data, status, created_at, updated_at)
                       VALUES ($1, $2, $3, $4, $5, $6, $7)""",
                    plan.id, plan.user_id, plan.symbol.upper(),
                    json.dumps(plan_data), plan.status, now, now
                )

        logger.info(f"Saved trading plan for {plan.symbol} (user: {plan.user_id})")
        return plan

    async def get_plan(self, user_id: str, symbol: str) -> Optional[TradingPlan]:
        """Get trading plan for a user and symbol from Postgres."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT plan_data FROM trading_plans WHERE user_id = $1 AND symbol = $2",
                user_id, symbol.upper()
            )
            if row and row["plan_data"]:
                data = json.loads(row["plan_data"]) if isinstance(row["plan_data"], str) else row["plan_data"]
                data = _migrate_plan_data(data)
                return TradingPlan(**data)
        return None

    async def get_active_plans(self, user_id: str) -> List[TradingPlan]:
        """Get all active trading plans for a user from Postgres."""
        from app.storage.postgres import get_connection

        plans = []
        async with get_connection() as conn:
            rows = await conn.fetch(
                """SELECT plan_data FROM trading_plans
                   WHERE user_id = $1 AND status = 'active'
                   ORDER BY updated_at DESC""",
                user_id
            )
            for row in rows:
                if row["plan_data"]:
                    data = json.loads(row["plan_data"]) if isinstance(row["plan_data"], str) else row["plan_data"]
                    data = _migrate_plan_data(data)
                    plans.append(TradingPlan(**data))
        return plans

    async def get_all_active_plans(self) -> List[TradingPlan]:
        """Get all active trading plans across all users from Postgres."""
        from app.storage.postgres import get_connection

        plans = []
        async with get_connection() as conn:
            rows = await conn.fetch(
                """SELECT plan_data FROM trading_plans
                   WHERE status = 'active'
                   ORDER BY updated_at DESC"""
            )
            for row in rows:
                if row["plan_data"]:
                    data = json.loads(row["plan_data"]) if isinstance(row["plan_data"], str) else row["plan_data"]
                    data = _migrate_plan_data(data)
                    plans.append(TradingPlan(**data))
        return plans

    async def get_plans_due_for_evaluation(self, minutes_threshold: int = 15) -> List[TradingPlan]:
        """Get all active plans across all users that are due for evaluation.

        A plan is due if:
        - It has a last_evaluation timestamp older than minutes_threshold, OR
        - It has no last_evaluation and created_at is older than minutes_threshold
        """
        from app.storage.postgres import get_connection

        cutoff = (datetime.utcnow() - timedelta(minutes=minutes_threshold)).isoformat()

        plans = []
        async with get_connection() as conn:
            rows = await conn.fetch(
                """SELECT plan_data FROM trading_plans
                   WHERE status = 'active'
                   AND (
                       (plan_data->>'last_evaluation' IS NULL OR plan_data->>'last_evaluation' = '')
                       AND (plan_data->>'created_at' < $1 OR created_at < $1)
                       OR
                       (plan_data->>'last_evaluation' IS NOT NULL AND plan_data->>'last_evaluation' != ''
                        AND plan_data->>'last_evaluation' < $1)
                   )
                   ORDER BY COALESCE(NULLIF(plan_data->>'last_evaluation', ''), plan_data->>'created_at', created_at) ASC""",
                cutoff
            )
            for row in rows:
                if row["plan_data"]:
                    data = json.loads(row["plan_data"]) if isinstance(row["plan_data"], str) else row["plan_data"]
                    data = _migrate_plan_data(data)
                    plans.append(TradingPlan(**data))
        return plans

    async def update_evaluation(
        self,
        user_id: str,
        symbol: str,
        notes: str,
        new_status: Optional[str] = None,
        adjustments: Optional[Dict[str, Any]] = None
    ) -> Optional[TradingPlan]:
        """Update the evaluation notes and optionally adjust plan values."""
        from app.storage.postgres import get_connection

        now = datetime.utcnow().isoformat()
        symbol = symbol.upper()

        # Get existing plan
        plan = await self.get_plan(user_id, symbol)
        if not plan:
            return None

        # Update fields
        plan.last_evaluation = now
        plan.evaluation_notes = notes
        plan.updated_at = now

        if new_status:
            plan.status = new_status

        # Apply adjustments
        if adjustments:
            allowed_fields = {
                "thesis",  # Can update thesis during evaluation
                "stop_loss", "target_1", "target_2", "target_3",
                "entry_zone_low", "entry_zone_high", "risk_reward",
                "stop_reasoning", "target_reasoning", "invalidation_criteria",
                "key_supports", "key_resistances"
            }
            for field_name, value in adjustments.items():
                if field_name in allowed_fields and value is not None:
                    setattr(plan, field_name, value)
                    logger.info(f"Adjusting {field_name} to {value} for {symbol}")

        # Save updated plan
        return await self.save_plan(plan)

    async def delete_plan(self, user_id: str, symbol: str) -> bool:
        """Delete a trading plan from Postgres."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            result = await conn.execute(
                "DELETE FROM trading_plans WHERE user_id = $1 AND symbol = $2",
                user_id, symbol.upper()
            )
            deleted = "DELETE 1" in result

        if deleted:
            logger.info(f"Deleted trading plan for {symbol} (user: {user_id})")
        return deleted


# Singleton instances
_sqlite_plan_store: Optional[PlanStore] = None
_postgres_plan_store: Optional[DatabasePlanStore] = None


def get_plan_store():
    """Get the appropriate plan store based on configuration.

    Returns SQLite-based PlanStore for development,
    PostgreSQL-based DatabasePlanStore for production.
    """
    settings = get_settings()

    if settings.use_postgres:
        global _postgres_plan_store
        if _postgres_plan_store is None:
            _postgres_plan_store = DatabasePlanStore()
        return _postgres_plan_store
    else:
        global _sqlite_plan_store
        if _sqlite_plan_store is None:
            _sqlite_plan_store = PlanStore()
        return _sqlite_plan_store
