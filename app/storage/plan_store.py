"""Trading Plan storage for the planning agent.

Stores AI-generated trading plans that evolve as the stock progresses.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import aiosqlite

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
    thesis: str = ""  # Why this trade makes sense
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

    # External sentiment (from web search)
    news_summary: str = ""  # Brief summary of recent news/catalysts
    reddit_sentiment: str = ""  # bullish, bearish, neutral, mixed, none
    reddit_buzz: str = ""  # Summary of Reddit discussion if found

    # Tracking
    created_at: str = ""
    updated_at: str = ""
    last_evaluation: str = ""
    evaluation_notes: str = ""  # Latest AI evaluation

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        # Convert lists to JSON strings for storage
        data["key_supports"] = json.dumps(data["key_supports"])
        data["key_resistances"] = json.dumps(data["key_resistances"])
        return data

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "TradingPlan":
        """Create from database row."""
        data = dict(row)
        # Parse JSON lists
        data["key_supports"] = json.loads(data.get("key_supports", "[]"))
        data["key_resistances"] = json.loads(data.get("key_resistances", "[]"))
        return cls(**data)


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
                    reddit_sentiment TEXT DEFAULT '',
                    reddit_buzz TEXT DEFAULT '',

                    created_at TEXT,
                    updated_at TEXT,
                    last_evaluation TEXT,
                    evaluation_notes TEXT,

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


# Singleton instance
_plan_store: Optional[PlanStore] = None


def get_plan_store() -> PlanStore:
    """Get singleton PlanStore instance."""
    global _plan_store
    if _plan_store is None:
        _plan_store = PlanStore()
    return _plan_store
