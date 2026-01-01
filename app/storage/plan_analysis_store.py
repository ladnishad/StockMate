"""Plan Analysis storage for V2 multi-agent trading plan generation.

Stores complete V2 analysis results including all sub-agent reports,
selected plan, alternatives, and context at analysis time.
Supports both SQLite (development) and PostgreSQL (production).
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field, asdict

import aiosqlite

from app.config import get_settings
from app.storage.database import get_db_path

logger = logging.getLogger(__name__)


@dataclass
class PlanAnalysis:
    """V2 multi-agent analysis result."""

    id: str
    user_id: str
    symbol: str
    analysis_version: str = "v2"
    status: str = "draft"  # draft, accepted, superseded

    # Selected plan info
    selected_style: Optional[str] = None  # day, swing, position
    selection_reasoning: Optional[str] = None

    # Full analysis data (contains selected_plan, alternatives, etc.)
    analysis_data: Dict[str, Any] = field(default_factory=dict)

    # Individual sub-agent reports for quick access
    day_trade_report: Optional[Dict[str, Any]] = None
    swing_trade_report: Optional[Dict[str, Any]] = None
    position_trade_report: Optional[Dict[str, Any]] = None

    # Context at analysis time
    market_context: Optional[Dict[str, Any]] = None
    news_context: Optional[Dict[str, Any]] = None

    # Tracking
    analysis_duration_ms: Optional[int] = None
    created_at: str = ""
    updated_at: str = ""

    # Link to approved trading_plan (when user accepts)
    linked_trading_plan_id: Optional[str] = None

    @classmethod
    def create(
        cls,
        user_id: str,
        symbol: str,
        selected_style: str,
        selection_reasoning: str,
        analysis_data: Dict[str, Any],
        day_trade_report: Optional[Dict[str, Any]] = None,
        swing_trade_report: Optional[Dict[str, Any]] = None,
        position_trade_report: Optional[Dict[str, Any]] = None,
        market_context: Optional[Dict[str, Any]] = None,
        news_context: Optional[Dict[str, Any]] = None,
        analysis_duration_ms: Optional[int] = None,
    ) -> "PlanAnalysis":
        """Create a new PlanAnalysis with auto-generated ID and timestamps."""
        now = datetime.utcnow().isoformat()
        return cls(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol.upper(),
            selected_style=selected_style,
            selection_reasoning=selection_reasoning,
            analysis_data=analysis_data,
            day_trade_report=day_trade_report,
            swing_trade_report=swing_trade_report,
            position_trade_report=position_trade_report,
            market_context=market_context,
            news_context=news_context,
            analysis_duration_ms=analysis_duration_ms,
            created_at=now,
            updated_at=now,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        # Convert dicts to JSON strings for SQLite
        for key in ["analysis_data", "day_trade_report", "swing_trade_report",
                    "position_trade_report", "market_context", "news_context"]:
            if data[key] is not None:
                data[key] = json.dumps(data[key])
            else:
                data[key] = None
        return data

    @classmethod
    def from_row(cls, row) -> "PlanAnalysis":
        """Create from database row."""
        data = dict(row)
        # Parse JSON fields
        for key in ["analysis_data", "day_trade_report", "swing_trade_report",
                    "position_trade_report", "market_context", "news_context"]:
            if data.get(key):
                if isinstance(data[key], str):
                    data[key] = json.loads(data[key])
            else:
                data[key] = None if key != "analysis_data" else {}
        return cls(**data)


class PlanAnalysisStore:
    """SQLite-backed storage for plan analyses (development)."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_db_path()
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the plan_analyses table exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS plan_analyses (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    analysis_version TEXT DEFAULT 'v2',
                    status TEXT DEFAULT 'draft',

                    selected_style TEXT,
                    selection_reasoning TEXT,

                    analysis_data TEXT NOT NULL,

                    day_trade_report TEXT,
                    swing_trade_report TEXT,
                    position_trade_report TEXT,

                    market_context TEXT,
                    news_context TEXT,

                    analysis_duration_ms INTEGER,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,

                    linked_trading_plan_id TEXT
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_analyses_user_symbol
                ON plan_analyses (user_id, symbol)
            """)
            await db.commit()

        self._initialized = True

    async def save_analysis(self, analysis: PlanAnalysis) -> PlanAnalysis:
        """Save a new analysis."""
        await self._ensure_table()

        now = datetime.utcnow().isoformat()
        analysis.updated_at = now

        data = analysis.to_dict()

        async with aiosqlite.connect(self.db_path) as db:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])

            await db.execute(f"""
                INSERT OR REPLACE INTO plan_analyses ({columns})
                VALUES ({placeholders})
            """, list(data.values()))
            await db.commit()

        logger.info(f"Saved plan analysis {analysis.id} for {analysis.symbol}")
        return analysis

    async def get_analysis(self, analysis_id: str) -> Optional[PlanAnalysis]:
        """Get analysis by ID."""
        await self._ensure_table()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM plan_analyses WHERE id = ?",
                (analysis_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PlanAnalysis.from_row(row)
        return None

    async def get_latest_analysis(self, user_id: str, symbol: str) -> Optional[PlanAnalysis]:
        """Get the most recent analysis for a symbol."""
        await self._ensure_table()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM plan_analyses
                   WHERE user_id = ? AND symbol = ?
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, symbol.upper())
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PlanAnalysis.from_row(row)
        return None

    async def get_analyses_for_symbol(
        self, user_id: str, symbol: str, limit: int = 10
    ) -> List[PlanAnalysis]:
        """Get all analyses for a symbol (for history)."""
        await self._ensure_table()

        analyses = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM plan_analyses
                   WHERE user_id = ? AND symbol = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, symbol.upper(), limit)
            ) as cursor:
                async for row in cursor:
                    analyses.append(PlanAnalysis.from_row(row))
        return analyses

    async def archive_previous_analyses(self, user_id: str, symbol: str, current_id: str):
        """Mark previous analyses as superseded when new one is created."""
        await self._ensure_table()

        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE plan_analyses
                   SET status = 'superseded', updated_at = ?
                   WHERE user_id = ? AND symbol = ? AND id != ? AND status = 'draft'""",
                (now, user_id, symbol.upper(), current_id)
            )
            await db.commit()

    async def accept_analysis(
        self, analysis_id: str, linked_plan_id: str
    ) -> Optional[PlanAnalysis]:
        """Mark analysis as accepted and link to trading plan."""
        await self._ensure_table()

        now = datetime.utcnow().isoformat()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """UPDATE plan_analyses
                   SET status = 'accepted', linked_trading_plan_id = ?, updated_at = ?
                   WHERE id = ?""",
                (linked_plan_id, now, analysis_id)
            )
            await db.commit()

        return await self.get_analysis(analysis_id)


class DatabasePlanAnalysisStore:
    """PostgreSQL-backed storage for plan analyses (production)."""

    def __init__(self):
        self._initialized = False

    async def _ensure_table(self):
        """Table is created by postgres.py init."""
        self._initialized = True

    async def save_analysis(self, analysis: PlanAnalysis) -> PlanAnalysis:
        """Save a new analysis to Postgres."""
        from app.storage.postgres import get_connection

        now = datetime.utcnow().isoformat()
        analysis.updated_at = now

        async with get_connection() as conn:
            # Check if exists
            existing = await conn.fetchval(
                "SELECT id FROM plan_analyses WHERE id = $1",
                analysis.id
            )

            if existing:
                # Update
                await conn.execute(
                    """UPDATE plan_analyses SET
                       status = $1, selected_style = $2, selection_reasoning = $3,
                       analysis_data = $4, day_trade_report = $5, swing_trade_report = $6,
                       position_trade_report = $7, market_context = $8, news_context = $9,
                       analysis_duration_ms = $10, updated_at = $11, linked_trading_plan_id = $12
                       WHERE id = $13""",
                    analysis.status, analysis.selected_style, analysis.selection_reasoning,
                    json.dumps(analysis.analysis_data),
                    json.dumps(analysis.day_trade_report) if analysis.day_trade_report else None,
                    json.dumps(analysis.swing_trade_report) if analysis.swing_trade_report else None,
                    json.dumps(analysis.position_trade_report) if analysis.position_trade_report else None,
                    json.dumps(analysis.market_context) if analysis.market_context else None,
                    json.dumps(analysis.news_context) if analysis.news_context else None,
                    analysis.analysis_duration_ms, now, analysis.linked_trading_plan_id,
                    analysis.id
                )
            else:
                # Insert
                await conn.execute(
                    """INSERT INTO plan_analyses
                       (id, user_id, symbol, analysis_version, status, selected_style,
                        selection_reasoning, analysis_data, day_trade_report, swing_trade_report,
                        position_trade_report, market_context, news_context, analysis_duration_ms,
                        created_at, updated_at, linked_trading_plan_id)
                       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)""",
                    analysis.id, analysis.user_id, analysis.symbol.upper(),
                    analysis.analysis_version, analysis.status, analysis.selected_style,
                    analysis.selection_reasoning, json.dumps(analysis.analysis_data),
                    json.dumps(analysis.day_trade_report) if analysis.day_trade_report else None,
                    json.dumps(analysis.swing_trade_report) if analysis.swing_trade_report else None,
                    json.dumps(analysis.position_trade_report) if analysis.position_trade_report else None,
                    json.dumps(analysis.market_context) if analysis.market_context else None,
                    json.dumps(analysis.news_context) if analysis.news_context else None,
                    analysis.analysis_duration_ms, analysis.created_at, now,
                    analysis.linked_trading_plan_id
                )

        logger.info(f"Saved plan analysis {analysis.id} for {analysis.symbol}")
        return analysis

    async def get_analysis(self, analysis_id: str) -> Optional[PlanAnalysis]:
        """Get analysis by ID from Postgres."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM plan_analyses WHERE id = $1",
                analysis_id
            )
            if row:
                return self._row_to_analysis(row)
        return None

    async def get_latest_analysis(self, user_id: str, symbol: str) -> Optional[PlanAnalysis]:
        """Get the most recent analysis for a symbol."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM plan_analyses
                   WHERE user_id = $1 AND symbol = $2
                   ORDER BY created_at DESC LIMIT 1""",
                user_id, symbol.upper()
            )
            if row:
                return self._row_to_analysis(row)
        return None

    async def get_analyses_for_symbol(
        self, user_id: str, symbol: str, limit: int = 10
    ) -> List[PlanAnalysis]:
        """Get all analyses for a symbol (for history)."""
        from app.storage.postgres import get_connection

        analyses = []
        async with get_connection() as conn:
            rows = await conn.fetch(
                """SELECT * FROM plan_analyses
                   WHERE user_id = $1 AND symbol = $2
                   ORDER BY created_at DESC LIMIT $3""",
                user_id, symbol.upper(), limit
            )
            for row in rows:
                analyses.append(self._row_to_analysis(row))
        return analyses

    async def archive_previous_analyses(self, user_id: str, symbol: str, current_id: str):
        """Mark previous analyses as superseded when new one is created."""
        from app.storage.postgres import get_connection

        now = datetime.utcnow().isoformat()
        async with get_connection() as conn:
            await conn.execute(
                """UPDATE plan_analyses
                   SET status = 'superseded', updated_at = $1
                   WHERE user_id = $2 AND symbol = $3 AND id != $4 AND status = 'draft'""",
                now, user_id, symbol.upper(), current_id
            )

    async def accept_analysis(
        self, analysis_id: str, linked_plan_id: str
    ) -> Optional[PlanAnalysis]:
        """Mark analysis as accepted and link to trading plan."""
        from app.storage.postgres import get_connection

        now = datetime.utcnow().isoformat()
        async with get_connection() as conn:
            await conn.execute(
                """UPDATE plan_analyses
                   SET status = 'accepted', linked_trading_plan_id = $1, updated_at = $2
                   WHERE id = $3""",
                linked_plan_id, now, analysis_id
            )

        return await self.get_analysis(analysis_id)

    def _row_to_analysis(self, row) -> PlanAnalysis:
        """Convert database row to PlanAnalysis."""
        data = dict(row)
        # Parse JSON fields
        for key in ["analysis_data", "day_trade_report", "swing_trade_report",
                    "position_trade_report", "market_context", "news_context"]:
            if data.get(key):
                if isinstance(data[key], str):
                    data[key] = json.loads(data[key])
            else:
                data[key] = None if key != "analysis_data" else {}
        return PlanAnalysis(**data)


# Singleton instances
_sqlite_store: Optional[PlanAnalysisStore] = None
_postgres_store: Optional[DatabasePlanAnalysisStore] = None


def get_plan_analysis_store():
    """Get the appropriate plan analysis store based on configuration.

    Returns SQLite-based store for development,
    PostgreSQL-based store for production.
    """
    settings = get_settings()

    if settings.use_postgres:
        global _postgres_store
        if _postgres_store is None:
            _postgres_store = DatabasePlanAnalysisStore()
        return _postgres_store
    else:
        global _sqlite_store
        if _sqlite_store is None:
            _sqlite_store = PlanAnalysisStore()
        return _sqlite_store
