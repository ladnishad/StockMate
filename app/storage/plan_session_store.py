"""Plan Session storage for interactive trading plan conversations.

Stores planning sessions that track the back-and-forth between user and AI
when creating, adjusting, and approving trading plans.
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from dataclasses import dataclass, field, asdict

import aiosqlite

from app.config import get_settings
from app.storage.database import get_db_path

logger = logging.getLogger(__name__)


@dataclass
class PlanMessage:
    """A single message in the planning conversation."""

    id: str
    role: Literal["user", "assistant", "system"]
    content: str
    message_type: Literal["question", "answer", "adjustment_request", "adjustment_response", "approval", "info"]
    timestamp: str
    # For adjustment responses, store the options presented
    options: List[Dict[str, Any]] = field(default_factory=list)
    # For selected options
    selected_option: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlanMessage":
        return cls(**data)


@dataclass
class PlanSession:
    """Interactive planning session for a trading plan."""

    id: str
    user_id: str
    symbol: str
    status: Literal["generating", "draft", "refining", "approved", "rejected", "expired"]

    # Conversation history
    messages: List[PlanMessage] = field(default_factory=list)

    # Draft plan data (JSON - the evolving plan)
    draft_plan_data: Optional[Dict[str, Any]] = None

    # Approved plan ID (references trading_plans table)
    approved_plan_id: Optional[str] = None

    # Tracking
    revision_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    approved_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data["messages"] = json.dumps([m.to_dict() for m in self.messages])
        data["draft_plan_data"] = json.dumps(self.draft_plan_data) if self.draft_plan_data else None
        return data

    @classmethod
    def from_row(cls, row: aiosqlite.Row) -> "PlanSession":
        """Create from database row."""
        data = dict(row)
        # Parse JSON fields
        messages_raw = data.get("messages", "[]")
        data["messages"] = [PlanMessage.from_dict(m) for m in json.loads(messages_raw)]
        draft_raw = data.get("draft_plan_data")
        data["draft_plan_data"] = json.loads(draft_raw) if draft_raw else None
        return cls(**data)

    def add_message(
        self,
        role: Literal["user", "assistant", "system"],
        content: str,
        message_type: str,
        options: List[Dict[str, Any]] = None,
        selected_option: str = None
    ) -> PlanMessage:
        """Add a message to the conversation."""
        msg = PlanMessage(
            id=str(uuid.uuid4()),
            role=role,
            content=content,
            message_type=message_type,
            timestamp=datetime.utcnow().isoformat(),
            options=options or [],
            selected_option=selected_option
        )
        self.messages.append(msg)
        return msg


class PlanSessionStore:
    """Async storage for planning sessions."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_db_path()
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the plan_sessions table exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS plan_sessions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT DEFAULT 'generating',

                    messages TEXT DEFAULT '[]',
                    draft_plan_data TEXT,
                    approved_plan_id TEXT,

                    revision_count INTEGER DEFAULT 0,
                    created_at TEXT,
                    updated_at TEXT,
                    approved_at TEXT
                )
            """)
            # Index for quick lookups
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_plan_sessions_user_symbol
                ON plan_sessions(user_id, symbol)
            """)
            await db.commit()

        self._initialized = True

    async def create_session(self, user_id: str, symbol: str) -> PlanSession:
        """Create a new planning session."""
        await self._ensure_table()

        now = datetime.utcnow().isoformat()
        session = PlanSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol.upper(),
            status="generating",
            created_at=now,
            updated_at=now
        )

        data = session.to_dict()

        async with aiosqlite.connect(self.db_path) as db:
            columns = ", ".join(data.keys())
            placeholders = ", ".join(["?" for _ in data])

            await db.execute(f"""
                INSERT INTO plan_sessions ({columns})
                VALUES ({placeholders})
            """, list(data.values()))
            await db.commit()

        logger.info(f"Created plan session {session.id} for {symbol} (user: {user_id})")
        return session

    async def get_session(self, session_id: str) -> Optional[PlanSession]:
        """Get a planning session by ID."""
        await self._ensure_table()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM plan_sessions WHERE id = ?",
                (session_id,)
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PlanSession.from_row(row)
        return None

    async def get_active_session(self, user_id: str, symbol: str) -> Optional[PlanSession]:
        """Get the most recent active (non-approved/rejected) session for a user and symbol."""
        await self._ensure_table()

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM plan_sessions
                   WHERE user_id = ? AND symbol = ? AND status NOT IN ('approved', 'rejected', 'expired')
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, symbol.upper())
            ) as cursor:
                row = await cursor.fetchone()
                if row:
                    return PlanSession.from_row(row)
        return None

    async def update_session(self, session: PlanSession) -> PlanSession:
        """Update an existing session."""
        await self._ensure_table()

        session.updated_at = datetime.utcnow().isoformat()
        data = session.to_dict()

        async with aiosqlite.connect(self.db_path) as db:
            set_clause = ", ".join([f"{k} = ?" for k in data.keys() if k != "id"])
            values = [v for k, v in data.items() if k != "id"]
            values.append(session.id)

            await db.execute(f"""
                UPDATE plan_sessions
                SET {set_clause}
                WHERE id = ?
            """, values)
            await db.commit()

        return session

    async def set_draft_plan(self, session_id: str, draft_plan: Dict[str, Any]) -> Optional[PlanSession]:
        """Set the draft plan for a session and update status to 'draft'."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.draft_plan_data = draft_plan
        session.status = "draft"
        return await self.update_session(session)

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        message_type: str,
        options: List[Dict[str, Any]] = None,
        selected_option: str = None
    ) -> Optional[PlanSession]:
        """Add a message to a session's conversation."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.add_message(role, content, message_type, options, selected_option)
        return await self.update_session(session)

    async def approve_session(self, session_id: str, approved_plan_id: str) -> Optional[PlanSession]:
        """Mark a session as approved and link to the final plan."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = "approved"
        session.approved_plan_id = approved_plan_id
        session.approved_at = datetime.utcnow().isoformat()
        return await self.update_session(session)

    async def reject_session(self, session_id: str) -> Optional[PlanSession]:
        """Mark a session as rejected."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = "rejected"
        return await self.update_session(session)

    async def increment_revision(self, session_id: str) -> Optional[PlanSession]:
        """Increment the revision count (when user requests adjustment)."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.revision_count += 1
        session.status = "refining"
        return await self.update_session(session)

    async def get_session_history(self, user_id: str, symbol: str, limit: int = 10) -> List[PlanSession]:
        """Get recent planning sessions for a user and symbol."""
        await self._ensure_table()

        sessions = []
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                """SELECT * FROM plan_sessions
                   WHERE user_id = ? AND symbol = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, symbol.upper(), limit)
            ) as cursor:
                async for row in cursor:
                    sessions.append(PlanSession.from_row(row))
        return sessions

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than specified days that are not approved."""
        await self._ensure_table()

        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """DELETE FROM plan_sessions
                   WHERE created_at < ? AND status != 'approved'""",
                (cutoff,)
            )
            await db.commit()
            return cursor.rowcount


class DatabasePlanSessionStore:
    """PostgreSQL storage for planning sessions (production)."""

    def __init__(self):
        # Import here to avoid circular imports
        from app.storage.postgres import get_connection
        self._get_connection = get_connection

    def _row_to_session(self, row) -> PlanSession:
        """Convert asyncpg Record to PlanSession."""
        data = dict(row)
        # Parse JSON fields - asyncpg returns dicts for JSONB, not strings
        messages_raw = data.get("messages", [])
        if isinstance(messages_raw, str):
            messages_raw = json.loads(messages_raw)
        data["messages"] = [PlanMessage.from_dict(m) for m in messages_raw]

        draft_raw = data.get("draft_plan_data")
        if isinstance(draft_raw, str):
            data["draft_plan_data"] = json.loads(draft_raw) if draft_raw else None
        # If it's already a dict (JSONB), use as-is
        elif isinstance(draft_raw, dict):
            data["draft_plan_data"] = draft_raw
        else:
            data["draft_plan_data"] = None

        return PlanSession(**data)

    async def create_session(self, user_id: str, symbol: str) -> PlanSession:
        """Create a new planning session."""
        now = datetime.utcnow().isoformat()
        session = PlanSession(
            id=str(uuid.uuid4()),
            user_id=user_id,
            symbol=symbol.upper(),
            status="generating",
            created_at=now,
            updated_at=now
        )

        async with self._get_connection() as conn:
            await conn.execute(
                """INSERT INTO plan_sessions
                   (id, user_id, symbol, status, messages, draft_plan_data,
                    approved_plan_id, revision_count, created_at, updated_at, approved_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (session.id, session.user_id, session.symbol, session.status,
                 json.dumps([]), None, None, 0, now, now, None)
            )

        logger.info(f"Created plan session {session.id} for {symbol} (user: {user_id})")
        return session

    async def get_session(self, session_id: str) -> Optional[PlanSession]:
        """Get a planning session by ID."""
        async with self._get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM plan_sessions WHERE id = ?",
                (session_id,)
            )
            if row:
                return self._row_to_session(row)
        return None

    async def get_active_session(self, user_id: str, symbol: str) -> Optional[PlanSession]:
        """Get the most recent active session for a user and symbol."""
        async with self._get_connection() as conn:
            row = await conn.fetchrow(
                """SELECT * FROM plan_sessions
                   WHERE user_id = ? AND symbol = ? AND status NOT IN ('approved', 'rejected', 'expired')
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id, symbol.upper())
            )
            if row:
                return self._row_to_session(row)
        return None

    async def update_session(self, session: PlanSession) -> PlanSession:
        """Update an existing session."""
        session.updated_at = datetime.utcnow().isoformat()

        messages_json = json.dumps([m.to_dict() for m in session.messages])
        draft_json = json.dumps(session.draft_plan_data) if session.draft_plan_data else None

        async with self._get_connection() as conn:
            await conn.execute(
                """UPDATE plan_sessions
                   SET user_id = ?, symbol = ?, status = ?, messages = ?,
                       draft_plan_data = ?, approved_plan_id = ?, revision_count = ?,
                       updated_at = ?, approved_at = ?
                   WHERE id = ?""",
                (session.user_id, session.symbol, session.status, messages_json,
                 draft_json, session.approved_plan_id, session.revision_count,
                 session.updated_at, session.approved_at, session.id)
            )

        return session

    async def set_draft_plan(self, session_id: str, draft_plan: Dict[str, Any]) -> Optional[PlanSession]:
        """Set the draft plan for a session and update status to 'draft'."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.draft_plan_data = draft_plan
        session.status = "draft"
        return await self.update_session(session)

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        message_type: str,
        options: List[Dict[str, Any]] = None,
        selected_option: str = None
    ) -> Optional[PlanSession]:
        """Add a message to a session's conversation."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.add_message(role, content, message_type, options, selected_option)
        return await self.update_session(session)

    async def approve_session(self, session_id: str, approved_plan_id: str) -> Optional[PlanSession]:
        """Mark a session as approved and link to the final plan."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = "approved"
        session.approved_plan_id = approved_plan_id
        session.approved_at = datetime.utcnow().isoformat()
        return await self.update_session(session)

    async def reject_session(self, session_id: str) -> Optional[PlanSession]:
        """Mark a session as rejected."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.status = "rejected"
        return await self.update_session(session)

    async def increment_revision(self, session_id: str) -> Optional[PlanSession]:
        """Increment the revision count."""
        session = await self.get_session(session_id)
        if not session:
            return None

        session.revision_count += 1
        session.status = "refining"
        return await self.update_session(session)

    async def get_session_history(self, user_id: str, symbol: str, limit: int = 10) -> List[PlanSession]:
        """Get recent planning sessions for a user and symbol."""
        sessions = []
        async with self._get_connection() as conn:
            rows = await conn.fetch(
                """SELECT * FROM plan_sessions
                   WHERE user_id = ? AND symbol = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, symbol.upper(), limit)
            )
            for row in rows:
                sessions.append(self._row_to_session(row))
        return sessions

    async def cleanup_old_sessions(self, days: int = 30) -> int:
        """Delete sessions older than specified days that are not approved."""
        from datetime import timedelta
        cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()

        async with self._get_connection() as conn:
            result = await conn.execute(
                """DELETE FROM plan_sessions
                   WHERE created_at < ? AND status != 'approved'""",
                (cutoff,)
            )
            return result.rowcount


# Singleton instances
_sqlite_store: Optional[PlanSessionStore] = None
_postgres_store: Optional[DatabasePlanSessionStore] = None


def get_plan_session_store():
    """Get the appropriate plan session store based on configuration."""
    settings = get_settings()

    if settings.use_postgres:
        global _postgres_store
        if _postgres_store is None:
            _postgres_store = DatabasePlanSessionStore()
        return _postgres_store
    else:
        global _sqlite_store
        if _sqlite_store is None:
            _sqlite_store = PlanSessionStore()
        return _sqlite_store
