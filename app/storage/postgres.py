"""PostgreSQL database setup and management.

Provides async PostgreSQL database for production:
- Connection pooling via asyncpg
- Same schema as SQLite for seamless switching
- Environment-based database selection
"""

import asyncpg
import logging
from typing import Optional, Any
from contextlib import asynccontextmanager

from app.config import get_settings

logger = logging.getLogger(__name__)

# Connection pool
_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create the connection pool."""
    global _pool
    if _pool is None:
        settings = get_settings()
        if not settings.database_url:
            raise ValueError("DATABASE_URL not configured")

        _pool = await asyncpg.create_pool(
            settings.database_url,
            min_size=2,
            max_size=10,
            command_timeout=60,
            statement_cache_size=0,  # Required for pgbouncer (Supabase) compatibility
        )
        logger.info("PostgreSQL connection pool created")
    return _pool


async def close_pool():
    """Close the connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


@asynccontextmanager
async def get_connection():
    """Get a connection from the pool.

    Usage:
        async with get_connection() as conn:
            await conn.execute(...)
    """
    pool = await get_pool()
    async with pool.acquire() as conn:
        yield conn


async def init_postgres_tables():
    """Create tables if they don't exist (PostgreSQL version)."""
    async with get_connection() as conn:
        # Positions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT DEFAULT 'watching',
                entry_price DOUBLE PRECISION,
                entry_date TEXT,
                entries JSONB DEFAULT '[]',
                avg_entry_price DOUBLE PRECISION,
                exits JSONB DEFAULT '[]',
                avg_exit_price DOUBLE PRECISION,
                current_size INTEGER DEFAULT 0,
                original_size INTEGER DEFAULT 0,
                stop_loss DOUBLE PRECISION,
                target_1 DOUBLE PRECISION,
                target_2 DOUBLE PRECISION,
                target_3 DOUBLE PRECISION,
                targets_hit JSONB DEFAULT '[]',
                cost_basis DOUBLE PRECISION,
                realized_pnl DOUBLE PRECISION,
                realized_pnl_pct DOUBLE PRECISION,
                unrealized_pnl DOUBLE PRECISION,
                unrealized_pnl_pct DOUBLE PRECISION,
                trade_type TEXT NOT NULL,
                notes TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, symbol)
            )
        """)

        # Create index for user lookups
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_positions_user_id
            ON positions (user_id)
        """)

        # Alerts table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                price_at_alert DOUBLE PRECISION NOT NULL,
                sent_at TEXT NOT NULL,
                acknowledged INTEGER DEFAULT 0
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_alerts_user_symbol_type
            ON alerts (user_id, symbol, alert_type, sent_at)
        """)

        # Device tokens table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS device_tokens (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                device_token TEXT NOT NULL UNIQUE,
                platform TEXT DEFAULT 'ios',
                created_at TEXT NOT NULL,
                last_used TEXT
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_device_tokens_user
            ON device_tokens (user_id)
        """)

        # Agent contexts table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_contexts (
                symbol TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                context_data TEXT NOT NULL,
                session_id TEXT,
                updated_at TEXT NOT NULL
            )
        """)

        # Watchlist table (migrated from JSON file)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                notes TEXT,
                alerts_enabled BOOLEAN DEFAULT true,
                added_at TEXT NOT NULL,
                UNIQUE(user_id, symbol)
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_watchlist_user_id
            ON watchlist (user_id)
        """)

        # Trading plans table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trading_plans (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                plan_data JSONB NOT NULL,
                status TEXT DEFAULT 'active',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, symbol)
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_trading_plans_user_id
            ON trading_plans (user_id)
        """)

        # Conversations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT,
                messages JSONB NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(user_id, symbol)
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_conversations_user_id
            ON conversations (user_id)
        """)

        # Plan sessions table (for interactive Claude Code-style planning)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS plan_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                status TEXT DEFAULT 'generating',
                messages JSONB DEFAULT '[]',
                draft_plan_data JSONB,
                approved_plan_id TEXT,
                revision_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                approved_at TEXT
            )
        """)

        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_plan_sessions_user_symbol
            ON plan_sessions (user_id, symbol)
        """)

        logger.info("PostgreSQL tables initialized")


class PostgresDatabase:
    """Async PostgreSQL database manager."""

    def __init__(self):
        """Initialize database."""
        pass

    @asynccontextmanager
    async def connection(self):
        """Get database connection context manager."""
        async with get_connection() as conn:
            yield PostgresConnection(conn)

    async def initialize(self) -> None:
        """Initialize database tables."""
        await init_postgres_tables()


class PostgresCursor:
    """Cursor-like wrapper to mimic SQLite cursor behavior for asyncpg."""

    def __init__(self, result: Any, status: str):
        self._result = result  # Can be a list of rows or None
        self._status = status
        self._index = 0

    @property
    def rowcount(self) -> int:
        """Get number of affected rows from status string."""
        # asyncpg returns strings like "DELETE 1", "UPDATE 2", "INSERT 0 1"
        if self._status:
            parts = self._status.split()
            if len(parts) >= 2 and parts[-1].isdigit():
                return int(parts[-1])
        return 0

    async def fetchone(self) -> Optional[Any]:
        """Fetch one row."""
        if isinstance(self._result, list) and self._index < len(self._result):
            row = self._result[self._index]
            self._index += 1
            return row
        return self._result if self._index == 0 else None

    async def fetchall(self) -> list:
        """Fetch all rows."""
        if isinstance(self._result, list):
            return self._result
        return [self._result] if self._result else []


class PostgresConnection:
    """Wrapper for asyncpg connection to provide consistent interface."""

    def __init__(self, conn: asyncpg.Connection):
        self._conn = conn

    def _normalize_args(self, args: tuple) -> tuple:
        """Normalize args to handle both SQLite-style tuple and individual args.

        SQLite: execute(query, (arg1, arg2))  ->  args = ((arg1, arg2),)
        asyncpg: execute(query, arg1, arg2)   ->  args = (arg1, arg2)

        This handles the SQLite-style tuple case.
        """
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            return tuple(args[0])
        return args

    async def execute(self, query: str, *args) -> PostgresCursor:
        """Execute a query and return a cursor-like object."""
        pg_query = self._convert_placeholders(query)
        normalized_args = self._normalize_args(args)

        # Determine if this is a SELECT query
        query_upper = query.strip().upper()
        if query_upper.startswith("SELECT"):
            # For SELECT, fetch all rows and wrap in cursor
            rows = await self._conn.fetch(pg_query, *normalized_args)
            return PostgresCursor(rows, f"SELECT {len(rows)}")
        else:
            # For INSERT/UPDATE/DELETE, execute and wrap status
            status = await self._conn.execute(pg_query, *normalized_args)
            return PostgresCursor(None, status)

    async def commit(self):
        """No-op for asyncpg (auto-commit by default)."""
        pass

    async def fetch(self, query: str, *args) -> list:
        """Fetch multiple rows."""
        pg_query = self._convert_placeholders(query)
        normalized_args = self._normalize_args(args)
        return await self._conn.fetch(pg_query, *normalized_args)

    async def fetchrow(self, query: str, *args) -> Optional[Any]:
        """Fetch a single row."""
        pg_query = self._convert_placeholders(query)
        normalized_args = self._normalize_args(args)
        return await self._conn.fetchrow(pg_query, *normalized_args)

    async def fetchval(self, query: str, *args) -> Any:
        """Fetch a single value."""
        pg_query = self._convert_placeholders(query)
        normalized_args = self._normalize_args(args)
        return await self._conn.fetchval(pg_query, *normalized_args)

    def _convert_placeholders(self, query: str) -> str:
        """Convert SQLite ? placeholders to PostgreSQL $1, $2, etc."""
        result = []
        count = 0
        i = 0
        while i < len(query):
            if query[i] == '?':
                count += 1
                result.append(f'${count}')
            else:
                result.append(query[i])
            i += 1
        return ''.join(result)


# Singleton instance
_postgres_db: Optional[PostgresDatabase] = None


def get_postgres_database() -> PostgresDatabase:
    """Get the singleton PostgreSQL database instance."""
    global _postgres_db
    if _postgres_db is None:
        _postgres_db = PostgresDatabase()
    return _postgres_db
