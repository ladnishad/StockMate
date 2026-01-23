"""Database setup and management.

Supports both SQLite (development) and PostgreSQL (production):
- SQLite: Local development, file-based
- PostgreSQL: Production, Supabase hosted

Database selection is automatic based on DATABASE_URL environment variable.
"""

import aiosqlite
import logging
from pathlib import Path
from typing import Optional, Protocol, Any
from contextlib import asynccontextmanager
from abc import ABC, abstractmethod

from app.config import get_settings

logger = logging.getLogger(__name__)

# Database location (SQLite)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
DB_PATH = DATA_DIR / "stockmate.db"


class DatabaseProtocol(Protocol):
    """Protocol for database implementations."""

    async def initialize(self) -> None:
        """Initialize database tables."""
        ...

    def connection(self):
        """Get database connection context manager."""
        ...


class Database:
    """Async SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        """Initialize database.

        Args:
            db_path: Path to SQLite database file. Defaults to data/stockmate.db
        """
        self.db_path = db_path or DB_PATH
        self._ensure_data_dir()

    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @asynccontextmanager
    async def connection(self):
        """Get database connection context manager.

        Usage:
            async with db.connection() as conn:
                await conn.execute(...)
        """
        async with aiosqlite.connect(self.db_path) as conn:
            conn.row_factory = aiosqlite.Row
            yield conn

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with self.connection() as conn:
            # Positions table - track user trades
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    status TEXT DEFAULT 'watching',
                    entry_price REAL,
                    entry_date TEXT,
                    entries TEXT DEFAULT '[]',
                    avg_entry_price REAL,
                    exits TEXT DEFAULT '[]',
                    avg_exit_price REAL,
                    current_size INTEGER DEFAULT 0,
                    original_size INTEGER DEFAULT 0,
                    stop_loss REAL,
                    target_1 REAL,
                    target_2 REAL,
                    target_3 REAL,
                    targets_hit TEXT DEFAULT '[]',
                    cost_basis REAL,
                    realized_pnl REAL,
                    realized_pnl_pct REAL,
                    unrealized_pnl REAL,
                    unrealized_pnl_pct REAL,
                    trade_type TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE(user_id, symbol)
                )
            """)

            # Migration: Add new columns if they don't exist (for existing databases)
            for column, col_type, default in [
                ("entries", "TEXT", "'[]'"),
                ("avg_entry_price", "REAL", "NULL"),
                ("exits", "TEXT", "'[]'"),
                ("avg_exit_price", "REAL", "NULL"),
                ("cost_basis", "REAL", "NULL"),
                ("realized_pnl", "REAL", "NULL"),
                ("realized_pnl_pct", "REAL", "NULL"),
                ("unrealized_pnl", "REAL", "NULL"),
                ("unrealized_pnl_pct", "REAL", "NULL"),
            ]:
                try:
                    await conn.execute(
                        f"ALTER TABLE positions ADD COLUMN {column} {col_type} DEFAULT {default}"
                    )
                except Exception:
                    pass  # Column already exists

            # Alerts table - history and deduplication
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    price_at_alert REAL NOT NULL,
                    sent_at TEXT NOT NULL,
                    acknowledged INTEGER DEFAULT 0
                )
            """)

            # Create index for alert deduplication queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_user_symbol_type
                ON alerts (user_id, symbol, alert_type, sent_at)
            """)

            # Device tokens table - for push notifications
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

            # Create index for user device lookups
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_device_tokens_user
                ON device_tokens (user_id)
            """)

            # Agent context cache - for session resumption
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_contexts (
                    symbol TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    context_data TEXT NOT NULL,
                    session_id TEXT,
                    updated_at TEXT NOT NULL
                )
            """)

            # User settings table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    user_id TEXT PRIMARY KEY,
                    model_provider TEXT DEFAULT 'claude',
                    subscription_tier TEXT DEFAULT 'base',
                    is_admin INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)

            # Migration: Add subscription_tier column if it doesn't exist
            try:
                await conn.execute(
                    "ALTER TABLE user_settings ADD COLUMN subscription_tier TEXT DEFAULT 'base'"
                )
            except Exception:
                pass  # Column already exists

            # Migration: Add is_admin column if it doesn't exist
            try:
                await conn.execute(
                    "ALTER TABLE user_settings ADD COLUMN is_admin INTEGER DEFAULT 0"
                )
            except Exception:
                pass  # Column already exists

            # API usage tracking table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS api_usage (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    operation_type TEXT NOT NULL,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    estimated_cost REAL DEFAULT 0,
                    tool_calls INTEGER DEFAULT 0,
                    tool_cost REAL DEFAULT 0,
                    symbol TEXT,
                    endpoint TEXT,
                    created_at TEXT NOT NULL
                )
            """)

            # Create indexes for efficient usage queries
            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_user
                ON api_usage (user_id, created_at)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_provider
                ON api_usage (provider, created_at)
            """)

            await conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_api_usage_date
                ON api_usage (created_at)
            """)

            await conn.commit()
            logger.info(f"Database initialized at {self.db_path}")


# Singleton instances
_sqlite_db: Optional[Database] = None
_postgres_db: Optional[Any] = None  # Lazy import to avoid circular deps


def get_db_path() -> str:
    """Get the database path as a string."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return str(DB_PATH)


def get_sqlite_database() -> Database:
    """Get the singleton SQLite database instance."""
    global _sqlite_db
    if _sqlite_db is None:
        _sqlite_db = Database()
    return _sqlite_db


def get_database():
    """Get the appropriate database based on configuration.

    Returns SQLite for development, PostgreSQL for production.
    """
    settings = get_settings()

    if settings.use_postgres:
        global _postgres_db
        if _postgres_db is None:
            from app.storage.postgres import get_postgres_database
            _postgres_db = get_postgres_database()
        return _postgres_db
    else:
        return get_sqlite_database()


async def init_database() -> None:
    """Initialize the database (call on app startup).

    Automatically selects SQLite or PostgreSQL based on configuration.
    """
    settings = get_settings()
    db = get_database()
    await db.initialize()

    if settings.use_postgres:
        logger.info("Using PostgreSQL database (production)")
    else:
        logger.info("Using SQLite database (development)")


async def delete_agent_context(symbol: str) -> bool:
    """Delete agent context cache for a specific symbol.

    Used when removing a stock from watchlist to clean up associated data.

    Args:
        symbol: Stock ticker symbol

    Returns:
        True if deleted, False if not found
    """
    symbol = symbol.upper()
    db = get_database()

    async with db.connection() as conn:
        cursor = await conn.execute(
            "DELETE FROM agent_contexts WHERE symbol = ?",
            (symbol,),
        )
        await conn.commit()

        deleted = cursor.rowcount > 0
        if deleted:
            logger.info(f"Deleted agent context for {symbol}")
        return deleted
