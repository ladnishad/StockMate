#!/usr/bin/env python3
"""
Migrate StockMate data from SQLite/JSON to Supabase PostgreSQL.

This script migrates:
- Watchlists (from JSON file)
- Positions (from SQLite)
- Alerts (from SQLite)
- Device tokens (from SQLite)
- Trading plans (from SQLite)
- Conversations (from SQLite)

Prerequisites:
1. Create Supabase project at https://supabase.com
2. Get your DATABASE_URL from Settings → Database → Connection string → URI
3. Add DATABASE_URL to your .env file
4. Create a user account via /auth/signup endpoint (or Supabase dashboard)
5. Run this script with your user ID

Usage:
    python scripts/migrate_to_supabase.py --user-id YOUR_SUPABASE_USER_UUID

The user ID is the UUID you get after signing up via the /auth/signup endpoint.
You can also find it in Supabase Dashboard → Authentication → Users.
"""

import argparse
import asyncio
import json
import logging
import sqlite3
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
SQLITE_DB = DATA_DIR / "stockmate.db"
WATCHLIST_JSON = DATA_DIR / "watchlists.json"


async def migrate_data(user_id: str, database_url: str):
    """Migrate all data to Supabase PostgreSQL."""
    logger.info(f"Starting migration for user: {user_id}")
    logger.info(f"Connecting to PostgreSQL...")

    # Connect to PostgreSQL
    conn = await asyncpg.connect(database_url)
    logger.info("Connected to PostgreSQL")

    try:
        # Create tables if they don't exist
        await create_tables(conn)

        # Migrate data
        await migrate_watchlist(conn, user_id)
        await migrate_positions(conn, user_id)
        await migrate_alerts(conn, user_id)
        await migrate_device_tokens(conn, user_id)
        await migrate_trading_plans(conn, user_id)
        await migrate_conversations(conn, user_id)
        await migrate_agent_contexts(conn, user_id)

        logger.info("Migration completed successfully!")
        logger.info("You can now delete the local data files:")
        logger.info(f"  - {SQLITE_DB}")
        logger.info(f"  - {WATCHLIST_JSON}")

    finally:
        await conn.close()


async def create_tables(conn: asyncpg.Connection):
    """Create tables in PostgreSQL if they don't exist."""
    logger.info("Creating tables...")

    # Watchlist table
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
            stop_loss DOUBLE PRECISION NOT NULL,
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

    # Trading plans table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS trading_plans (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            status TEXT DEFAULT 'active',
            bias TEXT,
            thesis TEXT,
            entry_zone_low DOUBLE PRECISION,
            entry_zone_high DOUBLE PRECISION,
            stop_loss DOUBLE PRECISION,
            stop_reasoning TEXT,
            target_1 DOUBLE PRECISION,
            target_2 DOUBLE PRECISION,
            target_3 DOUBLE PRECISION,
            target_reasoning TEXT,
            risk_reward DOUBLE PRECISION,
            position_size_pct DOUBLE PRECISION,
            key_supports TEXT DEFAULT '[]',
            key_resistances TEXT DEFAULT '[]',
            invalidation_criteria TEXT,
            trade_style TEXT,
            trade_style_reasoning TEXT,
            holding_period TEXT,
            confidence INTEGER DEFAULT 0,
            price_at_creation DOUBLE PRECISION,
            rsi_at_creation DOUBLE PRECISION,
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

    # Conversations table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            messages JSONB DEFAULT '[]',
            created_at TEXT,
            updated_at TEXT,
            UNIQUE(user_id, symbol)
        )
    """)

    # Agent contexts table
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_contexts (
            symbol TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            context_data JSONB NOT NULL,
            session_id TEXT,
            updated_at TEXT NOT NULL
        )
    """)

    # Create indexes
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_user ON positions(user_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_alerts_user ON alerts(user_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_devices_user ON device_tokens(user_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_plans_user ON trading_plans(user_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)")
    await conn.execute("CREATE INDEX IF NOT EXISTS idx_agent_contexts_user ON agent_contexts(user_id)")

    logger.info("Tables created successfully")


async def migrate_watchlist(conn: asyncpg.Connection, user_id: str):
    """Migrate watchlist from JSON file."""
    if not WATCHLIST_JSON.exists():
        logger.info("No watchlist JSON file found, skipping")
        return

    logger.info("Migrating watchlist...")

    with open(WATCHLIST_JSON, "r") as f:
        data = json.load(f)

    # Get the default user's data (our current data)
    user_data = data.get("default", {})
    if not user_data:
        logger.info("No watchlist data for default user")
        return

    symbols = user_data.get("symbols", [])
    metadata = user_data.get("metadata", {})

    count = 0
    for symbol in symbols:
        meta = metadata.get(symbol, {})
        item_id = f"wl_{symbol}_{user_id[:8]}"

        try:
            await conn.execute("""
                INSERT INTO watchlist (id, user_id, symbol, notes, alerts_enabled, added_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (user_id, symbol) DO NOTHING
            """,
                item_id,
                user_id,
                symbol,
                meta.get("notes"),
                meta.get("alerts_enabled", False),
                meta.get("added_at", datetime.utcnow().isoformat())
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate watchlist item {symbol}: {e}")

    logger.info(f"Migrated {count} watchlist items")


async def migrate_positions(conn: asyncpg.Connection, user_id: str):
    """Migrate positions from SQLite."""
    if not SQLITE_DB.exists():
        logger.info("No SQLite database found, skipping positions")
        return

    logger.info("Migrating positions...")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM positions WHERE user_id = 'default'")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.info("No positions table found")
        return
    finally:
        sqlite_conn.close()

    count = 0
    for row in rows:
        row_dict = dict(row)
        row_dict["user_id"] = user_id  # Update to new user ID

        # Handle JSON fields
        entries = row_dict.get("entries", "[]")
        exits = row_dict.get("exits", "[]")
        targets_hit = row_dict.get("targets_hit", "[]")

        try:
            await conn.execute("""
                INSERT INTO positions (
                    id, user_id, symbol, status, entry_price, entry_date,
                    entries, avg_entry_price, exits, avg_exit_price,
                    current_size, original_size, stop_loss,
                    target_1, target_2, target_3, targets_hit,
                    cost_basis, realized_pnl, realized_pnl_pct,
                    unrealized_pnl, unrealized_pnl_pct,
                    trade_type, notes, created_at, updated_at
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26
                )
                ON CONFLICT (user_id, symbol) DO NOTHING
            """,
                row_dict["id"], user_id, row_dict["symbol"], row_dict["status"],
                row_dict.get("entry_price"), row_dict.get("entry_date"),
                entries, row_dict.get("avg_entry_price"),
                exits, row_dict.get("avg_exit_price"),
                row_dict.get("current_size", 0), row_dict.get("original_size", 0),
                row_dict["stop_loss"],
                row_dict.get("target_1"), row_dict.get("target_2"), row_dict.get("target_3"),
                targets_hit,
                row_dict.get("cost_basis"), row_dict.get("realized_pnl"),
                row_dict.get("realized_pnl_pct"), row_dict.get("unrealized_pnl"),
                row_dict.get("unrealized_pnl_pct"),
                row_dict["trade_type"], row_dict.get("notes"),
                row_dict["created_at"], row_dict["updated_at"]
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate position {row_dict['symbol']}: {e}")

    logger.info(f"Migrated {count} positions")


async def migrate_alerts(conn: asyncpg.Connection, user_id: str):
    """Migrate alerts from SQLite."""
    if not SQLITE_DB.exists():
        logger.info("No SQLite database found, skipping alerts")
        return

    logger.info("Migrating alerts...")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM alerts WHERE user_id = 'default'")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.info("No alerts table found")
        return
    finally:
        sqlite_conn.close()

    count = 0
    for row in rows:
        row_dict = dict(row)

        try:
            await conn.execute("""
                INSERT INTO alerts (id, user_id, symbol, alert_type, message, price_at_alert, sent_at, acknowledged)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT DO NOTHING
            """,
                row_dict["id"], user_id, row_dict["symbol"],
                row_dict["alert_type"], row_dict["message"],
                row_dict["price_at_alert"], row_dict["sent_at"],
                row_dict.get("acknowledged", 0)
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate alert: {e}")

    logger.info(f"Migrated {count} alerts")


async def migrate_device_tokens(conn: asyncpg.Connection, user_id: str):
    """Migrate device tokens from SQLite."""
    if not SQLITE_DB.exists():
        logger.info("No SQLite database found, skipping device tokens")
        return

    logger.info("Migrating device tokens...")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM device_tokens WHERE user_id = 'default'")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.info("No device_tokens table found")
        return
    finally:
        sqlite_conn.close()

    count = 0
    for row in rows:
        row_dict = dict(row)

        try:
            await conn.execute("""
                INSERT INTO device_tokens (id, user_id, device_token, platform, created_at, last_used)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (device_token) DO UPDATE SET user_id = $2
            """,
                row_dict["id"], user_id, row_dict["device_token"],
                row_dict.get("platform", "ios"), row_dict["created_at"],
                row_dict.get("last_used")
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate device token: {e}")

    logger.info(f"Migrated {count} device tokens")


async def migrate_trading_plans(conn: asyncpg.Connection, user_id: str):
    """Migrate trading plans from SQLite."""
    if not SQLITE_DB.exists():
        logger.info("No SQLite database found, skipping trading plans")
        return

    logger.info("Migrating trading plans...")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM trading_plans WHERE user_id = 'default'")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.info("No trading_plans table found")
        return
    finally:
        sqlite_conn.close()

    count = 0
    for row in rows:
        row_dict = dict(row)

        try:
            await conn.execute("""
                INSERT INTO trading_plans (
                    id, user_id, symbol, status, bias, thesis,
                    entry_zone_low, entry_zone_high, stop_loss, stop_reasoning,
                    target_1, target_2, target_3, target_reasoning,
                    risk_reward, position_size_pct,
                    key_supports, key_resistances, invalidation_criteria,
                    trade_style, trade_style_reasoning, holding_period, confidence,
                    price_at_creation, rsi_at_creation, market_direction_at_creation,
                    technical_summary, news_summary, reddit_sentiment, reddit_buzz,
                    created_at, updated_at, last_evaluation, evaluation_notes
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                    $11, $12, $13, $14, $15, $16, $17, $18, $19, $20,
                    $21, $22, $23, $24, $25, $26, $27, $28, $29, $30,
                    $31, $32, $33, $34
                )
                ON CONFLICT (user_id, symbol) DO NOTHING
            """,
                row_dict["id"], user_id, row_dict["symbol"],
                row_dict.get("status", "active"), row_dict.get("bias"), row_dict.get("thesis"),
                row_dict.get("entry_zone_low"), row_dict.get("entry_zone_high"),
                row_dict.get("stop_loss"), row_dict.get("stop_reasoning"),
                row_dict.get("target_1"), row_dict.get("target_2"), row_dict.get("target_3"),
                row_dict.get("target_reasoning"), row_dict.get("risk_reward"),
                row_dict.get("position_size_pct"),
                row_dict.get("key_supports", "[]"), row_dict.get("key_resistances", "[]"),
                row_dict.get("invalidation_criteria"),
                row_dict.get("trade_style"), row_dict.get("trade_style_reasoning"),
                row_dict.get("holding_period"), row_dict.get("confidence", 0),
                row_dict.get("price_at_creation"), row_dict.get("rsi_at_creation"),
                row_dict.get("market_direction_at_creation"), row_dict.get("technical_summary"),
                row_dict.get("news_summary", ""), row_dict.get("reddit_sentiment", ""),
                row_dict.get("reddit_buzz", ""),
                row_dict.get("created_at"), row_dict.get("updated_at"),
                row_dict.get("last_evaluation"), row_dict.get("evaluation_notes")
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate trading plan {row_dict['symbol']}: {e}")

    logger.info(f"Migrated {count} trading plans")


async def migrate_conversations(conn: asyncpg.Connection, user_id: str):
    """Migrate conversations from SQLite."""
    if not SQLITE_DB.exists():
        logger.info("No SQLite database found, skipping conversations")
        return

    logger.info("Migrating conversations...")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM conversations WHERE user_id = 'default'")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.info("No conversations table found")
        return
    finally:
        sqlite_conn.close()

    count = 0
    for row in rows:
        row_dict = dict(row)

        try:
            await conn.execute("""
                INSERT INTO conversations (user_id, symbol, messages, created_at, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (user_id, symbol) DO NOTHING
            """,
                user_id, row_dict["symbol"],
                row_dict.get("messages", "[]"),
                row_dict.get("created_at"), row_dict.get("updated_at")
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate conversation {row_dict['symbol']}: {e}")

    logger.info(f"Migrated {count} conversations")


async def migrate_agent_contexts(conn: asyncpg.Connection, user_id: str):
    """Migrate agent contexts from SQLite."""
    if not SQLITE_DB.exists():
        logger.info("No SQLite database found, skipping agent contexts")
        return

    logger.info("Migrating agent contexts...")

    sqlite_conn = sqlite3.connect(SQLITE_DB)
    sqlite_conn.row_factory = sqlite3.Row
    cursor = sqlite_conn.cursor()

    try:
        cursor.execute("SELECT * FROM agent_contexts WHERE user_id = 'default'")
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        logger.info("No agent_contexts table found")
        return
    finally:
        sqlite_conn.close()

    count = 0
    for row in rows:
        row_dict = dict(row)

        try:
            await conn.execute("""
                INSERT INTO agent_contexts (symbol, user_id, context_data, session_id, updated_at)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (symbol) DO UPDATE SET
                    user_id = $2,
                    context_data = $3,
                    session_id = $4,
                    updated_at = $5
            """,
                row_dict["symbol"], user_id,
                row_dict.get("context_data", "{}"),
                row_dict.get("session_id"),
                row_dict.get("updated_at")
            )
            count += 1
        except Exception as e:
            logger.warning(f"Failed to migrate agent context {row_dict['symbol']}: {e}")

    logger.info(f"Migrated {count} agent contexts")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate StockMate data from SQLite/JSON to Supabase PostgreSQL"
    )
    parser.add_argument(
        "--user-id",
        required=True,
        help="Your Supabase user UUID (from /auth/signup or Supabase dashboard)"
    )
    parser.add_argument(
        "--database-url",
        help="PostgreSQL connection string (defaults to DATABASE_URL env var)"
    )

    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL not provided. Set it in .env or pass --database-url")
        sys.exit(1)

    # Run migration
    asyncio.run(migrate_data(args.user_id, database_url))


if __name__ == "__main__":
    main()
