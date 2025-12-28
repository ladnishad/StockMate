"""Watchlist storage with support for both JSON (dev) and PostgreSQL (prod).

Development: Uses JSON files for simplicity
Production: Uses PostgreSQL via Supabase for scalability
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from functools import lru_cache
import uuid

from app.config import get_settings

logger = logging.getLogger(__name__)

# Default storage location (for JSON file storage)
DATA_DIR = Path(__file__).parent.parent.parent / "data"
WATCHLIST_FILE = DATA_DIR / "watchlists.json"


class WatchlistStore:
    """Simple JSON-based watchlist storage.

    Structure:
    {
        "user_id": {
            "symbols": ["AAPL", "MSFT", ...],
            "metadata": {
                "AAPL": {"added_at": "2025-01-10T10:30:00Z", "notes": "...", "alerts_enabled": false},
                ...
            },
            "last_updated": "2025-01-10T10:30:00Z"
        }
    }
    """

    def __init__(self, file_path: Optional[Path] = None):
        """Initialize the store.

        Args:
            file_path: Path to JSON file. Defaults to data/watchlists.json
        """
        self.file_path = file_path or WATCHLIST_FILE
        self._ensure_data_dir()
        self._data: Dict = self._load()

    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> Dict:
        """Load data from JSON file."""
        if self.file_path.exists():
            try:
                with open(self.file_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning(f"Failed to load watchlist data: {e}")
                return {}
        return {}

    def _save(self) -> None:
        """Save data to JSON file."""
        try:
            with open(self.file_path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save watchlist data: {e}")

    def _ensure_user(self, user_id: str) -> None:
        """Ensure user entry exists."""
        if user_id not in self._data:
            self._data[user_id] = {
                "symbols": [],
                "metadata": {},
                "last_updated": datetime.utcnow().isoformat(),
            }

    def get_watchlist(self, user_id: str) -> List[dict]:
        """Get user's watchlist items.

        Args:
            user_id: User identifier

        Returns:
            List of watchlist items with symbol and metadata
        """
        self._ensure_user(user_id)
        user_data = self._data[user_id]
        items = []

        for symbol in user_data["symbols"]:
            metadata = user_data["metadata"].get(symbol, {})
            items.append({
                "symbol": symbol,
                "added_at": metadata.get("added_at", datetime.utcnow().isoformat()),
                "notes": metadata.get("notes"),
                "alerts_enabled": metadata.get("alerts_enabled", False),
            })

        return items

    def add_symbol(
        self,
        user_id: str,
        symbol: str,
        notes: Optional[str] = None,
        alerts_enabled: bool = False,
    ) -> dict:
        """Add symbol to user's watchlist.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            notes: Optional user notes
            alerts_enabled: Whether to enable alerts

        Returns:
            The created watchlist item
        """
        symbol = symbol.upper()
        self._ensure_user(user_id)

        user_data = self._data[user_id]

        # Don't add duplicates
        if symbol in user_data["symbols"]:
            logger.info(f"Symbol {symbol} already in watchlist for user {user_id}")
            return {
                "symbol": symbol,
                **user_data["metadata"].get(symbol, {}),
            }

        # Add symbol at the beginning (most recent first)
        user_data["symbols"].insert(0, symbol)

        # Add metadata
        added_at = datetime.utcnow().isoformat()
        user_data["metadata"][symbol] = {
            "added_at": added_at,
            "notes": notes,
            "alerts_enabled": alerts_enabled,
        }

        user_data["last_updated"] = added_at
        self._save()

        logger.info(f"Added {symbol} to watchlist for user {user_id}")

        return {
            "symbol": symbol,
            "added_at": added_at,
            "notes": notes,
            "alerts_enabled": alerts_enabled,
        }

    def remove_symbol(self, user_id: str, symbol: str) -> bool:
        """Remove symbol from user's watchlist.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol

        Returns:
            True if removed, False if not found
        """
        symbol = symbol.upper()
        self._ensure_user(user_id)

        user_data = self._data[user_id]

        if symbol not in user_data["symbols"]:
            logger.info(f"Symbol {symbol} not in watchlist for user {user_id}")
            return False

        user_data["symbols"].remove(symbol)
        user_data["metadata"].pop(symbol, None)
        user_data["last_updated"] = datetime.utcnow().isoformat()
        self._save()

        logger.info(f"Removed {symbol} from watchlist for user {user_id}")
        return True

    def update_symbol(
        self,
        user_id: str,
        symbol: str,
        notes: Optional[str] = None,
        alerts_enabled: Optional[bool] = None,
    ) -> Optional[dict]:
        """Update metadata for a symbol in the watchlist.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            notes: New notes (None to keep existing)
            alerts_enabled: New alerts setting (None to keep existing)

        Returns:
            Updated watchlist item, or None if not found
        """
        symbol = symbol.upper()
        self._ensure_user(user_id)

        user_data = self._data[user_id]

        if symbol not in user_data["symbols"]:
            return None

        metadata = user_data["metadata"].get(symbol, {})

        if notes is not None:
            metadata["notes"] = notes
        if alerts_enabled is not None:
            metadata["alerts_enabled"] = alerts_enabled

        user_data["metadata"][symbol] = metadata
        user_data["last_updated"] = datetime.utcnow().isoformat()
        self._save()

        return {
            "symbol": symbol,
            **metadata,
        }

    def reorder(self, user_id: str, symbols: List[str]) -> bool:
        """Reorder watchlist symbols.

        Args:
            user_id: User identifier
            symbols: New order of symbols

        Returns:
            True if successful
        """
        self._ensure_user(user_id)
        user_data = self._data[user_id]

        # Validate all symbols exist
        symbols = [s.upper() for s in symbols]
        current_symbols = set(user_data["symbols"])

        if set(symbols) != current_symbols:
            logger.warning("Reorder symbols don't match current watchlist")
            return False

        user_data["symbols"] = symbols
        user_data["last_updated"] = datetime.utcnow().isoformat()
        self._save()

        return True

    def get_symbol_count(self, user_id: str) -> int:
        """Get the number of symbols in user's watchlist."""
        self._ensure_user(user_id)
        return len(self._data[user_id]["symbols"])

    def has_symbol(self, user_id: str, symbol: str) -> bool:
        """Check if a symbol is in user's watchlist."""
        symbol = symbol.upper()
        self._ensure_user(user_id)
        return symbol in self._data[user_id]["symbols"]


class DatabaseWatchlistStore:
    """PostgreSQL-backed watchlist storage for production."""

    def __init__(self):
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the watchlist table exists (handled by postgres.py init)."""
        self._initialized = True

    async def get_watchlist(self, user_id: str) -> List[dict]:
        """Get user's watchlist items from database."""
        from app.storage.postgres import get_connection

        items = []
        async with get_connection() as conn:
            rows = await conn.fetch(
                """SELECT symbol, notes, alerts_enabled, added_at
                   FROM watchlist WHERE user_id = $1 ORDER BY added_at DESC""",
                user_id
            )
            for row in rows:
                items.append({
                    "symbol": row["symbol"],
                    "added_at": row["added_at"],
                    "notes": row["notes"],
                    "alerts_enabled": row["alerts_enabled"],
                })
        return items

    async def add_symbol(
        self,
        user_id: str,
        symbol: str,
        notes: Optional[str] = None,
        alerts_enabled: bool = False,
    ) -> dict:
        """Add symbol to user's watchlist in database."""
        from app.storage.postgres import get_connection

        symbol = symbol.upper()
        now = datetime.utcnow().isoformat()
        item_id = str(uuid.uuid4())

        async with get_connection() as conn:
            # Check if exists
            existing = await conn.fetchval(
                "SELECT symbol FROM watchlist WHERE user_id = $1 AND symbol = $2",
                user_id, symbol
            )
            if existing:
                logger.info(f"Symbol {symbol} already in watchlist for user {user_id}")
                row = await conn.fetchrow(
                    "SELECT * FROM watchlist WHERE user_id = $1 AND symbol = $2",
                    user_id, symbol
                )
                return {
                    "symbol": row["symbol"],
                    "added_at": row["added_at"],
                    "notes": row["notes"],
                    "alerts_enabled": row["alerts_enabled"],
                }

            # Insert new
            await conn.execute(
                """INSERT INTO watchlist (id, user_id, symbol, notes, alerts_enabled, added_at)
                   VALUES ($1, $2, $3, $4, $5, $6)""",
                item_id, user_id, symbol, notes, alerts_enabled, now
            )

        logger.info(f"Added {symbol} to watchlist for user {user_id}")
        return {
            "symbol": symbol,
            "added_at": now,
            "notes": notes,
            "alerts_enabled": alerts_enabled,
        }

    async def remove_symbol(self, user_id: str, symbol: str) -> bool:
        """Remove symbol from user's watchlist in database."""
        from app.storage.postgres import get_connection

        symbol = symbol.upper()

        async with get_connection() as conn:
            result = await conn.execute(
                "DELETE FROM watchlist WHERE user_id = $1 AND symbol = $2",
                user_id, symbol
            )
            deleted = "DELETE 1" in result

        if deleted:
            logger.info(f"Removed {symbol} from watchlist for user {user_id}")
        return deleted

    async def update_symbol(
        self,
        user_id: str,
        symbol: str,
        notes: Optional[str] = None,
        alerts_enabled: Optional[bool] = None,
    ) -> Optional[dict]:
        """Update metadata for a symbol in the watchlist."""
        from app.storage.postgres import get_connection

        symbol = symbol.upper()

        async with get_connection() as conn:
            # Check if exists
            row = await conn.fetchrow(
                "SELECT * FROM watchlist WHERE user_id = $1 AND symbol = $2",
                user_id, symbol
            )
            if not row:
                return None

            # Build update
            updates = []
            values = []
            param_count = 0

            if notes is not None:
                param_count += 1
                updates.append(f"notes = ${param_count}")
                values.append(notes)
            if alerts_enabled is not None:
                param_count += 1
                updates.append(f"alerts_enabled = ${param_count}")
                values.append(alerts_enabled)

            if updates:
                values.extend([user_id, symbol])
                await conn.execute(
                    f"""UPDATE watchlist SET {', '.join(updates)}
                        WHERE user_id = ${param_count + 1} AND symbol = ${param_count + 2}""",
                    *values
                )

            # Fetch updated row
            row = await conn.fetchrow(
                "SELECT * FROM watchlist WHERE user_id = $1 AND symbol = $2",
                user_id, symbol
            )
            return {
                "symbol": row["symbol"],
                "added_at": row["added_at"],
                "notes": row["notes"],
                "alerts_enabled": row["alerts_enabled"],
            }

    def get_symbol_count(self, user_id: str) -> int:
        """Get the number of symbols - sync wrapper for async."""
        # This is a sync method in the interface, but we need async
        # For now, return 0 - endpoints should use async version
        return 0

    async def get_symbol_count_async(self, user_id: str) -> int:
        """Get the number of symbols in user's watchlist."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            count = await conn.fetchval(
                "SELECT COUNT(*) FROM watchlist WHERE user_id = $1",
                user_id
            )
            return count or 0

    def has_symbol(self, user_id: str, symbol: str) -> bool:
        """Sync check - returns False, use async version."""
        return False

    async def has_symbol_async(self, user_id: str, symbol: str) -> bool:
        """Check if a symbol is in user's watchlist."""
        from app.storage.postgres import get_connection

        symbol = symbol.upper()
        async with get_connection() as conn:
            result = await conn.fetchval(
                "SELECT 1 FROM watchlist WHERE user_id = $1 AND symbol = $2",
                user_id, symbol
            )
            return result is not None


# Singleton instances
_json_store: Optional[WatchlistStore] = None
_db_store: Optional[DatabaseWatchlistStore] = None


def get_watchlist_store():
    """Get the appropriate watchlist store based on configuration.

    Returns JSON-based store for development, PostgreSQL for production.
    """
    settings = get_settings()

    if settings.use_postgres:
        global _db_store
        if _db_store is None:
            _db_store = DatabaseWatchlistStore()
        return _db_store
    else:
        global _json_store
        if _json_store is None:
            _json_store = WatchlistStore()
        return _json_store
