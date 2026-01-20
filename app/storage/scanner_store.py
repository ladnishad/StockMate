"""Scanner results storage for caching between scans.

Stores scanner results in memory with persistence to JSON file for development.
Production uses PostgreSQL for persistence across restarts.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime, timedelta
from functools import lru_cache

from app.config import get_settings
from app.models.scanner import (
    ScannerResult,
    ScannerResponse,
    ScanSession,
    TradingStyle,
    EXPIRATION_HOURS,
)

logger = logging.getLogger(__name__)

# Storage location
DATA_DIR = Path(__file__).parent.parent.parent / "data"
SCANNER_FILE = DATA_DIR / "scanner_results.json"


class ScannerStore:
    """In-memory scanner results store with JSON persistence.

    Stores:
    - Current results for each trading style
    - Previous results for detecting "new" setups
    - Scan session metadata
    - Symbol tracking for invalidation
    """

    def __init__(self, file_path: Optional[Path] = None):
        """Initialize the store.

        Args:
            file_path: Path to JSON persistence file. Defaults to data/scanner_results.json
        """
        self.file_path = file_path or SCANNER_FILE
        self._ensure_data_dir()

        # Current results by style
        self._results: Dict[TradingStyle, List[ScannerResult]] = {
            TradingStyle.DAY: [],
            TradingStyle.SWING: [],
            TradingStyle.POSITION: [],
        }

        # Previous results for "new" detection (symbol sets)
        self._previous_symbols: Dict[TradingStyle, Set[str]] = {
            TradingStyle.DAY: set(),
            TradingStyle.SWING: set(),
            TradingStyle.POSITION: set(),
        }

        # Scan metadata
        self._last_scan_time: Optional[datetime] = None
        self._next_scheduled_scan: Optional[datetime] = None
        self._current_scan_name: Optional[str] = None
        self._is_scanning: bool = False
        self._last_session: Optional[ScanSession] = None
        self._total_stocks_scanned: int = 0

        # Load persisted data
        self._load()

    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> None:
        """Load data from JSON file."""
        if not self.file_path.exists():
            return

        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)

            # Load results
            for style_str, results_data in data.get("results", {}).items():
                try:
                    style = TradingStyle(style_str)
                    self._results[style] = [
                        ScannerResult(**r) for r in results_data
                    ]
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to load results for {style_str}: {e}")

            # Load previous symbols
            for style_str, symbols in data.get("previous_symbols", {}).items():
                try:
                    style = TradingStyle(style_str)
                    self._previous_symbols[style] = set(symbols)
                except ValueError:
                    pass

            # Load metadata
            if data.get("last_scan_time"):
                self._last_scan_time = datetime.fromisoformat(data["last_scan_time"])
            if data.get("next_scheduled_scan"):
                self._next_scheduled_scan = datetime.fromisoformat(data["next_scheduled_scan"])
            self._current_scan_name = data.get("current_scan_name")
            self._total_stocks_scanned = data.get("total_stocks_scanned", 0)

            logger.info("Loaded scanner results from persistence")

        except (json.JSONDecodeError, IOError, KeyError) as e:
            logger.warning(f"Failed to load scanner data: {e}")

    def _save(self) -> None:
        """Save data to JSON file."""
        try:
            data = {
                "results": {
                    style.value: [r.model_dump(mode="json") for r in results]
                    for style, results in self._results.items()
                },
                "previous_symbols": {
                    style.value: list(symbols)
                    for style, symbols in self._previous_symbols.items()
                },
                "last_scan_time": self._last_scan_time.isoformat() if self._last_scan_time else None,
                "next_scheduled_scan": self._next_scheduled_scan.isoformat() if self._next_scheduled_scan else None,
                "current_scan_name": self._current_scan_name,
                "total_stocks_scanned": self._total_stocks_scanned,
            }

            with open(self.file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)

        except IOError as e:
            logger.error(f"Failed to save scanner data: {e}")

    # =========================================================================
    # Results management
    # =========================================================================

    def store_results(
        self,
        style: TradingStyle,
        results: List[ScannerResult],
        scan_name: Optional[str] = None,
        total_scanned: int = 0,
    ) -> None:
        """Store new scan results for a style.

        Args:
            style: Trading style
            results: List of scanner results
            scan_name: Name of the scan (e.g., "pre_market")
            total_scanned: Number of stocks that were scanned
        """
        # Track previous symbols for "new" detection
        self._previous_symbols[style] = {r.symbol for r in self._results[style]}

        # Mark new results
        previous_set = self._previous_symbols[style]
        for result in results:
            result.is_new = result.symbol not in previous_set

        # Sort by confidence score and limit to top 10
        results.sort(key=lambda r: r.confidence_score, reverse=True)
        self._results[style] = results[:10]

        # Update metadata
        self._last_scan_time = datetime.utcnow()
        if scan_name:
            self._current_scan_name = scan_name
        self._total_stocks_scanned = total_scanned

        self._save()
        logger.info(f"Stored {len(results)} results for {style.value} scan")

    def get_results(self, style: TradingStyle, user_watchlist: Optional[Set[str]] = None) -> List[ScannerResult]:
        """Get current results for a style.

        Args:
            style: Trading style
            user_watchlist: Optional set of symbols user is already watching

        Returns:
            List of scanner results with is_watching populated
        """
        results = self._results.get(style, [])

        # Remove expired results
        now = datetime.utcnow()
        expiration_hours = EXPIRATION_HOURS.get(style, 24)
        valid_results = []

        for result in results:
            # Check expiration
            if result.expires_at and result.expires_at < now:
                continue
            if (now - result.detected_at).total_seconds() > expiration_hours * 3600:
                continue

            # Mark if watching
            if user_watchlist:
                result.is_watching = result.symbol in user_watchlist

            valid_results.append(result)

        return valid_results

    def get_all_results(self, user_watchlist: Optional[Set[str]] = None) -> Dict[TradingStyle, List[ScannerResult]]:
        """Get results for all styles.

        Args:
            user_watchlist: Optional set of symbols user is already watching

        Returns:
            Dictionary of results by style
        """
        return {
            style: self.get_results(style, user_watchlist)
            for style in TradingStyle
        }

    def get_response(self, style: TradingStyle, user_watchlist: Optional[Set[str]] = None) -> ScannerResponse:
        """Get a ScannerResponse for an API endpoint.

        Args:
            style: Trading style
            user_watchlist: Optional set of symbols user is already watching

        Returns:
            ScannerResponse ready for API
        """
        return ScannerResponse(
            style=style,
            results=self.get_results(style, user_watchlist),
            scan_time=self._last_scan_time or datetime.utcnow(),
            next_scheduled_scan=self._next_scheduled_scan,
            total_stocks_scanned=self._total_stocks_scanned,
        )

    # =========================================================================
    # Invalidation
    # =========================================================================

    def invalidate_symbol(self, symbol: str, style: Optional[TradingStyle] = None) -> None:
        """Remove a symbol from results (e.g., setup invalidated).

        Args:
            symbol: Symbol to remove
            style: Specific style to remove from, or None for all
        """
        styles_to_check = [style] if style else list(TradingStyle)

        for s in styles_to_check:
            self._results[s] = [r for r in self._results[s] if r.symbol != symbol]

        self._save()
        logger.info(f"Invalidated {symbol} from scanner results")

    def clear_expired(self) -> int:
        """Remove expired results from all styles.

        Returns:
            Number of results removed
        """
        removed = 0
        now = datetime.utcnow()

        for style in TradingStyle:
            original_count = len(self._results[style])
            expiration_hours = EXPIRATION_HOURS.get(style, 24)

            self._results[style] = [
                r for r in self._results[style]
                if (not r.expires_at or r.expires_at >= now) and
                   (now - r.detected_at).total_seconds() <= expiration_hours * 3600
            ]

            removed += original_count - len(self._results[style])

        if removed:
            self._save()
            logger.info(f"Cleared {removed} expired scanner results")

        return removed

    def clear_all(self) -> None:
        """Clear all results."""
        for style in TradingStyle:
            self._results[style] = []
            self._previous_symbols[style] = set()

        self._save()
        logger.info("Cleared all scanner results")

    # =========================================================================
    # Scan state management
    # =========================================================================

    def start_scan(self, scan_name: str) -> None:
        """Mark that a scan is starting.

        Args:
            scan_name: Name of the scan (e.g., "pre_market")
        """
        self._is_scanning = True
        self._current_scan_name = scan_name
        logger.info(f"Starting {scan_name} scan")

    def end_scan(self, results_count: Dict[str, int]) -> None:
        """Mark that a scan has completed.

        Args:
            results_count: Number of results found per style
        """
        self._is_scanning = False
        self._last_scan_time = datetime.utcnow()
        logger.info(f"Completed scan: {results_count}")

    def set_next_scheduled_scan(self, next_scan: datetime) -> None:
        """Set when the next scheduled scan will run.

        Args:
            next_scan: Next scan datetime
        """
        self._next_scheduled_scan = next_scan
        self._save()

    def get_status(self) -> dict:
        """Get current scanner status.

        Returns:
            Status dictionary with scan metadata
        """
        return {
            "last_scan_time": self._last_scan_time,
            "next_scheduled_scan": self._next_scheduled_scan,
            "current_scan_name": self._current_scan_name,
            "is_scanning": self._is_scanning,
            "total_results": {
                style.value: len(results)
                for style, results in self._results.items()
            },
        }

    # =========================================================================
    # Symbol lookups
    # =========================================================================

    def get_result_by_symbol(self, symbol: str) -> Optional[ScannerResult]:
        """Find a result by symbol across all styles.

        Args:
            symbol: Symbol to find

        Returns:
            ScannerResult if found, None otherwise
        """
        symbol = symbol.upper()
        for style in TradingStyle:
            for result in self._results[style]:
                if result.symbol == symbol:
                    return result
        return None

    def has_symbol(self, symbol: str) -> bool:
        """Check if a symbol is in any scanner results.

        Args:
            symbol: Symbol to check

        Returns:
            True if found
        """
        return self.get_result_by_symbol(symbol) is not None


# Database-backed store for production
class DatabaseScannerStore:
    """PostgreSQL-backed scanner store for production."""

    def __init__(self):
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the scanner_results table exists."""
        from app.storage.postgres import get_connection

        async with get_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scanner_results (
                    id UUID PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    style TEXT NOT NULL,
                    confidence_grade TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    current_price REAL NOT NULL,
                    description TEXT NOT NULL,
                    pattern_type TEXT NOT NULL,
                    key_levels JSONB DEFAULT '{}',
                    detected_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    warnings JSONB DEFAULT '[]',
                    volume_multiple REAL,
                    gap_pct REAL,
                    fib_level REAL,
                    rsi_value REAL,
                    vwap REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, style)
                )
            """)

            await conn.execute("""
                CREATE TABLE IF NOT EXISTS scanner_metadata (
                    key TEXT PRIMARY KEY,
                    value JSONB NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

        self._initialized = True

    async def store_results(
        self,
        style: TradingStyle,
        results: List[ScannerResult],
        scan_name: Optional[str] = None,
        total_scanned: int = 0,
    ) -> None:
        """Store results in database."""
        from app.storage.postgres import get_connection
        import uuid

        if not self._initialized:
            await self._ensure_table()

        async with get_connection() as conn:
            # Clear old results for this style
            await conn.execute(
                "DELETE FROM scanner_results WHERE style = $1",
                style.value
            )

            # Insert new results
            for result in results[:10]:  # Limit to 10
                await conn.execute("""
                    INSERT INTO scanner_results
                    (id, symbol, style, confidence_grade, confidence_score,
                     current_price, description, pattern_type, key_levels,
                     detected_at, expires_at, warnings, volume_multiple,
                     gap_pct, fib_level, rsi_value, vwap)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                """,
                    str(uuid.uuid4()),
                    result.symbol,
                    style.value,
                    result.confidence_grade.value,
                    result.confidence_score,
                    result.current_price,
                    result.description,
                    result.pattern_type.value,
                    json.dumps(result.key_levels),
                    result.detected_at,
                    result.expires_at,
                    json.dumps(result.warnings),
                    result.volume_multiple,
                    result.gap_pct,
                    result.fib_level,
                    result.rsi_value,
                    result.vwap,
                )

            # Update metadata
            now = datetime.utcnow().isoformat()
            await conn.execute("""
                INSERT INTO scanner_metadata (key, value, updated_at)
                VALUES ('last_scan', $1, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = CURRENT_TIMESTAMP
            """, json.dumps({
                "scan_name": scan_name,
                "total_scanned": total_scanned,
                "timestamp": now,
            }))

    async def get_results(
        self,
        style: TradingStyle,
        user_watchlist: Optional[Set[str]] = None
    ) -> List[ScannerResult]:
        """Get results from database."""
        from app.storage.postgres import get_connection

        if not self._initialized:
            await self._ensure_table()

        results = []
        async with get_connection() as conn:
            rows = await conn.fetch("""
                SELECT * FROM scanner_results
                WHERE style = $1 AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
                ORDER BY confidence_score DESC
                LIMIT 10
            """, style.value)

            for row in rows:
                result = ScannerResult(
                    symbol=row["symbol"],
                    style=TradingStyle(row["style"]),
                    confidence_grade=row["confidence_grade"],
                    confidence_score=row["confidence_score"],
                    current_price=row["current_price"],
                    description=row["description"],
                    pattern_type=row["pattern_type"],
                    key_levels=json.loads(row["key_levels"]) if row["key_levels"] else {},
                    detected_at=row["detected_at"],
                    expires_at=row["expires_at"],
                    warnings=json.loads(row["warnings"]) if row["warnings"] else [],
                    volume_multiple=row["volume_multiple"],
                    gap_pct=row["gap_pct"],
                    fib_level=row["fib_level"],
                    rsi_value=row["rsi_value"],
                    vwap=row["vwap"],
                    is_watching=row["symbol"] in (user_watchlist or set()),
                )
                results.append(result)

        return results

    async def get_status(self) -> dict:
        """Get scanner status from database."""
        from app.storage.postgres import get_connection

        if not self._initialized:
            await self._ensure_table()

        status = {
            "last_scan_time": None,
            "next_scheduled_scan": None,
            "current_scan_name": None,
            "is_scanning": False,
            "total_results": {"day": 0, "swing": 0, "position": 0},
        }

        async with get_connection() as conn:
            # Get metadata
            row = await conn.fetchrow(
                "SELECT value FROM scanner_metadata WHERE key = 'last_scan'"
            )
            if row:
                data = json.loads(row["value"])
                status["last_scan_time"] = data.get("timestamp")
                status["current_scan_name"] = data.get("scan_name")

            # Get next scheduled scan
            next_row = await conn.fetchrow(
                "SELECT value FROM scanner_metadata WHERE key = 'next_scheduled_scan'"
            )
            if next_row:
                status["next_scheduled_scan"] = json.loads(next_row["value"])

            # Get counts
            for style in TradingStyle:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM scanner_results WHERE style = $1",
                    style.value
                )
                status["total_results"][style.value] = count or 0

        return status

    async def get_all_results(
        self,
        user_watchlist: Optional[Set[str]] = None
    ) -> Dict[TradingStyle, List[ScannerResult]]:
        """Get results for all styles from database.

        Args:
            user_watchlist: Optional set of symbols user is already watching

        Returns:
            Dictionary of results by style
        """
        return {
            style: await self.get_results(style, user_watchlist)
            for style in TradingStyle
        }

    async def get_response(
        self,
        style: TradingStyle,
        user_watchlist: Optional[Set[str]] = None
    ) -> ScannerResponse:
        """Get a ScannerResponse for an API endpoint.

        Args:
            style: Trading style
            user_watchlist: Optional set of symbols user is already watching

        Returns:
            ScannerResponse ready for API
        """
        from app.storage.postgres import get_connection

        if not self._initialized:
            await self._ensure_table()

        results = await self.get_results(style, user_watchlist)

        # Get scan metadata
        last_scan_time = None
        next_scheduled_scan = None
        total_stocks_scanned = 0

        async with get_connection() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM scanner_metadata WHERE key = 'last_scan'"
            )
            if row:
                data = json.loads(row["value"])
                last_scan_time = datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else None
                total_stocks_scanned = data.get("total_scanned", 0)

            next_row = await conn.fetchrow(
                "SELECT value FROM scanner_metadata WHERE key = 'next_scheduled_scan'"
            )
            if next_row:
                ts = json.loads(next_row["value"])
                if ts:
                    next_scheduled_scan = datetime.fromisoformat(ts) if isinstance(ts, str) else None

        return ScannerResponse(
            style=style,
            results=results,
            scan_time=last_scan_time or datetime.utcnow(),
            next_scheduled_scan=next_scheduled_scan,
            total_stocks_scanned=total_stocks_scanned,
        )

    async def set_next_scheduled_scan(self, next_scan: datetime) -> None:
        """Set when the next scheduled scan will run.

        Args:
            next_scan: Next scan datetime
        """
        from app.storage.postgres import get_connection

        if not self._initialized:
            await self._ensure_table()

        async with get_connection() as conn:
            await conn.execute("""
                INSERT INTO scanner_metadata (key, value, updated_at)
                VALUES ('next_scheduled_scan', $1, CURRENT_TIMESTAMP)
                ON CONFLICT (key) DO UPDATE SET value = $1, updated_at = CURRENT_TIMESTAMP
            """, json.dumps(next_scan.isoformat()))


# Singleton instances
_json_store: Optional[ScannerStore] = None
_db_store: Optional[DatabaseScannerStore] = None


def get_scanner_store():
    """Get the appropriate scanner store based on configuration.

    Returns JSON-based store for development, PostgreSQL for production.
    """
    settings = get_settings()

    if settings.use_postgres:
        global _db_store
        if _db_store is None:
            _db_store = DatabaseScannerStore()
        return _db_store
    else:
        global _json_store
        if _json_store is None:
            _json_store = ScannerStore()
        return _json_store
