"""Position tracking storage.

Tracks user trade positions including:
- Entry price and size
- Stop loss levels
- Target prices (up to 3)
- Position status (watching, entered, partial, stopped, closed)
"""

import json
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Literal
from pydantic import BaseModel

from app.storage.database import get_database

logger = logging.getLogger(__name__)

PositionStatus = Literal["watching", "entered", "partial", "stopped_out", "closed"]


class PositionEntry(BaseModel):
    """Individual entry into a position."""
    price: float
    shares: int
    date: str


class PositionExit(BaseModel):
    """Individual exit from a position."""
    price: float
    shares: int
    date: str
    reason: str  # "target_1", "target_2", "target_3", "stop_loss", "manual"


class Position(BaseModel):
    """Position data model."""

    id: str
    user_id: str
    symbol: str
    status: PositionStatus = "watching"

    # Legacy single entry (kept for backwards compatibility)
    entry_price: Optional[float] = None
    entry_date: Optional[str] = None

    # Multiple entries support
    entries: List[PositionEntry] = []
    avg_entry_price: Optional[float] = None

    # Multiple exits support
    exits: List[PositionExit] = []
    avg_exit_price: Optional[float] = None

    # Size tracking
    current_size: int = 0
    original_size: int = 0

    # Risk management
    stop_loss: Optional[float] = None
    target_1: Optional[float] = None
    target_2: Optional[float] = None
    target_3: Optional[float] = None
    targets_hit: List[int] = []

    # P&L tracking
    cost_basis: Optional[float] = None  # Total $ invested in current position
    realized_pnl: Optional[float] = None  # P&L from closed portions
    realized_pnl_pct: Optional[float] = None
    unrealized_pnl: Optional[float] = None  # P&L on open portion (needs current price)
    unrealized_pnl_pct: Optional[float] = None

    trade_type: str  # "day", "swing", "long"
    notes: Optional[str] = None
    created_at: str
    updated_at: str


class PositionStore:
    """Manages position storage in SQLite."""

    def __init__(self):
        """Initialize position store."""
        self.db = get_database()

    async def create_position(
        self,
        user_id: str,
        symbol: str,
        trade_type: str,
        stop_loss: Optional[float] = None,
        target_1: Optional[float] = None,
        target_2: Optional[float] = None,
        target_3: Optional[float] = None,
        notes: Optional[str] = None,
    ) -> Position:
        """Create a new position (initially in 'watching' status).

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            trade_type: Type of trade (day, swing, long)
            stop_loss: Optional stop loss price
            target_1: First profit target
            target_2: Second profit target
            target_3: Third profit target
            notes: Optional notes

        Returns:
            Created Position object
        """
        symbol = symbol.upper()
        position_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            await conn.execute(
                """
                INSERT INTO positions (
                    id, user_id, symbol, status, stop_loss,
                    target_1, target_2, target_3, targets_hit,
                    trade_type, notes, created_at, updated_at
                ) VALUES (?, ?, ?, 'watching', ?, ?, ?, ?, '[]', ?, ?, ?, ?)
                ON CONFLICT(user_id, symbol) DO UPDATE SET
                    status = 'watching',
                    stop_loss = excluded.stop_loss,
                    target_1 = excluded.target_1,
                    target_2 = excluded.target_2,
                    target_3 = excluded.target_3,
                    targets_hit = '[]',
                    trade_type = excluded.trade_type,
                    notes = excluded.notes,
                    entries = '[]',
                    exits = '[]',
                    avg_entry_price = NULL,
                    avg_exit_price = NULL,
                    entry_price = NULL,
                    entry_date = NULL,
                    current_size = 0,
                    original_size = 0,
                    cost_basis = NULL,
                    realized_pnl = NULL,
                    realized_pnl_pct = NULL,
                    unrealized_pnl = NULL,
                    unrealized_pnl_pct = NULL,
                    updated_at = excluded.updated_at
                """,
                (
                    position_id,
                    user_id,
                    symbol,
                    stop_loss,
                    target_1,
                    target_2,
                    target_3,
                    trade_type,
                    notes,
                    now,
                    now,
                ),
            )
            await conn.commit()

        logger.info(f"Created position for {symbol} (user: {user_id})")

        return Position(
            id=position_id,
            user_id=user_id,
            symbol=symbol,
            status="watching",
            stop_loss=stop_loss,
            target_1=target_1,
            target_2=target_2,
            target_3=target_3,
            targets_hit=[],
            trade_type=trade_type,
            notes=notes,
            created_at=now,
            updated_at=now,
        )

    async def enter_position(
        self,
        user_id: str,
        symbol: str,
        entry_price: float,
        size: int,
    ) -> Optional[Position]:
        """Mark position as entered with entry price and size.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            entry_price: Entry price
            size: Number of shares

        Returns:
            Updated Position or None if not found
        """
        symbol = symbol.upper()
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE positions
                SET status = 'entered',
                    entry_price = ?,
                    entry_date = ?,
                    current_size = ?,
                    original_size = ?,
                    updated_at = ?
                WHERE user_id = ? AND symbol = ?
                """,
                (entry_price, now, size, size, now, user_id, symbol),
            )
            await conn.commit()

            if cursor.rowcount == 0:
                return None

        logger.info(f"Entered position {symbol} @ ${entry_price} x {size} shares")
        return await self.get_position(user_id, symbol)

    async def add_entry(
        self,
        user_id: str,
        symbol: str,
        price: float,
        shares: int,
        date: Optional[str] = None,
    ) -> Optional[Position]:
        """Add an entry to a position (scale in or initial entry).

        Recalculates avg_entry_price and cost_basis automatically.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            price: Entry price per share
            shares: Number of shares bought
            date: Entry date (defaults to now)

        Returns:
            Updated Position or None if not found
        """
        symbol = symbol.upper()
        entry_date = date or datetime.utcnow().isoformat()
        now = datetime.utcnow().isoformat()

        position = await self.get_position(user_id, symbol)
        if not position:
            return None

        # Add new entry to list
        new_entry = {"price": price, "shares": shares, "date": entry_date}
        entries = [e.model_dump() for e in position.entries] + [new_entry]

        # Calculate new weighted average entry price
        total_shares = sum(e["shares"] for e in entries)
        total_cost = sum(e["price"] * e["shares"] for e in entries)
        avg_entry = total_cost / total_shares if total_shares > 0 else None
        cost_basis = total_cost

        # Update position status and sizes
        new_status = "entered" if position.status == "watching" else position.status
        new_current_size = position.current_size + shares
        new_original_size = max(position.original_size, new_current_size)

        async with self.db.connection() as conn:
            await conn.execute(
                """
                UPDATE positions
                SET status = ?,
                    entries = ?,
                    avg_entry_price = ?,
                    cost_basis = ?,
                    current_size = ?,
                    original_size = ?,
                    entry_price = ?,
                    entry_date = COALESCE(entry_date, ?),
                    updated_at = ?
                WHERE user_id = ? AND symbol = ?
                """,
                (
                    new_status,
                    json.dumps(entries),
                    avg_entry,
                    cost_basis,
                    new_current_size,
                    new_original_size,
                    avg_entry,  # Also update legacy entry_price
                    entry_date,
                    now,
                    user_id,
                    symbol,
                ),
            )
            await conn.commit()

        logger.info(
            f"Added entry to {symbol}: {shares} shares @ ${price:.2f} "
            f"(avg: ${avg_entry:.2f}, total: {new_current_size} shares)"
        )
        return await self.get_position(user_id, symbol)

    async def add_exit(
        self,
        user_id: str,
        symbol: str,
        price: float,
        shares: int,
        reason: str = "manual",
        date: Optional[str] = None,
    ) -> Optional[Position]:
        """Add an exit from a position (partial or full).

        Calculates realized P&L automatically.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            price: Exit price per share
            shares: Number of shares sold
            reason: Exit reason (target_1, target_2, target_3, stop_loss, manual)
            date: Exit date (defaults to now)

        Returns:
            Updated Position or None if not found
        """
        symbol = symbol.upper()
        exit_date = date or datetime.utcnow().isoformat()
        now = datetime.utcnow().isoformat()

        position = await self.get_position(user_id, symbol)
        if not position:
            return None

        if not position.avg_entry_price:
            logger.warning(f"Cannot exit {symbol}: no entry price set")
            return position

        # Cap shares at current position size
        shares_to_exit = min(shares, position.current_size)
        if shares_to_exit <= 0:
            return position

        # Add new exit to list
        new_exit = {"price": price, "shares": shares_to_exit, "date": exit_date, "reason": reason}
        exits = [e.model_dump() for e in position.exits] + [new_exit]

        # Calculate realized P&L for this exit
        exit_proceeds = price * shares_to_exit
        exit_cost_basis = position.avg_entry_price * shares_to_exit
        this_exit_pnl = exit_proceeds - exit_cost_basis

        # Add to cumulative realized P&L
        new_realized_pnl = (position.realized_pnl or 0) + this_exit_pnl

        # Calculate average exit price across all exits
        total_exit_shares = sum(e["shares"] for e in exits)
        total_exit_proceeds = sum(e["price"] * e["shares"] for e in exits)
        avg_exit = total_exit_proceeds / total_exit_shares if total_exit_shares > 0 else None

        # Calculate realized P&L percentage (based on cost basis of exited shares)
        total_exit_cost_basis = position.avg_entry_price * total_exit_shares
        realized_pnl_pct = (
            (new_realized_pnl / total_exit_cost_basis * 100)
            if total_exit_cost_basis > 0
            else None
        )

        # Update position sizes and status
        new_current_size = position.current_size - shares_to_exit
        new_cost_basis = position.avg_entry_price * new_current_size if new_current_size > 0 else 0

        # Determine new status
        if new_current_size == 0:
            new_status = "stopped_out" if reason == "stop_loss" else "closed"
        else:
            new_status = "partial"

        # Track targets hit
        targets_hit = position.targets_hit.copy()
        if reason in ["target_1", "target_2", "target_3"]:
            target_num = int(reason.split("_")[1])
            if target_num not in targets_hit:
                targets_hit.append(target_num)

        async with self.db.connection() as conn:
            await conn.execute(
                """
                UPDATE positions
                SET status = ?,
                    exits = ?,
                    avg_exit_price = ?,
                    current_size = ?,
                    cost_basis = ?,
                    realized_pnl = ?,
                    realized_pnl_pct = ?,
                    targets_hit = ?,
                    updated_at = ?
                WHERE user_id = ? AND symbol = ?
                """,
                (
                    new_status,
                    json.dumps(exits),
                    avg_exit,
                    new_current_size,
                    new_cost_basis,
                    new_realized_pnl,
                    realized_pnl_pct,
                    json.dumps(targets_hit),
                    now,
                    user_id,
                    symbol,
                ),
            )
            await conn.commit()

        logger.info(
            f"Added exit from {symbol}: {shares_to_exit} shares @ ${price:.2f} "
            f"(P&L: ${this_exit_pnl:+.2f}, remaining: {new_current_size} shares)"
        )
        return await self.get_position(user_id, symbol)

    def calculate_unrealized_pnl(
        self, position: Position, current_price: float
    ) -> Position:
        """Calculate unrealized P&L based on current market price.

        Args:
            position: Position to calculate P&L for
            current_price: Current market price

        Returns:
            Position with unrealized_pnl and unrealized_pnl_pct populated
        """
        if not position.avg_entry_price or position.current_size <= 0:
            return position

        current_value = current_price * position.current_size
        cost_basis = position.avg_entry_price * position.current_size

        unrealized_pnl = current_value - cost_basis
        unrealized_pnl_pct = (unrealized_pnl / cost_basis * 100) if cost_basis > 0 else None

        # Return new position with P&L calculated (immutable pattern)
        return Position(
            **{
                **position.model_dump(),
                "unrealized_pnl": round(unrealized_pnl, 2),
                "unrealized_pnl_pct": round(unrealized_pnl_pct, 2) if unrealized_pnl_pct else None,
            }
        )

    async def get_position_with_pnl(
        self,
        user_id: str,
        symbol: str,
        current_price: Optional[float] = None,
    ) -> Optional[Position]:
        """Get position with live unrealized P&L calculated.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            current_price: Current market price (if None, unrealized P&L won't be calculated)

        Returns:
            Position with P&L fields populated, or None if not found
        """
        position = await self.get_position(user_id, symbol)
        if not position:
            return None

        if current_price is not None:
            position = self.calculate_unrealized_pnl(position, current_price)

        return position

    async def scale_out(
        self,
        user_id: str,
        symbol: str,
        shares_sold: int,
        target_hit: int,
    ) -> Optional[Position]:
        """Scale out of position when target is hit.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            shares_sold: Number of shares sold
            target_hit: Which target was hit (1, 2, or 3)

        Returns:
            Updated Position or None if not found
        """
        symbol = symbol.upper()
        position = await self.get_position(user_id, symbol)

        if not position:
            return None

        new_size = max(0, position.current_size - shares_sold)
        new_status = "partial" if new_size > 0 else "closed"
        targets_hit = position.targets_hit.copy()
        if target_hit not in targets_hit:
            targets_hit.append(target_hit)

        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            await conn.execute(
                """
                UPDATE positions
                SET current_size = ?,
                    status = ?,
                    targets_hit = ?,
                    updated_at = ?
                WHERE user_id = ? AND symbol = ?
                """,
                (new_size, new_status, json.dumps(targets_hit), now, user_id, symbol),
            )
            await conn.commit()

        logger.info(f"Scaled out {shares_sold} shares of {symbol}, target {target_hit} hit")
        return await self.get_position(user_id, symbol)

    async def update_stop_loss(
        self,
        user_id: str,
        symbol: str,
        new_stop: float,
    ) -> Optional[Position]:
        """Update stop loss level (e.g., move to breakeven).

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            new_stop: New stop loss price

        Returns:
            Updated Position or None if not found
        """
        symbol = symbol.upper()
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE positions
                SET stop_loss = ?,
                    updated_at = ?
                WHERE user_id = ? AND symbol = ?
                """,
                (new_stop, now, user_id, symbol),
            )
            await conn.commit()

            if cursor.rowcount == 0:
                return None

        logger.info(f"Updated stop loss for {symbol} to ${new_stop}")
        return await self.get_position(user_id, symbol)

    async def close_position(
        self,
        user_id: str,
        symbol: str,
        reason: str = "manual",
    ) -> bool:
        """Close a position.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol
            reason: Reason for closing (manual, stopped_out, target_hit)

        Returns:
            True if closed, False if not found
        """
        symbol = symbol.upper()
        status = "stopped_out" if reason == "stopped_out" else "closed"
        now = datetime.utcnow().isoformat()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                """
                UPDATE positions
                SET status = ?,
                    current_size = 0,
                    notes = COALESCE(notes, '') || ' [Closed: ' || ? || ']',
                    updated_at = ?
                WHERE user_id = ? AND symbol = ?
                """,
                (status, reason, now, user_id, symbol),
            )
            await conn.commit()

            if cursor.rowcount == 0:
                return False

        logger.info(f"Closed position {symbol} (reason: {reason})")
        return True

    async def get_position(self, user_id: str, symbol: str) -> Optional[Position]:
        """Get position for a symbol.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol

        Returns:
            Position or None if not found
        """
        symbol = symbol.upper()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM positions WHERE user_id = ? AND symbol = ?",
                (user_id, symbol),
            )
            row = await cursor.fetchone()

            if not row:
                return None

            return self._row_to_position(row)

    async def get_active_positions(self, user_id: str) -> List[Position]:
        """Get all active positions (not closed or stopped out).

        Args:
            user_id: User identifier

        Returns:
            List of active Position objects
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                """
                SELECT * FROM positions
                WHERE user_id = ? AND status NOT IN ('closed', 'stopped_out')
                ORDER BY updated_at DESC
                """,
                (user_id,),
            )
            rows = await cursor.fetchall()

            return [self._row_to_position(row) for row in rows]

    async def get_all_positions(self, user_id: str) -> List[Position]:
        """Get all positions for a user.

        Args:
            user_id: User identifier

        Returns:
            List of all Position objects
        """
        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "SELECT * FROM positions WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            )
            rows = await cursor.fetchall()

            return [self._row_to_position(row) for row in rows]

    async def delete_position(self, user_id: str, symbol: str) -> bool:
        """Delete a position completely.

        Args:
            user_id: User identifier
            symbol: Stock ticker symbol

        Returns:
            True if deleted, False if not found
        """
        symbol = symbol.upper()

        async with self.db.connection() as conn:
            cursor = await conn.execute(
                "DELETE FROM positions WHERE user_id = ? AND symbol = ?",
                (user_id, symbol),
            )
            await conn.commit()

            return cursor.rowcount > 0

    def _row_to_position(self, row) -> Position:
        """Convert database row to Position object."""
        targets_hit = json.loads(row["targets_hit"]) if row["targets_hit"] else []

        # Parse entries list
        entries_raw = json.loads(row["entries"]) if row["entries"] else []
        entries = [PositionEntry(**e) for e in entries_raw]

        # Parse exits list
        exits_raw = json.loads(row["exits"]) if row["exits"] else []
        exits = [PositionExit(**e) for e in exits_raw]

        return Position(
            id=row["id"],
            user_id=row["user_id"],
            symbol=row["symbol"],
            status=row["status"],
            entry_price=row["entry_price"],
            entry_date=row["entry_date"],
            entries=entries,
            avg_entry_price=row["avg_entry_price"],
            exits=exits,
            avg_exit_price=row["avg_exit_price"],
            current_size=row["current_size"] or 0,
            original_size=row["original_size"] or 0,
            stop_loss=row["stop_loss"],
            target_1=row["target_1"],
            target_2=row["target_2"],
            target_3=row["target_3"],
            targets_hit=targets_hit,
            cost_basis=row["cost_basis"],
            realized_pnl=row["realized_pnl"],
            realized_pnl_pct=row["realized_pnl_pct"],
            unrealized_pnl=row["unrealized_pnl"],
            unrealized_pnl_pct=row["unrealized_pnl_pct"],
            trade_type=row["trade_type"],
            notes=row["notes"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# Singleton instance
_store: Optional[PositionStore] = None


def get_position_store() -> PositionStore:
    """Get the singleton position store instance."""
    global _store
    if _store is None:
        _store = PositionStore()
    return _store
