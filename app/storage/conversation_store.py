"""Conversation history storage for the planning agent.

Stores chat messages so the agent has memory of previous discussions.
"""

import json
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

import aiosqlite

from app.storage.database import get_db_path

logger = logging.getLogger(__name__)


@dataclass
class ConversationMessage:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: str = ""

    def to_claude_format(self) -> Dict[str, str]:
        """Convert to Claude API message format."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """A conversation thread for a stock."""

    user_id: str
    symbol: str  # "general" for non-stock-specific
    messages: List[ConversationMessage]
    created_at: str = ""
    updated_at: str = ""

    def to_claude_messages(self, max_messages: int = 20) -> List[Dict[str, str]]:
        """Convert to Claude API messages format, limited to recent messages."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        return [msg.to_claude_format() for msg in recent]

    def add_message(self, role: str, content: str):
        """Add a message to the conversation."""
        self.messages.append(ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.utcnow().isoformat()
        ))
        self.updated_at = datetime.utcnow().isoformat()


class ConversationStore:
    """Async storage for conversation history."""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or get_db_path()
        self._initialized = False

    async def _ensure_table(self):
        """Ensure the conversations table exists."""
        if self._initialized:
            return

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    messages TEXT DEFAULT '[]',
                    created_at TEXT,
                    updated_at TEXT,
                    UNIQUE(user_id, symbol)
                )
            """)
            await db.commit()

        self._initialized = True

    async def get_conversation(self, user_id: str, symbol: str) -> Conversation:
        """Get or create a conversation for a user and symbol."""
        await self._ensure_table()
        symbol = symbol.upper() if symbol != "general" else symbol

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(
                "SELECT * FROM conversations WHERE user_id = ? AND symbol = ?",
                (user_id, symbol)
            ) as cursor:
                row = await cursor.fetchone()

                if row:
                    messages_data = json.loads(row["messages"])
                    messages = [
                        ConversationMessage(**msg) for msg in messages_data
                    ]
                    return Conversation(
                        user_id=row["user_id"],
                        symbol=row["symbol"],
                        messages=messages,
                        created_at=row["created_at"],
                        updated_at=row["updated_at"]
                    )

        # Create new conversation
        now = datetime.utcnow().isoformat()
        return Conversation(
            user_id=user_id,
            symbol=symbol,
            messages=[],
            created_at=now,
            updated_at=now
        )

    async def save_conversation(self, conversation: Conversation) -> Conversation:
        """Save a conversation."""
        await self._ensure_table()

        now = datetime.utcnow().isoformat()
        conversation.updated_at = now
        if not conversation.created_at:
            conversation.created_at = now

        # Serialize messages
        messages_json = json.dumps([asdict(msg) for msg in conversation.messages])

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT INTO conversations (user_id, symbol, messages, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(user_id, symbol) DO UPDATE SET
                    messages = excluded.messages,
                    updated_at = excluded.updated_at
            """, (
                conversation.user_id,
                conversation.symbol,
                messages_json,
                conversation.created_at,
                conversation.updated_at
            ))
            await db.commit()

        return conversation

    async def add_message(
        self,
        user_id: str,
        symbol: str,
        role: str,
        content: str
    ) -> Conversation:
        """Add a message to a conversation and save."""
        conversation = await self.get_conversation(user_id, symbol)
        conversation.add_message(role, content)
        return await self.save_conversation(conversation)

    async def clear_conversation(self, user_id: str, symbol: str) -> bool:
        """Clear a conversation's messages."""
        await self._ensure_table()
        symbol = symbol.upper() if symbol != "general" else symbol

        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM conversations WHERE user_id = ? AND symbol = ?",
                (user_id, symbol)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def get_message_count(self, user_id: str, symbol: str) -> int:
        """Get the number of messages in a conversation."""
        conversation = await self.get_conversation(user_id, symbol)
        return len(conversation.messages)


# Singleton instance
_conversation_store: Optional[ConversationStore] = None


def get_conversation_store() -> ConversationStore:
    """Get singleton ConversationStore instance."""
    global _conversation_store
    if _conversation_store is None:
        _conversation_store = ConversationStore()
    return _conversation_store
