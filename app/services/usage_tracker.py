"""Usage tracking service for AI API calls.

Provides utilities to track AI provider usage with automatic cost calculation.
"""

import logging
from typing import Optional, Dict, Any

from app.storage.usage_store import get_usage_store, UsageStore
from app.models.usage import (
    UsageRecord,
    ModelProvider,
    OperationType,
)
from app.agent.providers import AIResponse

logger = logging.getLogger(__name__)


class UsageTracker:
    """Service for tracking AI API usage.

    Usage:
        tracker = get_usage_tracker()
        await tracker.track_ai_response(
            user_id="user-123",
            provider=ModelProvider.CLAUDE,
            model="claude-sonnet-4-20250514",
            operation_type=OperationType.PLAN_GENERATION,
            response=ai_response,
            symbol="AAPL",
        )
    """

    def __init__(self, store: Optional[UsageStore] = None):
        """Initialize the usage tracker.

        Args:
            store: Optional UsageStore instance (defaults to singleton)
        """
        self._store = store

    @property
    def store(self) -> UsageStore:
        """Get the usage store (lazy initialization)."""
        if self._store is None:
            self._store = get_usage_store()
        return self._store

    async def track_ai_response(
        self,
        user_id: str,
        provider: ModelProvider,
        model: str,
        operation_type: OperationType,
        response: AIResponse,
        symbol: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[UsageRecord]:
        """Track an AI API response with usage data.

        Args:
            user_id: User who made the request
            provider: AI provider (claude or grok)
            model: Model ID used
            operation_type: Type of operation
            response: AIResponse object containing usage data
            symbol: Stock symbol if applicable
            endpoint: API endpoint called

        Returns:
            Created UsageRecord or None if no usage data
        """
        if not response.usage:
            logger.debug(f"No usage data in response for {operation_type}")
            return None

        # Extract token counts from response
        # Note: Claude uses "input_tokens"/"output_tokens", OpenAI/Grok use "prompt_tokens"/"completion_tokens"
        usage = response.usage
        input_tokens = usage.get("input_tokens") if "input_tokens" in usage else usage.get("prompt_tokens", 0)
        output_tokens = usage.get("output_tokens") if "output_tokens" in usage else usage.get("completion_tokens", 0)

        # Count tool calls (from citations or direct tool_calls)
        tool_calls = 0
        if response.citations:
            # Each citation source likely came from a search
            tool_calls = 1  # Conservative estimate
        if response.tool_calls:
            tool_calls += len(response.tool_calls)

        try:
            record = await self.store.log_usage(
                user_id=user_id,
                provider=provider,
                model=model,
                operation_type=operation_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                symbol=symbol,
                endpoint=endpoint,
            )
            return record
        except Exception as e:
            logger.error(f"Failed to track usage: {e}")
            return None

    async def track_manual(
        self,
        user_id: str,
        provider: ModelProvider,
        model: str,
        operation_type: OperationType,
        input_tokens: int,
        output_tokens: int,
        tool_calls: int = 0,
        symbol: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[UsageRecord]:
        """Track usage with manual token counts.

        Use this when you have raw token counts instead of an AIResponse.

        Args:
            user_id: User who made the request
            provider: AI provider (claude or grok)
            model: Model ID used
            operation_type: Type of operation
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            tool_calls: Number of tool calls
            symbol: Stock symbol if applicable
            endpoint: API endpoint called

        Returns:
            Created UsageRecord or None on error
        """
        try:
            record = await self.store.log_usage(
                user_id=user_id,
                provider=provider,
                model=model,
                operation_type=operation_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                tool_calls=tool_calls,
                symbol=symbol,
                endpoint=endpoint,
            )
            return record
        except Exception as e:
            logger.error(f"Failed to track manual usage: {e}")
            return None

    async def track_streaming_usage(
        self,
        user_id: str,
        provider: ModelProvider,
        model: str,
        operation_type: OperationType,
        estimated_input_tokens: int,
        output_text: str,
        symbol: Optional[str] = None,
        endpoint: Optional[str] = None,
    ) -> Optional[UsageRecord]:
        """Track usage for streaming responses.

        For streaming, we estimate output tokens based on character count.
        A rough estimate is ~4 characters per token.

        Args:
            user_id: User who made the request
            provider: AI provider (claude or grok)
            model: Model ID used
            operation_type: Type of operation
            estimated_input_tokens: Estimated input tokens (from message length)
            output_text: The full streamed output text
            symbol: Stock symbol if applicable
            endpoint: API endpoint called

        Returns:
            Created UsageRecord or None on error
        """
        # Estimate output tokens (~4 chars per token)
        estimated_output_tokens = len(output_text) // 4

        return await self.track_manual(
            user_id=user_id,
            provider=provider,
            model=model,
            operation_type=operation_type,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            tool_calls=0,
            symbol=symbol,
            endpoint=endpoint,
        )


# Singleton instance
_tracker: Optional[UsageTracker] = None


def get_usage_tracker() -> UsageTracker:
    """Get the singleton usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker()
    return _tracker


# Convenience functions for direct use

async def track_usage(
    user_id: str,
    provider: ModelProvider,
    model: str,
    operation_type: OperationType,
    response: AIResponse,
    symbol: Optional[str] = None,
    endpoint: Optional[str] = None,
) -> Optional[UsageRecord]:
    """Convenience function to track AI response usage.

    Args:
        user_id: User who made the request
        provider: AI provider (claude or grok)
        model: Model ID used
        operation_type: Type of operation
        response: AIResponse object containing usage data
        symbol: Stock symbol if applicable
        endpoint: API endpoint called

    Returns:
        Created UsageRecord or None if no usage data
    """
    tracker = get_usage_tracker()
    return await tracker.track_ai_response(
        user_id=user_id,
        provider=provider,
        model=model,
        operation_type=operation_type,
        response=response,
        symbol=symbol,
        endpoint=endpoint,
    )
