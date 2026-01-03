"""AI Provider abstraction layer for StockMate.

This module provides a unified interface for different AI providers (Claude, Grok),
enabling easy switching between models and leveraging provider-specific features.
"""

import logging
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List, Dict, Any, AsyncIterator

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ModelProvider(str, Enum):
    """Supported AI model providers."""
    CLAUDE = "claude"
    GROK = "grok"


class ProviderConfig(BaseModel):
    """Configuration for an AI provider."""
    provider: ModelProvider
    planning_model: str = Field(..., description="Model ID for planning/analysis tasks")
    fast_model: str = Field(..., description="Model ID for quick responses/chat")
    api_key: str = Field(..., description="API key for the provider")
    base_url: Optional[str] = Field(None, description="Base URL override for the API")


class AIMessage(BaseModel):
    """A message in a conversation."""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ToolCall(BaseModel):
    """A tool call request from the AI."""
    id: str = Field(..., description="Unique identifier for this tool call")
    name: str = Field(..., description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool")


class AIResponse(BaseModel):
    """Response from an AI provider."""
    content: str = Field(..., description="Text content of the response")
    tool_calls: List[ToolCall] = Field(default_factory=list, description="Tool calls requested by the AI")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")
    citations: Optional[List[Dict[str, Any]]] = Field(None, description="Citations from web/X search (Grok)")
    raw_response: Optional[Any] = Field(None, description="Raw response object from the provider", exclude=True)


class SearchParameters(BaseModel):
    """Parameters for web/X search capabilities."""
    mode: str = Field("auto", description="Search mode: 'auto', 'on', or 'off'")
    sources: List[Dict[str, str]] = Field(
        default_factory=lambda: [{"type": "web"}],
        description="Search sources: web, x, news"
    )
    return_citations: bool = Field(True, description="Whether to return citations")


class AIProvider(ABC):
    """Abstract base class for AI providers.

    Implementations must handle:
    - Message creation (streaming and non-streaming)
    - Image analysis (vision capabilities)
    - Tool/function calling
    - Search capabilities (web, X/Twitter)
    """

    def __init__(self, config: ProviderConfig):
        """Initialize the provider with configuration.

        Args:
            config: Provider configuration including API keys and model names
        """
        self.config = config
        self._initialized = False
        logger.info(f"Initializing {config.provider.value} provider")

    @abstractmethod
    async def create_message(
        self,
        messages: List[AIMessage],
        system: Optional[str] = None,
        model_type: str = "fast",
        max_tokens: int = 2000,
        tools: Optional[List[Dict[str, Any]]] = None,
        search_parameters: Optional[SearchParameters] = None,
    ) -> AIResponse:
        """Create a message using the AI model.

        Args:
            messages: Conversation history
            system: System prompt
            model_type: "planning" for complex tasks, "fast" for quick responses
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            search_parameters: Parameters for web/X search

        Returns:
            AIResponse with content and optional tool calls
        """
        pass

    @abstractmethod
    async def create_message_stream(
        self,
        messages: List[AIMessage],
        system: Optional[str] = None,
        model_type: str = "fast",
        max_tokens: int = 2000,
        tools: Optional[List[Dict[str, Any]]] = None,
        search_parameters: Optional[SearchParameters] = None,
    ) -> AsyncIterator[str]:
        """Create a streaming message using the AI model.

        Args:
            messages: Conversation history
            system: System prompt
            model_type: "planning" for complex tasks, "fast" for quick responses
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            search_parameters: Parameters for web/X search

        Yields:
            Text chunks as they are generated
        """
        pass

    @abstractmethod
    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        model_type: str = "planning",
    ) -> AIResponse:
        """Analyze an image using the AI model's vision capabilities.

        Args:
            image_base64: Base64-encoded image data
            prompt: Analysis prompt
            model_type: Model to use for analysis

        Returns:
            AIResponse with analysis content
        """
        pass

    @property
    @abstractmethod
    def supports_web_search(self) -> bool:
        """Whether this provider supports web search."""
        pass

    @property
    @abstractmethod
    def supports_x_search(self) -> bool:
        """Whether this provider supports X (Twitter) search."""
        pass

    def get_model(self, model_type: str = "fast") -> str:
        """Get the model ID for the specified type.

        Args:
            model_type: "planning" or "fast"

        Returns:
            Model ID string
        """
        if model_type == "planning":
            return self.config.planning_model
        return self.config.fast_model


__all__ = [
    "ModelProvider",
    "ProviderConfig",
    "AIMessage",
    "ToolCall",
    "AIResponse",
    "SearchParameters",
    "AIProvider",
]
