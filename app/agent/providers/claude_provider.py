"""Claude (Anthropic) AI Provider implementation.

This provider uses the Anthropic SDK to interact with Claude models,
supporting message creation, streaming, vision, and web search.
"""

import logging
from typing import Optional, List, Dict, Any, AsyncIterator

import anthropic
from anthropic._exceptions import APIError, APIConnectionError, RateLimitError

from app.agent.providers import (
    AIProvider,
    ProviderConfig,
    AIMessage,
    AIResponse,
    ToolCall,
    SearchParameters,
)

logger = logging.getLogger(__name__)


class ClaudeProvider(AIProvider):
    """Claude AI provider using the Anthropic SDK.

    Features:
    - Streaming and non-streaming message creation
    - Vision (image analysis)
    - Web search via web_search_20250305 tool
    - Tool/function calling
    """

    def __init__(self, config: ProviderConfig):
        """Initialize the Claude provider.

        Args:
            config: Provider configuration with API key and model names
        """
        super().__init__(config)
        self._client: Optional[anthropic.AsyncAnthropic] = None

    def _get_client(self) -> anthropic.AsyncAnthropic:
        """Get or create the Anthropic async client."""
        if self._client is None:
            kwargs = {"api_key": self.config.api_key}
            if self.config.base_url:
                kwargs["base_url"] = self.config.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)
            self._initialized = True
        return self._client

    def _convert_messages(self, messages: List[AIMessage]) -> List[Dict[str, str]]:
        """Convert AIMessage list to Anthropic message format.

        Args:
            messages: List of AIMessage objects

        Returns:
            List of message dicts in Anthropic format
        """
        return [{"role": msg.role, "content": msg.content} for msg in messages]

    def _build_tools(
        self,
        tools: Optional[List[Dict[str, Any]]],
        search_parameters: Optional[SearchParameters],
    ) -> Optional[List[Dict[str, Any]]]:
        """Build the tools list including web search if enabled.

        Args:
            tools: Custom tools to include
            search_parameters: Search configuration

        Returns:
            Combined tools list or None
        """
        result_tools = []

        # Add web search tool if search is enabled
        if search_parameters and search_parameters.mode != "off":
            result_tools.append({
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,  # Default max uses
            })

        # Add custom tools
        if tools:
            result_tools.extend(tools)

        return result_tools if result_tools else None

    def _extract_response(self, response) -> AIResponse:
        """Extract content and tool calls from Anthropic response.

        Args:
            response: Anthropic API response

        Returns:
            AIResponse with extracted content
        """
        text_parts = []
        tool_calls = []

        for block in response.content:
            if hasattr(block, "type"):
                if block.type == "text":
                    text_parts.append(block.text)
                elif block.type == "tool_use":
                    tool_calls.append(ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input if hasattr(block, "input") else {},
                    ))

        # Build usage info
        usage = None
        if hasattr(response, "usage") and response.usage:
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }

        return AIResponse(
            content="\n".join(text_parts),
            tool_calls=tool_calls,
            usage=usage,
            raw_response=response,
        )

    async def create_message(
        self,
        messages: List[AIMessage],
        system: Optional[str] = None,
        model_type: str = "fast",
        max_tokens: int = 2000,
        tools: Optional[List[Dict[str, Any]]] = None,
        search_parameters: Optional[SearchParameters] = None,
    ) -> AIResponse:
        """Create a message using Claude.

        Args:
            messages: Conversation history
            system: System prompt
            model_type: "planning" or "fast"
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            search_parameters: Web search configuration

        Returns:
            AIResponse with content and optional tool calls

        Raises:
            APIError: If the API request fails
        """
        client = self._get_client()
        model = self.get_model(model_type)

        # Build request parameters
        request_params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages),
        }

        if system:
            request_params["system"] = system

        # Add tools (including web search if enabled)
        combined_tools = self._build_tools(tools, search_parameters)
        if combined_tools:
            request_params["tools"] = combined_tools

        try:
            logger.debug(f"Claude request: model={model}, messages={len(messages)}")
            response = await client.messages.create(**request_params)
            return self._extract_response(response)

        except RateLimitError as e:
            logger.warning(f"Claude rate limit hit: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Claude connection error: {e}")
            raise
        except APIError as e:
            logger.error(f"Claude API error: {e}")
            raise

    async def create_message_stream(
        self,
        messages: List[AIMessage],
        system: Optional[str] = None,
        model_type: str = "fast",
        max_tokens: int = 2000,
        tools: Optional[List[Dict[str, Any]]] = None,
        search_parameters: Optional[SearchParameters] = None,
    ) -> AsyncIterator[str]:
        """Create a streaming message using Claude.

        Args:
            messages: Conversation history
            system: System prompt
            model_type: "planning" or "fast"
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            search_parameters: Web search configuration

        Yields:
            Text chunks as they are generated
        """
        client = self._get_client()
        model = self.get_model(model_type)

        # Build request parameters
        request_params = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": self._convert_messages(messages),
        }

        if system:
            request_params["system"] = system

        # Add tools (including web search if enabled)
        combined_tools = self._build_tools(tools, search_parameters)
        if combined_tools:
            request_params["tools"] = combined_tools

        try:
            logger.debug(f"Claude streaming request: model={model}")

            async with client.messages.stream(**request_params) as stream:
                async for text in stream.text_stream:
                    yield text

        except RateLimitError as e:
            logger.warning(f"Claude rate limit hit during stream: {e}")
            raise
        except APIConnectionError as e:
            logger.error(f"Claude connection error during stream: {e}")
            raise
        except APIError as e:
            logger.error(f"Claude API error during stream: {e}")
            raise

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        model_type: str = "planning",
    ) -> AIResponse:
        """Analyze an image using Claude's vision capabilities.

        Args:
            image_base64: Base64-encoded image data (PNG, JPEG, GIF, or WebP)
            prompt: Analysis prompt describing what to look for
            model_type: Model to use (default: planning for better analysis)

        Returns:
            AIResponse with analysis content
        """
        client = self._get_client()
        model = self.get_model(model_type)

        # Determine media type (default to PNG for charts)
        media_type = "image/png"
        if image_base64.startswith("/9j/"):  # JPEG magic bytes
            media_type = "image/jpeg"
        elif image_base64.startswith("R0lGOD"):  # GIF magic bytes
            media_type = "image/gif"
        elif image_base64.startswith("UklGR"):  # WebP magic bytes
            media_type = "image/webp"

        message_content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": image_base64,
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

        try:
            logger.debug(f"Claude vision request: model={model}")
            response = await client.messages.create(
                model=model,
                max_tokens=2000,
                messages=[{"role": "user", "content": message_content}],
            )
            return self._extract_response(response)

        except APIError as e:
            logger.error(f"Claude vision error: {e}")
            raise

    @property
    def supports_web_search(self) -> bool:
        """Claude supports web search via the web_search_20250305 tool."""
        return True

    @property
    def supports_x_search(self) -> bool:
        """Claude does not support X (Twitter) search."""
        return False
