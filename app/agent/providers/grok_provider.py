"""Grok (xAI) AI Provider implementation.

This provider uses the xAI API (OpenAI-compatible) to interact with Grok models,
supporting message creation, streaming, vision, web search, and X (Twitter) search.
"""

import json
import logging
from typing import Optional, List, Dict, Any, AsyncIterator

import httpx

from app.agent.providers import (
    AIProvider,
    ProviderConfig,
    AIMessage,
    AIResponse,
    ToolCall,
    SearchParameters,
)

logger = logging.getLogger(__name__)

# Grok API base URL
GROK_BASE_URL = "https://api.x.ai/v1"


class GrokProvider(AIProvider):
    """Grok AI provider using the xAI API.

    Features:
    - Streaming and non-streaming message creation
    - Vision (image analysis)
    - Web search via search_parameters
    - X (Twitter) search - unique to Grok
    - Tool/function calling (OpenAI-compatible format)
    """

    def __init__(self, config: ProviderConfig):
        """Initialize the Grok provider.

        Args:
            config: Provider configuration with API key and model names
        """
        super().__init__(config)
        self._client: Optional[httpx.AsyncClient] = None
        self._base_url = config.base_url or GROK_BASE_URL

    def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP async client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=httpx.Timeout(60.0, connect=10.0),
            )
            self._initialized = True
        return self._client

    def _convert_messages(
        self,
        messages: List[AIMessage],
        system: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert AIMessage list to Grok/OpenAI message format.

        Args:
            messages: List of AIMessage objects
            system: Optional system prompt to prepend

        Returns:
            List of message dicts in OpenAI format
        """
        result = []

        # Add system message if provided
        if system:
            result.append({"role": "system", "content": system})

        # Convert messages
        for msg in messages:
            result.append({"role": msg.role, "content": msg.content})

        return result

    def _build_request_body(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        max_tokens: int,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        search_parameters: Optional[SearchParameters] = None,
    ) -> Dict[str, Any]:
        """Build the API request body.

        Args:
            messages: Formatted messages
            model: Model ID
            max_tokens: Maximum tokens
            stream: Whether to stream the response
            tools: Tool definitions
            search_parameters: Search configuration

        Returns:
            Request body dict
        """
        body = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        # Add search parameters if provided
        if search_parameters and search_parameters.mode != "off":
            body["search_parameters"] = {
                "mode": search_parameters.mode,
                "sources": search_parameters.sources,
                "return_citations": search_parameters.return_citations,
            }

        # Add tools if provided (OpenAI-compatible format)
        if tools:
            body["tools"] = tools

        return body

    def _extract_response(self, data: Dict[str, Any]) -> AIResponse:
        """Extract content and tool calls from Grok response.

        Args:
            data: JSON response data

        Returns:
            AIResponse with extracted content
        """
        content = ""
        tool_calls = []
        citations = None

        # Extract message content
        if "choices" in data and len(data["choices"]) > 0:
            choice = data["choices"][0]
            message = choice.get("message", {})

            content = message.get("content", "")

            # Extract tool calls if present
            if "tool_calls" in message:
                for tc in message["tool_calls"]:
                    tool_calls.append(ToolCall(
                        id=tc.get("id", ""),
                        name=tc.get("function", {}).get("name", ""),
                        arguments=json.loads(tc.get("function", {}).get("arguments", "{}")),
                    ))

        # Extract citations if present (from search)
        # Grok API returns citations as a list of URL strings
        # Convert to list of dicts for AIResponse compatibility
        if "citations" in data:
            raw_citations = data["citations"]
            if raw_citations:
                # Handle both string URLs and dict format
                citations = []
                for c in raw_citations:
                    if isinstance(c, str):
                        citations.append({"url": c})
                    elif isinstance(c, dict):
                        citations.append(c)
                    # Skip other types

        # Build usage info
        usage = None
        if "usage" in data:
            usage = {
                "prompt_tokens": data["usage"].get("prompt_tokens", 0),
                "completion_tokens": data["usage"].get("completion_tokens", 0),
                "total_tokens": data["usage"].get("total_tokens", 0),
            }

        return AIResponse(
            content=content,
            tool_calls=tool_calls,
            usage=usage,
            citations=citations,
            raw_response=data,
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
        """Create a message using Grok.

        Args:
            messages: Conversation history
            system: System prompt
            model_type: "planning" or "fast"
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            search_parameters: Search configuration (supports web, X, news)

        Returns:
            AIResponse with content and optional tool calls

        Raises:
            httpx.HTTPStatusError: If the API request fails
        """
        client = self._get_client()
        model = self.get_model(model_type)

        # Build request
        formatted_messages = self._convert_messages(messages, system)
        body = self._build_request_body(
            messages=formatted_messages,
            model=model,
            max_tokens=max_tokens,
            stream=False,
            tools=tools,
            search_parameters=search_parameters,
        )

        try:
            logger.debug(f"Grok request: model={model}, messages={len(messages)}")
            response = await client.post("/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            return self._extract_response(data)

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Grok rate limit hit: {e}")
            else:
                logger.error(f"Grok API error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Grok connection error: {e}")
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
        """Create a streaming message using Grok.

        Args:
            messages: Conversation history
            system: System prompt
            model_type: "planning" or "fast"
            max_tokens: Maximum tokens in response
            tools: Tool definitions for function calling
            search_parameters: Search configuration

        Yields:
            Text chunks as they are generated
        """
        client = self._get_client()
        model = self.get_model(model_type)

        # Build request
        formatted_messages = self._convert_messages(messages, system)
        body = self._build_request_body(
            messages=formatted_messages,
            model=model,
            max_tokens=max_tokens,
            stream=True,
            tools=tools,
            search_parameters=search_parameters,
        )

        try:
            logger.debug(f"Grok streaming request: model={model}")

            async with client.stream("POST", "/chat/completions", json=body) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    # Parse SSE format: "data: {json}\n\n"
                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix

                        # Check for end of stream
                        if data_str.strip() == "[DONE]":
                            break

                        try:
                            data = json.loads(data_str)

                            # Extract delta content
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content

                        except json.JSONDecodeError as e:
                            logger.warning(f"Failed to parse SSE chunk: {e}")
                            continue

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning(f"Grok rate limit hit during stream: {e}")
            else:
                logger.error(f"Grok API error during stream: {e.response.status_code}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Grok connection error during stream: {e}")
            raise

    async def analyze_image(
        self,
        image_base64: str,
        prompt: str,
        model_type: str = "planning",
    ) -> AIResponse:
        """Analyze an image using Grok's vision capabilities.

        Args:
            image_base64: Base64-encoded image data
            prompt: Analysis prompt describing what to look for
            model_type: Model to use (default: planning for better analysis)

        Returns:
            AIResponse with analysis content
        """
        client = self._get_client()
        model = self.get_model(model_type)

        # Determine media type (default to PNG)
        media_type = "image/png"
        if image_base64.startswith("/9j/"):  # JPEG magic bytes
            media_type = "image/jpeg"
        elif image_base64.startswith("R0lGOD"):  # GIF magic bytes
            media_type = "image/gif"
        elif image_base64.startswith("UklGR"):  # WebP magic bytes
            media_type = "image/webp"

        # Build message with image content (OpenAI vision format)
        message_content = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:{media_type};base64,{image_base64}",
                },
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]

        body = {
            "model": model,
            "messages": [{"role": "user", "content": message_content}],
            "max_tokens": 2000,
        }

        try:
            logger.debug(f"Grok vision request: model={model}")
            response = await client.post("/chat/completions", json=body)
            response.raise_for_status()
            data = response.json()
            return self._extract_response(data)

        except httpx.HTTPStatusError as e:
            logger.error(f"Grok vision error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Grok vision connection error: {e}")
            raise

    @property
    def supports_web_search(self) -> bool:
        """Grok supports web search via search_parameters."""
        return True

    @property
    def supports_x_search(self) -> bool:
        """Grok uniquely supports X (Twitter) search."""
        return True

    async def close(self):
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()


def get_x_search_parameters() -> SearchParameters:
    """Get search parameters optimized for X/Twitter and web search.

    This configuration is ideal for trading plan generation where you want
    both real-time X sentiment and web news.

    Returns:
        SearchParameters configured for X, web, and news sources
    """
    return SearchParameters(
        mode="on",
        sources=[
            {"type": "x"},      # X (Twitter) for real-time sentiment
            {"type": "web"},    # General web search
            {"type": "news"},   # News sources
        ],
        return_citations=True,
    )


def get_web_only_search_parameters() -> SearchParameters:
    """Get search parameters for web-only search.

    Returns:
        SearchParameters configured for web search only
    """
    return SearchParameters(
        mode="on",
        sources=[{"type": "web"}],
        return_citations=True,
    )
