"""Minimal tests for AI Provider implementations.

Tests that Claude and Grok providers can make basic API calls.
Uses fast models with short prompts to minimize costs.
"""

import asyncio
import pytest
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestProviderFactory:
    """Tests for the provider factory module."""

    def test_import_factory(self):
        """Test that factory module can be imported."""
        from app.agent.providers.factory import (
            get_provider_config,
            create_provider,
            get_provider,
            get_default_provider,
            get_available_providers,
            is_provider_available,
        )
        assert get_provider_config is not None
        assert create_provider is not None
        assert get_provider is not None
        assert get_default_provider is not None
        assert get_available_providers is not None
        assert is_provider_available is not None

    def test_import_providers(self):
        """Test that provider classes can be imported."""
        from app.agent.providers import (
            ModelProvider,
            ProviderConfig,
            AIMessage,
            AIResponse,
            AIProvider,
        )
        from app.agent.providers.claude_provider import ClaudeProvider
        from app.agent.providers.grok_provider import GrokProvider

        assert ModelProvider is not None
        assert ProviderConfig is not None
        assert AIMessage is not None
        assert AIResponse is not None
        assert AIProvider is not None
        assert ClaudeProvider is not None
        assert GrokProvider is not None

    def test_model_provider_enum(self):
        """Test ModelProvider enum values."""
        from app.agent.providers import ModelProvider

        assert ModelProvider.CLAUDE.value == "claude"
        assert ModelProvider.GROK.value == "grok"

    def test_get_available_providers(self):
        """Test that available providers are returned based on configuration."""
        from app.agent.providers.factory import get_available_providers
        from app.agent.providers import ModelProvider

        available = get_available_providers()
        logger.info(f"Available providers: {[p.value for p in available]}")

        # At least one provider should be available if tests are running
        assert isinstance(available, list)

    def test_get_default_provider(self):
        """Test that a default provider is returned."""
        from app.agent.providers.factory import get_default_provider
        from app.agent.providers import ModelProvider

        default = get_default_provider()
        logger.info(f"Default provider: {default.value}")

        assert default in [ModelProvider.CLAUDE, ModelProvider.GROK]

    def test_is_provider_available(self):
        """Test provider availability check."""
        from app.agent.providers.factory import is_provider_available
        from app.agent.providers import ModelProvider

        claude_available = is_provider_available(ModelProvider.CLAUDE)
        grok_available = is_provider_available(ModelProvider.GROK)

        logger.info(f"Claude available: {claude_available}")
        logger.info(f"Grok available: {grok_available}")

        # Just test they return booleans
        assert isinstance(claude_available, bool)
        assert isinstance(grok_available, bool)


class TestClaudeProvider:
    """Tests for Claude provider with real API calls."""

    @pytest.fixture
    def claude_provider(self) -> Optional["ClaudeProvider"]:
        """Create a Claude provider if API key is available."""
        from app.agent.providers.factory import (
            is_provider_available,
            create_provider,
        )
        from app.agent.providers import ModelProvider

        if not is_provider_available(ModelProvider.CLAUDE):
            pytest.skip("Claude API key not configured")
            return None

        return create_provider(ModelProvider.CLAUDE)

    @pytest.mark.asyncio
    async def test_claude_simple_message(self, claude_provider):
        """Test Claude can respond to a simple message."""
        from app.agent.providers import AIMessage

        if claude_provider is None:
            return

        messages = [
            AIMessage(role="user", content="Say hello in 5 words")
        ]

        response = await claude_provider.create_message(
            messages=messages,
            model_type="fast",  # Use fast model to minimize cost
            max_tokens=50,  # Limit tokens
        )

        logger.info(f"Claude response: {response.content}")
        logger.info(f"Claude usage: {response.usage}")

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.usage is not None

    @pytest.mark.asyncio
    async def test_claude_with_system_prompt(self, claude_provider):
        """Test Claude respects system prompts."""
        from app.agent.providers import AIMessage

        if claude_provider is None:
            return

        messages = [
            AIMessage(role="user", content="What are you?")
        ]

        response = await claude_provider.create_message(
            messages=messages,
            system="You are a friendly robot. Respond in 10 words or less.",
            model_type="fast",
            max_tokens=50,
        )

        logger.info(f"Claude system prompt response: {response.content}")

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

    @pytest.mark.asyncio
    async def test_claude_streaming(self, claude_provider):
        """Test Claude streaming works."""
        from app.agent.providers import AIMessage

        if claude_provider is None:
            return

        messages = [
            AIMessage(role="user", content="Count 1 to 3")
        ]

        chunks = []
        async for chunk in claude_provider.create_message_stream(
            messages=messages,
            model_type="fast",
            max_tokens=30,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        logger.info(f"Claude streaming response: {full_response}")

        assert len(chunks) > 0
        assert len(full_response) > 0

    def test_claude_properties(self, claude_provider):
        """Test Claude provider properties."""
        if claude_provider is None:
            return

        assert claude_provider.supports_web_search is True
        assert claude_provider.supports_x_search is False


class TestGrokProvider:
    """Tests for Grok provider with real API calls."""

    @pytest.fixture
    def grok_provider(self) -> Optional["GrokProvider"]:
        """Create a Grok provider if API key is available."""
        from app.agent.providers.factory import (
            is_provider_available,
            create_provider,
        )
        from app.agent.providers import ModelProvider

        if not is_provider_available(ModelProvider.GROK):
            pytest.skip("Grok API key not configured")
            return None

        return create_provider(ModelProvider.GROK)

    @pytest.mark.asyncio
    async def test_grok_simple_message(self, grok_provider):
        """Test Grok can respond to a simple message."""
        from app.agent.providers import AIMessage

        if grok_provider is None:
            return

        messages = [
            AIMessage(role="user", content="Say hello in 5 words")
        ]

        response = await grok_provider.create_message(
            messages=messages,
            model_type="fast",  # Use fast model to minimize cost
            max_tokens=50,  # Limit tokens
        )

        logger.info(f"Grok response: {response.content}")
        logger.info(f"Grok usage: {response.usage}")

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0
        assert response.usage is not None

        # Cleanup
        await grok_provider.close()

    @pytest.mark.asyncio
    async def test_grok_with_system_prompt(self, grok_provider):
        """Test Grok respects system prompts."""
        from app.agent.providers import AIMessage

        if grok_provider is None:
            return

        messages = [
            AIMessage(role="user", content="What are you?")
        ]

        response = await grok_provider.create_message(
            messages=messages,
            system="You are a friendly robot. Respond in 10 words or less.",
            model_type="fast",
            max_tokens=50,
        )

        logger.info(f"Grok system prompt response: {response.content}")

        assert response is not None
        assert response.content is not None
        assert len(response.content) > 0

        # Cleanup
        await grok_provider.close()

    @pytest.mark.asyncio
    async def test_grok_streaming(self, grok_provider):
        """Test Grok streaming works."""
        from app.agent.providers import AIMessage

        if grok_provider is None:
            return

        messages = [
            AIMessage(role="user", content="Count 1 to 3")
        ]

        chunks = []
        async for chunk in grok_provider.create_message_stream(
            messages=messages,
            model_type="fast",
            max_tokens=30,
        ):
            chunks.append(chunk)

        full_response = "".join(chunks)
        logger.info(f"Grok streaming response: {full_response}")

        assert len(chunks) > 0
        assert len(full_response) > 0

        # Cleanup
        await grok_provider.close()

    def test_grok_properties(self, grok_provider):
        """Test Grok provider properties."""
        if grok_provider is None:
            return

        assert grok_provider.supports_web_search is True
        assert grok_provider.supports_x_search is True

    @pytest.mark.asyncio
    async def test_grok_context_manager(self):
        """Test Grok provider works as context manager."""
        from app.agent.providers.factory import (
            is_provider_available,
            create_provider,
        )
        from app.agent.providers import ModelProvider, AIMessage

        if not is_provider_available(ModelProvider.GROK):
            pytest.skip("Grok API key not configured")
            return

        async with create_provider(ModelProvider.GROK) as provider:
            messages = [
                AIMessage(role="user", content="Hi")
            ]
            response = await provider.create_message(
                messages=messages,
                model_type="fast",
                max_tokens=20,
            )
            assert response.content is not None


class TestProviderComparison:
    """Tests comparing both providers."""

    @pytest.mark.asyncio
    async def test_both_providers_same_prompt(self):
        """Test that both providers can respond to the same prompt."""
        from app.agent.providers.factory import (
            is_provider_available,
            create_provider,
        )
        from app.agent.providers import ModelProvider, AIMessage

        prompt = "What is 2+2? Answer with just the number."
        messages = [AIMessage(role="user", content=prompt)]

        responses = {}

        # Test Claude
        if is_provider_available(ModelProvider.CLAUDE):
            provider = create_provider(ModelProvider.CLAUDE)
            response = await provider.create_message(
                messages=messages,
                model_type="fast",
                max_tokens=20,
            )
            responses["claude"] = response.content
            logger.info(f"Claude answer: {response.content}")

        # Test Grok
        if is_provider_available(ModelProvider.GROK):
            provider = create_provider(ModelProvider.GROK)
            response = await provider.create_message(
                messages=messages,
                model_type="fast",
                max_tokens=20,
            )
            responses["grok"] = response.content
            logger.info(f"Grok answer: {response.content}")
            await provider.close()

        # At least one provider should have responded
        assert len(responses) > 0, "No AI providers are configured"

        # Both should have "4" somewhere in their response
        for name, content in responses.items():
            assert "4" in content, f"{name} did not return correct answer: {content}"


if __name__ == "__main__":
    # Run with: python -m pytest tests/test_ai_providers.py -v -s
    pytest.main([__file__, "-v", "-s"])
