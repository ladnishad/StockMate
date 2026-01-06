"""AI Provider factory for creating and managing provider instances.

This module provides factory functions for creating AI providers based on
configuration and user preferences, with singleton caching for efficiency.
"""

import logging
from typing import Dict, Optional

from app.config import get_settings
from app.agent.providers import (
    ModelProvider,
    ProviderConfig,
    AIProvider,
)
from app.agent.providers.claude_provider import ClaudeProvider
from app.agent.providers.grok_provider import GrokProvider

logger = logging.getLogger(__name__)

# Singleton cache for provider instances
_provider_cache: Dict[ModelProvider, AIProvider] = {}


def get_provider_config(provider: ModelProvider) -> ProviderConfig:
    """Get the configuration for a specific provider.

    Reads API keys and model names from the application settings.

    Args:
        provider: The provider to get configuration for

    Returns:
        ProviderConfig with all settings populated

    Raises:
        ValueError: If the provider's API key is not configured
    """
    settings = get_settings()

    if provider == ModelProvider.CLAUDE:
        if not settings.claude_api_key:
            raise ValueError(
                "Claude API key not configured. "
                "Set CLAUDE_API_KEY in your environment or .env file."
            )
        return ProviderConfig(
            provider=provider,
            planning_model=settings.claude_model_planning,
            fast_model=settings.claude_model_fast,
            api_key=settings.claude_api_key,
            base_url=None,  # Use default Anthropic URL
        )

    elif provider == ModelProvider.GROK:
        if not settings.grok_api_key:
            raise ValueError(
                "Grok API key not configured. "
                "Set GROK_API_KEY (or XAI_API_KEY) in your environment or .env file."
            )
        return ProviderConfig(
            provider=provider,
            planning_model=settings.grok_model_planning,
            fast_model=settings.grok_model_fast,
            api_key=settings.grok_api_key,
            base_url="https://api.x.ai/v1",
        )

    else:
        raise ValueError(f"Unknown provider: {provider}")


def create_provider(provider: ModelProvider) -> AIProvider:
    """Create a new provider instance (not cached).

    Use this when you need a fresh provider instance, for example
    when testing or when you want to ensure the latest configuration.

    Args:
        provider: The provider type to create

    Returns:
        A new AIProvider instance

    Raises:
        ValueError: If the provider is not configured or unknown
    """
    config = get_provider_config(provider)

    if provider == ModelProvider.CLAUDE:
        return ClaudeProvider(config)
    elif provider == ModelProvider.GROK:
        return GrokProvider(config)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_provider(provider: ModelProvider) -> AIProvider:
    """Get a cached provider instance (singleton).

    This returns a shared provider instance for efficiency.
    The instance is created on first access and reused thereafter.

    Args:
        provider: The provider type to get

    Returns:
        A cached AIProvider instance

    Raises:
        ValueError: If the provider is not configured or unknown
    """
    global _provider_cache

    if provider not in _provider_cache:
        logger.info(f"Creating cached {provider.value} provider instance")
        _provider_cache[provider] = create_provider(provider)

    return _provider_cache[provider]


def clear_provider_cache() -> None:
    """Clear the provider cache.

    Call this when configuration changes or during testing.
    """
    global _provider_cache
    _provider_cache.clear()
    logger.info("Provider cache cleared")


async def get_user_provider(user_id: str) -> AIProvider:
    """Get the AI provider for a specific user based on their preferences and subscription tier.

    This function looks up the user's preferred AI provider from their
    settings, validates it against their subscription tier, and returns
    the appropriate provider instance.

    Args:
        user_id: The user's ID

    Returns:
        The user's preferred AIProvider instance (tier-validated)
    """
    from app.storage.user_settings_store import get_user_settings_store
    from app.services.subscription_service import get_subscription_service

    subscription_service = get_subscription_service()

    try:
        # Get user's subscription tier to determine available providers
        has_multi_model = await subscription_service.has_multi_model_access(user_id)
        available_provider_names = await subscription_service.get_available_providers(user_id)

        settings_store = get_user_settings_store()
        user_settings = await settings_store.get_settings(user_id)

        if user_settings and user_settings.model_provider:
            provider_name = user_settings.model_provider

            # Check if user's selected provider is allowed by their subscription tier
            if provider_name not in available_provider_names:
                logger.warning(
                    f"User {user_id} selected {provider_name} but their subscription "
                    f"tier only allows: {available_provider_names}. Falling back to Claude."
                )
                # Force Claude for users without multi-model access trying to use Grok
                provider_name = "claude"

            try:
                provider = ModelProvider(provider_name)
                # Check if the provider is actually configured (has API key)
                if is_provider_available(provider):
                    logger.debug(f"Using user-selected {provider.value} provider for user {user_id}")
                    return get_provider(provider)
                else:
                    logger.warning(
                        f"User {user_id} selected {provider_name} but it's not configured. "
                        "Falling back to default provider."
                    )
            except ValueError:
                logger.warning(f"Unknown provider '{provider_name}' for user {user_id}")
    except Exception as e:
        logger.warning(f"Error fetching user settings/subscription for {user_id}: {e}")

    # Fall back to the tier-aware default provider
    try:
        default_name = await subscription_service.get_default_provider(user_id)
        default_provider = ModelProvider(default_name)
        if is_provider_available(default_provider):
            logger.debug(f"Using tier-default {default_provider.value} provider for user {user_id}")
            return get_provider(default_provider)
    except Exception as e:
        logger.warning(f"Error getting tier-default provider for {user_id}: {e}")

    # Ultimate fallback: Claude (always safe for any tier)
    logger.debug(f"Using ultimate fallback (Claude) provider for user {user_id}")
    return get_provider(ModelProvider.CLAUDE)


def get_default_provider() -> ModelProvider:
    """Get the default AI provider based on configuration.

    The default is Grok if configured, otherwise Claude.

    Returns:
        The default ModelProvider
    """
    settings = get_settings()

    # Prefer Grok if configured (has X/Twitter search for real-time sentiment)
    if settings.grok_api_key:
        return ModelProvider.GROK

    # Fall back to Claude
    if settings.claude_api_key:
        return ModelProvider.CLAUDE

    # Neither configured - will raise error when actually used
    logger.warning("No AI provider API key configured")
    return ModelProvider.GROK


def get_available_providers() -> list[ModelProvider]:
    """Get a list of providers that are properly configured.

    Returns:
        List of ModelProvider values that have API keys configured
    """
    settings = get_settings()
    available = []

    if settings.claude_api_key:
        available.append(ModelProvider.CLAUDE)

    if settings.grok_api_key:
        available.append(ModelProvider.GROK)

    return available


def is_provider_available(provider: ModelProvider) -> bool:
    """Check if a provider is configured and available.

    Args:
        provider: The provider to check

    Returns:
        True if the provider's API key is configured
    """
    settings = get_settings()

    if provider == ModelProvider.CLAUDE:
        return bool(settings.claude_api_key)
    elif provider == ModelProvider.GROK:
        return bool(settings.grok_api_key)

    return False
