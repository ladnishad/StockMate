"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Alpaca API Configuration
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpaca_base_url: str = "https://paper-api.alpaca.markets"

    # AlgoTrader Plus Data Feed Configuration
    # "sip" = Securities Information Processor (all US exchanges, paid tier)
    # "iex" = Investors Exchange only (free tier)
    alpaca_data_feed: str = "sip"

    # Claude Agent SDK Configuration
    claude_api_key: str = ""
    claude_model_planning: str = "claude-sonnet-4-20250514"  # Smarter model for plan generation
    claude_model_fast: str = "claude-3-5-haiku-20241022"  # Fast model for chat/evaluation

    # Grok (xAI) Configuration
    # Get your API key from https://console.x.ai/team/default/api-keys
    grok_api_key: str = ""  # XAI_API_KEY environment variable
    grok_model_planning: str = "grok-4"  # Flagship model for plan generation
    grok_model_fast: str = "grok-4-1-fast"  # Fast model for chat/evaluation

    # Finnhub Configuration (Fundamental Data + News)
    # Get your API key from https://finnhub.io/dashboard
    # Free tier includes: fundamentals, earnings calendar, company news
    finnhub_api_key: str = ""  # FINNHUB_API_KEY environment variable

    # Supabase Configuration (Authentication + Database)
    supabase_url: str = ""  # https://xxxxx.supabase.co
    supabase_anon_key: str = ""  # Public anon key (for client-side)
    supabase_service_key: str = ""  # Service role key (backend only, full access)
    supabase_jwt_secret: str = ""  # JWT secret for token verification

    # Database Configuration
    # PostgreSQL connection string for production (from Supabase or Render)
    # Format: postgresql://user:password@host:5432/database
    database_url: Optional[str] = None

    # APNs (Apple Push Notification Service) Configuration
    apns_key_id: str = ""
    apns_team_id: str = ""
    apns_key_path: str = ""  # Path to .p8 key file
    apns_bundle_id: str = ""
    apns_use_sandbox: bool = True  # True for development, False for production

    # Agent Configuration
    agent_analysis_interval: int = 60  # Seconds between analysis cycles
    alert_cooldown_minutes: int = 15  # Prevent duplicate alerts
    max_concurrent_agents: int = 20  # Maximum stocks to monitor simultaneously

    # Web Search Configuration (Claude's built-in web search tool)
    web_search_enabled: bool = True  # Enable web/social search for news and sentiment
    web_search_max_uses: int = 5  # Max searches per plan generation (controls cost)

    # Rate Limiting Configuration
    rate_limit_per_minute: int = 60  # General API rate limit
    rate_limit_ai_per_minute: int = 10  # AI endpoints (expensive operations)

    # Application Configuration
    app_env: str = "development"  # development, staging, production
    log_level: str = "INFO"

    # CORS Configuration (comma-separated list of allowed origins for production)
    cors_origins: str = ""  # e.g., "https://yourdomain.com,capacitor://localhost"

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == "production"

    @property
    def use_postgres(self) -> bool:
        """Check if PostgreSQL should be used (production with database_url set)."""
        return bool(self.database_url)

    @property
    def cors_origin_list(self) -> list[str]:
        """Get list of CORS origins."""
        if self.cors_origins:
            return [origin.strip() for origin in self.cors_origins.split(",")]
        return []

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "extra": "ignore",  # Ignore extra env vars not defined above
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
