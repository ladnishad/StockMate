"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from functools import lru_cache


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

    # Application Configuration
    app_env: str = "development"
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
