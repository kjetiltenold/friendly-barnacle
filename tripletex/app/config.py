from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    anthropic_api_key: str
    model_name: str = "claude-sonnet-4-20250514"
    log_level: str = "INFO"
    max_agent_iterations: int = 15
    soft_timeout_seconds: int = 270

    model_config = {"env_file": ".env"}


@lru_cache
def get_settings() -> Settings:
    return Settings()
