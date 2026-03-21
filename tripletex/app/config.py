from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    model_name: str = "gpt-5.4"
    log_level: str = "INFO"
    max_agent_iterations: int = 15
    soft_timeout_seconds: int = 270
    azure_search_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("AZURE_SEARCH_ENDPOINT", "AISEARCH_URL"),
    )
    azure_search_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("AZURE_SEARCH_API_KEY", "AISEARCH_API_KEY"),
    )
    azure_search_index_name: str = "tripletex-endpoints"
    azure_search_api_version: str = "2024-07-01"
    azure_search_semantic_configuration: str | None = "default"
    endpoint_search_results: int = 5

    model_config = SettingsConfigDict(
        env_file=(PROJECT_DIR / "env", PROJECT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def azure_search_configured(self) -> bool:
        return bool(
            (self.azure_search_endpoint or "").strip()
            and (self.azure_search_api_key or "").strip()
        )


@lru_cache
def get_settings() -> Settings:
    settings_class: Any = Settings
    return settings_class()
