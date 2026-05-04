import os
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )

    openai_api_key: str = os.getenv("OPENAI_API_KEY")
    model_dev: str = "gpt-4o-mini"
    model_eval: str = "gpt-4.1"
    pipeline_timeout_s: int = 10
    session_db_path: str = "./valura_sessions.db"
    environment: str = "development"

    @property
    def model(self) -> str:
        if self.environment == "production":
            return self.model_eval
        return self.model_dev


@lru_cache
def get_settings() -> Settings:
    return Settings()