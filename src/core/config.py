"""Settings for AI service."""

from pydantic import model_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DEBUG: bool = False

    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    INTERNAL_API_KEY: str = ""

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    GROQ_MODEL: str = "llama3-8b-8192"
    GEMINI_MODEL: str = "gemini-1.5-pro"

    @model_validator(mode="after")
    def validate_secrets(self) -> "Settings":
        if not self.DEBUG:
            if not self.INTERNAL_API_KEY or self.INTERNAL_API_KEY in {"", "your_internal_api_key"}:
                raise ValueError("INTERNAL_API_KEY must be a secure, non-default string in production")
        return self

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
