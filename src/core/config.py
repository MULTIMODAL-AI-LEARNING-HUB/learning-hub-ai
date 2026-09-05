"""Settings for AI service."""

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEBUG: bool = False

    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    INTERNAL_API_KEY: str = ""

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    GROQ_MODEL: str = "llama3-8b-8192"
    GEMINI_MODEL: str = "gemini-2.0-flash-lite"

    @model_validator(mode="after")
    def validate_secrets(self) -> "Settings":
        weak_values = {"", "secret", "changeme", "your_internal_api_key", "your_internal_key"}
        if not self.INTERNAL_API_KEY or self.INTERNAL_API_KEY.lower() in weak_values or len(self.INTERNAL_API_KEY) < 16:
            if not self.DEBUG:
                raise ValueError("INTERNAL_API_KEY must be a secure, non-default string (min 16 chars) in production")
        for name, value in (("GROQ_API_KEY", self.GROQ_API_KEY), ("GEMINI_API_KEY", self.GEMINI_API_KEY)):
            if value and (value.lower() in weak_values or len(value) < 16):
                raise ValueError(f"{name} is too short or uses a weak placeholder")
        return self


settings = Settings()
