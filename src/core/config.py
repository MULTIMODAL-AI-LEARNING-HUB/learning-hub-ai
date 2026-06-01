"""Settings for AI service."""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GROQ_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333

    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    GROQ_MODEL: str = "llama3-8b-8192"
    GEMINI_MODEL: str = "gemini-1.5-pro"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
