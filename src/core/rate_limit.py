"""Redis-backed rate limiting for expensive AI endpoints."""

import logging
import time
from typing import Any

import redis.asyncio as redis
from fastapi import HTTPException, Request

from src.core.config import settings

logger = logging.getLogger("ai.rate_limit")
_redis_client: redis.Redis | None = None


def _client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        redis_url = settings.REDIS_URL or f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/0"
        kwargs: dict[str, Any] = {"decode_responses": True, "max_connections": 20}
        if redis_url.startswith("rediss://"):
            kwargs["ssl_cert_reqs"] = None
        _redis_client = redis.from_url(redis_url, **kwargs)
    return _redis_client


async def enforce_chat_rate_limit(request: Request) -> None:
    """Allow a bounded number of chat requests per client IP with graceful fallback."""
    client_ip = request.client.host if request.client else "unknown"
    window = int(time.time() // 60)
    key = f"rate-limit:ai-chat:{client_ip}:{window}"
    try:
        pipe = _client().pipeline(transaction=True)
        pipe.incr(key)
        pipe.expire(key, 61)
        count, _ = await pipe.execute()
        if int(count) > settings.AI_CHAT_RATE_LIMIT_PER_MINUTE:
            raise HTTPException(status_code=429, detail="Too many AI requests. Please try again later.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("AI rate limiter error (failing open): %s", exc)

