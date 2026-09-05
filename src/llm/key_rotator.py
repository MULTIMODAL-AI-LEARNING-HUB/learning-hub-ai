"""AI API Key Rotator and Failover Management for Multi-Providers (Gemini, Groq, OpenAI, etc.)."""

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("ai.key_rotator")


class KeyItem:
    def __init__(self, key: str, key_name: str = "Default", provider: str = "gemini", is_active: bool = True):
        self.key = key.strip()
        self.key_name = key_name
        self.provider = provider.lower().strip()
        self.is_active = is_active
        self.usage_count = 0
        self.last_used_at: Optional[float] = None
        self.cooldown_until: float = 0.0

    @property
    def masked_key(self) -> str:
        k = self.key
        if len(k) <= 10:
            return "***"
        return f"{k[:6]}...{k[-4:]}"

    @property
    def is_available(self) -> bool:
        return self.is_active and time.time() >= self.cooldown_until


class AIKeyRotator:
    _instance: Optional["AIKeyRotator"] = None

    def __init__(self):
        self.keys: list[KeyItem] = []
        self._current_indices: dict[str, int] = {}
        self._load_from_settings()

    @classmethod
    def get_instance(cls) -> "AIKeyRotator":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_from_settings(self):
        from src.core.config import settings
        # Load default Gemini keys
        raw_gemini = settings.GEMINI_API_KEY
        if raw_gemini:
            keys = [k.strip() for k in raw_gemini.replace(",", " ").split() if k.strip()]
            for i, k in enumerate(keys):
                self.add_key(k, key_name=f"Config Gemini {i+1}", provider="gemini")

        # Load default Groq keys
        raw_groq = settings.GROQ_API_KEY
        if raw_groq:
            keys = [k.strip() for k in raw_groq.replace(",", " ").split() if k.strip()]
            for i, k in enumerate(keys):
                self.add_key(k, key_name=f"Config Groq {i+1}", provider="groq")

        logger.info("Initialized AIKeyRotator with %d keys from settings", len(self.keys))

    def add_key(self, key_str: str, key_name: str = "Key", provider: str = "gemini") -> KeyItem:
        key_str = key_str.strip()
        provider = provider.lower().strip()
        for item in self.keys:
            if item.key == key_str:
                item.is_active = True
                item.provider = provider
                return item
        item = KeyItem(key=key_str, key_name=key_name, provider=provider)
        self.keys.append(item)
        return item

    def sync_keys(self, keys_data: list[dict[str, Any]]):
        """Synchronize the in-memory pool from API Gateway."""
        new_keys: list[KeyItem] = []
        for k in keys_data:
            api_key = k.get("api_key", "").strip()
            if not api_key:
                continue
            item = KeyItem(
                key=api_key,
                key_name=k.get("key_name", "Managed Key"),
                provider=k.get("provider", "gemini"),
                is_active=k.get("is_active", True),
            )
            item.usage_count = k.get("usage_count", 0)
            new_keys.append(item)
        
        if new_keys:
            self.keys = new_keys
            self._current_indices.clear()
            logger.info("Synced %d keys into AIKeyRotator", len(self.keys))

    def get_next_key(self, provider: str = "gemini") -> str:
        """Get the next available active API Key for the given provider via Round-Robin with cooldown handling."""
        provider = provider.lower().strip()
        now = time.time()
        provider_keys = [k for k in self.keys if k.provider == provider]
        available = [k for k in provider_keys if k.is_available]

        if not available:
            # If all active keys for this provider are on cooldown, pick the one whose cooldown expires soonest
            active_keys = [k for k in provider_keys if k.is_active]
            if active_keys:
                logger.warning("All active %s keys in cooldown. Re-activating the soonest expiring key.", provider)
                soonest = min(active_keys, key=lambda k: k.cooldown_until)
                soonest.cooldown_until = 0.0
                available = [soonest]
            else:
                from src.core.config import settings
                if provider == "gemini" and settings.GEMINI_API_KEY:
                    return settings.GEMINI_API_KEY.split(",")[0].strip()
                if provider == "groq" and settings.GROQ_API_KEY:
                    return settings.GROQ_API_KEY.split(",")[0].strip()
                raise RuntimeError(f"No active {provider.upper()} API keys configured in pool.")

        # Round Robin selection for this provider
        idx = self._current_indices.get(provider, 0)
        idx = (idx + 1) % len(available)
        self._current_indices[provider] = idx

        selected = available[idx]
        selected.usage_count += 1
        selected.last_used_at = now
        return selected.key

    def report_rate_limit(self, key_str: str, cooldown_seconds: float = 60.0):
        """Mark a key as rate-limited and place it in cooldown."""
        for item in self.keys:
            if item.key == key_str.strip():
                item.cooldown_until = time.time() + cooldown_seconds
                logger.warning("[%s] Key %s placed in cooldown for %.0fs due to rate limit/error", item.provider.upper(), item.masked_key, cooldown_seconds)
                break

    def get_status(self) -> dict[str, Any]:
        now = time.time()
        providers = sorted(list({k.provider for k in self.keys}))
        provider_stats = {}
        for p in providers:
            p_keys = [k for k in self.keys if k.provider == p]
            provider_stats[p] = {
                "total": len(p_keys),
                "active": len([k for k in p_keys if k.is_active]),
                "available": len([k for k in p_keys if k.is_available]),
            }

        return {
            "total_keys": len(self.keys),
            "active_keys": len([k for k in self.keys if k.is_active]),
            "available_keys": len([k for k in self.keys if k.is_available]),
            "by_provider": provider_stats,
            "keys": [
                {
                    "name": k.key_name,
                    "masked_key": k.masked_key,
                    "provider": k.provider,
                    "is_active": k.is_active,
                    "is_cooling_down": now < k.cooldown_until,
                    "cooldown_remaining_sec": max(0, int(k.cooldown_until - now)),
                    "usage_count": k.usage_count,
                }
                for k in self.keys
            ]
        }


# Alias for backward compatibility
GeminiKeyRotator = AIKeyRotator
