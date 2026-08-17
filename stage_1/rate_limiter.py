"""
stage_1/rate_limiter.py

Centralized Round-Robin Multi-Key Async Rate Limiter for LLM API calls.
Distributes calls evenly across multiple Google/Gemini API keys (API_KEY_1, API_KEY_2, API_KEY_3, API_KEY_4, API_KEY_5)
to achieve maximum concurrent throughput while strictly enforcing per-key RPM limits (e.g., 3 RPM per key).
Supports structured Pydantic output schemas.
"""

import asyncio
import os
import time
from collections import deque
from typing import List, Optional, Dict, Tuple, Any

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage

from stage_1.model import get_all_google_api_keys, init_model

load_dotenv()


class KeyBucket:
    """Tracking bucket for a single API key's sliding-window rate limit."""

    def __init__(self, key_name: str, api_key: str, max_rpm: int = 3, model_name: str = "gemma-4-31b-it"):
        self.key_name = key_name
        self.api_key = api_key
        self.max_rpm = max_rpm
        self.window_secs = 60.0
        self.timestamps: deque = deque()
        self.cooldown_until: float = 0.0
        
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            self._model = init_model(provider="gemini", model_name=self.model_name, api_key=self.api_key)
        return self._model

    def evict_old(self, now: float) -> None:
        while self.timestamps and (now - self.timestamps[0]) >= self.window_secs:
            self.timestamps.popleft()

    def is_available(self, now: float) -> bool:
        if now < self.cooldown_until:
            return False
        self.evict_old(now)
        return len(self.timestamps) < self.max_rpm

    def seconds_until_available(self, now: float) -> float:
        if now < self.cooldown_until:
            return max(self.cooldown_until - now, 0.1)
        self.evict_old(now)
        if len(self.timestamps) < self.max_rpm:
            return 0.0
        oldest = self.timestamps[0]
        return max(self.window_secs - (now - oldest) + 0.05, 0.1)

    def record_call(self, now: float) -> None:
        self.timestamps.append(now)

    def mark_cooldown(self, seconds: float = 5.0) -> None:
        self.cooldown_until = time.monotonic() + seconds
        print(f"  [MultiKeyLimiter] 429 Quota Exhausted on {self.key_name}. Placing key on {seconds:.1f}s cooldown...")


class MultiKeyAsyncRateLimiter:
    """
    Manages a pool of KeyBuckets in round-robin order.
    Directs async LLM calls across all configured API keys smoothly.
    Supports structured Pydantic schemas via the `schema` argument.
    """

    def __init__(self, per_key_rpm: Optional[int] = None):
        if per_key_rpm is None:
            per_key_rpm = int(os.getenv("RATE_LIMIT_PER_KEY_RPM", "3"))

        self.per_key_rpm = per_key_rpm
        model_name = os.getenv("GEMINI_MODEL", "gemma-4-31b-it")

        api_keys_list = get_all_google_api_keys()
        self.buckets: List[KeyBucket] = [
            KeyBucket(key_name=f"API_KEY_{idx+1}", api_key=key, max_rpm=self.per_key_rpm, model_name=model_name)
            for idx, key in enumerate(api_keys_list)
        ]

        if not self.buckets:
            default_key = os.getenv("GOOGLE_API_KEY", "")
            self.buckets = [KeyBucket(key_name="API_KEY_1", api_key=default_key, max_rpm=self.per_key_rpm, model_name=model_name)]

        self.total_capacity_rpm = len(self.buckets) * self.per_key_rpm
        self._lock: Optional[asyncio.Lock] = None
        self._rr_index: int = 0
        self.active_requests: int = 0
        self.total_completed_requests: int = 0

        print(f"[MultiKeyAsyncRateLimiter] Active API Keys: {len(self.buckets)} | Per-Key Capacity: {self.per_key_rpm} RPM | Total Throughput: {self.total_capacity_rpm} RPM (Round-Robin Pool)")

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def acquire_key_bucket(self) -> KeyBucket:
        """
        Selects the next available key bucket using Round-Robin dispatching.
        If all keys are currently at capacity, waits until the earliest slot opens.
        """
        lock = self._get_lock()
        printed = False

        while True:
            async with lock:
                now = time.monotonic()
                num_keys = len(self.buckets)

                for i in range(num_keys):
                    idx = (self._rr_index + i) % num_keys
                    bucket = self.buckets[idx]
                    if bucket.is_available(now):
                        bucket.record_call(now)
                        self._rr_index = (idx + 1) % num_keys
                        return bucket

                wait_times = [b.seconds_until_available(now) for b in self.buckets]
                min_wait = min(wait_times) if wait_times else 1.0

            if not printed and min_wait > 1.0:
                print(f"  [MultiKeyLimiter] All {len(self.buckets)} API keys reached {self.per_key_rpm} RPM capacity. Throttling queue for {min_wait:.1f}s ...")
                printed = True

            await asyncio.sleep(max(min_wait, 0.1))

    async def call(
        self,
        model_or_override,
        messages: List[BaseMessage],
        timeout: Optional[float] = None,
        schema: Optional[Any] = None,
    ) -> Any:
        """
        Executes a rate-limited async LLM call using Round-Robin API key dispatching.
        Supports structured Pydantic output when `schema` parameter is provided.
        """
        if timeout is None:
            timeout = float(os.getenv("API_TIMEOUT", "220.0"))

        max_retries = len(self.buckets) * 2
        last_exception = None

        for attempt in range(max_retries):
            bucket = await self.acquire_key_bucket()
            self.active_requests += 1
            print(f"  [Live Tracker] Active Requests: {self.active_requests} | Completed API Calls: {self.total_completed_requests} | Using {bucket.key_name}")
            try:
                if schema is not None:
                    target_model = bucket.model.with_structured_output(schema)
                else:
                    target_model = bucket.model

                response = await asyncio.wait_for(target_model.ainvoke(messages), timeout=timeout)
                self.total_completed_requests += 1
                self.active_requests = max(self.active_requests - 1, 0)
                return response
            except Exception as exc:
                self.active_requests = max(self.active_requests - 1, 0)
                err_msg = str(exc).lower()
                if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                    bucket.mark_cooldown(seconds=float(os.getenv("RETRY_WAIT_SECONDS", "5")))
                    last_exception = exc
                    await asyncio.sleep(0.5)
                    continue
                elif isinstance(exc, asyncio.TimeoutError):
                    print(f"  [MultiKeyLimiter] Call on {bucket.key_name} timed out after {timeout}s! Retrying on next key...")
                    last_exception = exc
                    await asyncio.sleep(0.5)
                    continue
                else:
                    raise exc

        raise last_exception or RuntimeError("All API keys in Round-Robin pool failed.")


# Global Singleton Rate Limiter instance
_GLOBAL_MULTI_KEY_LIMITER: Optional[MultiKeyAsyncRateLimiter] = None

def make_async_rate_limiter(per_key_rpm: Optional[int] = None) -> MultiKeyAsyncRateLimiter:
    """Returns the shared MultiKeyAsyncRateLimiter instance."""
    global _GLOBAL_MULTI_KEY_LIMITER
    if _GLOBAL_MULTI_KEY_LIMITER is None:
        _GLOBAL_MULTI_KEY_LIMITER = MultiKeyAsyncRateLimiter(per_key_rpm=per_key_rpm)
    return _GLOBAL_MULTI_KEY_LIMITER

# Alias for backwards compatibility
AsyncRateLimiter = MultiKeyAsyncRateLimiter
