"""LLM API client for OpenRouter with adaptive rate limiting and robust retries.

Key design decisions:
- OpenAI client's built-in retries are DISABLED (max_retries=0) because they
  bypass our rate limiter and cause 429 stampedes.
- `call_llm` is a RAW single-attempt call with no retries.
- `call_llm_rate_limited` wraps call_llm with both the rate limiter AND retries.
  Crucially, each retry attempt RE-ACQUIRES a rate limiter slot, so after a 429
  triggers a cooldown, the retry waits for the cooldown to clear before trying
  again. This prevents the "5 concurrent slots all retrying and hammering the
  API" problem that occurs when retries are inside the rate limiter.
- On 429 responses, we parse the rate limit headers (X-RateLimit-Reset,
  Retry-After) and dynamically slow the rate limiter.
- Token-bucket rate limiter enforces both req/s and max concurrent requests.
"""

import asyncio
import json
import os
import time
import logging
import random
import re

from openai import (
    APIError,
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    RateLimitError,
    AsyncOpenAI,
)
from openai.types.chat import ChatCompletionUserMessageParam
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Configure OpenRouter with NO built-in retries — we handle retries ourselves
# via the rate limiter so that retries respect rate limits.
api_key = os.getenv("OPENROUTER_API_KEY")
client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=api_key,
    max_retries=0,  # CRITICAL: disable built-in retries
    timeout=45.0,  # 45s hard timeout per request; triggers APITimeoutError (retryable)
)


# ---------------------------------------------------------------------------
# Token-bucket rate limiter with adaptive rate
# ---------------------------------------------------------------------------


class RateLimiter:
    """Async token-bucket rate limiter with adaptive backoff.

    Enforces a sustained requests-per-second rate AND a hard cap on concurrent
    in-flight requests. When 429 responses are detected, the rate automatically
    decreases and a cooldown period is applied.

    Usage:
        limiter = RateLimiter(requests_per_second=0.25, max_concurrent=5)
        async with limiter:
            await call_llm(...)
    """

    def __init__(
        self,
        requests_per_second: float = 0.25,
        max_concurrent: int = 5,
        burst: int | None = None,
    ):
        """
        Args:
            requests_per_second: Sustained rate limit (tokens refilled per second).
            max_concurrent: Hard cap on simultaneous in-flight requests.
            burst: Max tokens in bucket. Defaults to 1 to prevent initial stampede.
        """
        self.rate = requests_per_second
        self.max_concurrent = max_concurrent
        self.burst = burst if burst is not None else 1

        self._tokens = float(self.burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)

        # Cooldown: when a 429 is detected, block new requests until this time
        self._cooldown_until = 0.0

        # Stats
        self._total_acquired = 0
        self._total_429s = 0

    async def _refill(self):
        """Add tokens based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
        self._last_refill = now

    async def acquire(self, max_wait_s: float = 90.0):
        """Wait until both a token and a concurrency slot are available.

        max_wait_s caps the total time we'll loop here. Without this, a stuck
        cooldown that keeps getting re-upped will pin every concurrent task
        in the cell forever (saw 90+ min hangs on g3-27b OpenRouter).
        """
        await self._semaphore.acquire()
        start = time.monotonic()

        while True:
            async with self._lock:
                # Respect cooldown period from 429s
                now = time.monotonic()
                if now < self._cooldown_until:
                    wait_time = self._cooldown_until - now
                else:
                    wait_time = 0

                if wait_time <= 0:
                    await self._refill()
                    if self._tokens >= 1.0:
                        self._tokens -= 1.0
                        self._total_acquired += 1
                        return

            # Bail if we'd blow past max_wait_s.
            elapsed = time.monotonic() - start
            if elapsed >= max_wait_s:
                self._semaphore.release()
                raise asyncio.TimeoutError(
                    f"RateLimiter.acquire timed out after {elapsed:.1f}s "
                    f"(cooldown_until={self._cooldown_until:.1f}, rate={self.rate:.2f})"
                )

            # Wait for either cooldown or next token
            sleep_time = (
                max(wait_time, 1.0 / self.rate) if wait_time > 0 else (1.0 / self.rate)
            )
            sleep_time = min(sleep_time, max_wait_s - elapsed)
            await asyncio.sleep(sleep_time)

    def release(self):
        """Release the concurrency slot."""
        self._semaphore.release()

    def notify_429(self, retry_after_seconds: float | None = None):
        """Called when a 429 response is received. Triggers cooldown.

        Args:
            retry_after_seconds: Seconds to wait before retrying, from the
                429 response headers. If None, uses a default cooldown.
        """
        self._total_429s += 1
        # Cap cooldown at 60s — without this an upstream proxy (Cloudflare/
        # OpenRouter) can return absurd Retry-After values that pin every
        # task in the cell on a single async sleep. Saw 90+ min hangs.
        raw = retry_after_seconds or 5.0
        cooldown = min(60.0, max(0.0, raw))
        if raw > cooldown:
            logger.warning("Capping retry-after %.1fs → %.1fs", raw, cooldown)

        # Set a cooldown period so no new requests go out
        now = time.monotonic()
        new_cooldown = now + cooldown
        self._cooldown_until = max(self._cooldown_until, new_cooldown)

        # Reduce rate by 30% on each 429 (floor at 0.05 req/s = 3 RPM)
        old_rate = self.rate
        self.rate = max(0.05, self.rate * 0.7)
        if old_rate != self.rate:
            logger.info(
                "Rate limiter: 429 detected, reducing rate %.2f → %.2f req/s, "
                "cooldown %.1fs (total 429s: %d)",
                old_rate,
                self.rate,
                cooldown,
                self._total_429s,
            )

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, *exc):
        self.release()


# Module-level default rate limiter
_default_limiter: RateLimiter | None = None


def get_rate_limiter(
    requests_per_second: float = 2.0,
    max_concurrent: int = 20,
) -> RateLimiter:
    """Get or create the module-level rate limiter."""
    global _default_limiter
    if _default_limiter is None:
        _default_limiter = RateLimiter(
            requests_per_second=requests_per_second,
            max_concurrent=max_concurrent,
        )
    return _default_limiter


def reset_rate_limiter(
    requests_per_second: float = 2.0,
    max_concurrent: int = 20,
) -> RateLimiter:
    """Create a fresh rate limiter with new settings."""
    global _default_limiter
    _default_limiter = RateLimiter(
        requests_per_second=requests_per_second,
        max_concurrent=max_concurrent,
    )
    return _default_limiter


# ---------------------------------------------------------------------------
# 429 header parsing
# ---------------------------------------------------------------------------


def _parse_retry_after(error: RateLimitError) -> float | None:
    """Extract wait time from a 429 RateLimitError response.

    Checks (in order):
    1. Retry-After header (seconds)
    2. X-RateLimit-Reset header (epoch milliseconds → seconds to wait)
    3. 'retry after N seconds' in error message body
    """
    # Try to get headers/metadata from the error response
    body = getattr(error, "body", None) or {}
    metadata = {}
    if isinstance(body, dict):
        err = body.get("error", {})
        if isinstance(err, dict):
            metadata = err.get("metadata", {})

    headers = metadata.get("headers", {}) if isinstance(metadata, dict) else {}

    # Check Retry-After header
    retry_after = headers.get("Retry-After") or headers.get("retry-after")
    if retry_after:
        try:
            return float(retry_after)
        except (ValueError, TypeError):
            pass

    # Check X-RateLimit-Reset (epoch ms)
    reset_ms = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
    if reset_ms:
        try:
            reset_time = float(reset_ms) / 1000.0  # ms → s
            wait = reset_time - time.time()
            if 0 < wait < 120:  # sanity check
                return wait
        except (ValueError, TypeError):
            pass

    # Try to parse from error message text
    message = str(error)
    match = re.search(
        r"retry\s+(?:after|in)\s+(\d+(?:\.\d+)?)\s*s", message, re.IGNORECASE
    )
    if match:
        return float(match.group(1))

    return None


# Errors worth retrying (transient). Non-retryable errors (400, 401, 403, 404)
# are raised immediately — no point retrying a content moderation block 15 times.
_RETRYABLE_STATUS_CODES = {408, 500, 502, 503, 504, 520, 524}

_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError)


# ---------------------------------------------------------------------------
# Raw single-attempt LLM call (no retries, no rate limiting)
# ---------------------------------------------------------------------------


async def call_llm(
    prompt: str | list[dict],
    model: str,
    max_completion_tokens: int = None,
    response_format: dict = None,
):
    """Single-attempt call to OpenRouter (no retries, no rate limiting).

    For production use, prefer `call_llm_rate_limited` which adds both
    rate limiting and retries.

    Args:
        prompt: Either a plain user message string, or a fully-built messages
            list (e.g. from sae_auto_interp's build_prompt with system +
            few-shot turns).
        model: Model identifier (e.g. "google/gemma-2-9b-it").
        max_completion_tokens: Optional token limit.
        response_format: Optional structured output spec.
    """
    if isinstance(prompt, list):
        messages = prompt
    else:
        messages = [ChatCompletionUserMessageParam(role="user", content=prompt)]

    kwargs = dict(model=model, messages=messages)
    if max_completion_tokens is not None:
        kwargs["max_completion_tokens"] = max_completion_tokens
    if response_format is not None:
        kwargs["response_format"] = response_format

    response = await client.chat.completions.create(**kwargs)
    # OpenRouter occasionally returns a 200 with no choices (rate-limit /
    # provider-side 5xx leaking through). Treat as a retryable failure so
    # call_llm_rate_limited's retry loop catches it instead of crashing
    # mid-cell with "NoneType is not subscriptable".
    if not getattr(response, "choices", None):
        # Use APIConnectionError (in _RETRYABLE_ERRORS) so the retry loop
        # actually retries instead of crashing the cell after one bad
        # provider response. APIError (the base class) is too broad and
        # not retryable.
        raise APIConnectionError(
            message="OpenRouter returned response with empty choices",
            request=None,
        )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Rate-limited + retrying LLM call
# ---------------------------------------------------------------------------

MAX_RETRIES = 5


async def call_llm_rate_limited(
    prompt: str | list[dict],
    model: str,
    limiter: RateLimiter | None = None,
    max_completion_tokens: int = None,
    response_format: dict = None,
    wall_clock_budget_s: float = 900.0,
):
    """Call OpenRouter with rate limiting + retries.

    CRITICAL DESIGN: each retry attempt re-acquires a rate limiter slot.
    This means that when a 429 triggers a cooldown on the rate limiter,
    the retry will wait for the cooldown to clear before trying again —
    preventing the "all concurrent slots retrying simultaneously" stampede.

    Flow for each attempt:
        1. acquire rate limiter slot (waits for cooldown + token bucket)
        2. make API call
        3. release slot
        4. if 429: notify rate limiter (sets cooldown + reduces rate),
           compute backoff, sleep, then go to step 1
        5. if other transient error: backoff, go to step 1
        6. if success: return
    """
    if limiter is None:
        limiter = get_rate_limiter()

    last_exc = None
    wait = 0.0
    deadline = time.monotonic() + wall_clock_budget_s

    def _budget_left() -> float:
        return max(0.0, deadline - time.monotonic())

    for attempt in range(1, MAX_RETRIES + 1):
        if _budget_left() <= 0:
            raise (
                APITimeoutError(
                    request=None,
                )
                if last_exc is None
                else last_exc
            )
        # Acquire rate limiter slot (will wait for cooldown + token).
        # Wrap in wait_for so a stuck cooldown can't pin us forever.
        async with limiter:
            try:
                result = await asyncio.wait_for(
                    call_llm(
                        prompt=prompt,
                        model=model,
                        max_completion_tokens=max_completion_tokens,
                        response_format=response_format,
                    ),
                    timeout=min(120.0, _budget_left() or 120.0),
                )
                return result
            except RateLimitError as e:
                last_exc = e
                # Parse retry-after from headers and notify the rate limiter
                retry_after = _parse_retry_after(e)
                limiter.notify_429(retry_after)

                # Compute wait: use header if available, else exponential backoff
                if retry_after and retry_after > 0:
                    wait = retry_after + random.uniform(0.5, 2.0)
                else:
                    wait = min(120, (2**attempt) + random.uniform(0, 2))

                if attempt < MAX_RETRIES:
                    logger.warning(
                        "429 on attempt %d/%d (model=%s), waiting %.1fs "
                        "(rate limiter will also enforce cooldown on re-acquire)",
                        attempt,
                        MAX_RETRIES,
                        model,
                        wait,
                    )
                # Sleep OUTSIDE the rate limiter slot (slot released by __aexit__)

            except APIStatusError as e:
                # Log the ACTUAL status code and error body so we can debug
                status = e.status_code
                body = str(e.body)[:300] if e.body else str(e)[:300]

                if status == 429:
                    # Shouldn't reach here (RateLimitError above), but handle it
                    last_exc = e
                    limiter.notify_429(None)
                    wait = min(120, (2**attempt) + random.uniform(0, 2))
                    logger.warning(
                        "429 (APIStatusError) on attempt %d/%d (model=%s), "
                        "waiting %.1fs: %s",
                        attempt,
                        MAX_RETRIES,
                        model,
                        wait,
                        body,
                    )
                elif status in _RETRYABLE_STATUS_CODES:
                    # Genuinely transient server error — retry
                    last_exc = e
                    wait = min(60, (2**attempt) + random.uniform(0, 2))
                    if attempt < MAX_RETRIES:
                        logger.warning(
                            "Transient %d on attempt %d/%d (model=%s), "
                            "waiting %.1fs: %s",
                            status,
                            attempt,
                            MAX_RETRIES,
                            model,
                            wait,
                            body,
                        )
                else:
                    # Non-retryable error (400, 401, 403, 404, etc.)
                    # Log once with full details and raise immediately
                    logger.error(
                        "Non-retryable %d from %s: %s",
                        status,
                        model,
                        body,
                    )
                    raise

            except _RETRYABLE_ERRORS as e:
                last_exc = e
                wait = min(60, (2**attempt) + random.uniform(0, 2))
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Transient error on attempt %d/%d (model=%s): %s — %s, "
                        "waiting %.1fs",
                        attempt,
                        MAX_RETRIES,
                        model,
                        type(e).__name__,
                        str(e)[:200],
                        wait,
                    )

            except json.JSONDecodeError as e:
                # OpenRouter occasionally returns non-JSON bodies (proxy errors,
                # HTML error pages) — treat as transient and retry.
                last_exc = e
                wait = min(60, (2**attempt) + random.uniform(0, 2))
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Non-JSON response on attempt %d/%d (model=%s), "
                        "waiting %.1fs: %s",
                        attempt,
                        MAX_RETRIES,
                        model,
                        wait,
                        str(e)[:200],
                    )

            except asyncio.TimeoutError as e:
                # Our per-call wait_for fired — provider stalled mid-request.
                last_exc = APITimeoutError(request=None)
                wait = min(30, (2**attempt) + random.uniform(0, 2))
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Per-call timeout on attempt %d/%d (model=%s), waiting %.1fs",
                        attempt,
                        MAX_RETRIES,
                        model,
                        wait,
                    )

        # Wait between retries (outside the rate limiter context).
        # Don't sleep past the wall-clock deadline.
        if attempt < MAX_RETRIES:
            await asyncio.sleep(min(wait, _budget_left()))

    # All retries exhausted
    raise last_exc
