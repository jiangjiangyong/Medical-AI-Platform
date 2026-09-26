from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    """Raised when a dependency is temporarily unavailable after repeated failures."""


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_seconds: float = 30.0
    _failures: int = 0
    _opened_at: float | None = None

    def allow(self) -> bool:
        if self._opened_at is None:
            return True
        if time.monotonic() - self._opened_at >= self.recovery_seconds:
            self._opened_at = None
            self._failures = 0
            return True
        return False

    def record_success(self) -> None:
        self._failures = 0
        self._opened_at = None

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= max(1, self.failure_threshold):
            self._opened_at = time.monotonic()

    @property
    def state(self) -> str:
        if self._opened_at is not None and not self.allow():
            return "open"
        return "closed"


@dataclass(frozen=True)
class ResiliencePolicy:
    timeout_seconds: float = 45.0
    attempts: int = 2
    backoff_seconds: float = 0.25


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_seconds: float,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    total_attempts = max(1, int(attempts))
    for attempt in range(total_attempts):
        try:
            return operation()
        except retry_exceptions:
            if attempt + 1 >= total_attempts:
                raise
            delay = min(max(0.0, float(backoff_seconds)) * (2**attempt), 5.0)
            if delay:
                time.sleep(delay)
    raise RuntimeError("retry_call exhausted without a result")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    backoff_seconds: float,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    total_attempts = max(1, int(attempts))
    for attempt in range(total_attempts):
        try:
            return await operation()
        except retry_exceptions:
            if attempt + 1 >= total_attempts:
                raise
            delay = min(max(0.0, float(backoff_seconds)) * (2**attempt), 5.0)
            if delay:
                await asyncio.sleep(delay)
    raise RuntimeError("retry_async exhausted without a result")


def call_with_resilience(
    operation: Callable[[], T],
    *,
    policy: ResiliencePolicy,
    breaker: CircuitBreaker | None = None,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    if breaker is not None and not breaker.allow():
        raise CircuitOpenError("dependency circuit is open")
    try:
        result = retry_call(
            operation,
            attempts=policy.attempts,
            backoff_seconds=policy.backoff_seconds,
            retry_exceptions=retry_exceptions,
        )
    except BaseException:
        if breaker is not None:
            breaker.record_failure()
        raise
    if breaker is not None:
        breaker.record_success()
    return result


async def async_call_with_resilience(
    operation: Callable[[], Awaitable[T]],
    *,
    policy: ResiliencePolicy,
    breaker: CircuitBreaker | None = None,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    if breaker is not None and not breaker.allow():
        raise CircuitOpenError("dependency circuit is open")
    try:
        async def _timed_operation() -> T:
            return await asyncio.wait_for(operation(), timeout=policy.timeout_seconds)

        result = await retry_async(
            _timed_operation,
            attempts=policy.attempts,
            backoff_seconds=policy.backoff_seconds,
            retry_exceptions=retry_exceptions,
        )
    except BaseException:
        if breaker is not None:
            breaker.record_failure()
        raise
    if breaker is not None:
        breaker.record_success()
    return result
