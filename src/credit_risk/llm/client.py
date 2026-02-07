from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float


class LLMClient(Protocol):
    model: str

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> LLMResponse: ...


def approx_tokens(text: str) -> int:
    """Rough token count (about four characters per token for English)."""
    return max(1, len(text) // 4)


class MockLLM:
    """Deterministic offline client for tests and CI.

    ``responder`` maps (system, user) to the reply. The default returns a
    fixed string tagged with a hash of the prompt, so identical prompts
    always give identical output.
    """

    def __init__(
        self,
        responder: Callable[[str, str], str] | None = None,
        model: str = "mock-1",
    ) -> None:
        self.model = model
        self.responder = responder or self._default
        self.calls: list[tuple[str, str]] = []

    @staticmethod
    def _default(system: str, user: str) -> str:
        digest = hashlib.sha256((system + user).encode()).hexdigest()[:8]
        return f"mock response {digest}"

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> LLMResponse:
        self.calls.append((system, user))
        text = self.responder(system, user)
        return LLMResponse(
            text, self.model, approx_tokens(system + user), approx_tokens(text), 0.0
        )


class OpenAICompatibleLLM:
    """Client for any OpenAI-compatible chat endpoint, including local servers.

    Reads ``LLM_API_KEY``, ``LLM_BASE_URL`` and ``LLM_MODEL`` when arguments are
    not given. Requires the ``llm`` extra.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        from openai import OpenAI

        self.model = model or os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self._client = OpenAI(
            api_key=api_key or os.environ.get("LLM_API_KEY"),
            base_url=base_url or os.environ.get("LLM_BASE_URL"),
            timeout=timeout,
        )

    def complete(
        self,
        system: str,
        user: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
        json_mode: bool = False,
    ) -> LLMResponse:
        start = time.perf_counter()
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        )
        usage = resp.usage
        return LLMResponse(
            text=resp.choices[0].message.content or "",
            model=self.model,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_s=time.perf_counter() - start,
        )


@dataclass
class UsageLog:
    """Token, latency and cost accounting across calls."""

    price_per_1k_prompt: float = 0.0
    price_per_1k_completion: float = 0.0
    records: list[LLMResponse] = field(default_factory=list)

    def add(self, response: LLMResponse) -> None:
        self.records.append(response)

    @property
    def cost(self) -> float:
        return sum(
            r.prompt_tokens / 1000 * self.price_per_1k_prompt
            + r.completion_tokens / 1000 * self.price_per_1k_completion
            for r in self.records
        )

    def summary(self) -> dict[str, float]:
        n = len(self.records)
        return {
            "calls": n,
            "prompt_tokens": sum(r.prompt_tokens for r in self.records),
            "completion_tokens": sum(r.completion_tokens for r in self.records),
            "mean_latency_s": sum(r.latency_s for r in self.records) / n if n else 0.0,
            "cost": round(self.cost, 6),
        }


class RetryingClient:
    """Wraps a client with exponential backoff and usage logging."""

    def __init__(
        self,
        client: LLMClient,
        max_retries: int = 3,
        backoff_s: float = 1.0,
        usage: UsageLog | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self.model = client.model
        self.max_retries = max_retries
        self.backoff_s = backoff_s
        self.usage = usage or UsageLog()
        self._sleep = sleep

    def complete(self, system: str, user: str, **kwargs) -> LLMResponse:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.complete(system, user, **kwargs)
                self.usage.add(response)
                return response
            except Exception:
                if attempt == self.max_retries:
                    raise
                self._sleep(self.backoff_s * 2**attempt)
        raise RuntimeError("unreachable")


def get_client(provider: str | None = None) -> LLMClient:
    """Client selected by ``provider`` or the ``LLM_PROVIDER`` variable."""
    provider = (provider or os.environ.get("LLM_PROVIDER", "mock")).lower()
    if provider == "mock":
        return MockLLM()
    if provider in {"openai", "openai-compatible", "local"}:
        return OpenAICompatibleLLM()
    raise ValueError(f"unknown LLM provider '{provider}'")
