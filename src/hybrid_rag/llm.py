"""Thin LiteLLM wrapper that records token usage and cost for every call.

Set the provider key your model needs (``OPENAI_API_KEY``, ``MISTRAL_API_KEY``,
...) or point at a local model such as ``ollama/llama3.1``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

CompletionFn = Callable[..., Any]


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0


@dataclass
class UsageMeter:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, r: LLMResult) -> None:
        self.calls += 1
        self.prompt_tokens += r.prompt_tokens
        self.completion_tokens += r.completion_tokens
        self.cost_usd += r.cost_usd


def _default_completion(**kwargs: Any) -> Any:
    import litellm

    return litellm.completion(**kwargs)


def complete(
    model: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 600,
    completion_fn: CompletionFn | None = None,
    retries: int = 2,
    **kwargs: Any,
) -> LLMResult:
    fn = completion_fn or _default_completion
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            start = time.perf_counter()
            resp = fn(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, **kwargs)
            latency = time.perf_counter() - start
            break
        except Exception as err:  # noqa: BLE001 - provider errors vary by backend
            last_err = err
            if attempt == retries:
                raise
            time.sleep(2**attempt)
    else:  # pragma: no cover
        raise RuntimeError("unreachable") from last_err

    text = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    cost = 0.0
    try:
        import litellm

        cost = float(litellm.completion_cost(completion_response=resp) or 0.0)
    except Exception:  # noqa: BLE001 - unknown/local models have no price table
        cost = 0.0
    return LLMResult(
        text=text,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        cost_usd=cost,
        latency_s=latency,
    )

# Maintenance pass: 2026-07-20 14:10
