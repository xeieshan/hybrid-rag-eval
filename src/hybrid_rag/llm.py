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
