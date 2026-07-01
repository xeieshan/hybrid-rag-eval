"""Answer generation with citations, plus an optional LLM reranker."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import CompletionFn, LLMResult, complete
from .retrieval import Hit

NOT_FOUND = "I couldn't find this in the PEPs I retrieved."

SYSTEM_PROMPT = f"""You answer questions about Python using only the PEP excerpts provided.
Rules:
- Every factual sentence ends with a citation like [PEP 572].
- Cite only PEPs that appear in the excerpts.
- If the excerpts do not contain the answer, reply exactly: "{NOT_FOUND}"
- Be concise: 2-5 sentences, include code only when it helps."""

