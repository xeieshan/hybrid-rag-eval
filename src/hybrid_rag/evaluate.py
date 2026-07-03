"""Retrieval and answer-quality evaluation.

Retrieval metrics are document level: chunks are mapped back to their PEP and
de-duplicated in rank order, then compared with the question's gold PEP(s).

Answer metrics combine deterministic checks (does the answer cite a gold PEP?
does it cite only PEPs it was shown?) with an LLM judge that compares the answer
with a short human-written reference.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .generate import answer_question, rerank
from .llm import CompletionFn, UsageMeter, complete
from .retrieval import Hit, Retriever


@dataclass(frozen=True)
class Question:
    qid: str
    question: str
    gold_peps: tuple[int, ...]
    reference: str


def load_questions(path: str | Path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out.append(Question(rec["id"], rec["question"], tuple(rec["gold_peps"]), rec["reference"]))
    return out


def doc_ranking(hits: Iterable[Hit]) -> list[int]:
    seen: list[int] = []
    for h in hits:
        if h.chunk.number not in seen:
            seen.append(h.chunk.number)
    return seen


def hit_at_k(ranking: list[int], gold: Iterable[int], k: int) -> float:
    return float(bool(set(ranking[:k]) & set(gold)))
