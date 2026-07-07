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


def recall_at_k(ranking: list[int], gold: Iterable[int], k: int) -> float:
    gold = set(gold)
    return len(set(ranking[:k]) & gold) / len(gold) if gold else 0.0


def reciprocal_rank(ranking: list[int], gold: Iterable[int]) -> float:
    gold = set(gold)
    for i, doc in enumerate(ranking, start=1):
        if doc in gold:
            return 1.0 / i
    return 0.0


def evaluate_retrieval(
    retriever: Retriever, questions: list[Question], ks: tuple[int, ...] = (1, 3, 5, 10), depth: int = 50
) -> dict:
    per_q = []
    for q in questions:
        ranking = doc_ranking(retriever.retrieve(q.question, k=depth))
        rec = {"id": q.qid, "gold": list(q.gold_peps), "top5": ranking[:5], "rr": reciprocal_rank(ranking, q.gold_peps)}
        for k in ks:
            rec[f"hit@{k}"] = hit_at_k(ranking, q.gold_peps, k)
            rec[f"recall@{k}"] = recall_at_k(ranking, q.gold_peps, k)
        per_q.append(rec)
    n = len(per_q)
    summary = {"n": n, "mrr": sum(r["rr"] for r in per_q) / n}
    for k in ks:
        summary[f"hit@{k}"] = sum(r[f"hit@{k}"] for r in per_q) / n
        summary[f"recall@{k}"] = sum(r[f"recall@{k}"] for r in per_q) / n
    return {"summary": summary, "per_question": per_q}


def retrieval_table(results: dict[str, dict], ks: tuple[int, ...] = (1, 3, 5, 10)) -> str:
    header = "| Retriever | " + " | ".join(f"Hit@{k}" for k in ks) + " | MRR |"
    sep = "|---|" + "---:|" * (len(ks) + 1)
    rows = [header, sep]
    for name, res in results.items():
        s = res["summary"]
        cells = " | ".join(f"{s[f'hit@{k}']:.3f}" for k in ks)
        rows.append(f"| {name} | {cells} | {s['mrr']:.3f} |")
