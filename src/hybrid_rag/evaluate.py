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
    return "\n".join(rows)


JUDGE_PROMPT = """You grade answers about Python PEPs.
Question: {question}
Reference answer: {reference}
Candidate answer: {answer}

Is the candidate factually consistent with the reference and does it answer the question?
Extra correct detail is fine; contradictions or missing the core point are not.
Return only JSON: {{"correct": true|false, "reason": "<one sentence>"}}"""


def judge(question: Question, answer_text: str, model: str, completion_fn: CompletionFn | None = None):
    prompt = JUDGE_PROMPT.format(question=question.question, reference=question.reference, answer=answer_text)
    result = complete(
        model,
        [{"role": "user", "content": prompt}],
        completion_fn=completion_fn,
        max_tokens=150,
    )
    match = re.search(r"\{.*\}", result.text, re.DOTALL)
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    return bool(verdict.get("correct", False)), str(verdict.get("reason", "unparseable judge output")), result


def evaluate_answers(
    retriever: Retriever,
    questions: list[Question],
    model: str,
    judge_model: str | None = None,
    k: int = 6,
    use_rerank: bool = False,
    completion_fn: CompletionFn | None = None,
) -> dict:
    judge_model = judge_model or model
    meter = UsageMeter()
    per_q = []
    for q in questions:
        hits = retriever.retrieve(q.question, k=20 if use_rerank else k)
        if use_rerank:
            hits = rerank(q.question, hits, model, completion_fn=completion_fn)[:k]
        ans = answer_question(q.question, hits, model, completion_fn=completion_fn)
        if ans.llm:
            meter.add(ans.llm)
        correct, reason, jres = judge(q, ans.text, judge_model, completion_fn=completion_fn)
        meter.add(jres)
        shown = {h.chunk.number for h in hits}
        per_q.append(
            {
                "id": q.qid,
                "answer": ans.text,
                "cited": ans.cited_peps,
                "correct": correct,
                "judge_reason": reason,
                "cites_gold": bool(set(ans.cited_peps) & set(q.gold_peps)),
                "citations_grounded": set(ans.cited_peps) <= shown and bool(ans.cited_peps),
                "latency_s": ans.llm.latency_s if ans.llm else 0.0,
            }
        )
    n = len(per_q)
    summary = {
        "n": n,
        "accuracy": sum(r["correct"] for r in per_q) / n,
        "cites_gold": sum(r["cites_gold"] for r in per_q) / n,
        "citations_grounded": sum(r["citations_grounded"] for r in per_q) / n,
        "mean_latency_s": sum(r["latency_s"] for r in per_q) / n,
        "llm_calls": meter.calls,
        "total_cost_usd": round(meter.cost_usd, 4),
    }
    return {"summary": summary, "per_question": per_q}
