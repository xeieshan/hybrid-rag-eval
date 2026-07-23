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

_CITATION_RE = re.compile(r"\[PEP\s*(\d{1,4})[^\]]*\]", re.IGNORECASE)


@dataclass
class Answer:
    question: str
    text: str
    cited_peps: list[int]
    sources: list[Hit] = field(default_factory=list)
    llm: LLMResult | None = None

    @property
    def found(self) -> bool:
        return NOT_FOUND not in self.text


def format_context(hits: list[Hit], max_words_per_hit: int = 260) -> str:
    blocks = []
    for h in hits:
        words = h.chunk.text.split()
        text = " ".join(words[:max_words_per_hit]) + (" ..." if len(words) > max_words_per_hit else "")
        blocks.append(f"[PEP {h.chunk.number}: {h.chunk.title} | section: {h.chunk.section}]\n{text}")
    return "\n\n---\n\n".join(blocks)


def build_messages(question: str, hits: list[Hit]) -> list[dict[str, str]]:
    user = f"PEP excerpts:\n\n{format_context(hits)}\n\nQuestion: {question}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_citations(text: str) -> list[int]:
    seen: list[int] = []
    for m in _CITATION_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def answer_question(question: str, hits: list[Hit], model: str, completion_fn: CompletionFn | None = None) -> Answer:
    result = complete(model, build_messages(question, hits), completion_fn=completion_fn)
    return Answer(
        question=question,
        text=result.text.strip(),
        cited_peps=parse_citations(result.text),
        sources=hits,
        llm=result,
    )


RERANK_PROMPT = """Rate how well each passage answers the query.
0 = irrelevant, 1 = related topic, 2 = partially answers, 3 = directly answers.
Return only JSON: {{"scores": [<one integer per passage, in order>]}}

Query: {query}

{passages}"""


def rerank(query: str, hits: list[Hit], model: str, completion_fn: CompletionFn | None = None) -> list[Hit]:
    """Listwise LLM reranking. Falls back to the original order if the reply can't be parsed."""
    if len(hits) <= 1:
        return hits
    passages = "\n\n".join(
        f"Passage {i}: PEP {h.chunk.number} ({h.chunk.section})\n{' '.join(h.chunk.text.split()[:120])}"
        for i, h in enumerate(hits, start=1)
    )
    result = complete(
