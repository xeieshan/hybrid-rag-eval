import json

from hybrid_rag.evaluate import (
    Question,
    doc_ranking,
    evaluate_answers,
    evaluate_retrieval,
    hit_at_k,
    load_questions,
    recall_at_k,
    reciprocal_rank,
    retrieval_table,
)
from hybrid_rag.generate import NOT_FOUND, answer_question, build_messages, parse_citations, rerank
from hybrid_rag.retrieval import Retriever


def test_parse_citations_dedupes_in_order():
    text = "Use := [PEP 572]. Lines max 79 [PEP 8, Maximum Line Length]. Again [pep 572]."
    assert parse_citations(text) == [572, 8]


def test_prompt_contains_sources(bundle):
    hits = Retriever(bundle).retrieve("walrus", k=2)
    msgs = build_messages("What is :=?", hits)
    assert msgs[0]["role"] == "system"
    assert "PEP 572: Assignment Expressions" in msgs[1]["content"]
    assert msgs[1]["content"].endswith("Question: What is :=?")


def test_answer_question_uses_completion(bundle, fake_completion):
    fn = fake_completion("It is the walrus operator [PEP 572].")
    hits = Retriever(bundle).retrieve("walrus", k=2)
    ans = answer_question("What is :=?", hits, model="test-model", completion_fn=fn)
    assert ans.cited_peps == [572]
    assert ans.found
