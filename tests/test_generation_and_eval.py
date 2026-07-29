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
    assert fn.calls[0]["model"] == "test-model"
    assert fn.calls[0]["temperature"] == 0.0


def test_not_found_answer(bundle, fake_completion):
    ans = answer_question("Q", [], "m", completion_fn=fake_completion(NOT_FOUND))
    assert not ans.found


def test_rerank_reorders_by_scores(bundle, fake_completion):
    hits = Retriever(bundle).retrieve("python", k=3)
    reply = json.dumps({"scores": [0, 3, 1]})
    out = rerank("q", hits, "m", completion_fn=fake_completion(reply))
    assert len(hits) == 3
    expected = [hits[1].chunk.chunk_id, hits[2].chunk.chunk_id, hits[0].chunk.chunk_id]
    assert [h.chunk.chunk_id for h in out] == expected


def test_rerank_falls_back_on_bad_json(bundle, fake_completion):
    hits = Retriever(bundle).retrieve("python", k=3)
    out = rerank("q", hits, "m", completion_fn=fake_completion("not json"))
    assert out == hits
    out = rerank("q", hits, "m", completion_fn=fake_completion('{"scores": ["x", "y", "z"]}'))
    assert out == hits


def test_metric_functions():
    ranking = [20, 8, 572]
    assert hit_at_k(ranking, [8], 1) == 0.0
    assert hit_at_k(ranking, [8], 2) == 1.0
    assert recall_at_k(ranking, [8, 572], 2) == 0.5
    assert reciprocal_rank(ranking, [572]) == 1 / 3
    assert reciprocal_rank(ranking, [1]) == 0.0


def test_doc_ranking_dedupes(bundle):
    hits = Retriever(bundle, mode="bm25").retrieve("python", k=10)
    ranking = doc_ranking(hits)
    assert len(ranking) == len(set(ranking))


def test_evaluate_retrieval_end_to_end(bundle):
    qs = [
        Question("a", "walrus operator assignment expression", (572,), "ref"),
        Question("b", "maximum line length characters", (8,), "ref"),
    ]
    res = evaluate_retrieval(Retriever(bundle, mode="bm25"), qs, ks=(1, 3))
    assert res["summary"]["hit@1"] == 1.0
    assert res["summary"]["mrr"] == 1.0
    table = retrieval_table({"BM25": res}, ks=(1, 3))
    assert "| BM25 | 1.000 | 1.000 | 1.000 |" in table


def test_evaluate_answers_with_fake_llm(bundle, fake_completion):
    def reply(kwargs):
        content = kwargs["messages"][-1]["content"]
        if "Candidate answer" in content:
            return '{"correct": true, "reason": "matches"}'
        return "Use := [PEP 572]."

    qs = [Question("a", "walrus operator", (572,), "The := operator.")]
    res = evaluate_answers(Retriever(bundle), qs, model="m", completion_fn=fake_completion(reply))
    s = res["summary"]
    assert s["accuracy"] == 1.0
    assert s["cites_gold"] == 1.0
    assert s["citations_grounded"] == 1.0
    assert s["llm_calls"] == 2


def test_shipped_eval_set_is_well_formed():
    qs = load_questions("data/eval/questions.jsonl")
    assert len(qs) >= 40
    assert len({q.qid for q in qs}) == len(qs)
    assert all(q.gold_peps and q.reference for q in qs)

# Maintenance pass: 2026-07-29 17:41
