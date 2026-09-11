"""Command line interface: ``hybrid-rag <command> --help``."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .chunking import chunk_corpus
from .corpus import download_peps, load_corpus
from .embeddings import make_embedder
from .evaluate import evaluate_answers, evaluate_retrieval, load_questions, retrieval_table
from .generate import answer_question, rerank
from .retrieval import IndexBundle, Retriever

DEFAULT_QUESTIONS = "data/eval/questions.jsonl"


def cmd_download(args: argparse.Namespace) -> None:
    path = download_peps(args.dest)
    print(f"Downloaded PEPs to {path}")


def cmd_build_index(args: argparse.Namespace) -> None:
    t0 = time.perf_counter()
    docs = load_corpus(args.corpus)
    chunks = chunk_corpus(docs, max_words=args.max_words, overlap_words=args.overlap)
    embedder = make_embedder(args.embedder, cache_dir=Path(args.out) / "embed_cache")
    bundle = IndexBundle.build(chunks, embedder)
    bundle.save(args.out)
    elapsed = time.perf_counter() - t0
    print(f"Indexed {len(docs)} PEPs as {len(chunks)} chunks with {embedder.name} in {elapsed:.1f}s -> {args.out}")


def cmd_search(args: argparse.Namespace) -> None:
    hits = Retriever(IndexBundle.load(args.index), mode=args.mode).retrieve(args.query, k=args.k)
    for h in hits:
        preview = " ".join(h.chunk.text.split()[:30])
        print(f"{h.rank:>2}. PEP {h.chunk.number} [{h.chunk.section}] score={h.score:.4f}\n    {preview}...")


def cmd_ask(args: argparse.Namespace) -> None:
    retriever = Retriever(IndexBundle.load(args.index), mode="hybrid")
    hits = retriever.retrieve(args.question, k=20 if args.rerank else args.k)
    if args.rerank:
        hits = rerank(args.question, hits, args.model)[: args.k]
    ans = answer_question(args.question, hits, args.model)
    print(ans.text)
    print("\nSources: " + ", ".join(f"PEP {h.chunk.number} ({h.chunk.section})" for h in hits))


def _parse_named(values: list[str]) -> dict[str, str]:
    out = {}
    for v in values:
        name, sep, path = v.partition("=")
        out[name if sep else Path(v).name] = path if sep else v
    return out


def cmd_eval_retrieval(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)
    results: dict[str, dict] = {}
    bm25_done = False
    for name, path in _parse_named(args.index).items():
        bundle = IndexBundle.load(path)
        if not bm25_done:
            results["BM25"] = evaluate_retrieval(Retriever(bundle, mode="bm25"), questions)
            bm25_done = True
        results[f"Dense ({name})"] = evaluate_retrieval(Retriever(bundle, mode="dense"), questions)
        results[f"Hybrid RRF (BM25 + {name})"] = evaluate_retrieval(Retriever(bundle, mode="hybrid"), questions)
    table = retrieval_table(results)
    print(table)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(f"Questions: {len(questions)} ({args.questions})\n\n{table}\n", encoding="utf-8")
        out.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nWrote {out} and {out.with_suffix('.json')}")


def cmd_eval_answers(args: argparse.Namespace) -> None:
    questions = load_questions(args.questions)[: args.limit or None]
    retriever = Retriever(IndexBundle.load(args.index), mode="hybrid")
    res = evaluate_answers(
        retriever, questions, model=args.model, judge_model=args.judge_model, k=args.k, use_rerank=args.rerank
    )
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


def cmd_serve(args: argparse.Namespace) -> None:
    import os

    import uvicorn

    os.environ["RAG_INDEX_DIR"] = args.index
    uvicorn.run("hybrid_rag.api:app", host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hybrid-rag", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("download", help="download the pinned PEP corpus")
    s.add_argument("--dest", default="data/peps")
    s.set_defaults(func=cmd_download)

    s = sub.add_parser("build-index", help="chunk the corpus and build BM25 + dense indexes")
    s.add_argument("--corpus", default="data/peps")
    s.add_argument("--out", default="artifacts/index")
    s.add_argument("--embedder", default="lsa:256", help="lsa[:dims] | litellm:<model> | hashing")
    s.add_argument("--max-words", type=int, default=220)
    s.add_argument("--overlap", type=int, default=40)
    s.set_defaults(func=cmd_build_index)

    s = sub.add_parser("search", help="retrieve chunks for a query")
    s.add_argument("query")
    s.add_argument("--index", default="artifacts/index")
    s.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    s.add_argument("--k", type=int, default=5)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="answer a question with citations")
    s.add_argument("question")
    s.add_argument("--index", default="artifacts/index")
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--k", type=int, default=6)
    s.add_argument("--rerank", action="store_true")
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("eval-retrieval", help="hit@k / MRR for BM25, dense and hybrid")
    s.add_argument("--index", action="append", required=True, help="name=path, repeatable")
    s.add_argument("--questions", default=DEFAULT_QUESTIONS)
    s.add_argument("--out", default="results/retrieval.md")
    s.set_defaults(func=cmd_eval_retrieval)

    s = sub.add_parser("eval-answers", help="end-to-end answer accuracy with an LLM judge")
    s.add_argument("--index", default="artifacts/index")
    s.add_argument("--questions", default=DEFAULT_QUESTIONS)
    s.add_argument("--model", default="gpt-4o-mini")
    s.add_argument("--judge-model", default=None)
    s.add_argument("--k", type=int, default=6)
    s.add_argument("--rerank", action="store_true")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--out", default="results/answers.json")
    s.set_defaults(func=cmd_eval_answers)

    s = sub.add_parser("serve", help="run the HTTP API")
    s.add_argument("--index", default="artifacts/index")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=cmd_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Maintenance pass: 2026-09-11 14:37
