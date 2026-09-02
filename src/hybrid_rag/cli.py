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


