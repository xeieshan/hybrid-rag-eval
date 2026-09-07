"""HTTP API. Run with ``hybrid-rag serve`` or ``uvicorn hybrid_rag.api:app``.

Environment:
    RAG_INDEX_DIR   index directory written by ``hybrid-rag build-index`` (default: artifacts/index)
    RAG_LLM_MODEL   any LiteLLM model string (default: gpt-4o-mini)
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .generate import answer_question, rerank
from .llm import CompletionFn
from .retrieval import IndexBundle, Retriever


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=50)
    mode: Literal["bm25", "dense", "hybrid"] = "hybrid"


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=6, ge=1, le=20)
    rerank: bool = False


class Source(BaseModel):
    pep: int
    title: str
    section: str
    score: float
    text: str


class SearchResponse(BaseModel):
    results: list[Source]


class AskResponse(BaseModel):
    answer: str
    cited_peps: list[int]
    sources: list[Source]


def _source(hit) -> Source:
    c = hit.chunk
    return Source(pep=c.number, title=c.title, section=c.section, score=hit.score, text=c.text)


def create_app(
    bundle: IndexBundle | None = None, model: str | None = None, completion_fn: CompletionFn | None = None
) -> FastAPI:
    state: dict = {"bundle": bundle, "model": model or os.getenv("RAG_LLM_MODEL", "gpt-4o-mini")}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["bundle"] is None:
            state["bundle"] = IndexBundle.load(os.getenv("RAG_INDEX_DIR", "artifacts/index"))
