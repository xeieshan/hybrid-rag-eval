.PHONY: install test lint data index index-api eval eval-api answers ablation serve docker

EMBED_MODEL ?= text-embedding-3-small
LLM_MODEL ?= gpt-4o-mini

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

data:
	hybrid-rag download --dest data/peps

# Local index: BM25 + LSA, no API key needed.
index:
	hybrid-rag build-index --corpus data/peps --out artifacts/index-lsa --embedder lsa:256

# API embeddings through LiteLLM (needs the provider's key, e.g. OPENAI_API_KEY).
index-api:
	hybrid-rag build-index --corpus data/peps --out artifacts/index-api --embedder litellm:$(EMBED_MODEL)

eval:
	hybrid-rag eval-retrieval --index lsa=artifacts/index-lsa --out results/retrieval.md

eval-api:
	hybrid-rag eval-retrieval --index lsa=artifacts/index-lsa --index api=artifacts/index-api --out results/retrieval_api.md

answers:
	hybrid-rag eval-answers --index artifacts/index-api --model $(LLM_MODEL) --out results/answers.json

ablation:
	for mw in 120 400; do \
		hybrid-rag build-index --corpus data/peps --out artifacts/index-lsa-$$mw --embedder lsa:256 --max-words $$mw --overlap $$((mw / 5)); \
