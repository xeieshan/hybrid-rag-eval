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
