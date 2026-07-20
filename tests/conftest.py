from pathlib import Path

import pytest

from hybrid_rag.chunking import chunk_corpus
from hybrid_rag.corpus import load_corpus
from hybrid_rag.embeddings import HashingEmbedder
from hybrid_rag.retrieval import IndexBundle

PEPS = {
    "pep-0572.rst": """PEP: 572
Title: Assignment Expressions
Author: Chris Angelico <rosuav@gmail.com>,
        Tim Peters <tim.peters@gmail.com>
Status: Final
Python-Version: 3.8

Abstract
========

This is a proposal for creating a way to assign to variables within an
expression using the notation ``NAME := expr``.

Rationale
=========

Naming the result of an expression is an important part of programming.
The walrus operator is useful in while loops and comprehensions.
""",
    "pep-0008.rst": """PEP: 8
Title: Style Guide for Python Code
Status: Active

Maximum Line Length
===================

Limit all lines to a maximum of 79 characters. Docstrings and comments
should be limited to 72 characters.

Programming Recommendations
===========================

Always use a def statement instead of an assignment statement that binds a
lambda expression directly to an identifier.
""",
    "pep-0020.rst": """PEP: 20
Title: The Zen of Python
Status: Active

Abstract
========

Long time Pythoneer Tim Peters succinctly channels the BDFL's guiding
principles for Python's design into 20 aphorisms, only 19 of which
have been written down.
""",
}


@pytest.fixture()
def corpus_dir(tmp_path: Path) -> Path:
    for name, text in PEPS.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
    (tmp_path / "README.rst").write_text("not a pep", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def bundle(corpus_dir: Path) -> IndexBundle:
    chunks = chunk_corpus(load_corpus(corpus_dir), max_words=60, overlap_words=10)
    return IndexBundle.build(chunks, HashingEmbedder(256))


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeChoice:
    def __init__(self, content: str):
        self.message = FakeMessage(content)


class FakeUsage:
    prompt_tokens = 10
    completion_tokens = 5


class FakeResponse:
    def __init__(self, content: str):
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


@pytest.fixture()
def fake_completion():
    """Returns a factory: fake_completion(reply) -> completion_fn that always replies ``reply``."""

    def factory(reply):
        calls = []

        def fn(**kwargs):
            calls.append(kwargs)
            content = reply(kwargs) if callable(reply) else reply
            return FakeResponse(content)

        fn.calls = calls
        return fn

    return factory

# Maintenance pass: 2026-07-20 16:44
