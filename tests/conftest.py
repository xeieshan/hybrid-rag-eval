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

