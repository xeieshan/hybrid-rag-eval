"""Section-aware chunking for reStructuredText documents.

Each chunk stays inside one section, so a retrieved chunk can be cited as
"PEP 572, section Abstract". Every chunk also carries a short header with the
PEP number, title and section; this "contextual header" gives lexical and
dense retrievers the document-level context that a bare paragraph lacks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .corpus import Document

_UNDERLINE_RE = re.compile(r"^([=\-~^\"'`#*+])\1{2,}\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    number: int
    title: str
