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
    section: str
    text: str

    @property
    def indexed_text(self) -> str:
        return f"PEP {self.number}: {self.title}\nSection: {self.section}\n\n{self.text}"

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "number": self.number,
            "title": self.title,
            "section": self.section,
            "text": self.text,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Chunk:
        return cls(**data)


def split_sections(body: str) -> list[tuple[str, str]]:
    """Return ``[(heading, text), ...]``. Text before the first heading is "Preamble"."""
    lines = body.splitlines()
    sections: list[tuple[str, list[str]]] = [("Preamble", [])]
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        stripped = line.strip()
        is_heading = (
            stripped
            and not _UNDERLINE_RE.match(stripped)
            and _UNDERLINE_RE.match(nxt.strip())
            and len(nxt.strip()) >= len(stripped)
            and not line.startswith((" ", "\t"))
        )
        if is_heading:
            # Drop an overline directly above the heading, if present.
            current = sections[-1][1]
            if current and _UNDERLINE_RE.match(current[-1].strip()):
                current.pop()
            sections.append((stripped, []))
            i += 2
            continue
        sections[-1][1].append(line)
        i += 1
    result = []
    for heading, sec_lines in sections:
        text = "\n".join(sec_lines).strip()
        if text:
            result.append((heading, text))
