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
    return result


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _word_windows(words: list[str], max_words: int, overlap: int) -> list[list[str]]:
    step = max(1, max_words - overlap)
    return [words[i : i + max_words] for i in range(0, max(1, len(words) - overlap), step)]


def chunk_text(text: str, max_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Greedy paragraph packing with word overlap between consecutive chunks."""
    if max_words <= overlap_words:
        raise ValueError("max_words must be larger than overlap_words")
    pieces: list[str] = []
    for para in _paragraphs(text):
        words = para.split()
        if len(words) <= max_words:
            pieces.append(para)
        else:
            pieces.extend(" ".join(w) for w in _word_windows(words, max_words, overlap_words))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for piece in pieces:
        n = len(piece.split())
        if current and current_len + n > max_words:
            chunks.append("\n\n".join(current))
            # Carry trailing words forward as overlap.
            tail = " ".join("\n\n".join(current).split()[-overlap_words:]) if overlap_words else ""
            current = [tail] if tail else []
            current_len = len(tail.split())
        current.append(piece)
        current_len += n
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_document(doc: Document, max_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading, text in split_sections(doc.body):
        for piece in chunk_text(text, max_words=max_words, overlap_words=overlap_words):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    number=doc.number,
                    title=doc.title,
                    section=heading,
                    text=piece,
                )
            )
    return chunks


def chunk_corpus(docs: list[Document], max_words: int = 220, overlap_words: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_words=max_words, overlap_words=overlap_words))
    return out

# Maintenance pass: 2026-07-16 15:17
