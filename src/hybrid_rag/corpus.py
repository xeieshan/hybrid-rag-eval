"""Download and parse the Python Enhancement Proposals (PEPs) corpus.

PEPs are reStructuredText files with an RFC 822 style header block, e.g.::

    PEP: 572
    Title: Assignment Expressions
    Status: Final
    Python-Version: 3.8

The corpus is pinned to a specific commit of github.com/python/peps so that
evaluation numbers are reproducible.
"""

from __future__ import annotations

import io
import re
import tarfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

PEPS_REPO = "python/peps"
PEPS_COMMIT = "545c670f1e986162eb7161d63c944108e504bd09"
_PEP_FILE_RE = re.compile(r"^pep-(\d{4})\.rst$")


@dataclass(frozen=True)
class Document:
    doc_id: str
    number: int
    title: str
    body: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return self.metadata.get("Status", "")

    @property
    def python_version(self) -> str:
        return self.metadata.get("Python-Version", "")


def parse_header(text: str) -> tuple[dict[str, str], str]:
    """Split a PEP into (header fields, body). Continuation lines are joined."""
    lines = text.splitlines()
    meta: dict[str, str] = {}
    last_key: str | None = None
    end = len(lines)
    for idx, line in enumerate(lines):
        if not line.strip():
            end = idx
            break
        if line[:1] in (" ", "\t") and last_key is not None:
            meta[last_key] = f"{meta[last_key]} {line.strip()}".strip()
            continue
        key, sep, value = line.partition(":")
        if not sep:
            # Not a header line: treat the whole file as body.
            return {}, text
        last_key = key.strip()
        meta[last_key] = value.strip()
    body = "\n".join(lines[end + 1 :]).strip("\n")
    return meta, body


def parse_pep(path: Path) -> Document:
    match = _PEP_FILE_RE.match(path.name)
    if not match:
        raise ValueError(f"Not a PEP file name: {path.name}")
    number = int(match.group(1))
    meta, body = parse_header(path.read_text(encoding="utf-8"))
    title = meta.get("Title", path.stem)
    return Document(doc_id=f"pep-{number:04d}", number=number, title=title, body=body, metadata=meta)


def load_corpus(directory: str | Path) -> list[Document]:
    """Load every ``pep-NNNN.rst`` file in ``directory`` (searched recursively)."""
    directory = Path(directory)
    paths = sorted(p for p in directory.rglob("pep-*.rst") if _PEP_FILE_RE.match(p.name))
    if not paths:
        raise FileNotFoundError(f"No pep-NNNN.rst files found under {directory}")
    return [parse_pep(p) for p in paths]


def download_peps(dest: str | Path, commit: str = PEPS_COMMIT) -> Path:
    """Download the PEP sources at ``commit`` and extract the .rst files into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    url = f"https://codeload.github.com/{PEPS_REPO}/tar.gz/{commit}"
    with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 - fixed https URL
        payload = resp.read()
    count = 0
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as tar:
        for member in tar.getmembers():
            name = Path(member.name).name
            if member.isfile() and _PEP_FILE_RE.match(name):
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                (dest / name).write_bytes(extracted.read())
                count += 1
    if count == 0:
