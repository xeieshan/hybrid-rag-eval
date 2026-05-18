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

