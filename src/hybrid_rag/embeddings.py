"""Embedding backends.

* ``LSAEmbedder``     TF-IDF + truncated SVD. Runs locally, no API key; a useful
                      dense baseline that shows how much an API model actually adds.
* ``LiteLLMEmbedder`` Any embedding model LiteLLM supports (OpenAI, Cohere,
                      Voyage, Mistral, Ollama, ...). Results are cached on disk.
* ``HashingEmbedder`` Deterministic, dependency-free; used by the tests.

All embedders return L2-normalised float32 matrices, so cosine similarity is a dot product.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer


def _field(obj: object, key: str):
    return obj[key] if isinstance(obj, dict) else getattr(obj, key)


def _normalize(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)

