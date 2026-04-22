"""
File hashing utilities.

Provides deterministic SHA-256 hashing for artifacts produced by the
data pipeline. Hashes are used for provenance tracking and reproducibility,
allowing derived artifacts to be fingerprinted and verified.

Important:
- Uses chunked reads to support large files (e.g., videos).
- Pure utility functions (no side effects beyond reading files).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Compute the SHA-256 hash of a file without loading it fully into memory.
    
    Files are read in chunks so large artifacts (e.g., videos) can be hashed
    safely and deterministically.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)

    return h.hexdigest()
