"""
Artifact fingerprinting.

Create deterministic fingerprints for alignment and episode artifacts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .hashing import sha256_file

def compute_alignment_fingerprint(
    *,
    alignment_map_path: Path,
    alignment_health_path: Path,
    alignment_warnings_path: Path,
    alignment_examples_path: Path,
    episodes_path: Path,
    episode_invariant_path: Path,
    episode_health_path: Path,
    max_dt: float,
    policy: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Compute a deterministic fingerprint for derived artifacts and policy settings.

    The fingerprint records content hashes for produced artifacts together with
    key policy parameters used to generate them.
    """
    return {
        "artifacts": {
            "alignment_map_sha256": sha256_file(alignment_map_path),
            "alignment_health_sha256": sha256_file(alignment_health_path),
            "alignment_warnings_sha256": sha256_file(alignment_warnings_path),
            "alignment_examples_sha256": sha256_file(alignment_examples_path),
            "episodes_sha256": sha256_file(episodes_path),
            "episode_invariants_sha256": sha256_file(episode_invariant_path),
            "episode_health_sha256": sha256_file(episode_health_path),
        },
        "params": {
            "max_dt": float(max_dt),
        },
        "policy": policy,
    }
