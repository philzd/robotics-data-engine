"""
Hard invariants for episodes.json.

This module:
- Validates episode structure and semantics.
- Returns structured results (no printing, no file I/O).

Fail-fast behavior is handled by the CLI layer.
"""

from __future__ import annotations
from typing import Any, Dict, List

def check_episode_invariants(episode_artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate structural invariants for an episodes artifact.

    Expected artifact schema:
    {
        "episode_count": int,
        "summary": {...},
        "episodes": [
            {
                "episode_id": int,
                "start_frame_idx": int,
                "end_frame_idx": int,
                "length": int,
                "start_t": float,
                "end_t": float
            }, ...
        ] 
    }

    Returns:
    {
        "passed": bool,
        "violation_count": int,
        "violations": [...]    
    }
    """
    violations: List[Dict[str, Any]] = []

    # Invariant 1: required top-level keys.
    for k in ["episode_count", "episodes", "summary"]:
        if k not in episode_artifact:
            violations.append({
                "type": "MISSING_TOP_LEVEL_KEY",
                "message": f"episodes artifact missing key '{k}'",
            })

    if violations:
        return _result(violations)
    
    episodes = episode_artifact["episodes"]
    declared_count = episode_artifact["episode_count"]

    # Invariant 2: episode_count matches len(episodes).
    if not isinstance(episodes, list):
        violations.append({
            "type": "EPISODES_NOT_LIST",
            "message": "episodes must be a list",
            "actual_type": type(episodes).__name__,
        })
        return _result(violations)
    
    if declared_count != len(episodes):
        violations.append({
            "type": "EPISODE_COUNT_MISMATCH",
            "message": "episode_count must match len(episodes)",
            "declared": declared_count,
            "actual": len(episodes)
        })

    # Empty episodes are allowed (all frames missing).
    if len(episodes) == 0:
        return _result(violations)
    
    # Invariant 3: per-episode required keys and basic types.
    required_episode_keys = [
        "episode_id",
        "start_frame_idx",
        "end_frame_idx",
        "length",
        "start_t",
        "end_t",
    ]

    for ep in episodes:
        for k in required_episode_keys:
            if k not in ep:
                violations.append({
                    "type": "MISSING_EPISODE_KEY",
                    "message": f"episode missing key '{k}'",
                    "key": k,
                    "episode": ep,
                })

    if violations:
        return _result(violations)

    # Invariant 4: episode_id contiguous starting at 0.
    for expected_id, ep in enumerate(episodes):
        if ep["episode_id"] != expected_id:
            violations.append({
                "type": "EPISODE_ID_NON_CONTIGUOUS",
                "message": "episode_id must be contiguous starting at 0",
                "expected": expected_id,
                "actual": ep["episode_id"],
            })
            break

    # Invariant 5: frame bounds, length correctness, and time monotonicity.
    for ep in episodes:
        s = ep["start_frame_idx"]
        e = ep["end_frame_idx"]
        length = ep["length"]
        st = ep["start_t"]
        et = ep["end_t"]

        if not isinstance(s, int) or not isinstance(e, int) or not isinstance(length, int):
            violations.append({
                "type": "FRAME_FIELDS_NOT_INT",
                "message": "start_frame_idx/end_frame_idx/length must be int",
                "episode_id": ep["episode_id"],
            })
            continue

        if s > e:
            violations.append({
                "type": "START_AFTER_END",
                "message": "start_frame_idx must be <= end_frame_idx",
                "episode_id": ep["episode_id"],
                "start_frame_idx": s,
                "end_frame_idx": e,
            })

        expected_len = e - s + 1
        if length != expected_len:
            violations.append({
                "type": "LENGTH_MISMATCH",
                "message": "length must equal end_frame_idx - start_frame_idx + 1",
                "episode_id": ep["episode_id"],
                "expected": expected_len,
                "actual": length,
            })

        # Time should not go backwards inside an episode.
        try:
            if float(st) > float(et):
                violations.append({
                    "type": "TIME_NON_MONOTONIC",
                    "message": "start_t must be <= end_t",
                    "episode_id": ep["episode_id"],
                    "start_t": st,
                    "end_t": et,
                })
        except Exception:
            violations.append({
                "type": "TIME_NOT_NUMERIC",
                "message": "start_t/end_t must be numeric",
                "episode_id": ep["episode_id"],
            })
    
    # Invariant 6: episodes must be ordered and non-overlapping.
    # Frame ranges should strictly increase.
    prev_end = None
    for ep in episodes:
        s = ep["start_frame_idx"]
        e = ep["end_frame_idx"]

        if prev_end is not None and s <= prev_end:
            violations.append({
                "type": "EPISODES_OVERLAP_OR_UNSORTED",
                "message": "episodes must be ordered and non-overlapping (start_frame_idx must be > previous end_frame_idx)",
                "episode_id": ep["episode_id"],
                "start_frame_idx": s,
                "previous_end_frame_idx": prev_end,
            })
            break
        
        prev_end = e
    
    return _result(violations)


def _result(violations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build a standardized episode invariant result payload.
    """
    return {
        "passed": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations[:50],
    }
