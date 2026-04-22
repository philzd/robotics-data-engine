"""
Parquet dataset builder.

Converts per-session derived artifacts into structured, tabular
datasets suitable for Parquet storage and downstream ML workflows.

The outputs include:
- frame-level aligned data
- episode-level trajectory segments
- session-level health summaries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

def _load_json(path: Path) -> dict[str, Any]:
    """
    Load a JSON file into a dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """
    Load a JSONL file into a list of dictionaries.
    """
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def build_episode_index(episodes_artifact: dict[str, Any]) -> dict[int, int]:
    """
    Build a mapping from frame_idx to episode_id.

    Each frame that belongs to a contiguous matched segment (episode)
    is assigned the corresponding episode_id. Frames that are missing
    or not part of any episode will not appear in the mapping.
    """
    frame_to_episode: dict[int, int] = {}

    for ep in episodes_artifact.get("episodes", []):
        episode_id = int(ep["episode_id"])
        start_idx = int(ep["start_frame_idx"])
        end_idx = int(ep["end_frame_idx"])

        for frame_idx in range(start_idx, end_idx + 1):
            frame_to_episode[frame_idx] = episode_id
    
    return frame_to_episode


def build_frame_rows(
        session_id: str,
        alignment_rows: list[dict[str, Any]],
        episodes_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Construct frame-level rows for the dataset.

    Each row represents a single video frame and its alignment outcome,
    including matched sensor data (if available), alignment error,
    and episode membership.

    Fields include:
    - session_id
    - frame_idx
    - t_frame
    - status (matched / missing)
    - sensor_idx
    - t_sensor
    - dt (time difference)
    - missing_reason
    - episode_id (if part of a valid episode)
    """
    frame_to_episode = build_episode_index(episodes_artifact)
    out: list[dict[str, Any]] = []

    for row in alignment_rows:
        frame_idx = int(row["frame_idx"])

        out.append({
            "session_id": session_id,
            "frame_idx": frame_idx,
            "t_frame": row.get("t_frame"),
            "status": row.get("status"),
            "sensor_idx": row.get("sensor_idx"),
            "t_sensor": row.get("t_sensor"),
            "dt": row.get("dt"),
            "missing_reason": row.get("missing_reason"),
            "episode_id": frame_to_episode.get(frame_idx),
        })

    return out


def build_episode_rows(
    session_id: str,
    episodes_artifact: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Construct episode-level rows for the dataset.

    Each row represents a contiguous segment of valid (matched) frames
    forming a trajectory suitable for training or evaluation.

    Fields include:
    - session_id
    - episode_id
    - start_frame_idx
    - end_frame_idx
    - length
    - start_t
    - end_t
    """
    out: list[dict[str, Any]] = []

    for ep in episodes_artifact.get("episodes", []):
        out.append({
            "session_id": session_id,
            "episode_id": int(ep["episode_id"]),
            "start_frame_idx": int(ep["start_frame_idx"]),
            "end_frame_idx": int(ep["end_frame_idx"]),
            "length": int(ep["length"]),
            "start_t": ep.get("start_t"),
            "end_t": ep.get("end_t"),
        })

    return out


def build_session_health_row(
    session_id: str,
    alignment_health: dict[str, Any],
    episode_health: dict[str, Any],
    alignment_warnings: dict[str, Any],
) -> dict[str, Any]:
    """
    Construct a session-level health summary row.

    Aggregates alignment and episode metrics into a single record
    describing dataset quality for the session.

    Includes:
    - counts and ratios (matched, missing)
    - alignment error statistics (mean, max, p95)
    - episode statistics (count, lengths, fragmentation)
    - overall status from validation/warnings
    """
    return {
        "session_id": session_id,
        "total_frames": alignment_health.get("total_frames"),
        "matched_count": alignment_health.get("matched_count"),
        "missing_count": alignment_health.get("missing_count"),
        "matched_ratio": alignment_health.get("matched_ratio"),
        "missing_ratio": alignment_health.get("missing_ratio"),
        "dt_abs_mean": alignment_health.get("dt_abs_mean"),
        "dt_abs_max": alignment_health.get("dt_abs_max"),
        "dt_abs_p95": alignment_health.get("dt_abs_p95"),
        "max_consecutive_missing": alignment_health.get("max_consecutive_missing"),
        "episode_count": episode_health.get("episode_count"),
        "mean_episode_length": episode_health.get("mean_episode_length"),
        "max_episode_length": episode_health.get("max_episode_length"),
        "fragmentation_score": episode_health.get("fragmentation_score"),
        "overall_status": alignment_warnings.get("overall_status"),
    }


def build_session_tables(session_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """
    Build all dataset tables for a single session.

    Loads derived artifacts from disk and produces:
    - frame-level rows
    - episode-level rows
    - session-level health summary

    Returns a dictionary with keys:
    - "frames"
    - "episodes"
    - "session_health"
    """
    session_id = session_dir.name
    derived_dir = session_dir / "derived"

    alignment_rows = _load_jsonl(derived_dir / "alignment_map.jsonl")
    episodes_artifact = _load_json(derived_dir / "episodes.json")
    alignment_health = _load_json(derived_dir / "alignment_health.json")
    episode_health = _load_json(derived_dir / "episode_health.json")
    alignment_warnings = _load_json(derived_dir / "alignment_warnings.json")

    frame_rows = build_frame_rows(session_id, alignment_rows, episodes_artifact)
    episode_rows = build_episode_rows(session_id, episodes_artifact)
    session_health_row = build_session_health_row(
        session_id,
        alignment_health,
        episode_health,
        alignment_warnings,
    )

    return {
        "frames": frame_rows,
        "episodes": episode_rows,
        "session_health": [session_health_row],
    }
