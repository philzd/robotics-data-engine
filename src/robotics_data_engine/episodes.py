"""
Episode extraction from alignment_map rows.

An episode is a contiguous sequence of frames where status == "matched".
Episodes represent training-ready trajectory segments for downstream
ML pipelines.
"""
from __future__ import annotations
from typing import List, Dict, Any


def compute_episodes(
        alignment_rows: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Extract contiguous matched segments from alignment_map rows.

    Returns a list of episode dictionaries containing frame bounds,
    duration, and timestamps for each matched trajectory segment.
    """
    episodes = []

    current_start = None
    episode_id = 0

    for row in alignment_rows:
        frame_idx = row["frame_idx"]
        status = row["status"]
        t_frame = row["t_frame"]

        if status == "matched":
            # Start new episode.
            if current_start is None:
                current_start = {
                    "start_frame_idx": frame_idx,
                    "start_t": t_frame,
                }
            
            last_frame_idx = frame_idx
            last_t = t_frame
        
        else:
            # Close episode if inside one.
            if current_start is not None:
                episode = {
                    "episode_id": episode_id,
                    "start_frame_idx": current_start["start_frame_idx"],
                    "end_frame_idx": last_frame_idx,
                    "length": last_frame_idx - current_start["start_frame_idx"] + 1,
                    "start_t": current_start["start_t"],
                    "end_t": last_t,
                }

                episodes.append(episode)

                episode_id += 1
                current_start = None
        
    # Handle episode that reaches end of file.
    if current_start is not None:
        episode = {
            "episode_id": episode_id,
            "start_frame_idx": current_start["start_frame_idx"],
            "end_frame_idx": last_frame_idx,
            "length": last_frame_idx - current_start["start_frame_idx"] + 1,
            "start_t": current_start["start_t"],
            "end_t": last_t,
        }

        episodes.append(episode)
    
    return episodes

def compute_episodes_summary(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute deterministic summary statistics over extracted episodes.
    """
    lengths = [int(ep["length"]) for ep in episodes]

    if not lengths:
        return {
            "episode_count": 0,
            "total_frames_in_episodes": 0,
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0.0,
        }

    total = sum(lengths)
    return {
        "episode_count": int(len(lengths)),
        "total_frames_in_episodes": int(total),
        "min_length": int(min(lengths)),
        "max_length": int(max(lengths)),
        "mean_length": float(total / len(lengths)),
    }

def build_episodes_artifact(
        alignment_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Build the episodes.json payload:
    {
        "episode_count": int,
        "summary": {...},
        "episodes": [...]
    }
    """
    episodes = compute_episodes(alignment_rows)
    summary = compute_episodes_summary(episodes)

    return {
        "episode_count": summary["episode_count"],
        "summary": summary,
        "episodes": episodes,
    }
