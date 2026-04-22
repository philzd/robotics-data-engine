"""
Episode health metrics.

Summarizes trajectory fragmentation and episode statistics.
"""

from typing import Dict, Any

def compute_episode_health(episode_artifact: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute episode-level health metrics from an episodes artifact.

    Metrics summarize trajectory fragmentation and episode length
    distribution across the dataset.
    """
    episodes = episode_artifact["episodes"]

    if not episodes:
        return {
            "episode_count": 0,
            "total_frames_in_episodes": 0,
            "mean_episode_length": 0.0,
            "max_episode_length": 0,
            "largest_episode_ratio": 0.0,
            "fragmentation_score": 1.0,
        }
    
    lengths = [ep["length"] for ep in episodes]

    episode_count = len(lengths)
    total_frames = sum(lengths)
    max_len = max(lengths)
    mean_len = total_frames / episode_count

    largest_ratio = max_len / total_frames if total_frames > 0 else 0.0

    # Simple fragmentation heuristic.
    fragmentation_score = 1.0 - largest_ratio

    return {
        "episode_count": episode_count,
        "total_frames_in_episodes": total_frames,
        "mean_episode_length": mean_len,
        "max_episode_length": max_len,
        "largest_episode_ratio": largest_ratio,
        "fragmentation_score": fragmentation_score,
    }
