"""
Episode inspection tool.

Provides a lightweight inspection workflow over the Parquet outputs.

Allows users to:
- inspect episode-level trajectory segments for a session
- filter episodes by minimum length
- surface session-level dataset health metrics
- show review and labeling metadata for human-in-the-loop curation
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

# Resolve project root so paths work consistently regardless of where the script is run.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Parquet dataset locations.
DATASETS_ROOT = PROJECT_ROOT / "datasets"
EPISODES_ROOT = DATASETS_ROOT / "episodes"
SESSION_HEALTH_PATH = DATASETS_ROOT / "session_health" / "part-000.parquet"

# Label / review metadata location.
LABELS_PATH = PROJECT_ROOT / "data_labels" / "episode_labels.json"

def load_episodes(session_id: str) -> pd.DataFrame:
    """
    Load the episode-level Parquet table for a given session.

    The table contains one row per contiguous matched trajectory segment.
    """
    path = EPISODES_ROOT / f"session_id={session_id}" / "part-000.parquet"

    if not path.exists():
        raise FileNotFoundError(f"No episodes found for session {session_id}")
    
    return pd.read_parquet(path)


def load_health() -> pd.DataFrame:
    """
    Load the global session health Parquet table.

    The table contains one row per session with dataset quality metrics.
    """
    return pd.read_parquet(SESSION_HEALTH_PATH)


def load_labels() -> list[dict[str, Any]]:
    """
    Load episode-level label and review metadata.

    Returns an empty list if no labels have been created yet.
    """
    if not LABELS_PATH.exists():
        return []

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_index(labels: list[dict[str, Any]]) -> dict[tuple[str, int], dict[str, Any]]:
    """
    Build a lookup index for labels by (session_id, episode_id).

    This allows inspection output to quickly attach label / review metadata
    to each episode row.
    """
    return {
        (str(row["session_id"]), int(row["episode_id"])): row
        for row in labels
    }


def inspect_episodes(session_id: str, min_length: int = 0) -> None:
    """
    Print a readable summary of episodes for a session.

    Optionally filters out short episodes using min_length. If labels
    are available, they are displayed alongside each episode.
    """
    episodes = load_episodes(session_id)
    labels = load_labels()
    label_index = build_label_index(labels)

    # Optional filter to focus on longer, potentially more useful episodes.
    if min_length > 0:
        episodes = episodes[episodes["length"] >= min_length]

    print(f"\n=== Episodes for session: {session_id} ===\n")

    for _, row in episodes.iterrows():
        key = (session_id, int(row["episode_id"]))
        label_info = label_index.get(key)

        label = label_info["label"] if label_info else "UNLABELED"
        review_status = label_info["review_status"] if label_info else "-"
        notes = label_info["notes"] if label_info else ""

        print(
            f"episode_id={row['episode_id']} | "
            f"frames={row['start_frame_idx']}->{row['end_frame_idx']} | "
            f"length={row['length']} | "
            f"time={row['start_t']:.2f}->{row['end_t']:.2f} | "
            f"label={label} | "
            f"status={review_status} | "
            f"notes={notes}"
        )


def show_health() -> None:
    """
    Print a readable summary of session-level health metrics.

    Prints all session health rows.
    """
    health = load_health()

    print("\n=== Session Health ===\n")

    for _, row in health.iterrows():
        print(
            f"session={row['session_id']} | "
            f"missing_ratio={row['missing_ratio']:.3f} | "
            f"p95={row['dt_abs_p95']:.4f} | "
            f"fragmentation={row['fragmentation_score']:.3f} | "
            f"status={row['overall_status']}"
        )


def main() -> None:
    """
    Parse CLI arguments and run episode inspection.

    Required:
    - --session : session_id to inspect

    Optional:
    - --min-length : filter out short episodes
    - --show-health : also print session health summary
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--min-length", type=int, default=0)
    parser.add_argument("--show-health", action="store_true")

    args = parser.parse_args()

    # Print episode summaries for the requested session.
    inspect_episodes(args.session, args.min_length)

    # Optionally print health metrics for all sessions.
    if args.show_health:
        show_health()


if __name__ == "__main__":
    main()
