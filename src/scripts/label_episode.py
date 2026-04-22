"""
Episode labeling tool.

Provides a lightweight CLI for attaching human-in-the-loop review
metadata to trajectory episodes.

Supports:
- creating and updating labels for episodes
- tracking review status (e.g., reviewed, needs_review)
- storing free-form notes for debugging or annotation

Labels are persisted as a JSON file and act as a simple curation
layer on top of the Parquet datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Resolve project root so paths work consistently regardless of where the script is run.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Label / review metadata location.
LABELS_PATH = Path("data_labels/episode_labels.json")

def load_labels() -> list[dict[str, Any]]:
    """
    Load existing episode labels if they exist.

    Returns an empty list when no label file has been created yet.
    """
    if not LABELS_PATH.exists():
        return []
    
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_labels(labels: list[dict[str, Any]]) -> None:
    """
    Persist the full episode label list to disk.
    """
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(labels, f, indent=2)
        f.write("\n")


def upsert_episode_label(
    *,
    session_id: str,
    episode_id: int,
    label: str,
    review_status: str,
    notes: str,
) -> None:
    """
    Insert or update review metadata for one episode.

    The combination (session_id, episode_id) acts as the unique key.
    """
    labels = load_labels()

    updated = False
    for row in labels:
        if row["session_id"] == session_id and int(row["episode_id"]) == episode_id:
            row["label"] = label
            row["review_status"] = review_status
            row["notes"] = notes
            updated = True
            break

    if not updated:
        labels.append({
            "session_id": session_id,
            "episode_id": int(episode_id),
            "label": label,
            "review_status": review_status,
            "notes": notes,
        })

    save_labels(labels)


def main() -> None:
    """
    Parse CLI arguments and label a single episode.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--session", type=str, required=True)
    parser.add_argument("--episode-id", type=int, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--review-status", type=str, default="reviewed")
    parser.add_argument("--notes", type=str, default="")

    args = parser.parse_args()

    upsert_episode_label(
        session_id=args.session,
        episode_id=args.episode_id,
        label=args.label,
        review_status=args.review_status,
        notes=args.notes,
    )

    print(
        f"Saved label: session={args.session} | "
        f"episode_id={args.episode_id} | "
        f"label={args.label} | "
        f"review_status={args.review_status}"
    )


if __name__ == "__main__":
    main()
