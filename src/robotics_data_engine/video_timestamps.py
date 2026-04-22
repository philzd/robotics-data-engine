"""
Video timestamp generation.

This module produces deterministic frame timestamps for a video
recording using the policy: timestamp = frame_idx / fps

The output is written as a JSONL file where each row corresponds to a
video frame.
"""

from pathlib import Path
import json

def write_video_timestamps(
    raw_video_path: Path,
    output_path: Path,
    fps: float,
    frame_count: int,
) -> None:
    """
    Write deterministic video frame timestamps as JSONL.

    Each line contains:
    {
        "frame_idx": int,
        "timestamp_sec": float
    }

    The timestamp is computed as frame_idx / fps.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # raw_video_path is accepted so the function signature stays aligned with
    # the broader ingest pipeline, even though timestamp generation here only
    # depends on fps and frame_count.
    _ = raw_video_path

    with open(output_path, "w", encoding="utf-8") as f:
        for frame_idx in range(frame_count):
            timestamp_sec = frame_idx / fps
            row = {
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
            }
            f.write(json.dumps(row) + "\n")
