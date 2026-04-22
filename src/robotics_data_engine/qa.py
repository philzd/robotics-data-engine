"""
Quality assurance reporting utilities.

This module generates machine-readable QA reports summarizing basic
properties of ingested data, such as video duration estimates and
sensor normalization statistics.

The QA report provides lightweight diagnostics and warnings without
mutating dataset artifacts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

def write_qa_report(
    output_path: Path,
    *,
    video_fps: float,
    video_frame_count: int,
    sensor_stats: Optional[Dict[str, Any]],
    warnings: Optional[list[str]] = None,
) -> None:
    """
    Write a QA report summarizing basic dataset statistics.

    The report captures:
    - video frame count and duration estimate
    - sensor normalization statistics (if sensors are provided)
    - optional warning messages

    The output is a deterministic JSON artifact written to derived/.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    report = {

        "video": {
            "fps": float(video_fps),
            "frame_count": int(video_frame_count),
            "duration_sec_est": (video_frame_count - 1) / video_fps if video_frame_count > 0 else 0.0,
        },
        "sensor": sensor_stats, # None if no sensor provided.
        "warnings": warnings or [],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
