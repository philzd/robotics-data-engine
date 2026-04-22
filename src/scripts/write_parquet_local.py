"""
Local Parquet writer.

Scans session artifacts and materializes structured Parquet datasets
for downstream analytics and ML workflows.

Outputs:
- frame-level Parquet tables partitioned by session_id
- episode-level Parquet tables partitioned by session_id
- a global session_health Parquet table
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from robotics_data_engine.parquet_builder import build_session_tables

def main() -> None:
    """
    Build local Parquet datasets from all sessions with derived artifacts.

    The script expects each session to already contain the outputs required
    to build:
    - frames
    - episodes
    - session_health
    """
    # Root directories for input session artifacts and output datasets.
    sessions_root = Path("sessions")
    datasets_root = Path("datasets")

    if not sessions_root.exists():
        raise FileNotFoundError(f"Sessions directory not found: {sessions_root}")

    # Output dataset locations.
    frames_out = datasets_root / "frames"
    episodes_out = datasets_root / "episodes"
    session_health_out = datasets_root / "session_health"

    # Ensure top-level output directories exist.
    frames_out.mkdir(parents=True, exist_ok=True)
    episodes_out.mkdir(parents=True, exist_ok=True)
    session_health_out.mkdir(parents=True, exist_ok=True)

    # Accumulate all session-level health rows into one global table.
    all_health_rows = []

    # Iterate through every session directory.
    for session_dir in sorted(sessions_root.iterdir()):
        if not session_dir.is_dir():
            continue

        derived_dir = session_dir / "derived"
        if not derived_dir.exists():
            continue

        # Only process sessions that already have the required derived artifacts.
        if not (derived_dir / "alignment_map.jsonl").exists():
            continue
        if not (derived_dir / "episodes.json").exists():
            continue
        if not (derived_dir / "alignment_health.json").exists():
            continue
        if not (derived_dir / "episode_health.json").exists():
            continue
        if not (derived_dir / "alignment_warnings.json").exists():
            continue

        # Convert one session's artifacts into tabular row dictionaries.
        tables = build_session_tables(session_dir)

        session_id = session_dir.name

        # Convert row dictionaries into DataFrames for Parquet writing.
        frames_df = pd.DataFrame(tables["frames"])
        episodes_df = pd.DataFrame(tables["episodes"])

        # Partition-style output paths for per-session tables.
        session_frames_out = frames_out / f"session_id={session_id}"
        session_episodes_out = episodes_out / f"session_id={session_id}"

        # Ensure partition directories exist before writing Parquet.
        session_frames_out.mkdir(parents=True, exist_ok=True)
        session_episodes_out.mkdir(parents=True, exist_ok=True)

        # Write frame-level and episode-level tables for this session.
        frames_df.to_parquet(session_frames_out / "part-000.parquet", index=False)
        episodes_df.to_parquet(session_episodes_out / "part-000.parquet", index=False)

        # Collect the health row for the global session_health table.
        all_health_rows.extend(tables["session_health"])

        print(f"Wrote session: {session_id}")

    # Write a single combined session health table across all processed sessions.
    if all_health_rows:
        all_health_df = pd.DataFrame(all_health_rows)
        all_health_df.to_parquet(session_health_out / "part-000.parquet", index=False)
        print("Wrote session_health table")


if __name__ == "__main__":
    main()
