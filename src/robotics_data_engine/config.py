"""
Configuration constants for robotics_data_engine.

Centralizes directory names, standard filenames, and artifact names so
the session contract remains consistent across the codebase and README.
"""
from dataclasses import dataclass


# Session directory names.
SESSIONS_DIRNAME = "sessions"
RAW_DIRNAME = "raw"
DERIVED_DIRNAME = "derived"
MANIFESTS_DIRNAME = "manifests"
LOGS_DIRNAME = "logs"

# Standard filenames.
DEFAULT_VIDEO_FILENAME = "video.mp4"
DEFAULT_SENSOR_FILENAME = "sensor.csv"

VIDEO_TIMESTAMPS_FILENAME = "video_timestamps.jsonl"
SENSOR_NORMALIZED_FILENAME = "sensor_normalized.csv"
QA_REPORT_FILENAME = "qa_report.json"

SESSION_MANIFEST_FILENAME = "session_manifest.json"
INGEST_LOG_FILENAME = "ingest.log"

# Policy defaults.
VIDEO_TIMESTAMP_POLICY = "frame_idx_over_fps"

# Alignment artifacts.
ALIGNMENT_MAP_FILENAME = "alignment_map.jsonl"
ALIGNMENT_REPORT_FILENAME = "alignment_report.json"
ALIGNMENT_INVARIANTS_FILENAME = "alignment_invariants.json"
ALIGNMENT_HEALTH_FILENAME = "alignment_health.json"
ALIGNMENT_WARNINGS_FILENAME = "alignment_warnings.json"
ALIGNMENT_FINGERPRINT_FILENAME = "alignment_fingerprint.json"
ALIGNMENT_EXAMPLES_FILENAME = "alignment_examples.json"

# Episode artifacts.
EPISODES_FILENAME = "episodes.json"
EPISODE_INVARIANTS_FILENAME = "episode_invariants.json"
EPISODE_HEALTH_FILENAME = "episode_health.json"

@dataclass(frozen=True)
class SessionLayout:
    """
    Convenience bundle for session layout constants.

    Not required, but useful when passing grouped directory names around.
    """
    sessions_dirname: str = SESSIONS_DIRNAME
    raw_dirname: str = RAW_DIRNAME
    derived_dirname: str = DERIVED_DIRNAME
    manifests_dirname: str = MANIFESTS_DIRNAME
    logs_dirname: str = LOGS_DIRNAME
