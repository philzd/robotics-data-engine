"""
Session path contract.

A Session represents one robot run / one log bundle and provides
path helpers for the on-disk session layout defined in the README.
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from . import config

@dataclass(frozen=True)
class Session:
    """
    Immutable session identity + path helpers.

    Why frozen?
    - Session identity should not change after construction.
    - Makes accidental mutation harder (good for reproducibility).
    """
    session_id: str
    root: Path  # Root directory where session lives, e.g. Path("sessions").

    @staticmethod
    def from_root(session_id: str, root: str | Path = config.SESSIONS_DIRNAME) -> "Session":
        """Convenience constructor that accepts a string root."""
        return Session(session_id=session_id, root=Path(root))
    
    @property
    def session_dir(self) -> Path:
        return self.root / self.session_id
    
    @property
    def raw_dir(self) -> Path:
        return self.session_dir / config.RAW_DIRNAME
    
    @property
    def derived_dir(self) -> Path:
        return self.session_dir / config.DERIVED_DIRNAME
    
    @property
    def manifests_dir(self) -> Path:
        return self.session_dir / config.MANIFESTS_DIRNAME
    
    @property
    def logs_dir(self) -> Path:
        return self.session_dir / config.LOGS_DIRNAME
    
    @property
    def manifest_path(self) -> Path:
        return self.manifests_dir / config.SESSION_MANIFEST_FILENAME
    
    @property
    def qa_report_path(self) -> Path:
        return self.derived_dir / config.QA_REPORT_FILENAME
    
    @property
    def video_timestamps_path(self) -> Path:
        return self.derived_dir / config.VIDEO_TIMESTAMPS_FILENAME
    
    @property
    def sensor_normalized_path(self) -> Path:
        return self.derived_dir / config.SENSOR_NORMALIZED_FILENAME
    
    @property
    def alignment_map_path(self) -> Path:
        return self.derived_dir / config.ALIGNMENT_MAP_FILENAME
    
    @property
    def alignment_report_path(self) -> Path:
        return self.derived_dir / config.ALIGNMENT_REPORT_FILENAME
    
    @property
    def alignment_invariants_path(self) -> Path:
        """
        Path to derived/alignment_invariants.json

        This artifact records hard invariant checks over alignment_map.jsonl.
        """
        return self.derived_dir / config.ALIGNMENT_INVARIANTS_FILENAME
    
    @property
    def alignment_health_path(self) -> Path:
        """
        Path to derived/alignment_health.json
        
        This artifact summarizes alignment quality signals computed from
        alignment_map.jsonl.
        """
        return self.derived_dir / config.ALIGNMENT_HEALTH_FILENAME
    
    @property
    def alignment_warnings_path(self) -> Path:
        """
        Path to derived/alignment_warnings.json

        This artifact records structured warnings and failure signals derived
        from alignment health metrics.
        """
        return self.derived_dir / config.ALIGNMENT_WARNINGS_FILENAME
    
    @property
    def alignment_fingerprint_path(self) -> Path:
        """
        Path to manifests/alignment_fingerprint.json
        """
        return self.manifests_dir / config.ALIGNMENT_FINGERPRINT_FILENAME
    
    @property
    def alignment_examples_path(self) -> Path:
        """
        Path to derived/alignment_examples.json

        This file contains small explainability samples:
        - Worst matched frames by abs(dt).
        - Longest missing streaks.
        - Example missing rows per missing_reason.
        """
        return self.derived_dir / config.ALIGNMENT_EXAMPLES_FILENAME
    
    @property
    def episodes_path(self) -> Path:
        """
        Path to derived/episodes.json.
        """
        return self.derived_dir / config.EPISODES_FILENAME
    
    @property
    def episode_invariants_path(self) -> Path:
        """
        Path to derived/episode_invariants.json.

        Records validation results for episode artifacts.
        """
        return self.derived_dir / config.EPISODE_INVARIANTS_FILENAME
    
    @property
    def episode_health_path(self) -> Path:
        """
        Path to derived/episode_health.json.

        Records episode-level health and fragmentation metrics.
        """
        return self.derived_dir / config.EPISODE_HEALTH_FILENAME
    
    def create_dirs(self, overwrite: bool = False) -> None:
        """
        Create the session directory structure.

        When overwrite=False, refuse to continue if the session directory
        already exists. When overwrite=True, reuse the existing session
        directory without deleting contents.

        overwrite=True:
        - Allows using an existing session directory (use carefully).
        - Does not delete anything; it only ensures directories exist.
        """
        if self.session_dir.exists() and not overwrite:
            raise FileExistsError(
                f"Session already exists: {self.session_dir}."
                "Refusing to overwrite. Choose a new --session id."
            )
        
        # Create required directories (idempotent if already present).
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.derived_dir.mkdir(parents=True, exist_ok=True)
        self.manifests_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
