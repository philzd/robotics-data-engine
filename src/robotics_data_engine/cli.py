"""
Command-line interface.

This module defines:
- The top-level Typer application.
- CLI commands for dataset construction (ingest, align, validate, dataset-summary).
- Minimal argument validation and orchestration of dataset pipeline steps.

Important:
- The CLI only orchestrates the pipeline.
- Core data processing logic lives in dedicated modules (alignment, episodes, invariants, health, fingerprinting).
"""

import json
import shutil
import typer
from datetime import datetime, timezone
from pathlib import Path
from .session import Session
from .video_timestamps import write_video_timestamps
from .sensor_normalize import normalize_sensor_csv
from .qa import write_qa_report
from .hashing import sha256_file
from .alignment import (
    load_video_timestamps,
    load_sensor_times,
    align_nearest,
    write_alignment_map,
    write_alignment_report,
    compute_alignment_examples,
)
from .invariants import check_alignment_invariants
from .io_utils import read_jsonl, write_json
from .health import compute_alignment_health
from .warnings import compute_alignment_warnings
from .fingerprint import compute_alignment_fingerprint
from .episodes import build_episodes_artifact
from .episode_invariants import check_episode_invariants
from .episode_health import compute_episode_health

__version__ = "0.1.0"

app = typer.Typer(help="robotics_data_engine: multimodal robotics data engine.")

def _require_file(path_str: str, label: str) -> None:
    """
    Validate that a CLI argument points to an existing file.

    Raises a Typer error for clean CLI messaging.
    Does not read or parse the file.
    """
    p = Path(path_str)
    if not p.exists():
        raise typer.BadParameter (f"{label} file not found: {path_str}")
    if not p.is_file():
        raise typer.BadParameter(f"{label} path is not a file: {path_str}")


def _copy_to_raw(src: str, raw_dir: Path) -> Path:
    """
    Copy a file into the session's raw/ directory.

    The raw/ directory stores immutable copies of the original inputs.
    The original filename is preserved.

    shutil.copy2() retains file metadata (e.g., modification time),
    which is useful for debugging and provenance tracking.
    """
    src_path = Path(src)
    dst_path = raw_dir / src_path.name
    shutil.copy2(src_path, dst_path)
    return dst_path


def _write_manifest(session_obj: Session, copied: dict[str, Path], *, fps:float, frame_count: int) -> None:
    """
    Write the session manifest used for provenance tracking.

    The manifest records session identity, input file hashes,
    and extraction metadata needed for reproducibility.
    """
    manifest = {
        "session_id": session_obj.session_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "robotics_data_engine_version": __version__,
        "inputs": {
            k: {"path": str(v), "sha256": sha256_file(v)}
            for k, v in copied.items()
        },
        "extraction": {
            "video_timestamp_policy": "frame_idx/fps",
            "fps": float(fps),
            "frame_count": int(frame_count),
        },
        "warnings": [],
    }

    # Make sure manifests/ exists (Session class already created this -> backup).
    session_obj.manifests_dir.mkdir(parents=True, exist_ok=True)

    with open(session_obj.manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


@app.command()
def version():
    """
    Print the tool version.
    
    Kept as a separate command so tooling and users
    can quickly verify which version produced a dataset.
    """
    typer.echo(f"robotics_data_engine {__version__}")


@app.command()
def ingest(
    session: str = typer.Option(..., "--session", "-s", help="Session ID (e.g., demo_01)."),
    video: str | None = typer.Option(..., "--video", help="Path to input video (e.g., video.mp4)."),
    sensor: str | None = typer.Option(None, "--sensor", help="Path to sensor CSV (e.g., imu.csv)."),
    root: str = typer.Option("sessions", "--root", help="Root directory for sessions."),
):
    """
    Ingest raw inputs into a robotics_data_engine session.

    Creates the session directory structure, copies raw inputs into immutable
    storage, and generates deterministic derived artifacts and manifests.
    """
    if video is not None:
        _require_file(video, "video")
    if sensor is not None:
        _require_file(sensor, "sensor")
    
    session_obj = Session.from_root(session_id=session, root=root)
    # Create on-disk session folder structure (safe by default, no overwrite).
    session_obj.create_dirs(overwrite=False)

    # Copy raw inputs into raw/ (immutable snapshots of originals).
    copied = {}
    if video is not None:
        copied["video"] = _copy_to_raw(video, session_obj.raw_dir)
    if sensor is not None:
        copied["sensor"] = _copy_to_raw(sensor, session_obj.raw_dir)

    # Temporary deterministic video metadata used for demo dataset construction.
    # Later this should be read from the actual video file.
    FPS = 30.0
    FRAME_COUNT = 100

    write_video_timestamps(
        raw_video_path=copied["video"],
        output_path=session_obj.video_timestamps_path,
        fps=FPS,
        frame_count=FRAME_COUNT,
    )
    typer.echo(f"  wrote video timestamps: {session_obj.video_timestamps_path}")

    # Optional: normalize sensor timestamps to a canonical t_sec column.
    sensor_stats = None
    if sensor is not None:
        sensor_stats = normalize_sensor_csv(
            raw_sensor_path=copied["sensor"],
            output_path=session_obj.sensor_normalized_path,
        )
        typer.echo(f"  wrote sensor normalized: {session_obj.sensor_normalized_path}")

    # Emit machine-readable QA report (no mutation -> just signals).
    warnings: list[str] = []
    write_qa_report(
        session_obj.qa_report_path,
        video_fps=FPS,
        video_frame_count=FRAME_COUNT,
        sensor_stats=sensor_stats,
        warnings=warnings,
    )
    typer.echo(f"  wrote QA report: {session_obj.qa_report_path}")

    # Write the session manifest last (receipt of what was produced).
    _write_manifest(session_obj, copied, fps=FPS, frame_count=FRAME_COUNT)
    typer.echo(f"  wrote manifest: {session_obj.manifest_path}")
    
    # Print summary.
    typer.echo("INGEST")
    typer.echo(f"  session: {session_obj.session_id}")
    typer.echo(f"  session_dir: {session_obj.session_dir}")
    for k, v in copied.items():
        typer.echo(f"  copied {k}: {v}")


@app.command()
def align(
    session: str = typer.Option(..., "--session", "-s", help="Session ID (e.g., demo_01)."),
    root: str = typer.Option("sessions", "--root", help="Root directory for sessions."),
    max_dt: float = typer.Option(0.05, "--max-dt", help="Max allowed abs time delta (sec) for a match."),
):
    """
    Align multimodal data within a session.

    Reads canonical video timestamps and normalized sensor timestamps,
    then produces alignment artifacts, validation outputs, episode artifacts,
    health metrics, and a deterministic fingerprint.
    """
    session_obj = Session.from_root(session_id=session, root=root)

    # Make sure required derived inputs exist.
    if not session_obj.video_timestamps_path.exists():
        raise typer.BadParameter(
            f"Missing required file: {session_obj.video_timestamps_path}. Run ingest first."
        )
    if not session_obj.sensor_normalized_path.exists():
        raise typer.BadParameter(
            f"Missing required file: {session_obj.sensor_normalized_path}. Run ingest with --sensor first."
        )
    
    video_ts = load_video_timestamps(session_obj.video_timestamps_path)
    sensor_ts = load_sensor_times(session_obj.sensor_normalized_path)

    # Run nearest-neighbor alignment.
    rows, report = align_nearest(video_ts, sensor_ts, max_dt=max_dt)

    # Write artifacts (inspectable outputs).
    write_alignment_map(session_obj.alignment_map_path, rows)
    write_alignment_report(session_obj.alignment_report_path, report)

    # Load the alignment map rows (JSONL -> list[dict]).
    alignment_rows = read_jsonl(session_obj.alignment_map_path)
    # Validate invariants (hard contract).
    invariants = check_alignment_invariants(
        alignment_rows,
        max_dt_threshold=max_dt,    # Use the local variable passed to align().
    )
    # Write invariants report to derived/.
    write_json(session_obj.alignment_invariants_path, invariants)
    # Fail fast if invariants do not pass.
    # Stop the pipeline condition.
    if not invariants["passed"]:
        typer.echo("❌ Alignment invariants FAILED.")
        typer.echo(f"  violation_count: {invariants['violation_count']}")
        for v in invariants["violations"][:5]:
            typer.echo(f"  - {v.get('type')}: {v.get('message')} (frame_idx={v.get('frame_idx')})")
        raise typer.Exit(code=1)

    typer.echo("✅ Alignment invariants PASSED.")
    examples = compute_alignment_examples(alignment_rows, top_k=20)
    write_json(session_obj.alignment_examples_path, examples)
    typer.echo(f"  wrote examples: {session_obj.alignment_examples_path}")

    # Episode extraction from aligned frames.
    episode_artifact = build_episodes_artifact(alignment_rows)
    write_json(session_obj.episodes_path, episode_artifact)
    typer.echo(f"  wrote episodes: {session_obj.episodes_path}")
    typer.echo(f"  episode_count: {episode_artifact['episode_count']}")

    # Episode invariants (fail-fast contract).
    episode_invariants = check_episode_invariants(episode_artifact)

    write_json(session_obj.episode_invariants_path, episode_invariants)

    if not episode_invariants["passed"]:
        typer.echo(f"❌ Episode invariants FAILED.")
        typer.echo(f"  violation_count: {episode_invariants['violation_count']}")
        for v in episode_invariants["violations"][:5]:
            typer.echo(f"  - {v.get('type')}: {v.get('message')}")
        raise typer.Exit(code=1)
    
    typer.echo("✅ Episode invariants PASSED.")
    typer.echo(f"  wrote episode invariants: {session_obj.episode_invariants_path}")

    # Episode health / fragmentation metrics.
    episode_health = compute_episode_health(episode_artifact)
    write_json(session_obj.episode_health_path, episode_health)

    typer.echo(f"  wrote episode health: {session_obj.episode_health_path}")
    typer.echo(f"  episode_count: {episode_health['episode_count']}")
    typer.echo(f"  mean_episode_length: {episode_health['mean_episode_length']}")
    typer.echo(f"  max_episode_length: {episode_health['max_episode_length']}")
    typer.echo(f"  fragmentation_score: {episode_health['fragmentation_score']}")

    # Compute alignment health signals.
    health = compute_alignment_health(alignment_rows)
    write_json(session_obj.alignment_health_path, health)

    typer.echo(f"  wrote health: {session_obj.alignment_health_path}")
    typer.echo(f"  matched_ratio: {health['matched_ratio']}")
    typer.echo(f"  missing_ratio: {health['missing_ratio']}")
    typer.echo(f"  dt_abs_p95: {health['dt_abs_p95']}")
    typer.echo(f"  max_consecutive_missing: {health['max_consecutive_missing']}")

    # Warning semantics + overall status.
    warn_report = compute_alignment_warnings(
        health,
        max_dt_threshold=max_dt # Use CLI value.
    )
    write_json(session_obj.alignment_warnings_path, warn_report)

    typer.echo(f"  wrote warnings: {session_obj.alignment_warnings_path}")
    typer.echo(f"  overall_status: {warn_report['overall_status']}")

    # Add overall_status into alignment_report.json (quick visibility).
    report_path = session_obj.alignment_report_path

    with open(report_path, "r", encoding="utf-8") as f:
        report_obj = json.load(f)

    report_obj["overall_status"] = warn_report["overall_status"]

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, indent=2)
        f.write("\n")

    # Compute a deterministic fingerprint for the derived dataset artifacts.
    fingerprint = compute_alignment_fingerprint(
        alignment_map_path=session_obj.alignment_map_path,
        alignment_health_path=session_obj.alignment_health_path,
        alignment_warnings_path=session_obj.alignment_warnings_path,
        alignment_examples_path=session_obj.alignment_examples_path,
        episodes_path=session_obj.episodes_path,
        episode_invariant_path=session_obj.episode_invariants_path,
        episode_health_path=session_obj.episode_health_path,
        max_dt=max_dt,
        policy=warn_report["policy"],
    )
    write_json(session_obj.alignment_fingerprint_path, fingerprint)
    typer.echo(f"  wrote fingerprint: {session_obj.alignment_fingerprint_path}")
    
    typer.echo("PIPELINE SUMMARY")
    typer.echo(f"  session: {session_obj.session_id}")
    typer.echo(f"  wrote alignment map: {session_obj.alignment_map_path}")
    typer.echo(f"  wrote alignment report: {session_obj.alignment_report_path}")
    typer.echo(f"  matched: {report.matched_count}")
    typer.echo(f"  missing: {report.missing_count}")
    typer.echo(f"  dt_abs_mean: {report.dt_abs_mean}")
    typer.echo(f"  dt_abs_max: {report.dt_abs_max}")
    typer.echo(f"  max_dt_threshold: {report.max_dt_threshold}")
    typer.echo(f"  wrote invariants: {session_obj.alignment_invariants_path}")
    if report.warnings:
        typer.echo(f"  warnings: {', '.join(report.warnings)}")


@app.command()
def align_all(
    root: str = typer.Option("sessions", "--root", help="Root directory for sessions."),
    max_dt: float = typer.Option(0.05, "--max-dt", help="Max allowed abs time delta."),
):
    """
    Run alignment on all sessions under the root directory.
    """
    root_path = Path(root)

    if not root_path.exists():
        typer.echo(f"root directory not found: {root_path}")
        raise typer.Exit(code=1)
    
    session_dirs = [
        p for p in root_path.iterdir()
        if p.is_dir()
    ]

    typer.echo("ALIGN-ALL")
    typer.echo(f"  found_sessions: {len(session_dirs)}")

    success = 0
    failed = 0
    failed_sessions: list[str] = []

    for s in sorted(session_dirs):
        session_id = s.name

        try:
            typer.echo(f"  running align for: {session_id}")

            align(
                session=session_id,
                root=root,
                max_dt=max_dt,
            )

            success += 1
        
        except Exception as e:
            typer.echo(f"  ✖ failed: {session_id} ({e})")
            failed += 1
            failed_sessions.append(session_id)
    
    typer.echo("")
    typer.echo("ALIGN-ALL SUMMARY")
    typer.echo(f"  success: {success}")
    typer.echo(f"  failed: {failed}")

    if failed_sessions:
        typer.echo("  failed_sessions:")
        for s in failed_sessions:
            typer.echo(f"    - {s}")


@app.command()
def validate(
    session: str = typer.Option(..., "--session", "-s", help="Session ID to validate."),
    root: str = typer.Option("sessions", "--root", help="Root directory for sessions."),
    max_dt: float = typer.Option(0.05, "--max-dt", help="Max allowed abs time delta (sec)."),
):
    """
    Validate existing derived artifacts for a session without rebuilding them.
    """
    session_obj = Session.from_root(session_id=session, root=root)

    # Required artifacts.
    if not session_obj.alignment_map_path.exists():
        raise typer.BadParameter(
            f"Missing required file: {session_obj.alignment_map_path}. Run align first."
        )
    
    if not session_obj.episodes_path.exists():
        raise typer.BadParameter(
            f"Missing required file: {session_obj.episodes_path}. Run align first."
        )
    
    # Load artifacts.
    alignment_rows = read_jsonl(session_obj.alignment_map_path)

    with open(session_obj.episodes_path, "r", encoding="utf-8") as f:
        episode_artifact = json.load(f)
    
    # Re-run invariant checks.
    alignment_invariants = check_alignment_invariants(
        alignment_rows,
        max_dt_threshold=max_dt,
    )

    episode_invariants = check_episode_invariants(episode_artifact)

    # Print summary.
    typer.echo("VALIDATE")
    typer.echo(f"  session: {session_obj.session_id}")

    if alignment_invariants["passed"]:
        typer.echo("  ✅ alignment invariants PASSED")
    else:
        typer.echo("  ❌ alignment invariants FAILED")
        typer.echo(f"    violation_count: {alignment_invariants['violation_count']}")

    if episode_invariants["passed"]:
        typer.echo("  ✅ episode invariants PASSED")
    else:
        typer.echo("  ❌ episode invariants FAILED")
        typer.echo(f"    violation_count: {episode_invariants['violation_count']}")

    # Fail command if either invariant set fails.
    if not alignment_invariants["passed"] or not episode_invariants["passed"]:
        raise typer.Exit(code=1)

    typer.echo("  overall: PASSED")


@app.command()
def dataset_summary(
    root: str = typer.Option("sessions", "--root", help="Root directory for sessions.")
):
    """
    Summarize dataset health across all sessions with derived artifacts.
    """
    root_path = Path(root)

    if not root_path.exists():
        typer.echo(f"root directory not found: {root_path}")
        raise typer.Exit(code=1)
    
    session_dirs = [p for p in root_path.iterdir() if p.is_dir()]

    sessions_with_health = 0
    total_frames = 0
    total_matched_frames = 0
    total_missing_frames = 0
    matched_ratios: list[float] = []

    total_episodes = 0
    episode_lengths: list[float] = []

    best_session = None
    best_matched_ratio = None
    worst_session = None
    worst_matched_ratio = None

    for s in sorted(session_dirs):
        session_id = s.name

        alignment_health_path = s / "derived" / "alignment_health.json"
        episode_health_path = s / "derived" / "episode_health.json"

        if not alignment_health_path.exists():
            continue

        with open(alignment_health_path, "r", encoding="utf-8") as f:
            health = json.load(f)

        sessions_with_health += 1
        total_frames += int(health.get("total_frames", 0))
        total_matched_frames += int(health.get("matched_count", 0))
        total_missing_frames += int(health.get("missing_count", 0))

        matched_ratio = float(health.get("matched_ratio", 0.0))
        matched_ratios.append(matched_ratio)

        if best_matched_ratio is None or matched_ratio > best_matched_ratio:
            best_matched_ratio = matched_ratio
            best_session = session_id

        if worst_matched_ratio is None or matched_ratio < worst_matched_ratio:
            worst_matched_ratio = matched_ratio
            worst_session = session_id

        if episode_health_path.exists():
            with open(episode_health_path, "r", encoding="utf-8") as f:
                ep_health = json.load(f)

            total_episodes += int(ep_health.get("episode_count", 0))

            mean_len = float(ep_health.get("mean_episode_length", 0.0))
            ep_count = int(ep_health.get("episode_count", 0))

            # Weight by number of episodes so the global mean is more meaningful.
            for _ in range(ep_count):
                episode_lengths.append(mean_len)

    mean_matched_ratio = (
        sum(matched_ratios) / len(matched_ratios) if matched_ratios else 0.0
    )
    mean_missing_ratio = 1.0 - mean_matched_ratio if matched_ratios else 0.0
    mean_episode_length = (
        sum(episode_lengths) / len(episode_lengths) if episode_lengths else 0.0
    )

    typer.echo("DATASET SUMMARY")
    typer.echo(f"  sessions_total: {len(session_dirs)}")
    typer.echo(f"  sessions_with_alignment_health: {sessions_with_health}")
    typer.echo(f"  total_frames: {total_frames}")
    typer.echo(f"  total_matched_frames: {total_matched_frames}")
    typer.echo(f"  total_missing_frames: {total_missing_frames}")
    typer.echo(f"  mean_matched_ratio: {mean_matched_ratio}")
    typer.echo(f"  mean_missing_ratio: {mean_missing_ratio}")
    typer.echo(f"  total_episodes: {total_episodes}")
    typer.echo(f"  mean_episode_length: {mean_episode_length}")

    if best_session is not None:
        typer.echo(f"  best_session: {best_session}")
        typer.echo(f"  best_session_matched_ratio: {best_matched_ratio}")

    if worst_session is not None:
        typer.echo(f"  worst_session: {worst_session}")
        typer.echo(f"  worst_session_matched_ratio: {worst_matched_ratio}")


def main():
    app()


if __name__ == "__main__":
    main()
