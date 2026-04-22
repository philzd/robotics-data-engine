"""
Multimodal timestamp alignment.

Aligns sensor timestamps onto the canonical video frame timebase and
produces deterministic alignment artifacts for downstream dataset
construction and evaluation.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from bisect import bisect_left

@dataclass(frozen=True)
class AlignmentReport:
    """
    Summary statistics produced by an alignment run.

    These metrics describe alignment quality and help diagnose
    temporal drift or missing sensor coverage.
    """
    matched_count: int
    missing_count: int
    dt_abs_mean: float
    dt_abs_max: float
    max_dt_threshold: float
    warnings: list[str]


def load_video_timestamps(path: Path) -> list[tuple[int, float]]:
    """
    Load canonical video timestamps.

    Each row:
      frame_idx (int)
      timestamp_sec (float)

    This defines the canonical time base for alignment.
    """
    rows: list[tuple[int, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append((int(obj["frame_idx"]), float(obj["timestamp_sec"])))
    return rows


def load_sensor_times(path: Path) -> list[float]:
    """
    Load normalized sensor timestamps.

    Must contain column: t_sec

    This function enforces schema correctness.
    """
    times: list[float] = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        if "t_sec" not in (reader.fieldnames or []):
            raise ValueError(
                f"sensor_normalized.csv missing required column 't_sec': {path}"
            )

        for r in reader:
            times.append(float(r["t_sec"]))

    return times


def classify_missing_reason(
    *,
    t_frame: float,
    first_sensor_t: float | None,
    last_sensor_t: float | None,
    nearest_abs_dt: float | None,
    max_dt: float,
    sensor_count: int,
) -> str:
    """
    Determine structured reason for alignment failure.

    This models failure as a first-class system state,
    not just "None" or "error".

    Categories:
      - SENSOR_EMPTY
      - SENSOR_NOT_STARTED
      - SENSOR_ENDED
      - GAP_TOO_LARGE
    """
    if sensor_count == 0:
        return "SENSOR_EMPTY"

    if first_sensor_t is not None and t_frame < first_sensor_t:
        return "SENSOR_NOT_STARTED"

    if last_sensor_t is not None and t_frame > last_sensor_t:
        return "SENSOR_ENDED"

    if nearest_abs_dt is not None and nearest_abs_dt > max_dt:
        return "GAP_TOO_LARGE"

    return "UNKNOWN"


def find_nearest_sensor_index(
        sensor_ts: list[float],
        t_frame: float,
) -> tuple[int, float]:
    """
    Find the nearest sensor index to t_frame using binary search.

    Returns:
    - (best_index, best_abs_dt)

    Tie-break rule:
    if abs_dt ties, choose the smaller sensor_idx.
    """
    if not sensor_ts:
        raise ValueError("sensor_ts must not be empty")
    
    pos = bisect_left(sensor_ts, t_frame)

    candidates: list[int] = []

    if pos < len(sensor_ts):
        candidates.append(pos)
    if pos > 0:
        candidates.append(pos - 1)

    best_j = candidates[0]
    best_abs_dt = abs(sensor_ts[best_j] - t_frame)

    for j in candidates[1:]:
        abs_dt = abs(sensor_ts[j] - t_frame)
        if (abs_dt, j) < (best_abs_dt, best_j):
            best_abs_dt = abs_dt
            best_j = j
    
    return best_j, best_abs_dt


def align_nearest(
    video_ts: list[tuple[int, float]],
    sensor_ts: list[float],
    *,
    max_dt: float,
) -> tuple[list[dict], AlignmentReport]:
    """
    Deterministic nearest-neighbor alignment.

    For each video frame timestamp:
        choose sensor timestamp minimizing abs(sensor - frame)

    Deterministic tie-break:
        (abs_dt, index) comparison ensures stable ordering.

    No interpolation.
    No silent repair.
    """
    if max_dt <= 0:
        raise ValueError("max_dt must be > 0")

    out_rows: list[dict] = []

    matched = 0
    missing = 0
    dt_abs_sum = 0.0
    dt_abs_max = 0.0

    # Edge Case: No sensor data at all.
    if len(sensor_ts) == 0:

        evidence = {
            "max_dt": float(max_dt),
            "nearest_dt": None,
            "sensor_first_t": None,
            "sensor_last_t": None,
            "sensor_count": 0,
        }

        for frame_idx, t_frame in video_ts:
            out_rows.append({
                "frame_idx": frame_idx,
                "t_frame": t_frame,
                "sensor_idx": None,
                "t_sensor": None,
                "dt": None,
                "status": "missing",
                "reason": "SENSOR_EMPTY",
                "missing_reason": "SENSOR_EMPTY",
                "evidence": evidence,
            })

        report = AlignmentReport(
            matched_count=0,
            missing_count=len(video_ts),
            dt_abs_mean=0.0,
            dt_abs_max=0.0,
            max_dt_threshold=float(max_dt),
            warnings=["no_sensor_timestamps"],
        )

        return out_rows, report
    
    # Normal alignment case.
    first_sensor_t = sensor_ts[0]
    last_sensor_t = sensor_ts[-1]
    sensor_count = len(sensor_ts)

    for frame_idx, t_frame in video_ts:

        # Nearest Neighbor Search (O(N) simple baseline).
        # best_j = 0
        # best_abs_dt = abs(sensor_ts[0] - t_frame)

        # for j in range(1, sensor_count):
        #     abs_dt = abs(sensor_ts[j] - t_frame)

        #     # Deterministic tie-break:
        #     if (abs_dt, j) < (best_abs_dt, best_j):
        #         best_abs_dt = abs_dt
        #         best_j = j

        # Nearest neighbor search using binary search.
        # This replaces the earlier O(N) scan with an O(log N) lookup
        # while preserving deterministic tie-breaking behavior.
        best_j, best_abs_dt = find_nearest_sensor_index(sensor_ts, t_frame)

        # Missing case.
        if best_abs_dt > max_dt:

            missing += 1

            missing_reason = classify_missing_reason(
                t_frame=t_frame,
                first_sensor_t=first_sensor_t,
                last_sensor_t=last_sensor_t,
                nearest_abs_dt=best_abs_dt,
                max_dt=max_dt,
                sensor_count=sensor_count,
            )

            evidence = {
                "max_dt": float(max_dt),
                "nearest_dt": float(best_abs_dt),
                "sensor_first_t": float(first_sensor_t),
                "sensor_last_t": float(last_sensor_t),
                "sensor_count": int(sensor_count),
            }

            out_rows.append({
                "frame_idx": frame_idx,
                "t_frame": t_frame,
                "sensor_idx": None,
                "t_sensor": None,
                "dt": None,
                "status": "missing",
                "reason": missing_reason,
                "missing_reason": missing_reason,
                "evidence": evidence,
            })

        # Matched case.
        else:

            matched += 1

            t_sensor = sensor_ts[best_j]
            dt = t_sensor - t_frame
            abs_dt = abs(dt)

            dt_abs_sum += abs_dt
            dt_abs_max = max(dt_abs_max, abs_dt)

            evidence = {
                "max_dt": float(max_dt),
                "nearest_dt": float(abs_dt),
                "sensor_first_t": float(first_sensor_t),
                "sensor_last_t": float(last_sensor_t),
                "sensor_count": int(sensor_count),
            }

            out_rows.append({
                "frame_idx": frame_idx,
                "t_frame": t_frame,
                "sensor_idx": best_j,
                "t_sensor": float(t_sensor),
                "dt": float(dt),
                "status": "matched",
                "reason": "OK",
                "missing_reason": None,
                "evidence": evidence,
            })

    # Summary statistics.
    dt_abs_mean = (dt_abs_sum / matched) if matched > 0 else 0.0

    warnings: list[str] = []

    if missing > 0:
        warnings.append("missing_count > 0")

    if matched > 0 and dt_abs_max > 0.9 * max_dt:
        warnings.append("dt_abs_max close to threshold")

    report = AlignmentReport(
        matched_count=matched,
        missing_count=missing,
        dt_abs_mean=float(dt_abs_mean),
        dt_abs_max=float(dt_abs_max),
        max_dt_threshold=float(max_dt),
        warnings=warnings,
    )

    return out_rows, report


def compute_alignment_examples(alignment_rows: list[dict], *, top_k: int = 20) -> dict:
    """
    Produce small explainability samples for debugging and review.

    Outputs are intentionally small and deterministic.
    """
    # Matched rows where dt exists.
    matched = [r for r in alignment_rows if r.get("status") == "matched" and r.get("dt") is not None]
    # Missing rows.
    missing = [r for r in alignment_rows if r.get("status") == "missing"]

    # Worst matched by abs(dt) (largest time mismatch while still "matched").
    worst_matched = sorted(
        matched,
        key=lambda r: (-abs(float(r["dt"])), int(r.get("frame_idx", 0)))
    )[:top_k]

    # Longest missing streaks (contiguous runs of missing frames).
    streaks: list[list[dict]] = []
    current: list[dict] = []
    for r in alignment_rows:
        if r.get("status") == "missing":
            current.append(r)
        else:
            if current:
                streaks.append(current)
                current = []
    
    if current:
        streaks.append(current)

    longest_streaks = sorted(
        streaks,
        key=lambda s: (-len(s), int(s[0].get("frame_idx", 0)))
    )[:5]

    longest_missing_summary = []
    for s in longest_streaks:
        reasons = [x.get("missing_reason") or x.get("reason") for x in s]
        dominant = None
        if reasons:
            # Deterministic dominant reason: pick most frequent -> tie break by name.
            counts = {}
            for rr in reasons:
                counts[rr] = counts.get(rr, 0) + 1
            dominant = sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0])))[0][0]

        longest_missing_summary.append({
            "start_frame_idx": s[0].get("frame_idx"),
            "end_frame_idx": s[-1].get("frame_idx"),
            "length": len(s),
            "dominant_reason": dominant,
        })
    
    # One representative example per missing_reason.
    missing_reason_examples: dict[str, dict] = {}
    for r in missing:
        reason = r.get("missing_reason") or r.get("reason") or "UNKNOWN"
        if reason not in missing_reason_examples:
            missing_reason_examples[reason] = {
                "frame_idx": r.get("frame_idx"),
                "t_frame": r.get("t_frame"),
                "evidence": r.get("evidence"),
            }
    
    return {
        "worst_matched_by_abs_dt": worst_matched,
        "longest_missing_streaks": longest_missing_summary,
        "missing_reason_examples": missing_reason_examples,
        "top_k": int(top_k),
    }


def write_alignment_map(path: Path, rows: list[dict]) -> None:
    """
    Write one JSON object per line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def write_alignment_report(path: Path, report: AlignmentReport) -> None:
    """
    Write summary alignment report.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "matched_count": report.matched_count,
            "missing_count": report.missing_count,
            "dt_abs_mean": report.dt_abs_mean,
            "dt_abs_max": report.dt_abs_max,
            "max_dt_threshold": report.max_dt_threshold,
            "warnings": report.warnings,
        }, f, indent=2)
        f.write("\n")
