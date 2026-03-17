"""
Alignment invariants for robotics_data_engine.

This module validates structural guarantees over alignment_map rows.
It does not read or write files, print output, or mutate artifacts.
Instead, it returns structured validation results for the CLI layer.
"""
from typing import List, Dict, Any


def check_alignment_invariants(
        alignment_rows: List[Dict[str, Any]],
        *,
        max_dt_threshold: float,
) -> Dict[str, Any]:
    """
    Validate structural invariants for alignment_map rows.

    Expected minimal schema per row:
    - frame_idx: int
    - status: "matched" | "missing"
    - sensor_idx: int | None
    - t_sensor: float | None
    - dt: float | None

    Returns a dictionary with:
    - passed
    - violation_count
    - violations
    """

    violations = []

    # Invariant 1: frame_idx must be contiguous starting at 0.
    expected_idx = 0

    for row in alignment_rows:
        frame_idx = row.get("frame_idx")

        if frame_idx != expected_idx:
            violations.append({
                "type": "FRAME_INDEX_GAP",
                "message": "frame_idx must be contiguous starting at 0",
                "expected": expected_idx,
                "actual": frame_idx
            })
            # Stop early - if indices are broken, rest is unreliable.
            break

        expected_idx += 1

    # Invariant 2 and 3: Status semantics must be correct.
    for row in alignment_rows:
        frame_idx = row.get("frame_idx")
        status = row.get("status")

        sensor_idx = row.get("sensor_idx")
        t_sensor = row.get("t_sensor")
        dt = row.get("dt")

        t_frame = row.get("t_frame")
        missing_reason = row.get("missing_reason")
        evidence = row.get("evidence")

        if not isinstance(evidence, dict):
            violations.append({
                "type": "EVIDENCE_MISSING_OR_INVALID",
                "frame_idx": frame_idx,
                "message": "row must include evidence dict"
            })
            continue

        # Case 1: Matched rows.
        if status == "matched":

            # Must have non-null values.
            if sensor_idx is None or t_sensor is None or dt is None:
                violations.append({
                    "type": "INVALID_MATCH_ROW",
                    "frame_idx": frame_idx,
                    "message": "Matched rows must contain sensor_idx, t_sensor, and dt"
                })
                continue

            # dt must be within threshold.
            try:
                if abs(float(dt)) > float(max_dt_threshold):
                    violations.append({
                        "type": "DT_EXCEEDS_THRESHOLD",
                        "frame_idx": frame_idx,
                        "message": "abs(dt) exceeds max_dt_threshold",
                        "threshold": max_dt_threshold,
                        "actual_dt": dt
                    })
            except Exception:
                violations.append({
                    "type": "DT_NOT_NUMERIC",
                    "frame_idx": frame_idx,
                    "message": "dt must be numeric for matched rows"
                })

            if missing_reason is not None:
                violations.append({
                    "type": "MATCHED_HAS_MISSING_REASON",
                    "frame_idx": frame_idx,
                    "message": "matched rows must have missing_reason=None"
                })

        # Case 2: Missing rows.
        elif status == "missing":
            # Missing rows must have null Values.
            if sensor_idx is not None or t_sensor is not None or dt is not None:
                violations.append({
                    "type": "INVALID_MISSING_ROW",
                    "frame_idx": frame_idx,
                    "message": "Missing rows must have null sensor_idx, t_sensor, and dt"
                })
            
            if not missing_reason:
                violations.append({
                    "type": "MISSING_REASON_REQUIRED",
                    "frame_idx": frame_idx,
                    "message": "missing rows must have a missing_reason"
                })
                continue

            sensor_count = evidence.get("sensor_count")
            sensor_first_t = evidence.get("sensor_first_t")
            sensor_last_t = evidence.get("sensor_last_t")
            nearest_dt = evidence.get("nearest_dt")

            if missing_reason == "SENSOR_EMPTY":
                if sensor_count != 0:
                    violations.append({
                        "type": "SENSOR_EMPTY_INCONSISTENT",
                        "frame_idx": frame_idx,
                        "message": "SENSOR_EMPTY requires sensor_count == 0",
                        "sensor_count": sensor_count
                    })

            elif missing_reason == "SENSOR_NOT_STARTED":
                if t_frame is None or sensor_first_t is None or not (float(t_frame) < float(sensor_first_t)):
                    violations.append({
                        "type": "SENSOR_NOT_STARTED_INCONSISTENT",
                        "frame_idx": frame_idx,
                        "message": "SENSOR_NOT_STARTED requires t_frame < sensor_first_t",
                        "t_frame": t_frame,
                        "sensor_first_t": sensor_first_t
                    })

            elif missing_reason == "SENSOR_ENDED":
                if t_frame is None or sensor_last_t is None or not (float(t_frame) > float(sensor_last_t)):
                    violations.append({
                        "type": "SENSOR_ENDED_INCONSISTENT",
                        "frame_idx": frame_idx,
                        "message": "SENSOR_ENDED requires t_frame > sensor_last_t",
                        "t_frame": t_frame,
                        "sensor_last_t": sensor_last_t
                    })

            elif missing_reason == "GAP_TOO_LARGE":
                if nearest_dt is None or not (float(nearest_dt) > float(max_dt_threshold)):
                    violations.append({
                        "type": "GAP_TOO_LARGE_INCONSISTENT",
                        "frame_idx": frame_idx,
                        "message": "GAP_TOO_LARGE requires nearest_dt > max_dt_threshold",
                        "nearest_dt": nearest_dt,
                        "threshold": max_dt_threshold
                    })

        # Case 3: Unknown status.
        else:
            violations.append({
                "type": "UNKNOWN_STATUS",
                "frame_idx": frame_idx,
                "message": "status must be 'matched' or 'missing'",
                "actual_status": status
            })

    # Final Result.
    return {
        "passed": len(violations) == 0,
        "violation_count": len(violations),
        "violations": violations[:50],   # Limit size for safety.
    }
        
    
