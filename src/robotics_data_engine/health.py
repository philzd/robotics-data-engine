"""
Alignment health metrics.

Computes dataset halth signals from alignment_map. rows.

Important:
- Pure computation (no file I/O).
- Deterministic outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List

def _percentile(sorted_vals: List[float], p: float) -> float:
    """
    Compute percentile p (0..100) from a sorted list under linear interpolation.

    This implementation is deterministic and does not depend on numpy.
    """
    if not sorted_vals:
        return 0.0
    if p <= 0:
        return float(sorted_vals[0])
    if p >= 100:
        return float(sorted_vals[-1])
    
    k = (len(sorted_vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return float(sorted_vals[f])
    d0 = sorted_vals[f] * (c - k)
    d1 = sorted_vals[c] * (k - f)
    return float(d0 + d1)

def compute_alignment_health(alignment_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute health metrics for an alignment run.

    Expects each row to include:
    - status: "matched" | "missing"
    - dt: float | None
    - reason or missing_reason for missing-row breakdowns
    """
    total = len(alignment_rows)
    matched = 0
    missing = 0

    # dt stats (matched only).
    dt_abs_vals: List[float] = []

    # Missing structure.
    max_consecutive_missing = 0
    current_missing_run = 0

    # Breakdown.
    missing_by_reason: Dict[str, int] = {}

    for row in alignment_rows:
        status = row.get("status")
        reason = row.get("missing_reason") or row.get("reason", "UNKNOWN")

        if status == "matched":
            matched += 1
            current_missing_run = 0 # Reset run.

            dt = row.get("dt")
            if dt is not None:
                dt_abs_vals.append(abs(float(dt)))
                                  
        elif status == "missing":
            missing += 1

            # Track missing runs.
            current_missing_run += 1
            if current_missing_run > max_consecutive_missing:
                max_consecutive_missing = current_missing_run

            # Count reasons.
            missing_by_reason[reason] = missing_by_reason.get(reason, 0) + 1

        else:
            # Unknown status should not happen if invariants pass,
            # but avoid crashing here.
            missing_by_reason["UNKNOWN_STATUS"] = missing_by_reason.get("UNKNOWN_STATUS", 0) + 1

    dt_abs_vals.sort()

    dt_abs_mean = (sum(dt_abs_vals) / len(dt_abs_vals)) if dt_abs_vals else 0.0
    dt_abs_max = dt_abs_vals[-1] if dt_abs_vals else 0.0
    dt_abs_p95 = _percentile(dt_abs_vals, 95.0)

    matched_ratio = (matched / total) if total > 0 else 0.0
    missing_ratio = (missing / total) if total > 0 else 0.0

    return {
        "total_frames": total,
        "matched_count": matched,
        "missing_count": missing,
        "matched_ratio": matched_ratio,
        "missing_ratio": missing_ratio,
        "dt_abs_mean": float(dt_abs_mean),
        "dt_abs_max": float(dt_abs_max),
        "dt_abs_p95": float(dt_abs_p95),
        "max_consecutive_missing": int(max_consecutive_missing),
        "missing_by_reason": missing_by_reason,
    }
        