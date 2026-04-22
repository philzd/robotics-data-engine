"""
Alignment warning semantics.

Converts alignment health metrics into structured warnings and an
overall dataset status.

Important:
- Pure computation (no file I/O).
- Deterministic outputs.
"""

from __future__ import annotations

from typing import Any, Dict, List

def compute_alignment_warnings(
        health: Dict[str, Any],
        *,
        missing_ratio_fail: float = 0.05,
        max_consecutive_missing_fail: int = 5,
        dt_abs_p95_warn_frac: float = 0.9,
        max_dt_threshold: float,
) -> Dict[str, Any]:
    """
    Convert alignment health metrics into structured warnings and an overall status.

    Rules:
    - FAIL if missing_ratio > missing_ratio_fail
    - FAIL if max_consecutive_missing > max_consecutive_missing_fail
    - WARN if dt_abs_p95 > dt_abs_p95_warn_frac * max_dt_threshold
    """
    warnings: List[Dict[str, Any]] = []

    missing_ratio = float(health.get("missing_ratio", 0.0))
    max_run = int(health.get("max_consecutive_missing", 0))
    dt_abs_p95 = float(health.get("dt_abs_p95", 0.0))

    # FAIL conditions.
    if missing_ratio > missing_ratio_fail:
        warnings.append(
            {
                "code": "MISSING_RATIO_EXCEEDED",
                "level": "FAIL",
                "message": "Missing ratio exceeds allowed budget.",
                "evidence": {
                    "missing_ratio": missing_ratio,
                    "threshold": missing_ratio_fail,
                },
            }
        )
    
    if max_run > max_consecutive_missing_fail:
        warnings.append(
            {
                "code": "CONSECUTIVE_MISSING_EXCEEDED",
                "level": "FAIL",
                "message": "Too many consecutive missing frames.",
                "evidence": {
                    "max_consecutive_missing": max_run,
                    "threshold": max_consecutive_missing_fail,
                },
            }
        )

    # WARN condition.
    warn_threshold = dt_abs_p95_warn_frac * float(max_dt_threshold)
    if dt_abs_p95 > warn_threshold:
        warnings.append(
            {
                "code": "DT_P95_CLOSE_TO_THRESHOLD",
                "level": "WARN",
                "message": "dt_abs_p95 is close to the max_dt threshold (alignment is noisy).",
                "evidence": {
                    "dt_abs_p95": dt_abs_p95,
                    "warn_threshold": warn_threshold,
                    "max_dt_threshold": float(max_dt_threshold),
                },
            }
        )
    
    # Determine overall status.
    overall_status = "OK"
    if any(w["level"] == "FAIL" for w in warnings):
        overall_status = "FAIL"
    elif any(w["level"] == "WARN" for w in warnings):
        overall_status = "WARN"
    
    return {
        "overall_status": overall_status,
        "warnings": warnings,
        "policy": {
            "missing_ratio_fail": missing_ratio_fail,
            "max_consecutive_missing_fail": max_consecutive_missing_fail,
            "dt_abs_p95_warn_frac": dt_abs_p95_warn_frac,
            "max_dt_threshold": float(max_dt_threshold),
        },
    }
