"""
Sensor normalization utilities.

This module converts raw sensor CSV logs into a canonical format with a
standardized `t_sec` timestamp column. The normalized output ensures:

- timestamps are represented in seconds
- rows are sorted by time
- the format is consistent across sensor sources

The function returns basic statistics used for QA reporting.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List

def normalize_sensor_csv(
    raw_sensor_path: Path,
    output_path: Path,
) -> Dict[str, float | int]:
    """
    Normalize a raw sensor CSV into a canonical format.

    The normalized output:
    - includes a `t_sec` column representing time in seconds
    - sorts rows by `t_sec` ascending
    - preserves other columns without enforcing a strict schema

    Supported input time columns (first match wins):
    - t_sec
    - timestamp_sec
    - timestamp_ms
    - timestamp_ns

    Returns summary statistics used in the QA report.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(raw_sensor_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"Sensor CSV has no header: {raw_sensor_path}")
        
        fieldnames = list(reader.fieldnames)

        # Determine which time column is present.
        time_col = None
        scale = 1.0
        if "t_sec" in fieldnames:
            time_col = "t_sec"
            scale = 1.0
        elif "timestamp_sec" in fieldnames:
            time_col = "timestamp_sec"
            scale = 1.0
        elif "timestamp_ms" in fieldnames:
            time_col = "timestamp_ms"
            scale = 1e-3
        elif "timestamp_ns" in fieldnames:
            time_col = "timestamp_ns"
            scale = 1e-9
        else:
            raise ValueError(
                f"No supported time column found in {raw_sensor_path}."
                f"Expected one of: t_sec, timestamp_sec, timestamp_ms, timestamp_ns."
            )
        
        rows: List[Dict[str, str]] = []
        for row in reader:
            # Convert time to seconds.
            t = float(row[time_col]) * scale
            row["t_sec"] = f"{t:.9}"    # Stable formatting.
            rows.append(row)
        
        # Sort by time.
        rows.sort(key=lambda r: float(r["t_sec"]))

        # Output fields: ensure t_sec is first, keep all others.
        out_fields = ["t_sec"] + [c for c in fieldnames if c != "t_sec"]

        if time_col not in out_fields:
            # If original time_col not in header (unlikely), ignore.
            pass

        with open(output_path, "w", newline="", encoding="utf-8") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=out_fields)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k, "") for k in out_fields})

        # Summary stats for QA reporting.
        n = len(rows)
        t0 = float(rows[0]["t_sec"]) if n > 0 else 0.0
        t1 = float(rows[-1]["t_sec"]) if n > 0 else 0.0

        return {
            "rows": n,
            "t_start_sec": t0,
            "t_end_sec": t1,
        }
    