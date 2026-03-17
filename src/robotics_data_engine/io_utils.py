"""
I/O utilities for robotics_data_engine.

Provides small, deterministic helpers for reading and writing JSON
and JSONL artifacts used throughout the pipeline.

These helpers are centralized so modules do not duplicate slightly
different file-handling behavior.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Read a JSONL file (one JSON object per line) into a list of dictionaries.
    """
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    """
    Write a JSON file with stable formatting.

    Stable formatting (indentation, sorted keys, trailing newline) improves
    determinism and makes diffs easier to read.
    """
    if path is None:
        raise ValueError("write_json got path=None (check Session path properties / config constants).")
    
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
