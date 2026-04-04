# Multimodal Robotics Data Engine

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-portfolio-lightgrey)

A deterministic data pipeline that converts raw multimodal robot logs
(video + sensor streams) into aligned, validated datasets and metadata
for machine learning training and evaluation pipelines.

This project builds aligned, versioned robotics datasets from raw robot video and sensor logs. The pipeline performs ingestion, cross-modal alignment, episode extraction, dataset validation, and health analysis to simulate the core data infrastructure used in robotics ML systems.

----------

## TL;DR

This repository implements robotics_data_engine, a deterministic dataset
construction pipeline for multimodal robotics logs.

```text
Raw logs
(video + sensors)
↓
Deterministic ingestion
↓
Canonical timestamp extraction
↓
Cross-modal alignment
↓
Episode extraction
↓
Dataset validation + health metrics
(missing data ratio, alignment error, episode statistics)
↓
Artifact fingerprinting
(versioning)
↓
Versioned dataset artifacts
```

The goal is to model the data infrastructure used in robotics and embodied AI training pipelines.

----------

## Example Dataset Outputs

Below are representative artifacts produced by the pipeline.

### Alignment health metrics

Derived dataset health signals produced during cross-modal alignment.

`derived/alignment_health.json`

```json
{
  "dt_abs_max": 0.0016666666666669272,
  "dt_abs_mean": 0.0010999999999999966,
  "dt_abs_p95": 0.0016666666666669272,
  "matched_count": 100,
  "matched_ratio": 1.0,
  "max_consecutive_missing": 0,
  "missing_by_reason": {},
  "missing_count": 0,
  "missing_ratio": 0.0,
  "total_frames": 100
}
```

### Extracted trajectory episodes

Continuous sequences of aligned frames used for downstream training.

`derived/episodes.json`

```json
{
  "episode_count": 1,
  "episodes": [
    {
      "episode_id": 0,
      "start_frame_idx": 0,
      "end_frame_idx": 99,
      "length": 100,
      "start_t": 0.0,
      "end_t": 3.3
    }
  ],
  "summary": {
    "episode_count": 1,
    "total_frames_in_episodes": 100,
    "max_length": 100,
    "mean_length": 100.0,
    "min_length": 100
  }
}
```

### From Episodes to Training Data

The pipeline outputs trajectory episodes, which represent continuous aligned segments of multimodal robot data. These segments identify portions of the dataset where video frames and sensor streams remain consistently synchronized.

A downstream training pipeline can use `episodes.json` to gather the corresponding video frames and sensor measurements for each episode and convert them into sliding windows used as training samples in downstream ML pipelines.

```text
Raw robot logs
(video + IMU)
        │
        ▼
Cross-modal alignment
        │
        ▼
alignment_map.jsonl
(frame ↔ sensor mapping)
        │
        ▼
Episode extraction
        │
        ▼
episodes.json
(continuous trajectory segments)
        │
        ▼
Training data loader
(windowed frame + sensor sequences)
        │
        ▼
Model training
```

In practice, the training loader reads the episode boundaries, collects the corresponding frames and sensor values, and converts each trajectory segment into training batches used by the model.

### Alignment artifact fingerprint

Content hash manifest ensuring reproducible dataset artifacts.

`manifests/alignment_fingerprint.json`

```json
{
  "artifacts": {
    "alignment_examples_sha256": "272e36482d1974df6b556f0a12c0956181121fa2ad4e287981560edac7b29993",
    "alignment_health_sha256": "196cd9641a7b18c19bfb5e43e0fcb4f5de62f60adf8c0d2974fcb96b5d7d09e3",
    "alignment_map_sha256": "3913cd99439c1b9f608e0fdf09389da91965ec99352d6b8eecd54af5702cc579",
    "alignment_warnings_sha256": "dbeff72f6d63cdd992f76d821e56864ebe6293ceadcb9746f29001cf05faae43",
    "episode_health_sha256": "974e1c7617f029e31697a6d8273d6f4e9c33b375150dee122a854a4bd4ae0df5",
    "episode_invariants_sha256": "ca214f2ffa9771e4610247f41e7de26797edd3314fbd607391a6e2c84d44db82",
    "episodes_sha256": "bb9b891f52b306dc6ca7df731e760ab7d5ac593a12de1402e4d90284467d49fe"
  },
  "params": {
    "max_dt": 0.05
  },
  "policy": {
    "dt_abs_p95_warn_frac": 0.9,
    "max_consecutive_missing_fail": 5,
    "max_dt_threshold": 0.05,
    "missing_ratio_fail": 0.05
  }
}
```

----------

## Key Features

- Deterministic dataset construction
- Canonical timestamp extraction and cross-modal alignment
- Episode extraction for trajectory learning
- Structural invariant validation
- Dataset health metrics and quality signals
- Artifact fingerprinting for reproducibility

----------

## Table of Contents

- [Quick Demo](#quick-demo)
- [Getting Started](#getting-started)
- [Why This Project Exists](#why-this-project-exists)
- [What robotics_data_engine Is and Is Not](#what-robotics_data_engine-is--is-not)
- [System Architecture](#system-architecture)
- [Session Directory Contract](#session-directory-contract)
- [Pipeline Stages](#pipeline-stages)
- [CLI Commands](#cli-commands)
- [Core Guarantees](#core-guarantees)
- [Data Guarantees](#data-guarantees)
- [Technologies Used](#technologies-used)
- [Future Extensions](#future-extensions)
- [Project Scope](#project-scope)
- [Usage Notice](#usage-notice)
- [Author](#author)

----------

## Quick Demo

Example pipeline run:

```bash
python -m robotics_data_engine ingest \
    --session demo \
    --video examples/dummy_video.mp4 \
    --sensor examples/dummy_imu.csv

python -m robotics_data_engine align --session demo
```

Example pipeline output:

![Pipeline Run](docs/images/pipeline_output.png)

----------

## Getting Started

The following steps demonstrate how to run the data engine on a sample session.

Example assets in `examples/` are small synthetic files used for demonstration.
The pipeline itself is designed to operate on real robotics datasets containing
video and multi-sensor logs.

1. Clone the repository

```bash
git clone https://github.com/philzd/robotics-data-engine.git
cd robotics_data_engine
```

2. Create a Python environment and install dependencies:

```bash
python -m venv env
source env/bin/activate
pip install -e .
pip install -r requirements.txt
```

3. Run a demo session:

```bash
python -m robotics_data_engine ingest \
  --session demo \
  --video examples/dummy_video.mp4 \
  --sensor examples/dummy_imu.csv

python -m robotics_data_engine align --session demo
```

4. Inspect dataset health:
   
```bash
python -m robotics_data_engine dataset-summary
```

This will produce alignment artifacts, episode segmentation results, and dataset health metrics for the session.

----------

## Why This Project Exists

Modern robotics and embodied AI systems rely on large multimodal datasets containing:

- video
- IMU
- LiDAR
- robot state
- control signals

However, raw robot logs are often:
- temporally misaligned
- incomplete
- inconsistent across sensors
- difficult to reproduce or validate

This project demonstrates a deterministic data engine that:
- canonicalizes timestamps
- aligns multimodal signals
- validates dataset integrity
- extracts training episodes
- produces dataset health metrics

The system is designed to mirror the structure of real-world robotics ML data platforms.

----------

## What robotics_data_engine Is / Is Not

## Is

- A dataset construction layer for robotics logs
- A deterministic pipeline for multimodal alignment
- A framework for dataset validation and health metrics
- A reproducible artifact generation system

## Is Not

- A labeling / annotation UI
- A distributed compute system
- A cloud storage platform
- A full training pipeline

The focus is **dataset construction and validation infrastructure**.

----------

## System Architecture

The data engine converts raw multimodal robot logs into deterministic,
validated dataset artifacts used for downstream ML training.

```text
Inputs
  raw robot video + sensor logs
        │
        ▼

┌────────────────────────────────────┐
│   Multimodal Robotics Data Engine  │
│                                    │
│   • deterministic data ingestion   │
│   • canonical timestamp extraction │
│   • cross-modal alignment          │
│   • episode extraction             │
│   • invariant validation           │
│   • dataset health metrics         │
│   • artifact fingerprinting        │
└────────────────────────────────────┘
        │
        ▼

Outputs
  • aligned dataset artifacts
  • episode datasets
  • validation reports
  • dataset health metrics
  • reproducibility manifests


Operational Commands
  ingest | align | validate | align-all | dataset-summary
```

----------

## Session Directory Contract

Each robot run is represented as a **session**.

```
sessions/<session_id>/

    raw/
        video.mp4
        sensor.csv

    derived/
        video_timestamps.jsonl
        sensor_normalized.csv
        qa_report.json

        alignment_map.jsonl
        alignment_report.json
        alignment_examples.json
        alignment_invariants.json
        alignment_health.json
        alignment_warnings.json

        episodes.json
        episode_invariants.json
        episode_health.json

    manifests/
        session_manifest.json
        alignment_fingerprint.json

logs/
```

### Folder meanings

**raw/**

- Immutable copies of original inputs
- Never edited after ingest

**derived/**

- Generated artifacts derived from raw data
- Fully rebuildable

**manifests/**

- Dataset provenance and fingerprinting metadata

**logs/**

- CLI execution logs

This directory contract ensures datasets remain reproducible and auditable.

----------

## Pipeline Stages

### 1. Ingestion

Raw logs are copied into immutable session directories for deterministic processing.

Artifacts produced:
- `video_timestamps.jsonl`
- `sensor_normalized.csv`
- `qa_report.json`
- `session_manifest.json`

Video timestamps follow the deterministic rule:
`t_frame = frame_idx / fps`

### 2. Cross-Modal Alignment

Sensor timestamps are aligned to the video frame time base using nearest-neighbor matching.

Artifacts produced:
- `alignment_map.jsonl`
- `alignment_report.json`
- `alignment_examples.json`
- `alignment_invariants.json`
- `alignment_health.json`
- `alignment_warnings.json`
- `alignment_fingerprint.json`

Alignment explicitly records the time delta between modalities.
`dt = t_sensor - t_frame`

No hidden interpolation occurs to avoid introducing artificial
sensor values that could bias downstream model training.

### 3. Episode Extraction

Continuous sequences of successfully aligned frames are extracted into trajectory episodes.

Artifacts produced:
- `episodes.json`
- `episode_invariants.json`
- `episode_health.json`

Episodes represent training-ready trajectory segments.

### 4. Dataset Validation

Structural invariants verify that dataset artifacts satisfy expected schema and alignment contracts. Validation is applied after alignment and episode extraction artifacts are produced.

Examples:
- frame index continuity
- alignment threshold guarantees
- episode structure correctness

### 5. Dataset Health Metrics

The pipeline computes signals describing dataset quality:

Examples:
- matched frame ratio
- missing data ratio
- alignment error statistics
- trajectory fragmentation
- episode length statistics

These metrics enable quick evaluation of dataset quality.

### 6. Artifact Fingerprinting

Derived artifacts are hashed to produce a deterministic fingerprint.

The fingerprint hashes all derived artifacts using SHA-256 to ensure
that dataset outputs are content-addressable and reproducible.

This ensures:
- dataset reproducibility
- artifact provenance
- version traceability

## CLI Commands

The CLI provides commands for ingesting raw sessions, aligning multimodal data,
validating dataset artifacts, and inspecting dataset health. Commands can be
run individually depending on the workflow.

### Ingest a session

Copies raw inputs into a session and prepares normalized artifacts.

```bash
python -m robotics_data_engine ingest \
    --session demo_session \
    --video examples/dummy_video.mp4 \
    --sensor examples/dummy_imu.csv
```

### Align multimodal data

Performs cross-modal alignment and generates alignment artifacts, health metrics, and validation signals.

```bash
python -m robotics_data_engine align --session demo_session
```

### Validate dataset integrity

Runs invariant checks over alignment and episode artifacts.

```bash
python -m robotics_data_engine validate --session demo_session
```

### Process all sessions

Runs alignment across all sessions in the sessions directory.

```bash
python -m robotics_data_engine align-all
```

### Dataset summary

Aggregates dataset health metrics across sessions.

Summarize dataset health across sessions:
```bash
python -m robotics_data_engine dataset-summary
```

Example CLI output:

![Dataset Summary](docs/images/dataset_summary.png)

----------

## Core Guarantees

robotics_data_engine enforces three core guarantees.

- **Determinism**:
  Given identical inputs and configuration, derived artifacts are identical.

- **Reproducibility**:
  Datasets can be rebuilt from raw data and manifest metadata.

- **Provenance**:
  Every artifact records its origin, configuration, and creation time.

These guarantees are enforced through:

- immutable session identity
- content hashing
- explicit manifests
- rebuildable derived artifacts

----------

## Data Guarantees

robotics_data_engine enforces several guarantees over constructed datasets.

- **Deterministic dataset construction**:
  Given identical raw inputs and configuration, the pipeline produces identical derived artifacts.
  Mechanism: canonical timestamp extraction, deterministic alignment logic, and immutable session artifacts.

- **Explicit alignment semantics**:
  Cross-modal alignment always records the exact time delta between modalities. No hidden interpolation is performed.
  Mechanism: canonical timestamp extraction, deterministic alignment logic, and immutable session artifacts.

- **Invariant-validated datasets**:
  Alignment and episode artifacts are validated using structural invariants to detect corruption or schema violations.
  Mechanism: invariant checks applied to alignment and episode artifacts.

- **Observable dataset health**:
  The system computes dataset health metrics (alignment error, missing data ratio, episode fragmentation) to make dataset quality measurable.
  Mechanism: automated dataset health metrics generated during pipeline execution.

----------

## Tech Stack

- Python
- Typer (CLI framework)
- JSON / CSV artifact pipelines
- SHA-256 artifact fingerprinting

----------

## Future Extensions

Possible future extensions include:

- Parquet-based dataset storage
- Distributed batch processing (Spark or Ray)
- Streaming ingestion simulation (Kafka-style)
- Dataset partitioning and sharding
- Cloud-based storage and execution
- Dataset latency and performance measurement

These extensions would simulate aspects of large-scale robotics data infrastructure.

----------

## Project Scope

This project focuses on dataset construction and validation infrastructure.

It does not include:

- model training pipelines
- annotation systems
- distributed storage systems
- production deployment infrastructure

----------

## Usage Notice

This repository is shared for portfolio and evaluation purposes.

Please contact the author for permission before reusing or redistributing the code.

----------

## Author

Philippe Do

Focus areas:
- robotics ML infrastructure
- multimodal data pipelines
- distributed ML systems
