# MULTI-MODAL DEFENSIVE AVIONICS AI SIMULATOR

> **SYNTHETIC CLASSROOM SIMULATION — NO REAL-WORLD TARGETING**

An offline, software-only academic application that integrates multi-modal radio signal classification, computer vision object detection, reinforcement learning decision policy (PPO), sensor fusion, deterministic scenario generation, and a Streamlit engineering HUD dashboard.

---

## Scope & Ethical Boundaries

This repository is strictly designed for **academic simulation and algorithmic evaluation**.

### Strict Scope Constraints:
- ❌ **No real-world weapon targeting or firing**
- ❌ **No missile guidance, trajectory, or interception calculations**
- ❌ **No operational radar parameters or military frequency bands**
- ❌ **No electronic attack, jamming waveforms, or countermeasure physics**
- ❌ **No aircraft vulnerability models or kill-chain automation**
- ❌ **No external connections to live or military hardware**

All inputs and observations utilize synthetic geometric objects (circles, triangles, diamonds), communications-modulation classes from public datasets (RadioML), normalized dimensionless metrics $[0.0, 1.0]$, and abstract virtual system resources.

---

## Architecture & Data Flow

```text
Synthetic Signal Input (I/Q) ──┐
                               ├──> Sensor Fusion Engine ──> Abstract PPO Policy
Vision Detection (YOLO / Sky) ──┤   (Weighted Consensus &         (MlpPolicy)
                               │    Temporal Freshness)             │
Scenario Radar Contacts ───────┤                                   ▼
                               ├──> Multi-Modal Telemetry ──> Engineering HUD
Collaborative Nodes (Alpha/Bravo)┘                             (Streamlit Dashboard)
```

---

## Completed Modules

### 1. Signal Analysis (`defensive_avionics.signal`)
- **1D Residual CNN** for modulation classification (`best_iq_cnn.pt`).
- Classes: `8PSK`, `AM-DSB`, `BPSK`, `CPFSK`, `GFSK`, `PAM4`, `QAM16`, `QAM64`, `QPSK`, `WBFM`.
- Evaluates on RadioML test split (33,000 samples).
- Real observed performance: **88.63% accuracy above 0 dB SNR**.
- Exports STFT spectrograms, confusion matrices, and SNR performance curves.

### 2. Decision Policy (`defensive_avionics.policy`)
- **Proximal Policy Optimization (PPO)** in a normalized 7-state Gymnasium environment (`best_model.zip`).
- Abstract actions: `observe`, `virtual_resource_a`, `virtual_resource_b`, `virtual_signal_response`, `abstract_reposition`.
- Benchmark evaluation against rule-based heuristic baseline across fixed seeds.
- Real observed benchmark: **PPO $429.47 \pm 1.37$ vs Baseline $289.81 \pm 9.68$ (+139.66 advantage)**.

### 3. Vision Detection & Approach Tracking (`defensive_avionics.vision`)
- **YOLO Nano** trained on 500 synthetic sky images with geometric aerial objects (`synthetic_yolo_best.pt`).
- Temporal bounding-box expansion tracking (`ExpansionTracker`).
- Real observed performance on test split: **99.32% Precision, 97.93% Recall, 98.47% mAP50**.
- Categorizes visual trends into `insufficient_data`, `receding`, `stable`, `growing`, and `rapid_growth`.

### 4. Sensor Fusion Engine (`defensive_avionics.fusion`)
- Explainable confidence-weighted consensus across multi-modal inputs.
- Exponential temporal freshness decay with configurable half-life.
- Computes fused consensus label, fused confidence, uncertainty, urgency, and full evidence breakdown.

### 5. Synthetic Scenario Generator (`defensive_avionics.scenario`)
- Deterministic, offline scenario engine with normalized coordinate space.
- Simulates objects classified strictly as `Friendly`, `Neutral`, `Unknown`, and `Resource`.
- Telemetry simulation for 3 collaborative peer nodes (`Node-Alpha`, `Node-Bravo`, `Node-Charlie`).

### 6. Engineering HUD Dashboard (`app.dashboard`)
- Futuristic, dark navy/cyan cockpit tactical telemetry interface.
- 6 interactive tactical panels: Signal Waveform & Spectrogram, Polar Radar HUD, Vision Bounding Box & Expansion, PPO Policy State Meters, Sensor Fusion Consensus, and Collaborative Node Telemetry.
- Controls for Step (+1s), Reset, Difficulty preset (`low`, `medium`, `high`), and Seed configuration.

---

## Directory Structure

```text
defensive-avionics-ai-simulator/
├── app/
│   └── dashboard.py               # Streamlit Engineering HUD Dashboard
├── configs/                       # Module configurations (YAML)
├── data/
│   ├── processed/
│   │   ├── signal/                # Preprocessed RadioML data & metadata
│   │   └── vision_synthetic/      # 500-train / 100-val / 100-test synthetic YOLO dataset
├── models/
│   ├── policy/                    # Trained PPO policy models & metadata
│   ├── signal/                    # Trained 1D-CNN modulation classifier
│   └── vision/                    # Trained synthetic YOLO detector
├── outputs/
│   ├── figures/                   # Confusion matrix, SNR graph, PPO comparison, detection samples
│   └── reports/                   # Machine-readable evaluation JSONs
├── scripts/
│   ├── generate_synthetic_vision_data.py
│   └── run_dashboard.ps1          # One-click Windows PowerShell launcher
├── src/
│   └── defensive_avionics/
│       ├── common/                # Shared dataclasses & types
│       ├── fusion/                # Sensor fusion engine
│       ├── integration/           # Orchestrator combining all subsystems
│       ├── policy/                # Gymnasium environment, baseline, PPO training/eval
│       ├── scenario/              # Deterministic scenario & collaborative nodes
│       ├── signal/                # CNN architecture, preprocessing, dataset, evaluation
│       ├── ui/                    # Lightweight Pygame demo window
│       └── vision/                # YOLO detector, expansion tracking, training/eval
└── tests/                         # Fast unit & contract test suite
```

---

## Setup & Execution Guide

### Recommended Environment
- **Operating System:** Windows 10/11 or Linux
- **Python:** 3.11
- **Hardware:** CPU-only optimized (Intel i5, 8 GB RAM)

### Windows PowerShell Quickstart

```powershell
# 1. Activate virtual environment
.venv\Scripts\Activate.ps1

# 2. Run test suite
python -m pytest -v

# 3. Run lint check
python -m ruff check src tests scripts app
```

---

## Reproducible Command Reference

### Launch Engineering Dashboard
```powershell
# Option A: PowerShell Launcher
.\scripts\run_dashboard.ps1

# Option B: Direct Streamlit Run
streamlit run app/dashboard.py

# Option C: CLI Command
python -m defensive_avionics dashboard
```

### Run Module Evaluations
```powershell
# Module 1 — Signal Classifier Evaluation
python -m defensive_avionics.signal.evaluate

# Module 2 — Decision Policy Benchmark (PPO vs Baseline)
python -m defensive_avionics.policy.evaluate

# Module 3 — Vision Detector Evaluation
python -m defensive_avionics.vision.evaluate
```

### Retrain Models (Optional / Offline)
```powershell
# Generate synthetic vision dataset
python scripts/generate_synthetic_vision_data.py --train-count 500 --val-count 100 --test-count 100

# Train YOLO detector (CPU-optimized)
python -m defensive_avionics.vision.train --epochs 10 --image-size 320 --batch-size 8 --device cpu --workers 0

# Train PPO policy
python -m defensive_avionics.policy.train --total-timesteps 100000
```

---

## Known Limitations

1. **Synthetic RF Data:** RadioML represents communication waveforms rather than radar emitters. Modulation classifications must remain explicitly abstract.
2. **Visual Trend Approximation:** Approach urgency is derived from 2D bounding-box expansion ratios across successive frames, not physical rangefinding.
3. **Classroom Simulation Environment:** All aircraft models, tactics, and coordinates are normalized $[0, 1]$ educational demonstrations.
