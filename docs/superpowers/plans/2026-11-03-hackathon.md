# Peleng Hackathon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Follow RED → GREEN and record actual verification.

**Goal:** Deliver a causal five-camera tracking solution with 3D estimates, complete exports, evaluation and local demonstration.

**Architecture:** Existing Python modules remain, with independent image tracking, metric geometry and world-state estimation. The same estimator feeds live and sequential batch modes; truth is read only by the evaluator.

**Tech Stack:** Python 3.12, NumPy, OpenCV, Ultralytics YOLO11s, PyTorch MPS/CPU, FastAPI, vanilla HTML/JS.

**Spec:** `docs/superpowers/specs/2026-11-03-hackathon-design.md`

## Global Constraints

- No trajectory input at inference, no future observations/training/route fitting/cached inference.
- Preserve original input files and existing native coordinate convention (UE cm).
- One specified target vehicle; report missing model and unknown pivot explicitly.
- Local-only server; live drops reported, batch labeled independently.
- No claim of held-out verification or organizer submission.

## Review Focus

- Distractor with higher confidence must not replace a nearby established image track (Task 1).
- Edge-clipped boxes and nearly parallel rays must not create confident metric fixes (Task 2).
- Missing observations/outliers must not teleport the world track or predict indefinitely (Task 2).
- Batch output must include each source index exactly once, and exports must preserve lost states (Task 4).
- Evaluation must distinguish prediction coverage and Z error from observed horizontal error (Task 3).

### Task 1: Image association and detector upgrade

**Files:** `peleng/tracking.py`, `peleng/detector.py`, `tests/test_tracking.py`, optionally `tests/test_detector.py`.

**Interfaces:** Retain `CameraTrack.update(candidates, timestamp)` and detected/coasting/lost output. Preserve detector factory API `predict(frames)`/`warmup(camera_count)`. Extra candidate/source diagnostics are additive.

- [x] Add tests for nearby low-confidence target vs remote distractor, invalid boxes, scale changes, timeout/reacquisition and backwards/nonfinite time. Run and observe failures.
- [x] Implement conservative causal box association and reset; strengthen semantic default with YOLO11s (weights selected by caller); no image training.
- [x] Run full suite, self-review changes; report results for root integration.

### Task 2: Geometry and causal 3D state

**Files:** `peleng/geometry.py`, new `peleng/world.py`, `tests/test_geometry.py`, new `tests/test_world.py`.

**Interfaces:** Preserve `CameraRig`/legacy `estimate_position` API; new estimator may wrap it. `WorldTrack.update(measurement, timestamp)` produces a JSON-serializable world state including state, position_cm, method, age_s, uncertainty and raw_measurement; root supplies trail.

- [x] Incorporate Astra's geometry review; choose an observable center/contact convention and document any monocular assumptions.
- [x] Add failing behavioral tests for clipped boxes, geometric conditioning, known 3D point, noisy/outlier measurements, variable dt, loss/reacquisition and uncertainty growth.
- [x] Implement metric estimator and constant-velocity covariance filter with gating and bounded prediction. Do not consume object truth.
- [x] Verify full suite and independent code review.

### Task 3: Honest evaluation and report

**Files:** `peleng/evaluate.py`, `tests/test_evaluate.py`, optional focused `peleng/report.py`.

**Interfaces:** Preserve `evaluate_run(run_dir, truth_path)` and existing keys. Add observed/predicted/lost counts, axis MAE/RMSE, horizontal/3D quantiles, method breakdown; CLI writes evaluation.json plus standalone report.html and errors.csv.

- [x] Write failing tests with hand-derived 3/4/12 metre error (=13m 3D), predicted and lost records, malformed reference, empty overlap, exact frame matching; observe failures.
- [x] Implement additional scoring, graphs and coverage without leaking truth into inference. Embed report data safely.
- [x] Run full suite and self-review; root validates report visually.

### Task 4: Integration, export and delivery

**Files:** `peleng/engine.py`, `peleng/__main__.py`, `peleng/server.py`, `static/index.html`, `setup.sh`, README and docs, delivery script, engine/CLI tests.

**Interfaces:** Add `--batch`/`--check-data`; keep old commands working. All output coordinates in trajectory.csv labeled with units and state; no fabricated lost coordinates. Run folder contains saved config for repeatability.

- [x] Add failing batch ordering/export/data-validation tests; observe failures.
- [x] Integrate Tasks 1–3, YOLO11s, batch/full export, preflight and UI quality/method labels; update setup validation and portable launch.
- [x] Freeze configuration, run full wall-clock and full sequential passes, then score only after saving outputs. Compare baseline on shared source frames and report coverage.
- [x] Test Start/Stop/Restart in browser, inspect report, run compile/shell/JS/whitespace checks and full test suite.
- [x] Independent final review, fix important findings with regressions; package reproducible source, model and example results with Russian operating/demo notes.
