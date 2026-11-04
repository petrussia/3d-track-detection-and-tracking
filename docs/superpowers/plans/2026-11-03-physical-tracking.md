# Physical Tracking Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development / executing-plans with test-driven-development. Do not repeat the prior packaging-only outcome.

**Goal:** Fix actual missed-object and wrong-reference/range failures.
**Architecture:** Current-image evidence feeds a persistent 3D shape/pose estimator, then bounded world filtering. Inference remains causal and reference-free.
**Tech Stack:** Existing Python/OpenCV/NumPy/YOLO; SciPy 1.18.1 least_squares for small constrained shape fits.
**Spec:** `docs/superpowers/specs/2026-11-03-physical-tracking-design.md`.

## Global constraints

- No future frames, target trajectory, route/offset fitting or cached detections in inference.
- Original inputs unchanged. Static foreground and neural measurements distinct.
- Shared dimensions learned from silhouettes; front orientation from motion; retain visual center too.
- Validate actual real-frame behavior, not only implementation tests.

## Review focus

- Current foreground must not persist on a disappeared target or seed static rocks.
- Replay reset and frame skips must not leak old evidence or future knowledge.
- Degenerate/partial silhouettes must not collapse learned dimensions.
- Single-view range must depend on established image-derived scale and expose uncertainty.
- Reference/mode changes must not be counted as velocity or hidden from score.

### Task 1: Causal image recovery

Own `peleng/recovery.py`, `tests/test_recovery.py`; preserve other files for root.
Interface `ImageEvidence(rig).update(frames, camera_ids, frame_index, timestamp, neural_candidates) -> list[list[dict]]`, plus `reset()`. Returned candidates have bbox in original pixels, confidence/label, source, neural_detected, measured, quality, and optional crop_only. Process current frames in order, use current/past neural seeds only. Motion follow-ups can be measured if validated; indefinite old boxes cannot.

- [x] Write failing synthetic behavior tests; observe failures.
- [x] Implement static-camera foreground extraction/association, finite timestamps, gaps and reset; optional camera5 verified proposals only.
- [x] Verify on prototype real frames without target truth and provide visual examples/CPU timings.

### Task 2: Shared 3D shape and pose

Own `peleng/shape.py`, `tests/test_shape.py`; preserve engine/world/legacy geometry.
Interface `ShapeEstimator(rig).update(frame_index, timestamp, observations, prior_cm=None, velocity_cm_s=None) -> dict|None`, plus `reset()`. Return standard position measurement (position_cm, covariance_cm2, method, cameras) plus center_cm, front_cm, yaw_rad, dimensions_cm, fit diagnostics/reference/observability. Position reference must be selectable by root; derive both points. No truth input.

- [x] Tests: known two-view shape projection, forward sign from motion, single-view range after learned dimensions, degeneracy rejection, temporal prefix invariance and reset.
- [x] Fit projected bbox extrema with robust bounded least_squares; use held dimensions and heading regularization when appropriate. Validate quality and learn shared dimensions conservatively, not from a single bad silhouette.
- [x] Diagnose with saved image observations only and document remaining ambiguity. Root performs fresh detector runs/evaluation.

### Task 3: Integration and measured correction

Root owns detector.py, engine.py, tracking.py, world.py, CLI, evaluation, UI and delivery.

- [x] Add failed integration tests for image measurements versus neural detections and reference selection/reset.
- [x] Integrate recovery and shape; permit current image evidence for geometry while keeping source/status honest. Add dependency pin and restart resets.
- [x] Fresh full causal batch, visual inspection of previously lost frames, separate direct-frame truth scoring, paired comparison and source-specific metrics.
- [x] Independently review actual algorithm/data flow; fix important defects, rerun appropriate verification, and update ready-to-run artifacts only after quality improvement is demonstrated.

Checkpoint scope: user requested an immediate logical stopping point and ZIP in Downloads. The final current-code run uses batch-ui and all source frames. A new wall-clock live performance repeat is deferred and is not claimed in verification.md. No further refinement is planned for this checkpoint.
