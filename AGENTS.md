# Project decisions

- The user's target point (5 November 2026) is the **front of the truck cab**,
  following the organizer's training example, not the model centre.
- Read `docs/reference-point-contract.md`. `front_face_center` is the current
  cuboid proxy; its exact offset to the organizer pivot is not yet established.
- Do not silently substitute another point or relabel historical centre runs.
- Geometry replay from saved 2D observations does not test detector/recovery
  changes. Report scope, coverage, same-frame errors and missing outputs together.

## Test and experiment speed

- Before planning or running tests, Colab benchmarks, model comparisons or
  algorithm experiments, read and follow `docs/fast-experiments.md`.
- Use the smallest sufficient validation scope and compatible cached inputs.
  Do not rerun unchanged detector stages for geometry-only changes or launch
  a full GPU benchmark after every edit. Promote surviving candidates to full
  sequential, all-camera validation before claiming overall improvement.
- Profile before adding concurrency; preserve causal state, frame alignment,
  the cab-front reference and honest coverage/error comparisons.
- The guide distinguishes available commands from proposed optimizations.
  Do not assume proposed cache/window/pipeline flags already exist.
