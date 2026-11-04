# Physical target tracking correction

The prior delivery is not adequate: 12.57m mean 3D error, 25% missing world positions, and no paired-frame XY improvement. The user explicitly asks to continue implementation. Preserve causal/local inference and never load the target trajectory into inference. Offline truth projection was used to diagnose the output convention, not to fit an offset or route.

## Proven causes

- The exported reference projects near the forward cab/end of the truck. Box-center triangulation is consistently about 10–12m behind it despite subpixel ray consistency. No global coordinate or timing correction is justified; the wrong physical point is being reported.
- The truck remains plainly visible in static camera 2 through lost frames 173–316 and in camera 4 through later gaps. Causal foreground tracking seeded from past neural evidence recovers candidate image observations in 247/252 lost steps in an exploratory CPU prototype. This is not yet a validated accuracy result.
- Bearing-only extrapolation has no range observation and drifts when speed changes. A rigid shape estimated from simultaneous views can supply apparent-size constraints during later single-view periods.

## Chosen correction

Use current image evidence, not just neural box memory. Static-camera MOG2 foreground components are associated to an already seeded target with causal scale/proximity checks. Current foreground support is an image observation with its own source/status; do not call it neural detection. For a rotating camera, compensate known rotation and use flow/foreground only if verified; crop-only hints must not become metric boxes without validation. Stop on disappearance and reset on replay.

Fit a persistent oriented 3D cuboid to full-object bbox edges from multiple calibrated views. Learn dimensions only from current/past images and camera calibration, with robust shared estimates and fit-quality/observability checks; never learn dimensions from reference coordinates. Infer forward orientation from causal target motion, not truth yaw. Reuse established dimensions for single-view pose/range when sufficiently constrained. Reject bad/partial/clipped silhouettes and mark priors explicitly. A per-frame box fit with collapsing width or unconstrained shape is insufficient.

Report both geometric center and front-face center derived from the fitted volume, with an explicit reference label; never add a truth-fitted 10m translation. The user's earlier visual-center preference remains an available reference; task-coordinate output must make the front/cab convention visible. Keep mode/reference switches separate from motion and propagate uncertainty. When shape is not identifiable, fall back honestly rather than manufacture confidence.

## Acceptance

New regressions prove causal foreground continuation, disappearance, static-background rejection, reset and skipped frames; cuboid tests use independently projected known shapes and front orientation. Full real-frame runs must show materially better visibility/continuity and position error, including paired frames and all hard frames. Review representative recovered boxes visually, retain raw evidence and preserve before/after results. No completion claim based only on passing unit tests or packaging.
