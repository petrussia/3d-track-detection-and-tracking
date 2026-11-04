# Peleng hackathon solution

## Authority and outcome

The user asks to complete the existing algorithm and deliver a usable local hackathon solution. Use the two supplied lectures as requirements. Zuyenok slide 2 requires automatic detection/tracking of the specified object and X/Y/Z coordinates; slide 7 suggests detection → tracking → triangulation; slide 8 illustrates true/estimated X/Y/Z plots. Bespaltsev slides 13–14 motivate explicit loss/recovery, association, stability, latency and occlusion checks.

The slides set no numerical accuracy or FPS threshold and no exact submission schema. Zuyenok slide 10 announces a verification dataset and form on November 4 at 20:00, normal submission by midnight and a late deadline November 5 at 06:00. These times are as printed, with no timezone/year in the source. The linked project2.7z is the existing 11-file dataset; no 3D model is present despite slide 9 listing one.

## Constraints

- Keep the existing Python/OpenCV/Ultralytics/FastAPI application and all original inputs.
- Local inference only. Preserve causal processing: increasing source frames, no future observations, no backward smoothing, route fitting, training or cached detections.
- Never import or read the object trajectory in inference. Separate post-run evaluation only.
- Maintain real-time replay with explicit dropped frames and publication latency. Add clearly labeled sequential batch mode for complete causal frame-by-frame export; it makes no real-time throughput claim.
- One target vehicle, automatically initialized; target identity is not a general multi-object re-identification guarantee.
- Keep observations, predictions and loss distinguishable. Record raw measurements and rejected outliers.
- No fabricated accuracy, detector recall, held-out results or organizer submission. Provide ready-to-run materials, with final official export mapping deferred until the verification schema exists.

## Approaches considered and chosen

1. Tune the old detector and smooth ground-plane estimates. Smallest change, but it cannot solve systematic unknown-terrain/altitude errors.
2. Build a causal multi-view estimator with stronger pretrained detection, temporal association, robust geometric quality checks and a variable-dt 3D motion filter. Chosen as the best fit to the supplied inputs and local application.
3. Train a special model or reconstruct the entire scene. More data/3D assets and validation would be required; does not fit the current causal/no-training scope.

These are implementation decisions delegated by the user's turnkey request, not separately approved scientific assumptions. GPT-6 Astra Max reviewed geometry and independently reviewed the final implementation.

## Components and contracts

- Image tracking: validate candidates; associate to predicted box by proximity/overlap/scale; do not jump to a remote high-confidence distractor; short coasting expires; reacquisition after loss must be explicit. Stronger YOLO11s weights remain replaceable by CLI.
- Geometry: retain original CameraRig conventions; reject clipped boxes for metric estimation; multi-camera estimates expose conditioning, residuals and uncertainty. A bbox representative point is not a physical keypoint or guaranteed mesh pivot. Monocular assumptions must be explicit and conservative; no silently forced Z=0 altitude guarantee.
- Metric anchor: bbox visual volume centre for triangulation and bearing updates; ground fallback uses bbox bottom contact as an explicitly different proxy. Never add a learned pivot/route offset.
- World modes: current ground-plane pseudo-fixes with persistent terrain-bias uncertainty until valid triangulation; reset state/velocity/segment on ground→metric, depth expiry, or ground-camera handoff. Rank-two bearings only after a triangulated depth, at most 5 seconds; bound coast to 0.5 seconds, posterior speed to 50m/s and corrections to max(10m,50m/s*dt). These are model assumptions, not organizer thresholds.
- World tracking: causal variable-dt constant-velocity filter, covariance/innovation gating, bounded prediction gap, uncertainty growth, and reacquisition after long loss. Return state/position_cm/method/age_s/trail plus raw measurement and quality information.
- Execution: Engine live and batch modes use the same detector/tracker/geometry; live skips stale input, batch processes every index in order. Save observations.jsonl, trajectory.csv, summary.json and preview.
- Evaluation: report X/Y/Z and horizontal/3D errors, quantiles, observed/predicted/lost coverage, method counts and discontinuities; never call detection count recall. Generate a self-contained HTML report with X/Y/Z reference/estimate time graphs and XY path.
- Delivery: Russian README, requirements traceability, run/check/score commands, demonstration notes and a local ZIP with code, frozen environment, weights and example outputs. Check on original five videos and through browser Start/Stop/Restart.

## Acceptance

Existing tests stay green; new regression tests demonstrate distractor rejection, clipped-box/degenerate geometry, filter outlier rejection, motion prediction and timeout, batch causal ordering and export, and independent 3-axis evaluation. Use separate post-run truth scoring; correctness fixes and reference conventions were selected from synthetic failures and image-ray consistency, without a route/pivot fit. Report the development-set nature of all scores; report improvements with coverage and paired-frame comparisons, not MAE alone. A new verification dataset remains necessary to claim hackathon acceptance or generalization.

## Final scientific limit

The user confirmed using the visual centre when the pivot is unknown. Separate development-set scoring gives observed 3D mean error 12.57m, XYZ coverage 74.7% including predictions, and no demonstrated XY improvement on identical paired frames. The code and local delivery are complete; etalon accuracy, held-out validation and organizer submission remain unconfirmed.
