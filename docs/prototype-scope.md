> Historical spike scope, superseded by requirements.md and the 2026-11-03 hackathon design.

# Local real-time research prototype

User authorized a test implementation on the supplied inputs and local ML on this laptop.
This is an exploratory spike, not a final interpretation of the hackathon assignment.

- Five 1080p/30 FPS recordings emulate synchronized live streams in wall-clock time.
- Only frames whose source timestamps have arrived may be processed. No look-ahead,
  backward pass, trajectory-based tuning, offline model training, or cached detections.
- Pretrained YOLO detects vehicles. Past detections may support short causal coasting.
- Current camera calibration is an assumed permitted input. UE coordinates are assumed
  centimeters, X forward / Y right / Z up; this must be confirmed with the organizers.
- Object trajectory CSV is never read by the inference process. An optional separate
  evaluation command may read it only after inference outputs are saved.
- World position is approximate: bottom-center ground contact and z=0 unless sufficient
  geometrically consistent simultaneous views support triangulation. Render this limitation.
- Skipping old frames is allowed in this prototype to bound latency; dropped frames,
  output cadence, and end-to-end age must be reported, not hidden.
- Warm-up is timed separately. Runtime latency includes decoding, inference, drawing and
  publishing frames; it does not include browser display latency.

Implementation: causal reader + detector/tracker, camera geometry, local FastAPI dashboard.
Verification: focused causal/geometry/state tests, one full wall-clock stream run, browser QA,
saved per-step observations and summary metrics, independent code review.
