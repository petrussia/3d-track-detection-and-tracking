# Project decisions

- The user's target point (5 November 2026) is the **front of the truck cab**,
  following the organizer's training example, not the model centre.
- Read `docs/reference-point-contract.md`. `front_face_center` is the current
  cuboid proxy; its exact offset to the organizer pivot is not yet established.
- Do not silently substitute another point or relabel historical centre runs.
- Geometry replay from saved 2D observations does not test detector/recovery
  changes. Report scope, coverage, same-frame errors and missing outputs together.
