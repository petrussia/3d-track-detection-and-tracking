"""Physical-point and missing-frame policy shared by scored and unscored reports."""
SCORER_VERSION = 'named-point-v2'
ESTIMATE_STATES = {'observed', 'coasting', 'predicted'}


def named_reference(reference):
    return {'front': 'front_face_center', 'center': 'cuboid_center'}[reference]


def matches_reference(reference, actual, state):
    return actual == named_reference(reference) and state in ESTIMATE_STATES


def coverage_contract(frames, expected_frames, reference, fps=30.):
    frames = set(frames) & set(range(expected_frames))
    intervals, start = [], None
    for frame in range(expected_frames+1):
        absent = frame < expected_frames and frame not in frames
        if absent and start is None:
            start = frame
        if not absent and start is not None:
            intervals.append(dict(first_frame=start, last_frame=frame-1,
                                  frames=frame-start, duration_s=(frame-start)/fps))
            start = None
    return dict(scorer_version=SCORER_VERSION, selected_reference_point=named_reference(reference),
                expected_frames=expected_frames, coordinate_frames=len(frames),
                coordinate_coverage_pct=100*len(frames)/expected_frames if expected_frames else None,
                coordinate_frame_mask=sorted(frames), missing_intervals=intervals,
                longest_missing_frames=max((x['frames'] for x in intervals), default=0),
                missing_policy='Missing, wrong-reference and non-finite coordinates are failures in all-valid-truth threshold rates; invalid truth is excluded.',
                uncertainty_calibrated=False)
