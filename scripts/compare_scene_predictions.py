"""Compare availability and discontinuities of one named point, without GT."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path


def summarize(path, expected_frames, reference='cuboid_center', fps=30.):
    rows = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines()]
    if [r['source_frame'] for r in rows] != list(range(expected_frames)):
        raise ValueError('Comparison requires a complete, ordered run')
    points, references, resets = {}, Counter(), Counter()
    for row in rows:
        world = row['world']
        position = world.get('position_cm')
        if position is None or world['state'] not in {'observed', 'coasting', 'predicted'}:
            continue
        references[world.get('reference_point')] += 1
        if world.get('reference_point') != reference:
            continue
        if len(position) != 3 or not all(math.isfinite(v) for v in position):
            raise ValueError('Non-finite coordinate')
        points[row['source_frame']] = row
        if world.get('mode_reset'):
            resets[world.get('reset_reason') or 'unspecified_legacy_reset'] += 1
    missing, start = [], None
    for frame in range(expected_frames+1):
        absent = frame < expected_frames and frame not in points
        if absent and start is None:
            start = frame
        if not absent and start is not None:
            missing.append(dict(first_frame=start, last_frame=frame-1, frames=frame-start,
                                duration_s=(frame-start)/fps))
            start = None
    steps = []
    for frame, row in points.items():
        if frame-1 not in points:
            continue
        previous = points[frame-1]
        dt = row.get('source_time',frame/fps)-previous.get('source_time',(frame-1)/fps)
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError('Non-increasing timestamps')
        distance = math.dist(row['world']['position_cm'],previous['world']['position_cm'])/100
        steps.append(dict(frame=frame, displacement_m=distance, delta_s=dt,
            apparent_speed_m_s=distance/dt, reset_reason=row['world'].get('reset_reason'),
            method=row['world'].get('method')))
    return dict(expected_frames=expected_frames, reference=reference,
        named_coordinate_frames=len(points), named_coverage_pct=100*len(points)/expected_frames,
        all_available_references=dict(references),
        states=dict(Counter(r['world']['state'] for r in rows)),
        named_states=dict(Counter(r['world']['state'] for r in points.values())),
        named_methods=dict(Counter(r['world'].get('method') for r in points.values())),
        resets=dict(resets), missing_intervals=missing,
        longest_missing_frames=max((m['frames'] for m in missing),default=0),
        max_adjacent_step_m=max((s['displacement_m'] for s in steps),default=None),
        adjacent_steps_over_10m=[s for s in steps if s['displacement_m']>10]), set(points)


def compare(baseline, current, expected_frames=1932, reference='cuboid_center'):
    before, before_frames = summarize(baseline,expected_frames,reference)
    after, after_frames = summarize(current,expected_frames,reference)
    return dict(ground_truth_available=False, accuracy=None,
        limitation='Availability and discontinuities are not 3D accuracy; fewer jumps may result from more missing estimates.',
        common_named_coordinate_frames=len(before_frames & after_frames),
        baseline=before, current=after)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('current',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--expected-frames',type=int,default=1932)
    args=parser.parse_args()
    args.output.write_text(json.dumps(compare(args.baseline,args.current,args.expected_frames),
        ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
