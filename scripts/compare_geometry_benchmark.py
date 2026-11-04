"""Compare completed Way1 runs on the same named physical reference and frames."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.evaluate import _read_truth, _metrics


def diagnostics(named, truth):
    samples = []
    for frame, world in named.items():
        if frame not in truth:
            continue
        error = [(world['position_cm'][a]-truth[frame][a])/100 for a in range(3)]
        samples.append(dict(frame=frame, error=error, distance=math.hypot(*error), world=world))
    grouped = {}
    for field in ['state', 'method']:
        groups = {}
        for item in samples:
            groups.setdefault(item['world'].get(field, 'unknown'), []).append(item['error'])
        grouped[field] = {str(k): _metrics(v) for k, v in groups.items()}
    bins = []
    for low, high in [(0, 2), (2, 5), (5, 10), (10, math.inf)]:
        selected = [s for s in samples if isinstance(s['world'].get('uncertainty_m'), (int, float))
                    and low <= s['world']['uncertainty_m'] < high]
        bins.append(dict(lower_m=low, upper_m=high if math.isfinite(high) else None,
                         metrics=_metrics([s['error'] for s in selected])))
    total_squared = sum(s['distance']**2 for s in samples)
    worst = sorted(samples, key=lambda s:s['distance'], reverse=True)[:10]
    return dict(by=grouped, uncertainty_bins=bins,
        uncertainty_note='Bins use reported maximum-axis standard deviation, not a calibrated 3D confidence radius.',
        worst_frames=[dict(frame=s['frame'], error_m=s['distance'], error_xyz_m=s['error'],
                          method=s['world'].get('method'), state=s['world'].get('state')) for s in worst],
        worst_ten_squared_error_share_pct=(100*sum(s['distance']**2 for s in worst)/total_squared
                                            if total_squared else 0.))


def compare(root, truth_path, expected_frames=1001):
    truth, invalid_frames, invalid_rows = _read_truth(truth_path)
    truth = {f:p for f,p in truth.items() if 0 <= f < expected_frames}
    runs, points = {}, {}
    for name in ['way1-baseline', 'way1-generic', 'way1-known']:
        folders = sorted(p.parent for p in (root/name).glob('*/observations.jsonl'))
        if not folders:
            raise FileNotFoundError(name)
        folder = folders[-1]
        rows = [json.loads(s) for s in (folder/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
        frames = [r['source_frame'] for r in rows]
        if frames != list(range(expected_frames)):
            raise ValueError(f'{name}: incomplete or out-of-order run')
        available = {r['source_frame']:r['world'] for r in rows
                     if r['world']['position_cm'] is not None}
        named = {f:w for f,w in available.items() if w['reference_point']=='front_face_center'}
        points[name] = {f:w['position_cm'] for f,w in named.items() if f in truth}
        errors = [[(p[a]-truth[f][a])/100 for a in range(3)] for f,p in points[name].items()]
        errors_3d = [math.hypot(*e) for e in errors]
        runs[name] = dict(run=str(folder), available_frames=len(available),
            named_reference_frames=len(named), matched_named_reference_frames=len(errors),
            named_reference_coverage_pct=100*len(named)/expected_frames,
            excluded_other_references=dict(Counter(w['reference_point'] for f,w in available.items() if f not in named)),
            own_available_metrics=_metrics(errors),
            diagnostics=diagnostics(named, truth),
            thresholds=[dict(tolerance_m=t,success_frames=sum(e<=t for e in errors_3d),
                pct_of_all_valid_truth=100*sum(e<=t for e in errors_3d)/len(truth)) for t in [1,2,5,10,15,20]])
    common = set.intersection(*(set(p) for p in points.values()))
    for name, positions in points.items():
        errors = [[(positions[f][a]-truth[f][a])/100 for a in range(3)] for f in sorted(common)]
        runs[name]['common_frame_metrics'] = _metrics(errors)
    return dict(kind='end_to_end_same_named_reference_common_frames',
        reference='front_face_center',
        reference_limitation='Front-face center is a visually supported proxy for organizer pivot; exact actor pivot is not verified.',
        valid_truth_frames=len(truth), invalid_truth_frames=invalid_frames,
        invalid_truth_rows=invalid_rows, common_frames=len(common), variants=runs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.root, args.truth)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
