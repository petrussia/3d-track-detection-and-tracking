"""Separate, exact-frame post-run scoring; never imported by inference.

Legacy horizontal metrics remain observed-only. Coverage measures availability of
saved world estimates, not detector recall. No alignment search or imputation.
"""
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import statistics


PREDICTED_STATES = {'coasting', 'predicted'}
ESTIMATE_STATES = {'observed', *PREDICTED_STATES}


def _frame(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError('Frame must be a nonnegative integer')
    frame = int(value)
    if frame < 0:
        raise ValueError('Negative frame')
    return frame


def _xyz(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Expected three coordinates')
    if any(isinstance(v, bool) for v in value):
        raise ValueError('Boolean coordinate')
    xyz = [float(v) for v in value]
    if not all(math.isfinite(v) for v in xyz):
        raise ValueError('Nonfinite coordinate')
    return xyz


def _read_truth(path):
    content = path.read_bytes()
    try:
        text = content.decode('utf-8-sig')
    except UnicodeDecodeError:
        text = content.decode('cp1251')
    truth, invalid_frames, invalid_rows = {}, [], []
    reader = csv.DictReader(io.StringIO(text), delimiter=';')
    required = {'Frame', 'Loc_X', 'Loc_Y', 'Loc_Z'}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError('Truth CSV requires Frame;Loc_X;Loc_Y;Loc_Z columns')
    for row in reader:
        frame = None
        try:
            frame = _frame(row['Frame'])
            xyz = _xyz([row[k] for k in ('Loc_X', 'Loc_Y', 'Loc_Z')])
            if frame in truth:
                raise ValueError('Duplicate truth frame; first valid row retained')
            truth[frame] = xyz
        except (TypeError, ValueError, OverflowError) as exc:
            if frame is not None:
                invalid_frames.append(frame)
            invalid_rows.append({'line': reader.line_num, 'frame': frame, 'reason': str(exc)})
    return truth, invalid_frames, invalid_rows


def _read_run(run_dir):
    summary = {}
    summary_path = run_dir/'summary.json'
    if summary_path.exists():
        try:
            candidate = json.loads(summary_path.read_text(encoding='utf-8'))
            if isinstance(candidate, dict):
                summary = candidate
        except (ValueError, UnicodeDecodeError):
            pass
    try:
        fps = float(summary.get('source_fps', 30))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('Invalid FPS')
    except (TypeError, ValueError, OverflowError):
        fps = 30.
    records, invalid_rows, invalid_estimates = {}, [], 0
    with (run_dir/'observations.jsonl').open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError('Expected a JSON object')
                frame = _frame(item['source_frame'])
                world = item['world']
                if not isinstance(world, dict) or not isinstance(world.get('state'), str):
                    raise ValueError('Missing world state')
                if frame in records:
                    raise ValueError('Duplicate source frame; first valid row retained')
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                invalid_rows.append({'line': line_number, 'reason': str(exc)})
                continue
            state = world['state']
            position = None
            if state in ESTIMATE_STATES:
                try:
                    position = _xyz(world.get('position_cm'))
                except (TypeError, ValueError, OverflowError):
                    invalid_estimates += 1
            # A lost state never exposes stale coordinates, even if its trace has them.
            time_s = item.get('source_time', frame/fps)
            try:
                time_s = float(time_s)
                if not math.isfinite(time_s) or time_s < 0:
                    raise ValueError('Invalid time')
            except (TypeError, ValueError, OverflowError):
                time_s = frame/fps
            method = world.get('method')
            mode = world.get('mode')
            segment = world.get('segment_id')
            if isinstance(segment, bool) or not isinstance(segment, int) or segment < 0:
                segment = None
            records[frame] = {'frame': frame, 'state': state, 'position_cm': position,
                              'method': method if isinstance(method, str) else 'unknown',
                              'mode': mode if isinstance(mode, str) else 'unknown',
                              'segment_id': segment, 'mode_reset': world.get('mode_reset') is True,
                              'source_time': time_s}
    return records, fps, invalid_rows, invalid_estimates


def _percentile(values, q):
    """Linear percentile between sorted samples (same convention as NumPy default)."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered)-1)*q
    left = math.floor(index)
    right = math.ceil(index)
    return ordered[left] + (ordered[right]-ordered[left])*(index-left)


def _distribution(values):
    return {'mean_m': statistics.mean(values) if values else None,
            'mse_m2': statistics.mean(v*v for v in values) if values else None,
            'rmse_m': math.sqrt(statistics.mean(v*v for v in values)) if values else None,
            'median_m': statistics.median(values) if values else None,
            'p90_m': _percentile(values, .90), 'p95_m': _percentile(values, .95),
            'p99_m': _percentile(values, .99), 'max_m': max(values) if values else None}


def _metrics(errors):
    axes = {}
    for axis, name in enumerate('xyz'):
        values = [e[axis] for e in errors]
        axes[name] = {'mae_m': statistics.mean(abs(v) for v in values) if values else None,
                      'mse_m2': statistics.mean(v*v for v in values) if values else None,
                      'rmse_m': math.sqrt(statistics.mean(v*v for v in values)) if values else None}
    return {'count': len(errors), 'axes': axes,
            'horizontal': _distribution([math.hypot(e[0], e[1]) for e in errors]),
            'three_d': _distribution([math.hypot(*e) for e in errors])}


def _errors(records, truth, states=ESTIMATE_STATES, frames=None):
    return [[(record['position_cm'][axis]-truth[frame][axis])/100 for axis in range(3)]
            for frame, record in sorted(records.items())
            if frame in truth and record['position_cm'] is not None
            and record['state'] in states and (frames is None or frame in frames)]


def _segment_boundary(previous, current):
    return (current['mode_reset'] or current['segment_id'] != previous['segment_id']
            or current['mode'] != previous['mode'])


def _movement(records, within_segment=False):
    pairs = []
    previous_record, previous_estimate = None, None
    boundary, boundary_pairs = False, 0
    for _, current in sorted(records.items()):
        if previous_record is not None:
            boundary = boundary or _segment_boundary(previous_record, current)
        previous_record = current
        if current['position_cm'] is None:
            continue
        if previous_estimate is not None:
            boundary_pairs += int(boundary)
            if not (within_segment and boundary):
                dt = current['source_time']-previous_estimate['source_time']
                jump = math.dist(current['position_cm'], previous_estimate['position_cm'])/100
                pairs.append({'from_frame': previous_estimate['frame'], 'to_frame': current['frame'],
                              'from_segment_id': previous_estimate['segment_id'],
                              'to_segment_id': current['segment_id'],
                              'from_mode': previous_estimate['mode'], 'to_mode': current['mode'],
                              'crosses_segment_boundary': boundary,
                              'dt_s': dt, 'jump_m': jump,
                              'speed_m_s': jump/dt if dt > 0 else None})
        previous_estimate, boundary = current, False
    speeds = [p['speed_m_s'] for p in pairs if p['speed_m_s'] is not None]
    return {'pair_count': len(pairs), 'jump_threshold_m': 10., 'speed_threshold_m_s': 50.,
            'large_jump_count': sum(p['jump_m'] > 10 for p in pairs),
            'large_speed_count': sum(s > 50 for s in speeds),
            'max_jump_m': max((p['jump_m'] for p in pairs), default=None),
            'max_speed_m_s': max(speeds, default=None),
            'mean_speed_m_s': statistics.mean(speeds) if speeds else None,
            'nonpositive_dt_pairs': sum(p['dt_s'] <= 0 for p in pairs),
            'segment_boundary_pairs': boundary_pairs,
            'excluded_boundary_pairs': boundary_pairs if within_segment else 0,
            'definition': ('Within-segment only: mode_reset, segment_id changes and mode changes (including in intervening lost records) exclude a pair. ' if within_segment else 'All estimates, including mode/segment resets; reset corrections are not necessarily physical motion. ')
                          + 'Between consecutive available estimates, including across gaps; actual source_time dt, or frame/source_fps when absent. Thresholds are diagnostics, not accuracy criteria.',
            'pairs': pairs}


def _fraction(numerator, denominator):
    return numerator/denominator if denominator else None


def _group_metrics(records, truth, field):
    groups = {}
    for label in sorted({r[field] for r in records.values()}):
        subset = {frame: r for frame, r in records.items() if r[field] == label}
        available = [r for r in subset.values() if r['position_cm'] is not None]
        observed = _metrics(_errors(subset, truth, {'observed'}))
        predicted = _metrics(_errors(subset, truth, PREDICTED_STATES))
        all_estimates = _metrics(_errors(subset, truth))
        groups[label] = {
            'processed_steps': len(subset), 'estimate_steps': len(available),
            'state_counts': dict(sorted(Counter(r['state'] for r in subset.values()).items())),
            'observed_metrics': observed, 'predicted_metrics': predicted,
            'all_estimate_metrics': all_estimates,
            'coverage': {
                'estimate_of_processed': _fraction(len(available), len(records)),
                'estimate_of_truth': _fraction(all_estimates['count'], len(truth)),
                'observed_of_processed': _fraction(sum(r['state'] == 'observed' for r in available), len(records)),
                'observed_of_truth': _fraction(observed['count'], len(truth)),
                'estimate_of_group': _fraction(len(available), len(subset)),
            },
        }
    return groups


def _analyse(run_dir, truth):
    records, fps, invalid_rows, invalid_estimates = _read_run(run_dir)
    counts = Counter(r['state'] for r in records.values())
    available = [r for r in records.values() if r['position_cm'] is not None]
    observed = _metrics(_errors(records, truth, {'observed'}))
    predicted = _metrics(_errors(records, truth, PREDICTED_STATES))
    all_estimates = _metrics(_errors(records, truth))
    observed_available = sum(r['state'] == 'observed' for r in available)
    predicted_available = sum(r['state'] in PREDICTED_STATES for r in available)
    total = len(records)
    result = {
        'processed_steps': total, 'truth_frames': len(truth),
        'processed_truth_steps': len(records.keys() & truth.keys()),
        'observed_steps': counts['observed'],
        'predicted_steps': sum(counts[s] for s in PREDICTED_STATES),
        'lost_steps': counts['lost'],
        'unknown_state_steps': sum(n for s, n in counts.items() if s not in ESTIMATE_STATES | {'lost'}),
        'state_counts': dict(sorted(counts.items())),
        'estimate_steps': len(available), 'invalid_estimate_steps': invalid_estimates,
        'matched_observed_steps': observed['count'], 'matched_predicted_steps': predicted['count'],
        'matched_estimate_steps': all_estimates['count'],
        'observed_metrics': observed, 'predicted_metrics': predicted,
        'all_estimate_metrics': all_estimates,
        'axis_mae_m': {axis: values['mae_m'] for axis, values in observed['axes'].items()},
        'axis_rmse_m': {axis: values['rmse_m'] for axis, values in observed['axes'].items()},
        'horizontal_mae_m': observed['horizontal']['mean_m'],
        'horizontal_median_m': observed['horizontal']['median_m'],
        'horizontal_max_m': observed['horizontal']['max_m'],
        'horizontal_rmse_m': observed['horizontal']['rmse_m'],
        'horizontal_p95_m': observed['horizontal']['p95_m'],
        'three_d_mae_m': observed['three_d']['mean_m'],
        'three_d_rmse_m': observed['three_d']['rmse_m'],
        'three_d_median_m': observed['three_d']['median_m'],
        'three_d_p95_m': observed['three_d']['p95_m'],
        'three_d_max_m': observed['three_d']['max_m'],
        'coverage': {
            'observed_of_processed': _fraction(observed_available, total),
            'predicted_of_processed': _fraction(predicted_available, total),
            'estimate_of_processed': _fraction(len(available), total),
            'lost_of_processed': _fraction(counts['lost'], total),
            'processed_of_truth': _fraction(len(records.keys() & truth.keys()), len(truth)),
            'observed_of_truth': _fraction(observed['count'], len(truth)),
            'predicted_of_truth': _fraction(predicted['count'], len(truth)),
            'estimate_of_truth': _fraction(all_estimates['count'], len(truth)),
        },
        'coverage_definition': 'Observed/predicted coverage counts finite XYZ estimates; processed denominator is unique valid trace frames, truth denominator is unique valid reference frames. Missing rows are not imputed. This is world-estimate availability, not detection recall.',
        'method_counts': dict(sorted(Counter(r['method'] for r in available).items())),
        'observed_method_counts': dict(sorted(Counter(r['method'] for r in available if r['state'] == 'observed').items())),
        'mode_counts': dict(sorted(Counter(r['mode'] for r in records.values()).items())),
        'metrics_by_method': _group_metrics(records, truth, 'method'),
        'metrics_by_mode': _group_metrics(records, truth, 'mode'),
        'group_coverage_definition': 'Group estimate/observed_of_processed and _of_truth use the entire run/reference denominators; estimate_of_group uses the records in that group. All modes, including assumed_ground, remain in the main aggregate.',
        'segment_count': len({r['segment_id'] for r in records.values() if r['segment_id'] is not None}),
        'segment_reset_steps': sum(r['mode_reset'] for r in records.values()),
        'source_fps': fps,
        'source_span_s': (max(records)-min(records))/fps if records else 0.,
        'sample_durations_s': {s: n/fps for s, n in sorted(counts.items())},
        'duration_definition': 'Sample durations are count/source_fps, not continuous state durations across skipped frames. Source span is (last-first source_frame)/source_fps; FPS defaults to 30 when absent or invalid.',
        'invalid_observation_rows': invalid_rows,
        'discontinuities': _movement(records),
        'within_segment_discontinuities': _movement(records, within_segment=True),
        'assumptions': 'Direct CSV/video source_frame matching; cm converted to m; no offset search, interpolation or tuning. Reference is an object pivot; an image bbox representative point need not coincide with that pivot. Inspect the saved method/config for geometric assumptions. XYZ errors, including altitude, are scored. Legacy top-level errors are observed-only; predictions are reported separately.',
        'ground_truth_used_for_inference': False,
    }
    points = []
    previous = None
    for frame in sorted(records.keys() | truth.keys()):
        record = records.get(frame)
        target = truth.get(frame)
        estimate = record['position_cm'] if record else None
        error = [(estimate[i]-target[i])/100 for i in range(3)] if target is not None and estimate is not None else None
        points.append({'frame': frame, 'time_s': frame/fps,
                       'record_time_s': record['source_time'] if record else None,
                       'state': record['state'] if record else 'unprocessed',
                       'method': record['method'] if record else '',
                       'mode': record['mode'] if record else None,
                       'segment_id': record['segment_id'] if record else None,
                       'mode_reset': record['mode_reset'] if record else False,
                       'connect_estimate': bool(previous is not None and record is not None
                                                and previous['frame']+1 == frame
                                                and previous['position_cm'] is not None and estimate is not None
                                                and not _segment_boundary(previous, record)),
                       'truth_m': [v/100 for v in target] if target is not None else None,
                       'estimate_m': [v/100 for v in estimate] if estimate is not None else None,
                       'error_m': error,
                       'horizontal_error_m': math.hypot(error[0], error[1]) if error is not None else None,
                       'three_d_error_m': math.hypot(*error) if error is not None else None})
        previous = record
    return result, points, records


def _paired(current, baseline, truth, states):
    frames = {frame for frame, record in current.items()
              if frame in truth and frame in baseline
              and record['position_cm'] is not None and baseline[frame]['position_cm'] is not None
              and record['state'] in states and baseline[frame]['state'] in states}
    return {'source_frames': sorted(frames), 'count': len(frames),
            'current': _metrics(_errors(current, truth, states, frames)),
            'baseline': _metrics(_errors(baseline, truth, states, frames))}


def _evaluate(run_dir, truth_path, baseline_run=None):
    truth, invalid_frames, invalid_rows = _read_truth(truth_path)
    result, points, records = _analyse(run_dir, truth)
    result.update(invalid_truth_frames=invalid_frames, invalid_truth_rows=invalid_rows)
    if baseline_run is not None:
        baseline_result, _, baseline_records = _analyse(Path(baseline_run), truth)
        result['baseline_comparison'] = {
            'run': str(baseline_run),
            'definition': 'Both runs scored on exactly the same source frames with valid reference and estimate; no time shift or parameter selection. Observed-paired requires observed in both; all-estimate-paired permits predictions.',
            'observed_paired': _paired(records, baseline_records, truth, {'observed'}),
            'all_estimate_paired': _paired(records, baseline_records, truth, ESTIMATE_STATES),
            'baseline_coverage': baseline_result['coverage'],
            'baseline_processed_steps': baseline_result['processed_steps'],
            'baseline_invalid_observation_rows': baseline_result['invalid_observation_rows'],
        }
    return result, points


def evaluate_run(run_dir: Path, truth_path: Path, baseline_run: Path | None = None):
    """Score saved results without modifying inference, alignment or algorithm settings."""
    return _evaluate(Path(run_dir), Path(truth_path), baseline_run)[0]


def _write_errors(path, points):
    fields = ['source_frame', 'time_s', 'record_time_s', 'state', 'method', 'mode', 'segment_id', 'mode_reset',
              'truth_x_m', 'truth_y_m', 'truth_z_m', 'estimate_x_m', 'estimate_y_m', 'estimate_z_m',
              'error_x_m', 'error_y_m', 'error_z_m', 'error_horizontal_m', 'error_3d_m']
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for point in points:
            row = {'source_frame': point['frame'], 'time_s': point['time_s'],
                   'record_time_s': point['record_time_s'], 'state': point['state'], 'method': point['method'],
                   'mode': point['mode'], 'segment_id': point['segment_id'], 'mode_reset': point['mode_reset'],
                   'error_horizontal_m': point['horizontal_error_m'], 'error_3d_m': point['three_d_error_m']}
            for key, prefix in [('truth_m', 'truth'), ('estimate_m', 'estimate'), ('error_m', 'error')]:
                values = point[key]
                for axis, name in enumerate('xyz'):
                    row[f'{prefix}_{name}_m'] = values[axis] if values is not None else None
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description='Score saved estimates after a completed run, using exact source frames')
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--truth', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Optional saved run; quality compared only on paired source frames')
    args = parser.parse_args()
    result, points = _evaluate(args.run, args.truth, args.baseline)
    output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    (args.run/'evaluation.json').write_text(output, encoding='utf-8')
    _write_errors(args.run/'errors.csv', points)
    from peleng.report import write_report
    write_report(args.run/'report.html', result, points, str(args.run))
    print(output)


if __name__ == '__main__':
    main()
