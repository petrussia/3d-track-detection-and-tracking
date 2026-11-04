"""Diagnose the world filter on frozen measurements, not a full pipeline benchmark.

Example: python scripts/replay_world.py RUN_DIR --output replay.json
RUN_DIR needs observations.jsonl and errors.csv from the same saved run.
Reference coordinates are used only after all filter updates have completed.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.world import WorldTrack


def replay(run_dir):
    track = WorldTrack()
    results = []
    with (run_dir / 'observations.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            result = track.update(row['world']['raw_measurement'], row['source_time'])
            results.append((row['source_frame'], result))
    # Ground truth must never influence filter updates or parameter selection here.
    with (run_dir / 'errors.csv').open(encoding='utf-8') as stream:
        truth = {int(r['source_frame']): np.array([float(r['truth_'+k+'_m']) for k in 'xyz'])
                 for r in csv.DictReader(stream) if all(r['truth_'+k+'_m'] for k in 'xyz')}
    frames = {frame for frame, _ in results}
    valid_reference = len(frames.intersection(truth))
    errors, jumps = [], []
    previous = None
    coverage = resets = 0
    for frame, result in results:
        resets += bool(result['mode_reset'])
        if result['position_cm'] is None:
            previous = None
            continue
        coverage += 1
        position = np.array(result['position_cm']) / 100
        if previous is not None:
            jumps.append(float(np.linalg.norm(position-previous)))
        previous = position
        if frame in truth:
            errors.append(float(np.linalg.norm(position-truth[frame])))
    return {
        'kind': 'frozen_measurement_filter_replay_not_end_to_end',
        'frames': len(results), 'coordinate_frames': coverage,
        'valid_reference_frames': valid_reference, 'matched_frames': len(errors),
        'rmse_m': float(np.sqrt(np.mean(np.square(errors)))) if errors else None,
        'within_5m_all_valid_pct': 100*sum(e <= 5 for e in errors)/valid_reference if valid_reference else None,
        'resets': resets, 'adjacent_jumps_over_10m': sum(j > 10 for j in jumps),
        'max_adjacent_jump_m': max(jumps, default=None),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    content = json.dumps(replay(args.run_dir), indent=2, allow_nan=False)
    print(content)
    if args.output:
        args.output.write_text(content, encoding='utf-8')
