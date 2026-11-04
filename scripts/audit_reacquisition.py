"""Compare world filters on frozen saved inputs; this is NOT end-to-end accuracy."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.world import WorldTrack


def audit(run, revision):
    original = types.ModuleType('baseline_world')
    source = subprocess.check_output(['git', 'show', f'{revision}:peleng/world.py'], text=True)
    exec(compile(source, f'{revision}:peleng/world.py', 'exec'), original.__dict__)
    tracks = {'baseline': original.WorldTrack(), 'candidate': WorldTrack()}
    stats = {key: dict(states=Counter(), resets=[], jumps=[], positions={}) for key in tracks}
    mismatches = []
    previous = dict.fromkeys(tracks)
    count = 0
    with (run/'observations.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            row = json.loads(line)
            frame, timestamp = row['source_frame'], row['source_time']
            count += 1
            for name, track in tracks.items():
                result = track.update(row['world']['raw_measurement'], timestamp)
                point = result['position_cm']
                stat = stats[name]
                stat['states'][result['state']] += 1
                if name == 'baseline':
                    saved = row['world']['position_cm']
                    if (point is None) != (saved is None) or (point is not None and not np.allclose(point, saved, atol=1e-6)):
                        mismatches.append(frame)
                if result['mode_reset']:
                    stat['resets'].append(dict(frame=frame, time_s=timestamp,
                        reason=result.get('reset_reason') or result.get('reacquisition_reason'),
                        method=(result.get('raw_measurement') or {}).get('method')))
                if point is None:
                    previous[name] = None
                    continue
                position = np.asarray(point)/100
                stat['positions'][frame] = position
                if previous[name] is not None:
                    prev_frame, prev_time, prev_point = previous[name]
                    distance = float(np.linalg.norm(position-prev_point))
                    if frame == prev_frame+1 and distance > 10:
                        stat['jumps'].append(dict(frame=frame, displacement_m=distance, dt_s=timestamp-prev_time))
                previous[name] = (frame, timestamp, position)
    baseline, candidate = (stats[k]['positions'] for k in ('baseline','candidate'))
    common = sorted(baseline.keys() & candidate.keys())
    changes = [float(np.linalg.norm(candidate[f]-baseline[f])) for f in common]
    for stat in stats.values():
        stat['coordinate_frames'] = len(stat.pop('positions'))
        stat['coverage_pct'] = 100*stat['coordinate_frames']/count
    return dict(kind='frozen_measurement_replay_not_end_to_end_not_accuracy', frames=count,
        baseline_revision=revision, baseline_mismatch_frames=mismatches, variants=stats,
        common_coordinate_frames=len(common),
        changed_by_over_1m_on_common_frames=sum(v>1 for v in changes))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--baseline', default='23decea3ae146e2f2a08b2fcad66bf39870d5c51')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.run, args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
