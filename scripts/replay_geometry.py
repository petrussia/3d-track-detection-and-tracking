"""Recompute shape AND body state from saved image observations, not saved 3D.

This is an isolated geometry experiment: detector/recovery changes are NOT tested.
No reference trajectory is opened. Source timestamps and camera poses are retained.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.geometry import CameraRig
from peleng.shape import ShapeEstimator
from peleng.scene_geometry import SceneGeometry
from peleng.colab_support import export_metres


def replay(data, observations, output, reference='front'):
    output.mkdir(parents=True, exist_ok=False)
    rig = CameraRig(data)
    pipeline = SceneGeometry(rig, ShapeEstimator(rig), reference)
    counts = Counter()
    start = time.perf_counter()
    with observations.open(encoding='utf-8') as source, \
            (output/'observations.jsonl').open('w', encoding='utf-8') as journal, \
            (output/'trajectory.csv').open('w', encoding='utf-8', newline='') as trajectory:
        writer = csv.DictWriter(trajectory, fieldnames=['Frame','Time_s','Loc_X','Loc_Y','Loc_Z',
            'state','method','reference_point','uncertainty_m'])
        writer.writeheader()
        for index, line in enumerate(source):
            row = json.loads(line)
            if row['source_frame'] != index:
                raise ValueError('Replay requires a complete ordered source journal')
            raw = row['camera_observations']
            world, qualified, excluded = pipeline.update(index, row['source_time'], raw)
            counts[world['state']] += 1
            saved = dict(source_frame=index, source_time=row['source_time'], world=world,
                         camera_observations=qualified, excluded_center_observations=excluded,
                         replay_scope='geometry_from_saved_2d_observations')
            journal.write(json.dumps(saved, ensure_ascii=False, allow_nan=False)+'\n')
            position = world['position_cm'] or ['', '', '']
            writer.writerow(dict(Frame=index, Time_s=row['source_time'],
                **dict(zip(['Loc_X','Loc_Y','Loc_Z'],position)),
                **{k:world[k] for k in ['state','method','reference_point','uncertainty_m']}))
            if index % 100 == 0:
                print(index, dict(counts), f'{time.perf_counter()-start:.1f}s', flush=True)
    summary = dict(status='finished', processed_steps=sum(counts.values()), world_states=dict(counts),
        wall_duration_s=time.perf_counter()-start, ground_truth_used_for_inference=False,
        reference=reference, scope='geometry_only_saved_detector_and_2d_tracking',
        source_journal_sha256=hashlib.sha256(observations.read_bytes()).hexdigest())
    (output/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    export_metres(output)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--observations',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference',choices=['front','center'],default='front')
    a=p.parse_args()
    replay(a.data,a.observations,a.output,a.reference)
