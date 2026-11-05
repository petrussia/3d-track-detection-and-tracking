"""Filter/triangulation drill using SYNTHETIC annotated centre projections.

This is not detector performance: oracle 2D points isolate mathematical recovery.
The schedule is fixed by frame number before reading the reference trajectory.
"""
import argparse
import csv
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.geometry import CameraRig, estimate_measurement
from peleng.world import WorldTrack
from evaluate_experiments import compare


def run(dataset, output):
    manifest=json.loads((dataset/'manifest.json').read_text())
    count,fps=manifest['frames'],manifest['fps']
    rig=CameraRig(dataset/'videos',width=manifest['width'],height=manifest['height'])
    observations={f:[] for f in range(count)}
    with (dataset/'labels/visibility.csv').open() as stream:
        for row in csv.DictReader(stream):
            f=int(row['frame'])
            if f not in observations or int(row['visible_pixels'])<100 or row['center_in_image']!='1':continue
            x,y=float(row['center_px_x']),float(row['center_px_y'])
            observations[f].append(dict(camera_id=int(row['camera_id']),bbox=[x-.5,y-.5,x+.5,y+.5],
                confidence=.99,anchor_reference='body_center'))
    paths={}
    for variant in ['control','dropouts']:
        path=output/variant;path.mkdir(parents=True,exist_ok=False);paths[variant]=path
        track=WorldTrack()
        with (path/'observations.jsonl').open('w') as stream:
            for f in range(count):
                phase=min(4,f*5//count)
                limit=[2,1,0,1,2][phase] if variant=='dropouts' else 2
                views=sorted(observations[f],key=lambda v:v['camera_id'])[:limit]
                fix=estimate_measurement(rig,f,views,prior_cm=track.predict_position(f/fps),allow_ground=False)
                if fix:
                    fix=dict(fix,reference_point='cuboid_center',reacquisition_qualified=(
                        fix['method']=='triangulated' and fix.get('ray_angle_deg',0)>=10
                        and fix.get('reprojection_max_px',float('inf'))<=8
                        and fix.get('uncertainty_m',float('inf'))<=3))
                world=track.update(fix,f/fps)
                stream.write(json.dumps(dict(source_frame=f,source_time=f/fps,world=world,
                    oracle_observations=True,phase=phase,cameras=[v['camera_id'] for v in views]))+'\n')
        (path/'summary.json').write_text(json.dumps(dict(source_fps=fps,processed_steps=count)))
    # Reference is read only after both causal runs finish.
    result=compare(paths,count,'center',dataset/'labels/object_transform_SM_AA_track.csv')
    result.update(scope='Oracle synthetic centre projections: triangulation and WorldTrack only; no YOLO, recovery, cuboid fitting or cab offset.',
        schedule='Fixed fifths of the video: 2 -> 1 -> 0 -> 1 -> 2 cameras; lowest visible camera IDs.',
        oracle_observations=True)
    rows=[json.loads(s) for s in (paths['dropouts']/'observations.jsonl').read_text().splitlines()]
    return_frame=(4*count+4)//5
    recovered=next((r['source_frame'] for r in rows[return_frame:] if r['world']['state']=='observed'),None)
    result['recovery_after_second_camera_return_s']=(recovered-return_frame)/fps if recovered is not None else None
    (output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(dict(oracle=True,recovery_s=result['recovery_after_second_camera_return_s'],
        coverage={n:r['coordinate_coverage_pct'] for n,r in result['runs'].items()})))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.dataset,a.output)
