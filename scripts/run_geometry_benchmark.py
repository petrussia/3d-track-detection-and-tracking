"""Run fresh-process end-to-end geometry comparisons on a Colab GPU.

Detector weights stay unchanged between Way1 variants. Ground truth is opened
only after inference, never passed to Engine. Scene2 has no accuracy score.
"""
import argparse
import json
from pathlib import Path
import signal
import subprocess
import sys


WORKER = r'''
import hashlib,json,shutil,sys,time
from pathlib import Path
args=json.loads(sys.argv[1])
sys.path.insert(0,args['code'])
from peleng.engine import Engine
from peleng.inputs import inspect_dataset
from peleng.colab_support import export_metres,score_run
from peleng.prediction_report import write_prediction_report
info=inspect_dataset(args['data'])
kwargs=dict(device='cuda:0',image_size=960,replay_mode='batch',reference=args['reference'],confidence=args['confidence'])
if args['dimensions'] is not None: kwargs['shape_dimensions_m']=args['dimensions']
engine=Engine(Path(args['data']),Path(args['weights']),Path(args['output']),**kwargs)
engine.start()
try:
    while True:
        state=engine.snapshot()
        print(args['name'],state['status'],state['processed_steps'],'/',info['frames_per_camera'],state['message'],flush=True)
        if state['status'] in {'finished','stopped','error'}: break
        time.sleep(10)
finally:
    if engine.snapshot()['status'] not in {'finished','stopped','error'}: engine.stop()
    engine.wait()
    backup=Path(args['backup']);backup.mkdir(parents=True,exist_ok=True)
    for folder in Path(args['output']).glob('*'):
        if folder.is_dir(): shutil.copytree(folder,backup/folder.name,dirs_exist_ok=True)
state=engine.snapshot()
if state['status']!='finished': raise RuntimeError(state['message'])
run=Path(state['summary']['run_dir'])
export_metres(run)
if args['truth']:
    scores=score_run(run,Path(args['truth']),info['frames_per_camera'],'organizer_pivot_front_face_proxy_not_exactly_verified')
else:
    scores=write_prediction_report(run,info['frames_per_camera'])
environment=dict(args,weights_sha256=hashlib.sha256(Path(args['weights']).read_bytes()).hexdigest(),
    git_commit=__import__('subprocess').check_output(['git','-C',args['code'],'rev-parse','HEAD'],text=True).strip())
(run/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
shutil.copytree(run,backup/run.name,dirs_exist_ok=True)
print('RESULT',json.dumps(dict(name=args['name'],run=str(run),backup=str(backup/run.name),scores=scores)),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--way1', type=Path, required=True)
    parser.add_argument('--scene2', type=Path, required=True)
    parser.add_argument('--way1-weights', type=Path, required=True)
    parser.add_argument('--scene2-weights', type=Path, required=True)
    parser.add_argument('--baseline-code', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--only', choices=['way1-baseline','way1-generic','way1-known','scene2-front','scene2-generic','scene2-known'])
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    profile = json.loads((root/'configs/truck-scene2-geometry.json').read_text())
    cases = [
        ('way1-baseline', args.baseline_code, args.way1, args.way1_weights, 'front', .2, None),
        ('way1-generic', root, args.way1, args.way1_weights, 'front', .2, None),
        ('way1-known', root, args.way1, args.way1_weights, 'front', .2, profile['dimensions_m']),
        ('scene2-front', root, args.scene2, args.scene2_weights, 'front', .5, None),
    ]
    if args.only == 'scene2-known':
        cases.append(('scene2-known', root, args.scene2, args.scene2_weights, 'center', .5, profile['dimensions_m']))
    if args.only == 'scene2-generic':
        # Explicit historical centre diagnostic; not the requested cab output.
        cases.append(('scene2-generic', root, args.scene2, args.scene2_weights, 'center', .5, None))
    for name, code, data, weights, reference, confidence, dimensions in cases:
        if args.only and args.only != name:
            continue
        truth = (data/'object_transform_SM_AA_track.csv') if name.startswith('way1') else None
        if truth is not None and not truth.is_file():
            raise FileNotFoundError(truth)
        request = dict(name=name,code=str(code),data=str(data),weights=str(weights),reference=reference,
            confidence=confidence,dimensions=dimensions,truth=str(truth) if truth else None,
            output=str(args.outputs/name),backup=str(args.backup/name))
        process = subprocess.Popen([sys.executable,'-u','-c',WORKER,json.dumps(request)],cwd=code)
        try:
            if process.wait():
                raise RuntimeError(f'{name} failed; inspect its saved output and progress log')
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise


if __name__ == '__main__':
    main()
