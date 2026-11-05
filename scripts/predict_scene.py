"""Fresh-process GPU inference with progress and durable, truth-free output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.colab_support import export_metres
from peleng.engine import Engine
from peleng.inputs import inspect_dataset
from peleng.prediction_report import write_prediction_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--confidence', type=float, default=.20)
    parser.add_argument('--reference', choices=['center', 'front'], default='front')
    parser.add_argument('--shape-dimensions-m', nargs=3, type=float, metavar=('L', 'W', 'H'))
    args = parser.parse_args()
    info = inspect_dataset(args.data)
    print(json.dumps(info), flush=True)
    engine = Engine(args.data, args.weights, args.outputs, device='cuda:0', image_size=960,
                    replay_mode='batch', reference=args.reference, confidence=args.confidence,
                    shape_dimensions_m=args.shape_dimensions_m)
    engine.start()
    try:
        while True:
            state = engine.snapshot()
            print(f"{state['status']} | {state['processed_steps']}/{info['frames_per_camera']} | {state['message']}", flush=True)
            if state['status'] in {'finished', 'stopped', 'error'}: break
            time.sleep(10)
    finally:
        if engine.snapshot()['status'] not in {'finished', 'stopped', 'error'}: engine.stop()
        engine.wait()
        args.backup.mkdir(parents=True, exist_ok=True)
        for folder in args.outputs.iterdir():
            if folder.is_dir(): shutil.copytree(folder, args.backup / folder.name, dirs_exist_ok=True)
    state = engine.snapshot()
    if state['status'] != 'finished': raise RuntimeError(state['message'])
    run = Path(state['summary']['run_dir'])
    export_metres(run)
    diagnostics = write_prediction_report(run, info['frames_per_camera'])
    environment = {'weights': str(args.weights), 'weights_sha256': hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                   'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                   'dataset': str(args.data), 'reference': args.reference, 'imgsz': 960,
                   'confidence': args.confidence, 'shape_dimensions_m': args.shape_dimensions_m}
    (run / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    shutil.copytree(run, args.backup / run.name, dirs_exist_ok=True)
    print(json.dumps(diagnostics), flush=True)
    print('SAVED:', args.backup / run.name, flush=True)


if __name__ == '__main__':
    main()
