import argparse
import json
from pathlib import Path

from .engine import Engine


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description='Peleng causal multi-camera 3D tracking')
    parser.add_argument('--data',type=Path,default=root/'data')
    parser.add_argument('--weights',type=Path,default=root/'models/yolo11s.pt')
    parser.add_argument('--outputs',type=Path,default=root/'outputs')
    parser.add_argument('--device',default='auto',choices=['auto','cpu','mps','cuda','cuda:0'])
    parser.add_argument('--imgsz',type=int,default=960)
    parser.add_argument('--port',type=int,default=8765)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--batch',action='store_true', help='Process every frame sequentially; no real-time latency claim')
    parser.add_argument('--batch-ui',action='store_true', help='Browser demonstration without dropping frames; suitable for CPU')
    parser.add_argument('--check-data',action='store_true', help='Validate camera inputs without model inference or truth')
    parser.add_argument('--reference', choices=['front','center'], default='front',
                        help='Front-face task reference or geometric volume center; both are exported')
    args = parser.parse_args()
    from .inputs import inspect_dataset
    try:
        inspection = inspect_dataset(args.data)
    except (ValueError, FileNotFoundError, OSError) as exc:
        parser.exit(2, f'Input error: {exc}\n')
    if args.check_data:
        print(json.dumps(inspection, indent=2, ensure_ascii=False))
        return
    engine = Engine(args.data,args.weights,args.outputs,args.device,args.imgsz,
                    replay_mode='batch' if args.batch or args.batch_ui else 'live',reference=args.reference)
    if args.headless or args.batch:
        engine.start()
        try:
            engine.wait()
        except KeyboardInterrupt:
            engine.stop()
            engine.wait()
        state = engine.snapshot()
        print(json.dumps(state['summary'] or state,indent=2,ensure_ascii=False))
        raise SystemExit(1 if state['status']=='error' else 0)
    import uvicorn
    from .server import create_app
    uvicorn.run(create_app(engine),host='127.0.0.1',port=args.port,log_level='warning')


if __name__ == '__main__':
    main()
