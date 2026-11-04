"""Notebook utilities. Reference coordinates are loaded only after inference."""
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile


def unpack_bundles(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source/'bundle-manifest.json').read_text(encoding='utf-8'))
    for item in manifest:
        marker = destination/f".{item['name']}.sha256"
        if marker.exists() and marker.read_text() == item['sha256'] and (destination/'datasets'/item['name']).is_dir():
            print('Already unpacked:', item['name']); continue
        archive = destination/(item['name']+'.zip')
        digest = hashlib.sha256()
        with archive.open('wb') as dst:
            for name in item['parts']:
                if Path(name).name != name:
                    raise ValueError('Invalid archive part name')
                with (source/name).open('rb') as src:
                    while chunk := src.read(4*1024*1024):
                        digest.update(chunk); dst.write(chunk)
        if archive.stat().st_size != item['bytes'] or digest.hexdigest() != item['sha256']:
            raise ValueError(f"Checksum mismatch: {item['name']}")
        with zipfile.ZipFile(archive) as bundle:
            for member in bundle.infolist():
                target = (destination/member.filename).resolve()
                if not target.is_relative_to(destination.resolve()):
                    raise ValueError('Archive path escapes destination')
            bundle.extractall(destination)
        marker.write_text(item['sha256'])
        print('Verified and unpacked:', item['name'])
    return destination/'datasets'


def discover_datasets(root):
    result = {}
    for folder in sorted(Path(root).iterdir()):
        if not folder.is_dir(): continue
        videos = folder/'videos' if (folder/'videos').is_dir() else folder
        if not list(videos.glob('object_transform_CineCameraActor*.csv')): continue
        truth = folder/'labels/trajectory_m.csv'
        if not truth.exists(): truth = folder/'object_transform_SM_AA_track.csv'
        result[folder.name] = {'videos': videos, 'truth': truth if truth.exists() else None,
                               'truth_reference': 'center' if (folder/'labels/trajectory_m.csv').exists() else 'organizer_unspecified'}
    return result


def normalize_truth(source, destination):
    """Convert synthetic metre CSV for legacy evaluator; preserve damaged original rows."""
    source, destination = Path(source), Path(destination)
    data = source.read_bytes()
    try: text = data.decode('utf-8-sig')
    except UnicodeDecodeError: text = data.decode('cp1251')
    if {'Frame','Loc_X','Loc_Y','Loc_Z'}.issubset(next(csv.reader(io.StringIO(text), delimiter=';'))):
        destination.write_bytes(data); return destination
    rows = csv.DictReader(io.StringIO(text))
    if not {'Frame','X_m','Y_m','Z_m'}.issubset(rows.fieldnames or []):
        raise ValueError('Unknown truth schema')
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, delimiter=';')
        writer.writerow(['Frame','Loc_X','Loc_Y','Loc_Z'])
        for row in rows:
            xyz = [float(row[k])*100 for k in ['X_m','Y_m','Z_m']]
            if not all(math.isfinite(v) for v in xyz): raise ValueError('Nonfinite reference')
            writer.writerow([row['Frame'], *xyz])
    return destination


def score_run(run_dir, truth_path, expected_frames, truth_reference):
    from .evaluate import _evaluate, _write_errors
    from .report import write_report
    run_dir = Path(run_dir)
    truth = normalize_truth(truth_path, run_dir/'evaluation_reference.csv')
    result, points = _evaluate(run_dir, truth)
    # Do not count the extra organizer CSV row beyond the end of the video.
    valid = [p for p in points if 0 <= p['frame'] < expected_frames and p['truth_m'] is not None]
    matched = [p for p in valid if p['three_d_error_m'] is not None]
    config = json.loads((run_dir/'config.json').read_text())
    scores = {'expected_frames': expected_frames, 'valid_reference_video_frames': len(valid),
              'matched_frames': len(matched), 'truth_reference': truth_reference,
              'output_reference': config['reference_point'],
              'coordinate_coverage_pct': 100*result['estimate_steps']/expected_frames if expected_frames else None,
              'all_estimate_metrics': result['all_estimate_metrics'],
              'thresholds': []}
    for limit in [1,2,5,10,15,20]:
        count = sum(p['three_d_error_m'] <= limit for p in matched)
        scores['thresholds'].append({'tolerance_m':limit,'success_frames':count,
            'pct_of_valid_reference_frames':100*count/len(valid) if valid else None,
            'pct_of_matched_frames':100*count/len(matched) if matched else None})
    scores['mse_per_axis_mean_m2'] = (result['all_estimate_metrics']['three_d']['mse_m2']/3
        if matched else None)
    (run_dir/'evaluation.json').write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    (run_dir/'scores.json').write_text(json.dumps(scores,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    _write_errors(run_dir/'errors.csv', points)
    write_report(run_dir/'report.html', result, points, str(run_dir))
    return scores


def export_metres(run_dir):
    run_dir = Path(run_dir)
    with (run_dir/'trajectory.csv').open(encoding='utf-8') as src, (run_dir/'trajectory_m.csv').open('w',encoding='utf-8',newline='') as dst:
        reader=csv.DictReader(src)
        writer=csv.DictWriter(dst,fieldnames=['Frame','Time_s','X_m','Y_m','Z_m','state','method','reference_point'])
        writer.writeheader()
        for row in reader:
            out={k:row[k] for k in ['Frame','Time_s','state','method','reference_point']}
            out.update({axis+'_m':float(row['Loc_'+axis])/100 if row['Loc_'+axis] else '' for axis in 'XYZ'})
            writer.writerow(out)


def show_report(run_dir):
    from IPython.display import HTML, Image, display
    run_dir=Path(run_dir)
    if (run_dir/'scores.json').exists():
        scores=json.loads((run_dir/'scores.json').read_text())
        import pandas as pd
        display(pd.DataFrame(scores['thresholds']))
        display(pd.DataFrame(scores['all_estimate_metrics']['axes']).T)
        print('3D, including predictions:', scores['all_estimate_metrics']['three_d'])
        print('Output reference:',scores['output_reference'],'; reference data:',scores['truth_reference'])
    if (run_dir/'preview.jpg').exists(): display(Image(filename=str(run_dir/'preview.jpg')))
    if (run_dir/'report.html').exists():
        content=html.escape((run_dir/'report.html').read_text(encoding='utf-8'),quote=True)
        display(HTML(f'<iframe sandbox="allow-scripts" srcdoc="{content}" style="width:100%;height:950px;border:0"></iframe>'))


def run_notebook(dataset, weights, outputs, backup, imgsz=960, reference='center'):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU not connected. Runtime > Change runtime type > T4 GPU.')
    from IPython.display import display
    import ipywidgets as widgets
    from .engine import Engine
    from .inputs import inspect_dataset
    info=inspect_dataset(dataset['videos'])
    print('Input:',info)
    engine=Engine(dataset['videos'],Path(weights),Path(outputs),device='cuda:0',
                  image_size=imgsz,replay_mode='batch',reference=reference)
    progress=widgets.IntProgress(min=0,max=info['frames_per_camera'],description='Frames')
    status=widgets.HTML(value='Loading model…')
    views={i:widgets.Image(format='jpeg',width=480) for i in info['camera_ids']}
    display(widgets.VBox([progress,status,widgets.GridBox(list(views.values()),
        layout=widgets.Layout(grid_template_columns='repeat(2, 1fr)'))]))
    engine.start()
    try:
        while True:
            state=engine.snapshot(); progress.value=state['processed_steps']
            status.value=html.escape(f"{state['status']} | frame {state['source_frame']} | {state['message']}")
            for i,view in views.items():
                data=engine.image(i)
                if data: view.value=data
            if state['status'] in {'finished','stopped','error'}: break
            time.sleep(1)
    except BaseException:
        engine.stop(); engine.wait(); raise
    finally:
        engine.wait()
        # Includes partial runs if execution was interrupted or failed.
        Path(backup).mkdir(parents=True,exist_ok=True)
        for candidate in Path(outputs).glob('*'):
            if candidate.is_dir(): shutil.copytree(candidate,Path(backup)/candidate.name,dirs_exist_ok=True)
    state=engine.snapshot()
    if state['status']=='error': raise RuntimeError(state['message'])
    run_dir=Path(state['summary']['run_dir'])
    export_metres(run_dir)
    environment={'python':sys.version,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
                 'dataset':str(dataset['videos']),'reference':reference,'imgsz':imgsz,'weights_name':Path(weights).name,
                 'weights_sha256':hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                 'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    (run_dir/'environment.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    (run_dir/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
    try:
        if dataset['truth']: score_run(run_dir,dataset['truth'],info['frames_per_camera'],dataset['truth_reference'])
        else:
            from .prediction_report import write_prediction_report
            write_prediction_report(run_dir, info['frames_per_camera'])
    finally:
        shutil.copytree(run_dir,Path(backup)/run_dir.name,dirs_exist_ok=True)
    show_report(run_dir)
    return run_dir


def compare_runs(root):
    import pandas as pd
    rows=[]
    for file in Path(root).rglob('scores.json'):
        s=json.loads(file.read_text());env=file.parent/'environment.json'
        env=json.loads(env.read_text()) if env.exists() else {}
        config_path=file.parent/'config.json'
        config=json.loads(config_path.read_text()) if config_path.exists() else {}
        rows.append({'run':file.parent.name,'dataset':env.get('dataset'),
                     'model':env.get('weights_name',config.get('model_path',config.get('weights'))),
                     'imgsz':env.get('imgsz'),
                     'reference':s['output_reference'],'RMSE_m':s['all_estimate_metrics']['three_d']['rmse_m'],
                     'MSE_m2':s['all_estimate_metrics']['three_d']['mse_m2'],
                     'coverage_pct':s['coordinate_coverage_pct'],
                     'within_5m_pct':next(x['pct_of_valid_reference_frames'] for x in s['thresholds'] if x['tolerance_m']==5)})
    return pd.DataFrame(rows)


def prepare_detection_data(datasets_root, destination, stride=6):
    """Scenario split, all views of a frame stay in one split. Original data excluded."""
    import cv2
    import yaml
    splits={'train':['Peleng-01-valley','Peleng-02-occlusion','Peleng-03-moving'],
            'val':['Peleng-04-stop-go'],'test':['Peleng-05-reverse']}
    root,destination=Path(datasets_root),Path(destination)
    counts={}
    for split,names in splits.items():
        image_dir=destination/'images'/split;label_dir=destination/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True);label_dir.mkdir(parents=True,exist_ok=True)
        count=0
        for name in names:
            folder=root/name
            with (folder/'labels/visibility.csv').open() as stream: rows=list(csv.DictReader(stream))
            cameras=sorted({int(r['camera_id']) for r in rows})
            for camera in cameras:
                annotations={int(r['frame']):r for r in rows if int(r['camera_id'])==camera}
                cap=cv2.VideoCapture(str(folder/'videos'/f'Way1.FinalImage.CineCameraActor{camera}.mp4'))
                if not cap.isOpened(): raise ValueError(f'Missing video {name}, camera {camera}')
                index=0
                try:
                    while cap.grab():
                        if index%stride==0:
                            ok,image=cap.retrieve()
                            if not ok or index not in annotations: raise ValueError('Missing frame or label')
                            h,w=image.shape[:2];r=annotations[index];text=''
                            if float(r['visible_pixels'])>0 and r['x_min']:
                                x1,y1,x2,y2=[float(r[k]) for k in ['x_min','y_min','x_max','y_max']]
                                if not (0<=x1<x2<=w and 0<=y1<y2<=h): raise ValueError('Invalid detection box')
                                text=f'0 {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}\n'
                            stem=f'{name}_cam{camera}_{index:05d}'
                            if not cv2.imwrite(str(image_dir/(stem+'.jpg')),image): raise OSError('Cannot write training image')
                            (label_dir/(stem+'.txt')).write_text(text);count+=1
                        index+=1
                finally: cap.release()
        counts[split]=count
    config={'path':str(destination.resolve()),'train':'images/train','val':'images/val','test':'images/test','names':{0:'truck'}}
    path=destination/'dataset.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
    print('Scenario-separated dataset:',counts)
    return path
