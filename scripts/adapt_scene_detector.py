"""Train a scene-specific detector from reviewed boxes; keep frames grouped by split."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil

import cv2


def prepare(source, annotations, destination):
    manifest = json.loads(Path(annotations).read_text(encoding='utf-8'))
    rows = manifest['images']
    seen, groups = set(), {}
    for row in rows:
        if row['split'] not in {'train', 'val', 'test'}:
            raise ValueError('Unknown dataset split')
        if len(row['boxes']) > 1:
            raise ValueError('This crop pipeline supports one truck per image')
        key = (row['frame'], row['camera_id'])
        if key in seen: raise ValueError('Duplicate camera/frame')
        seen.add(key)
        if groups.setdefault(row['frame'], row['split']) != row['split']:
            raise ValueError('The same frame is in multiple splits')
        for x1, y1, x2, y2 in row['boxes']:
            if not (0 <= x1 < x2 <= 1920 and 0 <= y1 < y2 <= 1080):
                raise ValueError('Invalid annotation')
    destination = Path(destination)
    if (destination/'dataset.yaml').exists(): raise ValueError('Use a new destination to avoid stale labels')
    for split in ('train', 'val', 'test'):
        for kind in ('images', 'labels'): (destination/kind/split).mkdir(parents=True, exist_ok=True)

    def save(im, boxes, split, stem):
        h, w = im.shape[:2]
        assert cv2.imwrite(str(destination/'images'/split/(stem+'.jpg')), im)
        text = ''.join(f'0 {(a+c)/(2*w):.8f} {(b+d)/(2*h):.8f} {(c-a)/w:.8f} {(d-b)/h:.8f}\n' for a,b,c,d in boxes)
        (destination/'labels'/split/(stem+'.txt')).write_text(text)

    rng = random.Random(42)
    for index, row in enumerate(rows):
        cap = cv2.VideoCapture(str(Path(source)/f"Way2.FinalImage.CineCameraActor{row['camera_id']}.mp4"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, row['frame'])
        ok, im = cap.read(); cap.release()
        if not ok or im.shape[:2] != (1080, 1920): raise ValueError('Missing or wrong-size frame')
        stem = f"cam{row['camera_id']}_{row['frame']:04d}"
        save(im, row['boxes'], row['split'], stem)
        # Training-only crops preserve the full visible box and change its context/scale.
        if row['split'] == 'train' and row['boxes']:
            a,b,c,d = row['boxes'][0]
            for j, size in enumerate((960, 1280)):
                cw, ch = max(size, c-a+24), max(int(size*9/16), d-b+24)
                cw, ch = min(1920,cw), min(1080,ch)
                left = rng.randint(max(0,int(c-cw)), min(int(a),1920-cw))
                top = rng.randint(max(0,int(d-ch)), min(int(b),1080-ch))
                save(im[top:top+ch,left:left+cw], [[a-left,b-top,c-left,d-top]], 'train', stem+f'_crop{j}')
        if index % 30 == 0: print('Extracting', index, '/', len(rows), flush=True)
    config = destination/'dataset.yaml'
    config.write_text(f'path: {destination.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: truck\n')
    shutil.copy2(annotations, destination/'annotations.json')
    print('Split counts:', {s: len(list((destination/'images'/s).glob('*.jpg'))) for s in ('train','val','test')}, flush=True)
    return config


def inspect_predictions(model, dataset, output, split, conf=.2):
    rows = []
    tp = fp = fn = 0
    for path in sorted((dataset/'images'/split).glob('*.jpg')):
        im = cv2.imread(str(path)); h,w = im.shape[:2]
        labels = (dataset/'labels'/split/(path.stem+'.txt')).read_text().splitlines()
        truth = []
        for line in labels:
            _,x,y,bw,bh=map(float,line.split());truth.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        result = model.predict(im,imgsz=960,conf=conf,device=0,verbose=False)[0]
        boxes = result.boxes.xyxy.cpu().tolist()
        matched = set()
        for box in boxes:
            candidates=[]
            for i,g in enumerate(truth):
                inter=max(0,min(box[2],g[2])-max(box[0],g[0]))*max(0,min(box[3],g[3])-max(box[1],g[1]))
                union=(box[2]-box[0])*(box[3]-box[1])+(g[2]-g[0])*(g[3]-g[1])-inter
                if i not in matched: candidates.append((inter/max(union,1e-9),i))
            overlap, idx = max(candidates, default=(0,-1))
            if overlap >= .5: tp+=1;matched.add(idx)
            else: fp+=1
        fn += len(truth)-len(matched)
        rows.append({'image':path.name,'truth':truth,'predictions':boxes,'confidence':result.boxes.conf.cpu().tolist()})
        if truth or boxes:
            preview=result.plot()
            for a,b,c,d in truth: cv2.rectangle(preview,(int(a),int(b)),(int(c),int(d)),(0,255,0),2)
            cv2.imwrite(str(output/f'{split}_{path.stem}.jpg'),preview)
    report={'split':split,'images':len(rows),'confidence_threshold':conf,'iou_threshold':.5,'tp':tp,'fp':fp,'fn':fn,
            'precision':tp/(tp+fp) if tp+fp else None,'recall':tp/(tp+fn) if tp+fn else None,
            'scope':'Small held-out frames from the same scene, manual approximate 2D boxes; not 3D accuracy', 'rows':rows}
    (output/f'{split}_fixed_threshold.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print({k:v for k,v in report.items() if k!='rows'},flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--annotations',required=True,type=Path)
    p.add_argument('--dataset',required=True,type=Path);p.add_argument('--weights',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    data=prepare(a.source,a.annotations,a.dataset)
    from ultralytics import YOLO
    model=YOLO(str(a.weights))
    model.train(data=str(data),epochs=60,patience=15,imgsz=960,batch=8,device=0,workers=2,
                project=str(a.output),name='detector',exist_ok=False,seed=42,deterministic=True,
                optimizer='AdamW',lr0=.001,freeze=10,cos_lr=True,mosaic=.5,close_mosaic=10,
                degrees=5,translate=.1,scale=.3,fliplr=.5,hsv_s=.25,hsv_v=.25,plots=True)
    weights=Path(model.trainer.best)
    best=YOLO(str(weights))
    for split in ('val','test'):
        metrics=best.val(data=str(data),split=split,imgsz=960,batch=8,device=0,workers=2,plots=True,project=str(a.output),name=split)
        (a.output/f'{split}_metrics.json').write_text(json.dumps(metrics.results_dict,indent=2),encoding='utf-8')
        inspect_predictions(best,a.dataset,a.output,split)
    (a.output/'provenance.json').write_text(json.dumps({'weights':str(weights),'weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
        'annotations_sha256':hashlib.sha256(a.annotations.read_bytes()).hexdigest(),'reference':'2D truck boxes only; no 3D ground truth'},indent=2))
    print('BEST_WEIGHTS:',weights,flush=True)


if __name__=='__main__': main()
