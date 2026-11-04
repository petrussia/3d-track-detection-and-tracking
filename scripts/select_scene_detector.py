"""Select checkpoint/operating threshold on validation only; then report test."""
import argparse
import hashlib
import json
from pathlib import Path

from adapt_scene_detector import inspect_predictions


def score(rows, threshold):
    tp = fp = fn = 0
    for row in rows:
        used = set()
        for box, confidence in zip(row['predictions'], row['confidence']):
            if confidence < threshold:
                continue
            overlaps = []
            for i, truth in enumerate(row['truth']):
                if i in used:
                    continue
                inter = max(0, min(box[2], truth[2])-max(box[0], truth[0])) * max(0, min(box[3], truth[3])-max(box[1], truth[1]))
                area = (box[2]-box[0])*(box[3]-box[1])+(truth[2]-truth[0])*(truth[3]-truth[1])-inter
                overlaps.append((inter/max(area, 1e-9), i))
            overlap, idx = max(overlaps, default=(0, -1))
            if overlap >= .5:
                tp += 1
                used.add(idx)
            else:
                fp += 1
        fn += len(row['truth'])-len(used)
    return {'tp': tp, 'fp': fp, 'fn': fn,
            'precision': tp/(tp+fp) if tp+fp else 0,
            'recall': tp/(tp+fn) if tp+fn else 0,
            'f1': 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--dataset', type=Path, required=True)
    a = p.parse_args()
    from ultralytics import YOLO
    candidates = []
    for checkpoint in ('best', 'last'):
        weights = a.output / f'detector/weights/{checkpoint}.pt'
        folder = a.output / f'selection_{checkpoint}'
        folder.mkdir(exist_ok=True)
        inspect_predictions(YOLO(str(weights)), a.dataset, folder, 'val', conf=.01)
        rows = json.loads((folder/'val_fixed_threshold.json').read_text())['rows']
        for threshold in (.05, .1, .15, .2, .3, .4, .5, .6, .7, .8):
            candidates.append({'weights': str(weights), 'confidence': threshold, **score(rows, threshold)})
    selected = max(candidates, key=lambda r: (r['f1'], r['precision'], r['confidence']))
    selected['weights_sha256'] = hashlib.sha256(Path(selected['weights']).read_bytes()).hexdigest()
    report = {'selection_rule': 'Maximum validation F1 at IoU=.5; ties: precision, then confidence',
              'selection_split': 'val', 'selected': selected, 'candidates': candidates,
              'limitation': 'Only 6 validation objects from the same scene. Not 3D accuracy.'}
    (a.output/'selection.json').write_text(json.dumps(report, indent=2))
    print('SELECTED', selected, flush=True)
    folder = a.output/'selected_evaluation'
    folder.mkdir(exist_ok=True)
    model = YOLO(selected['weights'])
    for split in ('val', 'test'):
        inspect_predictions(model, a.dataset, folder, split, conf=selected['confidence'])
    (a.output/'selection_complete.json').write_text(json.dumps(selected, indent=2))


if __name__ == '__main__':
    main()
