import json
from pathlib import Path
import importlib.util

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'adapt_detector', Path(__file__).parents[1] / 'scripts/adapt_scene_detector.py')
adapt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapt)


def test_prevent_same_timestamp_in_train_and_test(tmp_path):
    annotations = tmp_path / 'annotations.json'
    annotations.write_text(json.dumps({'images': [
        {'frame': 0, 'camera_id': 1, 'split': 'train', 'boxes': []},
        {'frame': 0, 'camera_id': 2, 'split': 'test', 'boxes': []},
    ]}))
    with pytest.raises(ValueError, match='multiple splits'):
        adapt.prepare(tmp_path, annotations, tmp_path / 'dataset')


def test_crops_keep_edge_box_and_negatives_stay_empty(tmp_path, monkeypatch):
    class Capture:
        def __init__(self, path): pass
        def set(self, key, value): pass
        def read(self): return True, np.zeros((1080, 1920, 3), dtype=np.uint8)
        def release(self): pass
    monkeypatch.setattr(adapt.cv2, 'VideoCapture', Capture)
    annotations = tmp_path / 'annotations.json'
    annotations.write_text(json.dumps({'images': [
        {'frame': 0, 'camera_id': 1, 'split': 'train', 'boxes': [[0, 0, 1200, 700]]},
        {'frame': 1, 'camera_id': 1, 'split': 'val', 'boxes': []},
        {'frame': 2, 'camera_id': 1, 'split': 'test', 'boxes': [[1800, 900, 1920, 1080]]},
    ]}))
    target = tmp_path / 'dataset'
    adapt.prepare(tmp_path, annotations, target)
    assert len(list((target / 'images/train').glob('*.jpg'))) == 3
    assert len(list((target / 'images/test').glob('*.jpg'))) == 1
    assert (target / 'labels/val/cam1_0001.txt').read_text() == ''
    for label in (target / 'labels/train').glob('*.txt'):
        _, x, y, w, h = map(float, label.read_text().split())
        assert x-w/2 >= -1e-7 and y-h/2 >= -1e-7
        assert x+w/2 <= 1+1e-7 and y+h/2 <= 1+1e-7
        image = cv2.imread(str(target / 'images/train' / (label.stem+'.jpg')))
        assert w*image.shape[1] == pytest.approx(1200, abs=.001)
        assert h*image.shape[0] == pytest.approx(700, abs=.001)
    with pytest.raises(ValueError, match='new destination'):
        adapt.prepare(tmp_path, annotations, target)
