import json
from pathlib import Path
import importlib.util
import sys

import cv2
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    'adapt_detector', Path(__file__).parents[1] / 'scripts/adapt_scene_detector.py')
adapt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapt)
sys.modules['adapt_scene_detector'] = adapt
selection_spec = importlib.util.spec_from_file_location(
    'select_detector', Path(__file__).parents[1] / 'scripts/select_scene_detector.py')
selection = importlib.util.module_from_spec(selection_spec)
selection_spec.loader.exec_module(selection)


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


def test_operating_threshold_counts_duplicate_and_background_boxes():
    rows = [
        {'truth': [[0, 0, 10, 10]], 'predictions': [[0, 0, 10, 10], [0, 0, 10, 10]], 'confidence': [.9, .3]},
        {'truth': [], 'predictions': [[0, 0, 10, 10]], 'confidence': [.4]},
        {'truth': [[0, 0, 10, 10]], 'predictions': [], 'confidence': []},
    ]
    low = selection.score(rows, .2)
    high = selection.score(rows, .5)
    assert (low['tp'], low['fp'], low['fn']) == (1, 2, 1)
    assert (high['tp'], high['fp'], high['fn']) == (1, 0, 1)
    assert high['f1'] == pytest.approx(2/3)


def test_ignored_examples_are_not_exported_as_background(tmp_path, monkeypatch):
    def forbidden_capture(_):
        raise AssertionError('Ignored image must not be decoded or exported')
    monkeypatch.setattr(adapt.cv2,'VideoCapture',forbidden_capture)
    annotations=tmp_path/'annotations.json'
    annotations.write_text(json.dumps(dict(excluded_ambiguous=[[300,3]],images=[
        dict(frame=300,camera_id=3,split='train',boxes=[]),
        dict(frame=450,camera_id=4,split='val',boxes=[],ignore=True)])))
    destination=tmp_path/'dataset'
    adapt.prepare(tmp_path,annotations,destination)
    assert not list((destination/'images').rglob('*.jpg'))
    assert not list((destination/'labels').rglob('*.txt'))
