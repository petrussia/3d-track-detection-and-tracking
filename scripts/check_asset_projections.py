"""Fit asset scale/pose to training annotations only, without trajectory labels.

This is an offline geometry diagnostic, not the causal inference algorithm.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from peleng.geometry import CameraRig
from peleng.inputs import open_camera_capture
from peleng.shape import ShapeEstimator, _axes


def check(data, annotations, asset, output):
    vertices = np.load(asset)['vertices_asset_m'][:, [2, 0, 1]]
    dimensions = np.ptp(vertices, axis=0)
    vertices -= (vertices.min(axis=0)+vertices.max(axis=0))/2
    vertices = vertices[ConvexHull(vertices).vertices]
    labels = json.loads(annotations.read_text(encoding='utf-8'))
    rig = CameraRig(data)
    estimator = ShapeEstimator(rig)
    groups = {}
    for row in labels['images']:
        if row['split'] != 'train':
            continue
        for box in row['boxes']:
            groups.setdefault(row['frame'], []).append(dict(camera_id=row['camera_id'],
                bbox=box, confidence=1., measured=True))
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for frame, observations in sorted(groups.items()):
        views = [v for v in estimator._views(frame, observations) if v['learnable']]
        seed = estimator._seed(views)
        if seed is None:
            continue
        target = np.concatenate([v['edges'] for v in views])
        sigma = np.concatenate([v['sigma'] for v in views])

        def project(parameters):
            points = parameters[:3] + math.exp(parameters[4])*vertices @ _axes(parameters[3]).T
            projected = []
            for view in views:
                local = (points-view['origin']) @ view['rotation']
                depth = local[:, 0]
                if depth.min() <= .05:
                    return None
                u = rig.width/2+view['fx']*local[:, 1]/depth
                v = rig.height/2-view['fy']*local[:, 2]/depth
                projected.extend([u.min(), u.max(), v.min(), v.max()])
            return np.asarray(projected)

        def residual(parameters):
            predicted = project(parameters)
            return ((predicted-target)/sigma if predicted is not None else np.full(len(target), 1e5))

        candidates = []
        for yaw in np.arange(0., 2*math.pi, math.pi/6):
            fit = least_squares(residual, np.r_[seed, yaw, math.log(3.)],
                bounds=([-np.inf]*4+[math.log(.2)], [np.inf]*4+[math.log(20.)]),
                loss='soft_l1', f_scale=2., max_nfev=100, x_scale='jac')
            if project(fit.x) is not None:
                candidates.append((float(np.mean(residual(fit.x)**2)), fit))
        if not candidates:
            continue
        score, fit = min(candidates, key=lambda item: item[0])
        predicted = project(fit.x)
        scale = math.exp(fit.x[4])
        covariance = estimator._parameter_covariance(fit.jac)
        result = dict(frame=frame, split='train', cameras=[v['camera_id'] for v in views],
            center_m=fit.x[:3].tolist(), yaw_rad=float(fit.x[3]), scale=scale,
            dimensions_m=(scale*dimensions).tolist(), normalized_rms=math.sqrt(score),
            rms_px=float(np.sqrt(np.mean((predicted-target)**2))),
            scale_sigma=(scale*math.sqrt(covariance[4,4]) if covariance is not None else None),
            per_camera_rms_px=[float(np.sqrt(np.mean(e**2))) for e in (predicted-target).reshape(-1,4)])
        results.append(result)
        tiles = []
        for index, view in enumerate(views):
            capture = open_camera_capture(data, view['camera_id'])
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
            ok, image = capture.read()
            capture.release()
            if not ok:
                continue
            actual = view['box']
            fitted = predicted[index*4:(index+1)*4][[0,2,1,3]]
            for box, color in [(actual, (0,200,0)), (fitted, (255,100,0))]:
                cv2.rectangle(image, tuple(np.rint(box[:2]).astype(int)),
                    tuple(np.rint(box[2:]).astype(int)), color, 3)
            lo = np.floor(np.minimum(actual[:2],fitted[:2])-40).astype(int)
            hi = np.ceil(np.maximum(actual[2:],fitted[2:])+40).astype(int)
            lo = np.maximum(lo,0); hi = np.minimum(hi,[rig.width,rig.height])
            tile = image[lo[1]:hi[1],lo[0]:hi[0]]
            if not tile.size:
                continue
            tile = cv2.resize(tile, (480,360))
            cv2.putText(tile, f"frame {frame} cam {view['camera_id']} scale {scale:.3f}",
                (8,24),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
            tiles.append(tile)
        if tiles:
            cv2.imwrite(str(output/f'frame-{frame}.jpg'),np.hstack(tiles))
        print(json.dumps(result), flush=True)
    (output/'results.json').write_text(json.dumps(dict(
        kind='offline_training_boxes_asset_geometry_diagnostic_no_ground_truth',
        hull_vertices=len(vertices), asset_dimensions_m=dimensions.tolist(), fits=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--annotations', type=Path, required=True)
    parser.add_argument('--asset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.data, args.annotations, args.asset, args.output)
