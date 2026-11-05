"""Keep partial silhouettes out of the legacy common-center ray estimator.

Only a past image-derived volume and a causal body prediction are used here.
The independent current multi-view shape solver still receives every view, so
this conservative guard cannot veto a newly observed stereo pose.
"""
from itertools import product
import math

import numpy as np


def center_observations(rig, frame, observations, model, prior_cm, position_sigma_cm=0.):
    eligible,rejected=[],[]
    for observation in observations:
        box=observation['bbox']
        partial=(observation.get('occluded') or observation.get('truncated')
                 or (observation.get('quality') or {}).get('partial'))
        clipped=min(box[0],box[1],rig.width-box[2],rig.height-box[3])<=2.
        permitted = observation.get('geometry_eligibility', {}).get('center_bearing', True)
        if partial or clipped or not permitted:
            rejected.append({'camera_id':observation['camera_id'],
                             'reason':'geometry_ineligible' if not permitted else 'explicit_partial' if partial else 'truncated_center'})
        else:
            eligible.append(observation)
    if not model or prior_cm is None or model.get('dimensions_cm') is None:
        return eligible, rejected
    dimensions = np.asarray(model['dimensions_cm'], dtype=float)
    prior = np.asarray(prior_cm, dtype=float)
    yaw = float(model.get('yaw_rad', 0.))
    if (dimensions.shape != (3,) or prior.shape != (3,)
            or not np.isfinite(np.r_[dimensions,prior,yaw]).all()
            or min(dimensions) <= 0 or position_sigma_cm > max(dimensions)):
        return eligible, rejected
    c,s=math.cos(yaw),math.sin(yaw)
    axes=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    corners=prior+(np.asarray(list(product((-1.,1.),repeat=3)))*dimensions/2)@axes.T
    kept=[]
    for observation in eligible:
        camera=observation['camera_id']
        box=np.asarray(observation['bbox'],dtype=float)
        points=[rig.project(camera,frame,point) for point in corners]
        if any(point is None for point in points):
            kept.append(observation)
            continue
        points=np.asarray(points)
        low,high=points.min(axis=0),points.max(axis=0)
        expected=high-low
        actual=box[2:]-box[:2]
        reason=None
        if min(box[0],box[1],rig.width-box[2],rig.height-box[3]) <= 2.:
            reason='truncated_center'
        elif min(expected) >= 10.:
            ratio=actual/expected
            center=(box[:2]+box[2:])/2
            if min(ratio)<.45 or max(ratio)>2.2:
                reason='partial_extent'
            elif np.any(np.abs(center-(low+high)/2)>2.*expected):
                reason='incompatible_center'
        if reason:
            rejected.append({'camera_id':camera,'reason':reason,
                             'observed_extent_px':actual.tolist(),'expected_extent_px':expected.tolist()})
        else:
            kept.append(observation)
    return kept,rejected
