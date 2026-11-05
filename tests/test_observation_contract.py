import copy
import numpy as np

from peleng.observation_quality import ObservationQuality, pose_change_reason
from peleng.reference import reference_from_body_state
from peleng.world import WorldTrack


def test_box_center_triangulation_without_shape_is_not_exported_as_body_center(monkeypatch):
    from types import SimpleNamespace
    import peleng.scene_geometry as module
    raw=dict(position_cm=[1000.,0.,400.],method='triangulated',cameras=[1,2],
             covariance_cm2=(np.eye(3)*10000).tolist(),ray_angle_deg=45.,
             reprojection_max_px=0.,uncertainty_m=1.)
    monkeypatch.setattr(module,'estimate_measurement',lambda *a,**k:dict(raw))
    shape=SimpleNamespace(update=lambda *a,**k:None)
    pipeline=module.SceneGeometry(SimpleNamespace(width=1920,height=1080),shape,'center')
    world,_,_=pipeline.update(0,0.,[box([100,100,500,300])])
    assert world['position_cm'] is None and world['object_center_cm'] is None
    assert world['body_center_cm'] is None
    assert world['internal_body_reference']=='visual_box_center_proxy'
    assert not world['untyped_center_candidate']['reacquisition_qualified']


def box(bounds, confidence=.8, **extra):
    return dict(camera_id=1,bbox=bounds,confidence=confidence,source='neural',**extra)


def test_dense_fragment_and_weak_candidates_can_track_but_not_update_geometry():
    quality=ObservationQuality(1920,1080)
    quality.update([box([100,100,500,300])],0.)
    items=quality.update([box([300,120,450,280],quality={'foreground_fill':.99})],.03)
    eligibility=items[0]['geometry_eligibility']
    assert eligibility['track_2d'] and not eligibility['depth'] and not eligibility['center_bearing']
    assert 'abrupt_extent_loss' in eligibility['reasons']
    weak=quality.update([box([100,100,500,300],confidence=.25)],.06)[0]
    assert weak['geometry_eligibility']['track_2d']
    assert not weak['geometry_eligibility']['shape']


def test_low_foreground_fill_alone_is_not_partial_evidence():
    item=ObservationQuality(1920,1080).update(
        [box([100,100,500,300],quality={'foreground_fill':.39})],0.)[0]
    assert item['geometry_eligibility']['depth']
    assert item['geometry_eligibility']['full_extent_status']=='assumed'


def test_target_reappears_with_body_without_a_second_reacquisition():
    track=WorldTrack()
    raw=dict(position_cm=[1000.,0.,400.],method='triangulated',cameras=[1,2],
             reference_point='body_center',covariance_cm2=(np.eye(3)*10000).tolist())
    body=track.update(raw,0.)
    model=dict(center_cm=[1000.,0.,400.],front_cm=[2000.,0.,400.],front_observable=True,
               dimensions_cm=[2000.,600.,700.],yaw_sigma_rad=.1)
    result=reference_from_body_state(body,track.p[:3,:3],model,'front')
    assert result['position_cm']==[2000.,0.,400.]
    assert result['state']==body['state'] and result['measurement_accepted']
    assert result['segment_id']==body['segment_id']
    assert result['reference_point']=='front_face_center'
    held=reference_from_body_state(body,track.p[:3,:3],model,'front',model_age_s=2.)
    assert held['uncertainty_m']>result['uncertainty_m']
    assert model['yaw_sigma_rad']==.1
    unknown=reference_from_body_state(body,track.p[:3,:3],None,'front')
    assert unknown['position_cm'] is None and unknown['state']=='lost'


def test_stale_orientation_uses_elapsed_pose_time_not_one_frame():
    old=dict(center_cm=[0,0,0],front_cm=[100,0,0],front_observable=True)
    new=dict(center_cm=[0,0,0],front_cm=[0,100,0],front_observable=True)
    assert pose_change_reason(old,new,.03)
    assert pose_change_reason(old,new,2.) is None


def test_qualified_triangulation_needs_sequence_to_reacquire():
    def measurement(x):
        return dict(position_cm=[x,0,400],method='triangulated',cameras=[1,2],
            reference_point='body_center',covariance_cm2=(np.eye(3)*10000).tolist(),
            metric_depth_observed=True,reacquisition_qualified=True)
    track=WorldTrack()
    track.update(measurement(0),0.)
    assert not track.update(measurement(10000),6.)['measurement_accepted']
    assert not track.update(measurement(10000),6.05)['measurement_accepted']
    result=track.update(measurement(10000),6.10)
    assert result['measurement_accepted']
    assert result['reset_reason']=='consistent_qualified_triangulation'


def test_unqualified_triangulation_cannot_reset_metric_state():
    track=WorldTrack()
    raw=dict(position_cm=[0,0,400],method='triangulated',cameras=[1,2],
             covariance_cm2=(np.eye(3)*10000).tolist())
    track.update(raw,0.)
    for time in [6.,6.05,6.1,6.15]:
        assert not track.update(dict(raw,position_cm=[10000,0,400]),time)['measurement_accepted']
