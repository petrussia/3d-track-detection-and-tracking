"""One causal body state shared by live inference and saved-observation replay."""
import math
import numpy as np
from .geometry import estimate_measurement
from .world import WorldTrack
from .reference import select_reference, reference_from_center, orient_reference, reference_from_body_state
from .silhouette import center_observations
from .observation_quality import ObservationQuality, pose_change_reason


class SceneGeometry:
    def __init__(self, rig, shape_estimator, reference='front'):
        self.rig, self.shape_estimator, self.reference = rig, shape_estimator, reference
        self.body_track = WorldTrack()
        self.observation_quality = ObservationQuality(rig.width, rig.height)
        self.previous_body = self.previous_timestamp = self.last_shape = self.last_shape_time = None

    def update(self, frame_index, timestamp, observations):
        observations = self.observation_quality.update(observations, timestamp)
        dt = timestamp-self.previous_timestamp if self.previous_timestamp is not None else 0.
        center_prior = self.body_track.predict_position(timestamp)
        prior_sigma=(math.sqrt(max(0.,float(np.linalg.eigvalsh(self.body_track.p[:3,:3]).max())))
                     if self.body_track.p is not None else 0.)
        center_views,excluded_centers=center_observations(self.rig,frame_index,observations,
                                                        self.last_shape,center_prior,prior_sigma)
        # A stale position prior may reject centre rays, but must not veto an
        # independent current multi-view shape that can recover that position.
        raw_center_fix = estimate_measurement(self.rig, frame_index, center_views,
                                   prior_cm=center_prior,
                                   allow_ground=not self.body_track.depth_prior_active(timestamp))
        if raw_center_fix and raw_center_fix.get('method') == 'triangulated':
            raw_center_fix['reacquisition_qualified'] = bool(
                center_views and all(v.get('anchor_reference') == 'body_center' for v in center_views)
                and
                len(set(raw_center_fix.get('cameras', []))) >= 2
                and raw_center_fix.get('ray_angle_deg', 0.) >= 10.
                and raw_center_fix.get('reprojection_max_px', math.inf) <= 8.
                and raw_center_fix.get('uncertainty_m', math.inf) <= 3.)
        velocity_hint=None
        if self.previous_body and self.previous_body['velocity_cm_s'] is not None and not self.previous_body['mode_reset']:
            sigma=math.sqrt(max(0.,float(np.linalg.eigvalsh(self.body_track.p[3:5,3:5]).max())))
            if np.linalg.norm(self.previous_body['velocity_cm_s'][:2]) >= max(100.,3*sigma):
                velocity_hint=self.previous_body['velocity_cm_s']
        shape_checkpoint = (self.shape_estimator.geometry_checkpoint()
                            if hasattr(self.shape_estimator, 'geometry_checkpoint') else None)
        shape = self.shape_estimator.update(frame_index, timestamp, observations,
            prior_cm=center_prior,
            velocity_cm_s=velocity_hint)
        shape_rejection = pose_change_reason(self.last_shape, shape,
            timestamp-self.last_shape_time if self.last_shape_time is not None else dt)
        if shape_rejection:
            shape = None
        body_fix=(dict(shape) if shape else reference_from_center(raw_center_fix,self.last_shape,'center')
                  if self.last_shape else raw_center_fix)
        untyped_center_candidate = None
        if not shape and not self.last_shape and body_fix and body_fix['method'] != 'ground_plane':
            untyped_center_candidate = body_fix
        if body_fix:
            body_fix=dict(body_fix)
            body_fix['source_frame'] = frame_index
            # This is a private filter anchor, not an exported physical-point
            # assertion. A provisional bbox-centre prior may seed geometry but
            # is never published as a cuboid centre before a shape is accepted.
            body_fix['reference_point'] = ('body_contact_proxy'
                if body_fix['method']=='ground_plane' and not self.last_shape else 'body_track_anchor')
        body=self.body_track.update(body_fix,timestamp)
        rejected_shape = bool(shape is None or shape_rejection or not body['measurement_accepted'])
        if shape_checkpoint is not None and rejected_shape:
            self.shape_estimator.restore_geometry(shape_checkpoint)
        if shape and body['measurement_accepted']:
            self.last_shape = dict(shape)
            self.last_shape_time = timestamp
        elif self.last_shape and body['state']=='observed' and not body['mode_reset']:
            self.last_shape = orient_reference(self.last_shape,body['velocity_cm_s'],self.body_track.p[3:5,3:5],dt)
        covariance = (self.body_track.p[:3,:3]+self.body_track.systematic_covariance
                      if self.body_track.p is not None else np.eye(3)*1e6)
        physical_reference = ('body_center' if self.last_shape else 'body_contact_proxy'
            if body.get('reference_point')=='body_contact_proxy' else 'visual_box_center_proxy')
        export_body = dict(body, reference_point=physical_reference)
        if body.get('raw_measurement'):
            export_body['raw_measurement'] = dict(body['raw_measurement'], reference_point=physical_reference)
        world = reference_from_body_state(export_body, covariance, self.last_shape, self.reference,
            model_age_s=timestamp-self.last_shape_time if self.last_shape_time is not None else 0.)
        world.update(shape_rejection_reason=shape_rejection,
                     untyped_center_candidate=untyped_center_candidate,
                     blocked_reference_measurement=(select_reference(shape,self.reference)
                         if shape and not body['measurement_accepted'] else None),
                     dimensions_cm=self.last_shape.get('dimensions_cm') if self.last_shape else None,
                     yaw_rad=self.last_shape.get('yaw_rad') if self.last_shape else None,
                     body_center_cm=body['position_cm'] if self.last_shape else None,
                     internal_body_reference=physical_reference,
                     internal_body_position_cm=body['position_cm'],body_velocity_cm_s=body['velocity_cm_s'])
        self.previous_body=body
        self.previous_timestamp=timestamp
        return world, observations, excluded_centers
