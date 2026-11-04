"""Causal constant-velocity state with metric and rank-two bearing updates.

All units here are UE centimetres and seconds. Bearings cannot determine depth:
their information matrix has rank two. A finite depth lifetime prevents a motion
prior from being presented indefinitely as a new metric measurement.
"""
import math

import numpy as np


class WorldTrack:
    def __init__(self, max_coast_s=.5, max_bearing_s=5., acceleration_cm_s2=600.):
        if min(max_coast_s, max_bearing_s, acceleration_cm_s2) <= 0:
            raise ValueError('World tracking timeouts/noise must be positive')
        self.max_coast_s = max_coast_s
        self.max_bearing_s = max_bearing_s
        self.acceleration = acceleration_cm_s2
        self.x = None
        self.p = None
        self.timestamp = None
        self.last_seen = None
        self.last_metric = None
        self.last_independent_depth = None
        self.last_method = None
        self.depth_source = None
        self.mode = None
        self.segment_id = -1
        self.systematic_covariance = np.zeros((3, 3))
        self.ground_cameras = None
        self.reference_point = None
        self._metric_candidates = []

    @property
    def initialized(self):
        return self.x is not None

    def predict_position(self, timestamp):
        if not self.depth_prior_active(timestamp):
            return None
        return (self.x[:3] + (timestamp-self.timestamp) * self.x[3:]).tolist()

    def depth_prior_active(self, timestamp):
        return (self.initialized and self.mode == 'metric_3d'
                and timestamp-self.last_metric <= self.max_bearing_s)

    def _predict(self, dt):
        transition = np.eye(6)
        transition[:3, 3:] = np.eye(3) * dt
        noise = np.vstack((np.eye(3) * dt**2 / 2, np.eye(3) * dt))
        self.x = transition @ self.x
        self.p = transition @ self.p @ transition.T + noise @ noise.T * self.acceleration**2

    @staticmethod
    def _metric(measurement):
        point = np.asarray(measurement['position_cm'], dtype=float)
        covariance = np.asarray(measurement.get('covariance_cm2', np.eye(3)*300**2), dtype=float)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError('World measurement must contain three finite coordinates')
        if (covariance.shape != (3, 3) or not np.isfinite(covariance).all()
                or not np.allclose(covariance, covariance.T)
                or np.linalg.eigvalsh(covariance).min() <= 0):
            raise ValueError('World covariance must be symmetric positive definite')
        return point, covariance

    def update(self, measurement, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp) or (self.timestamp is not None and timestamp < self.timestamp):
            raise ValueError('World time must be finite and cannot go backwards')
        point = covariance = None
        raw_measurement = measurement
        blocked_reason = None
        # A clipped silhouette can fit a box perfectly without observing range.
        if (measurement is not None and measurement.get('method') == 'cuboid_singleview'
                and measurement.get('partial_silhouette', False)):
            blocked_reason = 'partial_singleview_depth'
            measurement = None
        bearing = measurement is not None and measurement.get('method') == 'bearing_update'
        direction = origin = None
        if measurement is not None:
            point, covariance = self._metric(measurement)
            if bearing:
                try:
                    direction = np.asarray(measurement['direction'], dtype=float)
                    origin = np.asarray(measurement['origin_cm'], dtype=float)
                    sigma = float(measurement['transverse_sigma_cm'])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError('Bearing requires origin, unit direction and positive transverse sigma') from exc
                if (direction.shape != (3,) or origin.shape != (3,)
                        or not np.isfinite(direction).all() or not np.isfinite(origin).all()
                        or abs(np.linalg.norm(direction)-1) > 1e-5
                        or not math.isfinite(sigma) or sigma <= 0):
                    raise ValueError('Bearing requires origin, unit direction and positive transverse sigma')
        was_initialized = self.initialized
        expired = self.last_seen is not None and timestamp - self.last_seen > self.max_coast_s
        dt = timestamp-self.timestamp if self.timestamp is not None else 0.
        if self.initialized:
            self._predict(timestamp-self.timestamp)
        predicted_position = self.x[:3].tolist() if self.initialized else None
        self.timestamp = timestamp
        accepted = False
        reason = blocked_reason
        reinitialized = False
        reset_reason = None
        depth_expired = self.mode == 'metric_3d' and timestamp-self.last_metric > self.max_bearing_s
        next_mode = ('assumed_ground' if measurement and measurement['method'] == 'ground_plane'
                     else 'metric_3d')
        mode_changed = measurement is not None and not bearing and next_mode != self.mode
        reference_changed = (measurement is not None and not bearing and self.reference_point is not None
                             and measurement.get('reference_point', self.reference_point) != self.reference_point)
        independent_depth = bool(measurement is not None and not bearing and
            measurement.get('metric_depth_observed', measurement['method'] == 'triangulated'))
        transition_reasons = [name for name, active in (
            ('track_expired', expired), ('depth_expired', depth_expired),
            ('mode_changed', mode_changed), ('reference_changed', reference_changed)) if active]
        # A brief visibility gap may resume via the ordinary innovation/motion
        # gates, without relocating or resetting the filter. A bearing alone
        # cannot revive a lost track, nor can incompatible reference/depth modes.
        needs_confirmation = self.initialized and (
            depth_expired or mode_changed or reference_changed or (expired and bearing))
        if measurement is not None and not bearing and not self.initialized:
            self.x = np.r_[point, np.zeros(3)]
            self.p = np.zeros((6, 6))
            self.p[:3, :3] = covariance
            self.p[3:, 3:] = np.eye(3) * 1500**2
            accepted = True
            reinitialized = was_initialized
            self.mode = next_mode
            self.last_metric = timestamp
            self.depth_source = measurement['method']
            self.segment_id += 1
        elif measurement is not None and needs_confirmation:
            # Expiration is not evidence that the next range estimate is correct.
            # Keep the old state separate from the candidate until confirmed.
            reason = 'reacquisition_pending:' + ','.join(transition_reasons)
        elif measurement is not None and self.initialized and not depth_expired and (not bearing or self.mode == 'metric_3d'):
            h = np.zeros((3, 6))
            h[:, :3] = np.eye(3)
            z, r = point, covariance
            if bearing:
                # The last two rows of V^T span the plane normal to d.
                _, _, basis = np.linalg.svd(direction.reshape(1, 3))
                b = basis[1:]
                h = np.zeros((2, 6))
                h[:, :3] = b
                z = b @ origin
                r = np.eye(2) * float(measurement['transverse_sigma_cm'])**2
            innovation = z - h @ self.x
            s = h @ self.p @ h.T + r
            mahalanobis = float(innovation @ np.linalg.solve(s, innovation))
            if mahalanobis <= (13.82 if len(z) == 2 else 16.27):
                gain = np.linalg.solve(s, h @ self.p).T
                posterior = self.x + gain @ innovation
                if bearing and (posterior[:3]-origin) @ direction <= 0:
                    reason = 'behind_camera'
                elif (np.linalg.norm(posterior[3:]) > 5000.
                      or np.linalg.norm(posterior[:3]-self.x[:3]) > max(1000., 5000.*dt)):
                    reason = 'motion_gate'
                else:
                    self.x = posterior
                    remainder = np.eye(6) - gain @ h
                    self.p = remainder @ self.p @ remainder.T + gain @ r @ gain.T
                    self.p = (self.p + self.p.T) / 2
                    accepted = True
            else:
                reason = 'innovation_gate'
        elif measurement is not None:
            reason = 'depth_prior_expired' if depth_expired else 'bearing_has_no_metric_initialization'
        self._metric_candidates = [c for c in self._metric_candidates if timestamp-c['time'] <= .5]
        # Reliable full-volume poses can recover a depth prior that accepted
        # bearings kept alive. Epipolar ray agreement alone is not enough.
        independent = (measurement is not None and measurement.get('method') == 'cuboid_multiview'
                       and len(set(measurement.get('cameras',[]))) >= 2
                       and measurement.get('metric_depth_observed',False)
                       and measurement.get('scale_usable',False)
                       and not measurement.get('partial_silhouette',False)
                       and measurement.get('fit_normalized_rms',math.inf) <= 2.5)
        # Moving from an assumed ground plane to measured depth can also use
        # repeated triangulations. They cannot reset an existing metric track.
        ground_to_metric = (measurement is not None and self.mode == 'assumed_ground'
                            and measurement.get('method') == 'triangulated'
                            and len(set(measurement.get('cameras', []))) >= 2
                            and independent_depth)
        reacquisition_reason = None
        if not accepted and (independent or ground_to_metric):
            candidate = {'time':timestamp,'point':point.copy(),'covariance':covariance,
                         'reference':measurement.get('reference_point',self.reference_point)}
            if self._metric_candidates:
                previous=self._metric_candidates[-1]
                delta=timestamp-previous['time']
                noise=math.sqrt(float(np.linalg.eigvalsh(covariance+previous['covariance']).max()))
                if (delta<=0 or previous['reference']!=candidate['reference'] or
                        np.linalg.norm(point-previous['point']) > 5000.*delta+3.*noise):
                    self._metric_candidates=[]
            self._metric_candidates.append(candidate)
            self._metric_candidates=self._metric_candidates[-8:]
            if len(self._metric_candidates)>=3 and timestamp-self._metric_candidates[0]['time'] >= .08:
                self.x=np.r_[point,np.zeros(3)]
                self.p=np.zeros((6,6))
                self.p[:3,:3]=covariance
                self.p[3:,3:]=np.eye(3)*1500.**2
                self.mode='metric_3d'
                self.segment_id+=1
                accepted=reinitialized=True
                reason=None
                reacquisition_reason=('consistent_multiview_pose' if independent
                                      else 'consistent_triangulation_from_ground')
                reset_reason = reacquisition_reason
                self._metric_candidates=[]
        elif accepted and not bearing:
            self._metric_candidates=[]
        if accepted:
            self.last_seen = timestamp
            self.last_method = measurement['method']
            self.reference_point = measurement.get('reference_point', self.reference_point)
            if not bearing or 'systematic_covariance_cm2' in measurement:
                # Common size/reference errors survive repeated updates. R is
                # a conservative per-observation marginal covariance; retaining
                # this extra floor is conservative, not an exact bias-state filter.
                self.systematic_covariance = np.asarray(measurement.get('systematic_covariance_cm2', np.zeros((3, 3))))
            if not bearing:
                # A full silhouette with learned size supports range, but does not
                # renew independent multi-camera depth. Keep the clocks separate.
                self.last_metric = timestamp
                self.depth_source = measurement['method']
                if independent_depth:
                    self.last_independent_depth = timestamp
                self.ground_cameras = measurement.get('cameras') if self.mode == 'assumed_ground' else None
        age = timestamp - self.last_seen if self.last_seen is not None else math.inf
        depth_age = timestamp-self.last_metric if self.last_metric is not None else math.inf
        alive = self.initialized and age <= self.max_coast_s and (self.mode == 'assumed_ground' or depth_age <= self.max_bearing_s)
        state = 'observed' if accepted and alive else 'coasting' if alive else 'lost'
        uncertainty = float(math.sqrt(max(0., np.linalg.eigvalsh(self.p[:3, :3]+self.systematic_covariance).max())) / 100) if self.initialized else None
        return {'state': state, 'position_cm': self.x[:3].tolist() if alive else None,
                'velocity_cm_s': self.x[3:].tolist() if alive else None,
                'method': self.last_method if state == 'observed' else 'prediction' if alive else None,
                'age_s': age if math.isfinite(age) else None,
                'depth_age_s': depth_age if math.isfinite(depth_age) else None,
                'depth_support_age_s': depth_age if math.isfinite(depth_age) else None,
                'independent_depth_age_s': (timestamp-self.last_independent_depth
                    if self.last_independent_depth is not None else None),
                'depth_source': self.depth_source, 'uncertainty_m': uncertainty,
                'mode': self.mode, 'segment_id': self.segment_id, 'mode_reset': bool(reinitialized),
                'reset_reason': reset_reason, 'transition_reasons': transition_reasons,
                'predicted_position_cm': predicted_position,
                'reference_point': self.reference_point,
                'measurement_accepted': accepted, 'raw_measurement': raw_measurement,
                'rejection_reason': reason, 'reinitialized': reinitialized,
                'reacquisition_reason':reacquisition_reason,
                'metric_depth_observed': bool(accepted and not bearing and
                    measurement.get('metric_depth_observed', measurement['method'] == 'triangulated'))}
