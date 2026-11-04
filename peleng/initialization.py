"""Causal leading contact before an image-derived physical shape is available.

A complete elongated side silhouette moving horizontally identifies a leading
image end. Its bottom corner is only a front ground-contact proxy: ground z=0,
unknown vehicle width, occlusion, and an unknown mesh pivot remain assumptions.
There are no object coordinates, dimensions, routes, or camera-specific rules.
"""
from collections import deque
import math

import numpy as np


class LeadingContact:
    """Select a current leading corner after observing stable image motion.

    Call on every processed frame, even if the base measurement is absent.
    Inputs must advance strictly; call ``reset`` before replay. The caller must
    enable this only for a front reference before accepting any physical shape.
    Unknown/unsuitable motion leaves the base measurement unchanged.
    """
    def __init__(self, rig):
        self.rig = rig
        self.reset()

    def reset(self):
        self._frame = None
        self._time = None
        self._history = {}
        self._signs = {}

    def _observation(self, item, frame):
        try:
            quality = item.get('quality') or {}
            if (item.get('crop_only') or item.get('measured') is False
                    or item.get('occluded') or item.get('truncated')
                    or quality.get('partial') or float(quality.get('foreground_fill', 1.)) < .6):
                return None
            camera_id = int(item['camera_id'])
            box = np.asarray(item['bbox'], dtype=float)
            confidence = float(item['confidence'])
            if (box.shape != (4,) or not np.isfinite(box).all()
                    or not math.isfinite(confidence) or not 0 < confidence <= 1):
                return None
            x1, y1, x2, y2 = box
            margin = max(2., .002*self.rig.width)
            if not (margin < x1 < x2 < self.rig.width-margin
                    and margin < y1 < y2 < self.rig.height-margin
                    and x2-x1 >= 1.6*(y2-y1) and y2-y1 >= 10.):
                return None
            _, origin, rotation = self.rig._pose(camera_id, frame)
            return camera_id, box, np.r_[origin, rotation.ravel()], confidence
        except (TypeError, ValueError, KeyError, OverflowError):
            return None

    def _direction(self, camera_id, timestamp, box, pose):
        history = self._history.setdefault(camera_id, deque())
        if history and not np.allclose(history[-1][2], pose, rtol=0, atol=1e-7):
            history.clear()
            self._signs.pop(camera_id, None)
        center = (box[:2]+box[2:])/2
        history.append((timestamp, center, pose))
        while history and timestamp-history[0][0] > .5:
            history.popleft()
        if len(history) >= 3 and timestamp-history[0][0] >= .08:
            times = np.array([sample[0]-timestamp for sample in history])
            points = np.array([sample[1] for sample in history])
            times -= times.mean()
            centered = points-points.mean(axis=0)
            velocity = times @ centered / float(times @ times)
            fitted = times[:, None]*velocity
            residual = float(np.sum((centered-fitted)**2))
            variation = float(np.sum(centered**2))
            span = history[-1][0]-history[0][0]
            movement = abs(velocity[0])*span
            # At least several current measurements and actual displacement;
            # a static residual, vertical motion, or noisy sign is not a front.
            stable = (abs(velocity[0]) >= max(3., .025*(box[2]-box[0]))
                      and abs(velocity[0]) >= 1.5*abs(velocity[1])
                      and movement >= max(1.5, .005*(box[2]-box[0]))
                      and variation > 0 and residual <= .35*variation)
            if stable:
                sign = 1 if velocity[0] > 0 else -1
                old = self._signs.get(camera_id)
                # A reversal needs a consistent recent window, not one jittered
                # centroid following a confirmed direction.
                recent = np.diff(points[:, 0])
                agrees = np.count_nonzero(recent*sign > .1) >= .7*len(recent)
                if old is None or old[0] == sign or agrees:
                    self._signs[camera_id] = (sign, timestamp)
        sign = self._signs.get(camera_id)
        return sign[0] if sign and timestamp-sign[1] <= .2 else None

    def update(self, frame_index, timestamp, observations, base_measurement):
        timestamp = float(timestamp)
        if (not isinstance(frame_index, (int, np.integer)) or frame_index < 0
                or not math.isfinite(timestamp)
                or self._frame is not None and (frame_index <= self._frame or timestamp <= self._time)):
            raise ValueError('Leading-contact inputs must advance; reset before replay')
        self._frame, self._time = int(frame_index), timestamp
        eligible = {}
        for item in observations:
            parsed = self._observation(item, frame_index)
            if parsed is not None:
                camera_id, box, pose, confidence = parsed
                if camera_id not in eligible or confidence > eligible[camera_id][2]:
                    eligible[camera_id] = box, pose, confidence
        for camera_id in list(self._history):
            if camera_id not in eligible:
                self._history.pop(camera_id, None)
                self._signs.pop(camera_id, None)
        directions = {camera_id: self._direction(camera_id, timestamp, box, pose)
                      for camera_id, (box, pose, _) in eligible.items()}
        if not base_measurement or base_measurement.get('method') != 'ground_plane':
            return base_measurement
        cameras = base_measurement.get('cameras', [])
        if len(cameras) != 1 or cameras[0] not in eligible or directions[cameras[0]] is None:
            return base_measurement
        camera_id = cameras[0]
        box = eligible[camera_id][0]
        sign = directions[camera_id]
        pixel = (float(box[2] if sign > 0 else box[0]), float(box[3]))
        origin, direction = self.rig.ray(camera_id, frame_index, pixel)
        if direction[2] >= -1e-4:
            return base_measurement
        distance = -origin[2]/direction[2]
        if not 0 < distance <= 10_000_000.:
            return base_measurement
        point = origin + distance*direction
        point[2] = 0.
        if not np.isfinite(point).all():
            return base_measurement
        # A visible corner can be on either side of the physical centerline.
        # Keep a metre-scale ambiguity in addition to the original ground bias;
        # repeated samples must not present this proxy as a precise mesh pivot.
        previous = np.asarray(base_measurement['position_cm'], dtype=float)
        sigma = max(150., .15*float(np.linalg.norm(point[:2]-previous[:2])))
        covariance = np.asarray(base_measurement.get('covariance_cm2', np.eye(3)*150.**2), dtype=float).copy()
        covariance += np.diag([sigma**2, sigma**2, 150.**2])
        systematic = np.asarray(base_measurement.get('systematic_covariance_cm2', np.zeros((3, 3))), dtype=float).copy()
        systematic += np.diag([sigma**2, sigma**2, 150.**2])
        result = dict(base_measurement)
        result.update(position_cm=point.tolist(), reference_point='front_ground_contact',
                      covariance_cm2=covariance.tolist(), systematic_covariance_cm2=systematic.tolist(),
                      uncertainty_m=math.sqrt(float(np.linalg.eigvalsh(covariance+systematic).max()))/100.,
                      quality='assumed_leading_ground_contact', metric_depth_observed=False,
                      leading_pixel=list(pixel), image_motion_sign=sign,
                      assumption='Leading bbox bottom corner on assumed z=0 ground; height and vehicle centerline are not observed')
        return result
