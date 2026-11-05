"""Separate image association from eligibility for metric geometry.

A confidence or foreground fill is not proof of a complete truck silhouette.
Temporal extent changes are conservative diagnostics, not new measurements.
"""
import math

import numpy as np


class ObservationQuality:
    def __init__(self, width, height):
        self.width, self.height = width, height
        self._anchors = {}

    def update(self, observations, timestamp):
        result = []
        for observation in observations:
            item = dict(observation)
            quality = dict(item.get('quality') or {})
            box = np.asarray(item['bbox'], dtype=float)
            size = box[2:] - box[:2]
            camera = item['camera_id']
            reasons = []
            if item.get('occluded') or item.get('truncated') or quality.get('partial'):
                reasons.append('explicit_partial')
            if min(box[0], box[1], self.width-box[2], self.height-box[3]) <= 2:
                reasons.append('image_boundary')
            previous = self._anchors.get(camera)
            if previous is not None and 0 <= timestamp-previous[0] <= .5:
                ratio = size / previous[1]
                # Do not update the anchor from a fragment: otherwise the next
                # small box would immediately be declared complete again.
                if np.min(ratio) < .65:
                    reasons.append('abrupt_extent_loss')
            weak = item.get('source', 'neural') == 'neural' and item['confidence'] < .45
            if weak:
                reasons.append('weak_neural_candidate')
            complete = not reasons
            capabilities = dict(track_2d=True, center_bearing=complete,
                                depth=complete, shape=complete,
                                full_extent_status='assumed' if complete else 'untrusted',
                                reasons=reasons)
            item['geometry_eligibility'] = capabilities
            quality['partial'] = bool(quality.get('partial') or
                                      any(r != 'weak_neural_candidate' for r in reasons))
            item['quality'] = quality
            if complete:
                self._anchors[camera] = (timestamp, size.copy())
            result.append(item)
        return result


def pose_change_reason(previous, proposed, dt):
    """Do not turn a front/back ambiguity into an instantaneous cab jump."""
    if not previous or not proposed or dt <= 0:
        return None
    if previous.get('front_observable') and proposed.get('front_observable'):
        old = np.asarray(previous['front_cm']) - np.asarray(previous['center_cm'])
        new = np.asarray(proposed['front_cm']) - np.asarray(proposed['center_cm'])
        angle = abs(math.atan2(old[0]*new[1]-old[1]*new[0], old[:2] @ new[:2]))
        sigma = math.hypot(float(previous.get('yaw_sigma_rad', .2)),
                           float(proposed.get('yaw_sigma_rad', .2)))
        if angle > min(math.pi, 1.5*dt + 3*sigma):
            return 'implausible_front_orientation_change'
    return None
