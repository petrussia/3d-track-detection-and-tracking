"""Causal single-vehicle image association with bounded, explicit prediction.

An established track admits weaker detections only inside its motion/scale gate.
Strong detections initialize immediately; weak detections need three consistent
frames. After loss, strong detections also need two frames. This is not visual
re-identification and cannot guarantee identity among similar vehicles.
"""
import math
from collections.abc import Mapping


class CameraTrack:
    def __init__(self, max_coast_s=0.5, *, acquisition_confidence=.45,
                 continuation_confidence=.20, min_box_size=8., min_box_area=144.,
                 max_center_distance=1.25, max_scale_ratio=2.):
        values = (max_coast_s, acquisition_confidence, continuation_confidence,
                  min_box_size, min_box_area, max_center_distance, max_scale_ratio)
        if not all(math.isfinite(value) for value in values):
            raise ValueError('Track parameters must be finite')
        if not (0 < max_coast_s <= .5 and
                0 <= continuation_confidence <= acquisition_confidence <= 1 and
                min_box_size > 0 and min_box_area > 0 and
                max_center_distance > 0 and max_scale_ratio > 1):
            raise ValueError('Invalid track parameters')
        self.max_coast_s = max_coast_s
        self.acquisition_confidence = acquisition_confidence
        self.continuation_confidence = continuation_confidence
        self.min_box_size = min_box_size
        self.min_box_area = min_box_area
        self.max_center_distance = max_center_distance
        self.max_scale_ratio = max_scale_ratio
        self.last = None
        self.seen_at = None
        self.updated_at = -math.inf
        self.velocity = (0., 0.)
        self._ever_acquired = False
        self._pending = None
        self._pending_at = None
        self._pending_count = 0

    def _valid(self, candidate):
        if not isinstance(candidate, Mapping):
            return None
        if candidate.get('crop_only') or (candidate.get('measured') is False and candidate.get('neural_detected') is False):
            return None
        try:
            box = [float(value) for value in candidate['bbox']]
            confidence = float(candidate['confidence'])
        except (KeyError, TypeError, ValueError, OverflowError):
            return None
        if (len(box) != 4 or not all(math.isfinite(value) for value in box) or
                not math.isfinite(confidence) or not 0 <= confidence <= 1):
            return None
        width, height = box[2] - box[0], box[3] - box[1]
        if (width < self.min_box_size or height < self.min_box_size or
                not math.isfinite(width * height) or width * height < self.min_box_area):
            return None
        return {**candidate, 'bbox': box, 'confidence': confidence}

    def _association(self, candidate, anchor):
        """Return a proximity-first score, or None outside the spatial gate."""
        box = candidate['bbox']
        width, height = anchor[2] - anchor[0], anchor[3] - anchor[1]
        ratios = ((box[2] - box[0]) / width, (box[3] - box[1]) / height)
        if any(r < 1 / self.max_scale_ratio or r > self.max_scale_ratio for r in ratios):
            return None
        distance = math.hypot((box[0] + box[2] - anchor[0] - anchor[2]) / 2,
                              (box[1] + box[3] - anchor[1] - anchor[3]) / 2)
        distance /= math.hypot(width, height)
        if distance > self.max_center_distance:
            return None
        overlap = max(0., min(box[2], anchor[2]) - max(box[0], anchor[0])) * max(
            0., min(box[3], anchor[3]) - max(box[1], anchor[1]))
        union = width * height + (box[2]-box[0]) * (box[3]-box[1]) - overlap
        return 2 * overlap / union - distance + .15 * candidate['confidence']

    def _predicted_box(self, age):
        dx, dy = (value * age for value in self.velocity)
        x1, y1, x2, y2 = self.last['bbox']
        return [x1 + dx, y1 + dy, x2 + dx, y2 + dy]

    def _accept(self, chosen, timestamp):
        if self.last is not None and timestamp - self.seen_at > 1e-6:
            old, new = self.last['bbox'], chosen['bbox']
            dt = timestamp - self.seen_at
            self.velocity = ((new[0]+new[2]-old[0]-old[2]) / (2*dt),
                             (new[1]+new[3]-old[1]-old[3]) / (2*dt))
        else:
            # Duplicate/nearly duplicate timestamps carry no useful velocity.
            self.velocity = (0., 0.)
        self.last = chosen
        self.seen_at = timestamp
        self._ever_acquired = True
        self._pending = self._pending_at = None
        self._pending_count = 0
        neural = bool(chosen.get('neural_detected', True))
        return {**chosen, 'bbox': list(chosen['bbox']), 'state': 'detected' if neural else 'tracked',
                'detected': neural, 'neural_detected': neural, 'measured': True, 'age_s': 0.}

    def update(self, candidates, timestamp):
        if not math.isfinite(timestamp):
            raise ValueError('Track time must be finite')
        if timestamp < self.updated_at:
            raise ValueError('Track time cannot go backwards')
        self.updated_at = timestamp
        valid = [box for candidate in candidates if (box := self._valid(candidate)) is not None]
        age = timestamp - self.seen_at if self.seen_at is not None else math.inf
        if self.last is not None and age <= self.max_coast_s:
            anchor = self._predicted_box(age)
            scored = [(score, box) for box in valid
                      if box['confidence'] >= self.continuation_confidence
                      and (score := self._association(box, anchor)) is not None]
            if scored:
                return self._accept(max(scored, key=lambda item: item[0])[1], timestamp)
            return {**self.last, 'bbox': anchor,
                    'confidence': self.last['confidence'] * max(0., 1-age/self.max_coast_s),
                    'state': 'coasting', 'detected': False, 'neural_detected': False,
                    'measured': False, 'age_s': age}

        # Expired motion is discarded before considering acquisition elsewhere.
        self.last = self.seen_at = None
        self.velocity = (0., 0.)
        eligible = [box for box in valid if box['confidence'] >= self.continuation_confidence]
        if eligible:
            chosen = None
            if self._pending is not None and 0 < timestamp-self._pending_at <= self.max_coast_s:
                scored = [(score, box) for box in eligible
                          if (score := self._association(box, self._pending['bbox'])) is not None]
                if scored:
                    chosen = max(scored, key=lambda item: item[0])[1]
                    self._pending_count += 1
            if chosen is None:
                chosen = max(eligible, key=lambda box: box['confidence'])
                self._pending_count = 1
            strong = chosen['confidence'] >= self.acquisition_confidence
            if ((strong and not self._ever_acquired) or
                    self._pending_count >= (2 if strong else 3)):
                return self._accept(chosen, timestamp)
            self._pending = chosen
            self._pending_at = timestamp
        else:
            self._pending = self._pending_at = None
            self._pending_count = 0
        return {'bbox': None, 'confidence': None, 'state': 'lost', 'detected': False,
                'neural_detected': False, 'measured': False, 'age_s': None}
