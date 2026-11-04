"""Causal current-image measurements for a neurally seeded moving vehicle.

Foreground is image evidence, not a neural classification or motion prediction.
Fixed cameras use MOG2. Pure rotations warp a persistent clean background
using current/past camera poses; translation invalidates that background. Only
current residual pixels associated with the seeded target are measurements. No target telemetry, future pose, or saved detection
file is an input. Scores below are association heuristics, not probabilities.
"""
from collections.abc import Mapping
from dataclasses import dataclass, field
import math

import cv2
import numpy as np


def _new_background():
    return cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=20,
                                             detectShadows=False)


@dataclass
class _View:
    background: object = field(default_factory=_new_background)
    pose: object = None
    samples: int = 0
    box: object = None
    measured_at: float | None = None
    seed_frame: int | None = None
    neural_at: float | None = None
    label: str = 'vehicle'
    appearance: object = None
    image_background: object = None
    background_valid: object = None
    rotating: bool = False
    previous_image: object = None
    saturation_limit: float | None = None
    neural_box: object = None
    neural_appearance: object = None
    strong_box: object = None
    strong_frame: int | None = None


def _valid(candidate, width, height):
    if not isinstance(candidate, Mapping):
        return None
    try:
        box = np.asarray(candidate['bbox'], dtype=float)
        confidence = float(candidate['confidence'])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if (box.shape != (4,) or not np.isfinite(box).all() or
            not math.isfinite(confidence) or not 0 <= confidence <= 1 or
            not 0 <= box[0] < box[2] <= width or
            not 0 <= box[1] < box[3] <= height):
        return None
    return {**candidate, 'bbox': box.tolist(), 'confidence': confidence,
            'label': candidate.get('label', 'vehicle'),
            'source': candidate.get('source', 'neural'),
            'neural_detected': True, 'measured': not candidate.get('crop_only', False),
            'quality': dict(candidate.get('quality') or {})}


def _area(box):
    return float((box[2]-box[0]) * (box[3]-box[1]))


def _overlap(a, b):
    return float(np.maximum(0, np.minimum(a[2:], b[2:]) -
                            np.maximum(a[:2], b[:2])).prod())


def _distance(a, b):
    return float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2))


def _near(anchor, box, gap):
    size, other = anchor[2:]-anchor[:2], box[2:]-box[:2]
    ratio = other / size
    return (ratio.min() >= .4 and ratio.max() <= 2.5 and
            _distance(anchor, box) / np.linalg.norm(size) <= .75 + min(.5, gap))


def _appearance(image, box):
    height, width = image.shape[:2]
    x, y, x2, y2 = np.rint(box).astype(int)
    crop = image[max(0, y):min(height, y2), max(0, x):min(width, x2)]
    if not crop.size:
        return None
    return cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (32, 24)).astype(np.float32)


def _seed_saturation(image, box):
    height, width = image.shape[:2]
    x, y, x2, y2 = np.rint(box).astype(int)
    crop = image[max(0, y):min(height, y2), max(0, x):min(width, x2)]
    if not crop.size:
        return None
    saturation = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[:, :, 1]
    return min(255., max(35., float(np.percentile(saturation, 75))+20.))


def _appearance_support(previous, current):
    if previous is None or current is None:
        return False
    old_std, new_std = float(previous.std()), float(current.std())
    # A learned initial target can leave a MOG ghost when it disappears. Its
    # silhouette alone is not evidence if current pixels lost the target texture.
    if old_std > 4 and new_std < 2:
        return False
    if old_std < 2 or new_std < 2:
        return abs(float(previous.mean()-current.mean())) < 30
    correlation = float(cv2.matchTemplate(current, previous, cv2.TM_CCOEFF_NORMED)[0, 0])
    return math.isfinite(correlation) and correlation > .05


def _rotation_homography(camera, previous_rotation, rotation, width, height, rig):
    # UE local X-forward/Y-right/Z-up -> image X-right/Y-down/Z-forward.
    axes = np.array([[0., 1., 0.], [0., 0., -1.], [1., 0., 0.]])
    intrinsic = np.array([[camera.fx*width/rig.width, 0., width/2],
                          [0., camera.fy*height/rig.height, height/2], [0., 0., 1.]])
    return intrinsic @ axes @ rotation.T @ previous_rotation @ axes.T @ np.linalg.inv(intrinsic)


def _warped_box(box, homography, factors):
    if box is None:
        return None
    a = box / factors
    corners = np.float32([[a[0], a[1]], [a[2], a[1]], [a[2], a[3]], [a[0], a[3]]])
    points = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), homography).reshape(-1, 2)
    if not np.isfinite(points).all():
        return None
    return np.r_[points.min(axis=0), points.max(axis=0)]*factors


def _warp_background(background, valid, homography, size):
    weights = valid.astype(np.float32) / 255.
    warped_weights = cv2.warpPerspective(weights, homography, size)
    weighted = cv2.warpPerspective(background*weights[:, :, None], homography, size)
    normalized = weighted / np.maximum(warped_weights[:, :, None], 1e-6)
    return normalized, (warped_weights > .8).astype(np.uint8)*255


def _clean_background(view, small, homography, foreground_mask, neural, factors, rate):
    """Warp only previously observed background; never EMA a known foreground."""
    height, width = small.shape[:2]
    if view.image_background is None:
        view.image_background = small.astype(np.float32)
        view.background_valid = np.full((height, width), 255, np.uint8)
        initial = True
    else:
        initial = False
        if homography is not None:
            view.image_background, view.background_valid = _warp_background(
                view.image_background, view.background_valid, homography, (width, height))
    blocked = cv2.dilate(foreground_mask, np.ones((7, 7), np.uint8))
    protect = [np.array(candidate['bbox']) for candidate in neural]
    if view.box is not None and view.measured_at is not None:
        protect.append(view.box)
    for box in protect:
        x, y, x2, y2 = np.rint(box/factors).astype(int)
        # The measurement excludes dark shadows; they still must never enter
        # the clean background and become a bright ghost on later frames.
        pad_x = 2
        pad_bottom = 2
        blocked[max(0, y-2):min(height, y2+pad_bottom),
                max(0, x-pad_x):min(width, x2+pad_x)] = 255
    if initial:
        view.background_valid[blocked > 0] = 0
    valid = view.background_valid > 0
    initialize = ~valid & (blocked == 0)
    view.image_background[initialize] = small[initialize]
    view.background_valid[initialize] = 255
    update = valid & (blocked == 0)
    cv2.accumulateWeighted(small, view.image_background, rate,
                           mask=update.astype(np.uint8))
    view.previous_image = small


def _full_neural_scale(reference, candidate):
    ratio = (candidate[2:]-candidate[:2]) / (reference[2:]-reference[:2])
    return ratio.min() >= .6 and ratio.max() <= 2.25 and ratio.prod() >= .4


def _relative_motion(previous, current, previous_box, box, homography, factors, dt):
    """Validate independent image motion; LK never supplies the measured box."""
    if previous is None or previous_box is None or homography is None:
        return False, None
    old_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    now_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    x, y, x2, y2 = np.rint(previous_box/factors).astype(int)
    mask = np.zeros_like(old_gray)
    mask[max(0, y):min(mask.shape[0], y2), max(0, x):min(mask.shape[1], x2)] = 255
    points = cv2.goodFeaturesToTrack(old_gray, 80, .01, 3, mask=mask, blockSize=5)
    if points is None or len(points) < 6:
        return False, None
    moved, status, error = cv2.calcOpticalFlowPyrLK(old_gray, now_gray, points, None,
                                                  winSize=(21, 21), maxLevel=3)
    back, reverse, _ = cv2.calcOpticalFlowPyrLK(now_gray, old_gray, moved, None,
                                              winSize=(21, 21), maxLevel=3)
    good = ((status.flatten() > 0) & (reverse.flatten() > 0) &
            (np.linalg.norm(back-points, axis=2).flatten() < 1.) & (error.flatten() < 35.))
    old, new = points.reshape(-1, 2)[good], moved.reshape(-1, 2)[good]
    if len(old) < 6:
        return False, None
    current_box = box/factors
    inside = ((new[:, 0] >= current_box[0]-2) & (new[:, 0] <= current_box[2]+2) &
              (new[:, 1] >= current_box[1]-2) & (new[:, 1] <= current_box[3]+2))
    old, new = old[inside], new[inside]
    if len(old) < 6:
        return False, None
    ego = cv2.perspectiveTransform(old.reshape(-1, 1, 2), homography).reshape(-1, 2)
    residual = new-ego
    median = np.median(residual, axis=0)
    magnitude = float(np.linalg.norm(median))
    limit = .3*(current.shape[1]/640)*max(1., (dt or 1/30)*30)
    fraction = float(np.mean(np.linalg.norm(residual, axis=1) > limit))
    return magnitude > limit and fraction >= .5, magnitude


class ImageEvidence:
    """Keep one causal foreground identity per camera, with explicit replay reset.

    ``update`` accepts only the frames being processed now and neural candidates
    for those same frames. The return preserves camera order. ``foreground``
    candidates describe connected current foreground pixels and expose their
    neural seed age; they never extend a missing box by extrapolation. Background
    subtraction cannot establish identity among multiple similar moving objects.
    """
    def __init__(self, rig):
        self.rig = rig
        self.reset()

    def reset(self):
        self._views = {}
        self._frame = None
        self._time = None

    def update(self, frames, camera_ids, frame_index, timestamp, neural_candidates):
        if (not isinstance(frame_index, (int, np.integer)) or frame_index < 0 or
                not math.isfinite(timestamp)):
            raise ValueError('Image evidence needs a nonnegative frame and finite time')
        if self._frame is not None and (frame_index <= self._frame or timestamp <= self._time):
            raise ValueError('Image evidence must advance; reset before replay')
        if not len(frames) == len(camera_ids) == len(neural_candidates):
            raise ValueError('Image evidence camera inputs must have equal lengths')
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError('Image evidence camera IDs must be unique')
        # Validate before mutating any background state.
        for camera_id, image in zip(camera_ids, frames):
            if camera_id not in self.rig.camera_ids:
                raise ValueError(f'Unknown camera {camera_id}')
            if (not isinstance(image, np.ndarray) or image.dtype != np.uint8 or
                    image.ndim != 3 or image.shape[2] != 3 or
                    image.shape[:2] != (self.rig.height, self.rig.width)):
                raise ValueError('Image evidence expects calibrated uint8 BGR frames')
        dt = timestamp-self._time if self._time is not None else None
        if dt is not None and dt > .5:
            self._views.clear()
        result = []
        for image, camera_id, candidates in zip(frames, camera_ids, neural_candidates):
            result.append(self._update_view(image, camera_id, frame_index,
                                            timestamp, dt, candidates))
        self._frame, self._time = int(frame_index), float(timestamp)
        return result

    def _update_view(self, image, camera_id, frame, timestamp, dt, candidates):
        view = self._views.get(camera_id)
        if view is None:
            view = self._views[camera_id] = _View()
        camera, origin, rotation = self.rig._pose(camera_id, frame)
        pose = np.r_[origin, rotation.flatten()]
        previous_pose = view.pose
        translated = previous_pose is not None and not np.allclose(origin, previous_pose[:3], rtol=0, atol=1e-7)
        rotated = previous_pose is not None and not np.allclose(rotation.flatten(), previous_pose[3:], rtol=0, atol=1e-7)
        if translated:
            view = self._views[camera_id] = _View()
            previous_pose = None
        view.rotating = view.rotating or (rotated and not translated)
        neural = [valid for candidate in candidates
                  if (valid := _valid(candidate, self.rig.width, self.rig.height)) is not None]
        scale = min(1., 640/self.rig.width)
        width, height = round(self.rig.width*scale), round(self.rig.height*scale)
        small = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        factors = np.array([self.rig.width/width, self.rig.height/height]*2)
        # Time-scaled learning keeps live skipped frames and batch replay comparable.
        rate = 1. if view.samples == 0 else min(.03, 1-math.exp(-.06*(dt or 1/30)))
        previous_image = view.previous_image
        previous_box = view.box.copy() if view.box is not None else (
            view.neural_box.copy() if view.neural_box is not None else None)
        homography = None
        if view.rotating and previous_pose is not None:
            homography = _rotation_homography(camera, previous_pose[3:].reshape(3, 3),
                                               rotation, width, height, self.rig)
            view.box = _warped_box(view.box, homography, factors)
            view.neural_box = _warped_box(view.neural_box, homography, factors)
            view.strong_box = _warped_box(view.strong_box, homography, factors)
        if view.rotating and view.image_background is not None:
            background, valid = _warp_background(view.image_background, view.background_valid,
                                                  homography, (width, height))
            # Compare smoothed colour to suppress subpixel tree/TAA edge residue.
            current_blur = cv2.GaussianBlur(small.astype(np.float32), (5, 5), 1.)
            background_blur = cv2.GaussianBlur(background, (5, 5), 1.)
            residual = np.max(np.abs(current_blur-background_blur), axis=2)
            valid = cv2.erode(valid, np.ones((3, 3), np.uint8))
            mask = ((residual > 22.) & (valid > 0)).astype(np.uint8)*255
            # Dark cast shadows change brightness without supplying the vehicle's
            # silhouette. Keep their mask for background protection, not geometry.
            current_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            background_gray = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)
            shadow = (current_gray < .5*background_gray) & (current_gray < 100)
            measurement_mask = mask.copy()
            measurement_mask[shadow] = 0
            if view.saturation_limit is not None:
                saturation = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[:, :, 1]
                measurement_mask[saturation > view.saturation_limit] = 0
        else:
            mask = view.background.apply(small, learningRate=rate)
            measurement_mask = mask
        _clean_background(view, small, homography, mask if view.samples else np.zeros_like(mask),
                          neural, factors, max(.2 if view.rotating else .002, rate if rate < 1 else .002))
        view.pose = pose
        view.samples += 1
        mask = measurement_mask
        components = []
        if view.samples > 1 and not translated:
            close_size = 5
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((close_size, close_size), np.uint8))
            open_size = 3
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((open_size, open_size), np.uint8))
            if np.count_nonzero(mask) < .4 * width * height:
                for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)[0]:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w*h < 60 or min(w, h) < 5 or w*h > .35*width*height:
                        continue
                    fill = cv2.countNonZero(mask[y:y+h, x:x+w]) / (w*h)
                    if fill >= .15:
                        components.append((np.array([x, y, x+w, y+h])*factors, fill))
        # A weak static rock is not corroborated by unrelated moving foreground.
        neural = [candidate for candidate in neural
                  if candidate['confidence'] >= .45 or not components or
                  any(_near(np.array(candidate['bbox']), box, 0)
                      for box, _ in components)]
        if view.rotating and view.strong_box is not None:
            for candidate in neural:
                size = np.array(candidate['bbox'])[2:]-np.array(candidate['bbox'])[:2]
                reference_size = view.strong_box[2:]-view.strong_box[:2]
                if candidate['confidence'] < .45 and np.min(size/reference_size) < .75:
                    candidate['crop_only'] = True
                    candidate['measured'] = False
                    candidate['quality'] = {**candidate['quality'], 'partial': True,
                                            'reason': 'weak_neural_partial_silhouette'}
        # Current semantic evidence takes precedence over a shadow/texture
        # component. Never continue an image fragment while ignoring a current
        # neural vehicle elsewhere inside the same association gate.
        if view.rotating:
            eligible = [candidate for candidate in neural if not candidate.get('crop_only')
                        and (candidate['confidence'] >= .45 or view.box is not None
                             and _near(view.box, np.array(candidate['bbox']), .1))]
            if eligible:
                candidate = max(eligible, key=lambda item: item['confidence'])
                view.box = np.array(candidate['bbox'])
                view.appearance = _appearance(image, view.box)
                view.saturation_limit = _seed_saturation(image, view.box)
                view.neural_box = view.box.copy()
                view.neural_appearance = view.appearance.copy()
                if candidate['confidence'] >= .45:
                    view.strong_box = view.box.copy()
                    view.strong_frame = frame
                if view.seed_frame is None:
                    view.seed_frame = frame
                view.measured_at = view.neural_at = timestamp
                view.label = candidate['label']
                return neural
        rejected_hints = []
        motion_px = None
        if view.rotating and view.neural_box is not None:
            confirmed = []
            neural_age = timestamp-view.neural_at if view.neural_at is not None else math.inf
            for box, fill in components:
                reason = None
                scale_reference = view.strong_box if view.strong_box is not None else view.neural_box
                if not _full_neural_scale(scale_reference, box):
                    reason = 'inconsistent_with_neural_silhouette'
                elif not _appearance_support(view.neural_appearance, _appearance(image, box)):
                    reason = 'appearance_changed_from_neural_seed'
                elif neural_age > .5:
                    moving, motion_px = _relative_motion(previous_image, small, previous_box,
                                                         box, homography, factors, dt)
                    if not moving:
                        reason = 'no_independent_vehicle_motion'
                if reason is None:
                    confirmed.append((box, fill))
                elif _near(view.neural_box, box, .5):
                    rejected_hints.append({'bbox': box.tolist(), 'confidence': .2,
                                           'label': view.label, 'source': 'foreground',
                                           'neural_detected': False, 'measured': False,
                                           'crop_only': True,
                                           'quality': {'partial': True, 'reason': reason,
                                                       'measurement': 'rotation_compensated_background'}})
            components = confirmed
        chosen = None
        anchor = appearance = None
        gap = 0.
        if view.box is not None and timestamp-view.measured_at <= .5:
            anchor, appearance = view.box, view.appearance
            gap = timestamp-view.measured_at
        elif (view.rotating and view.neural_box is not None and view.neural_at is not None
              and timestamp-view.neural_at <= 3.):
            # A retained identity proposes an association only; it does not renew
            # the measurement time. Current pixels/scale/appearance/flow must pass.
            anchor, appearance = view.neural_box, view.neural_appearance
            gap = .5
        if anchor is not None:
            nearby = [(box, fill) for box, fill in components
                      if _near(anchor, box, gap)
                      and _appearance_support(appearance, _appearance(image, box))]
            if nearby:
                chosen = min(nearby, key=lambda item: _distance(anchor, item[0]))
        matched_neural = None
        if chosen is None:
            for candidate in sorted(neural, key=lambda item: item['confidence'], reverse=True):
                if candidate.get('crop_only'):
                    continue
                anchor = np.array(candidate['bbox'])
                nearby = [(box, fill) for box, fill in components if _near(anchor, box, 0)]
                if nearby:
                    chosen = min(nearby, key=lambda item: _distance(anchor, item[0]))
                    matched_neural = candidate
                    view.seed_frame = frame
                    break
        if chosen is not None:
            box, fill = chosen
            # A neural observation may validate identity while providing only a cab.
            supporting = [candidate for candidate in neural if not candidate.get('crop_only')
                          and _overlap(np.array(candidate['bbox']), box) /
                          _area(np.array(candidate['bbox'])) >= .6]
            if supporting:
                matched_neural = max(supporting, key=lambda item: item['confidence'])
            # Colour residual can join a real vehicle to a distant tree shadow.
            # A current neural silhouette is safer than that boundary/low-fill blob.
            merged = view.rotating and (fill < .55 or box[0] < 2 or box[1] < 2 or
                                       box[2] > self.rig.width-2 or box[3] > self.rig.height-2)
            if merged and matched_neural is not None:
                box = np.array(matched_neural['bbox'])
                x, y, x2, y2 = np.rint(box/factors).astype(int)
                fill = float(np.count_nonzero(mask[y:y2, x:x2]) / max(1, (x2-x)*(y2-y)))
            if matched_neural is not None:
                view.neural_at = timestamp
                view.label = matched_neural['label']
                view.saturation_limit = _seed_saturation(image, matched_neural['bbox'])
                view.neural_box = np.array(matched_neural['bbox'])
                view.neural_appearance = _appearance(image, view.neural_box)
                if matched_neural['confidence'] >= .45:
                    view.strong_box = view.neural_box.copy()
                    view.strong_frame = frame
                if view.seed_frame is None:
                    view.seed_frame = frame
            view.box, view.measured_at = box, timestamp
            view.appearance = _appearance(image, box)
            x, y, x2, y2 = np.rint(box/factors).astype(int)
            local = mask[max(0, y):min(height, y2), max(0, x):min(width, x2)]
            exterior = np.zeros_like(local)
            contours = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(exterior, contours, -1, 255, cv2.FILLED)
            silhouette_fill = float(np.count_nonzero(exterior)/max(1, local.size))
            quality = {'foreground_fill': float(fill), 'silhouette_fill': silhouette_fill,
                       'seed_frame': view.seed_frame, 'neural_reference_frame': view.strong_frame,
                       'measurement': ('rotation_compensated_background' if view.rotating else 'static_mog2'),
                       'shadow_filtered': bool(view.rotating),
                       'relative_motion_px': motion_px,
                       'neural_age_s': timestamp-view.neural_at if view.neural_at is not None else None,
                       'confidence_kind': 'association_score',
                       'partial': bool(box[0] < 2 or box[1] < 2 or
                                       box[2] > self.rig.width-2 or box[3] > self.rig.height-2)}
            adequate_neural = [candidate for candidate in supporting
                               if (candidate['bbox'][2]-candidate['bbox'][0]) >= .75*(box[2]-box[0])
                               and (candidate['bbox'][3]-candidate['bbox'][1]) >= .65*(box[3]-box[1])]
            if not adequate_neural:
                # Suppress partial duplicates so association can use the fuller
                # current silhouette instead of repeatedly selecting the cab.
                neural = [candidate for candidate in neural if candidate not in supporting]
                neural.append({'bbox': box.tolist(), 'confidence': .5, 'label': view.label,
                               'source': 'foreground', 'neural_detected': False,
                               'measured': True, 'quality': quality})
        else:
            # Strong current neural evidence may initialize immediately, including
            # the first frame while no meaningful background model exists yet.
            strong = [candidate for candidate in neural
                      if candidate['confidence'] >= .45 and not candidate.get('crop_only')]
            if strong:
                candidate = max(strong, key=lambda item: item['confidence'])
                view.box = np.array(candidate['bbox'])
                view.appearance = _appearance(image, view.box)
                view.saturation_limit = _seed_saturation(image, view.box)
                view.neural_box = view.box.copy()
                view.neural_appearance = view.appearance.copy()
                view.strong_box = view.box.copy()
                view.strong_frame = frame
                view.measured_at, view.neural_at = timestamp, timestamp
                view.seed_frame, view.label = frame, candidate['label']
            elif view.measured_at is not None and timestamp-view.measured_at > .5:
                view.box = None
        return neural + rejected_hints[:1]
