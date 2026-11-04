"""Approximate pinhole geometry using only exported camera calibration/poses.

Assumptions: UE world coordinates are centimetres, X forward/Y right/Z up,
left-handed; Rot_Roll/Pitch/Yaw are UE Euler degrees. Positive yaw turns +X
toward +Y, positive pitch points +X toward +Z, and positive roll tips +Y
toward -Z. The reported filmback and focal length describe the full image,
the principal point is its centre, and lens distortion is ignored.

Observations must refer to the same current frame and the same object. The
quality-aware estimator uses bbox visual centres for multi-view/bearing 3D;
ground-assumed fallback uses bottom-centres on z=0 as a contact proxy. Legacy
estimate_position defaults to bottom-centres. Neither anchor identifies a mesh
pivot, and the ground plane is not a measured altitude.
No object transform, dimensions, route, future detections, or trajectory fit is
read. A pose lookup holds the latest pose at or before the requested frame;
it never interpolates with a future pose (including camera 5's pan).

Rejection thresholds below are prototype heuristics, not accuracy guarantees.
Residuals measure agreement of rays, not error against a reference position.
"""

from bisect import bisect_right
import csv
from dataclasses import dataclass
from itertools import combinations
import math
from pathlib import Path
import re

import numpy as np


_COLUMNS = ("Frame", "Loc_X", "Loc_Y", "Loc_Z", "Rot_Roll", "Rot_Pitch", "Rot_Yaw")
_MIN_PAIR_SIN_SQUARED = math.sin(math.radians(2.0)) ** 2
_MAX_RAY_SEPARATION_CM = 250.0
_MIN_GROUND_DIRECTION_Z = 1e-4
_MAX_GROUND_DISTANCE_CM = 10_000_000.0


@dataclass(frozen=True)
class _Camera:
    frames: tuple[int, ...]
    locations: np.ndarray
    angles: np.ndarray
    fx: float
    fy: float


def _number(value: str) -> float:
    result = float(value.strip().replace(",", "."))
    if not math.isfinite(result):
        raise ValueError("non-finite number")
    return result


def _read_camera(path: Path, width: int, height: int) -> _Camera:
    """Parse the CP1251 semicolon camera export, including trailing metadata."""
    try:
        with path.open(encoding="cp1251", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=";"))
        if len(rows) < 2:
            raise ValueError("missing header or pose rows")
        header = [cell.strip() for cell in rows[0]]
        missing = [name for name in _COLUMNS if name not in header]
        if missing:
            raise ValueError("missing columns: " + ", ".join(missing))
        indexes = [header.index(name) for name in _COLUMNS]
        focal_index = next((index for index, cell in enumerate(header)
                            if cell.lower() == "focal length"), None)
        if 'Focal_Length_mm' in header:
            column = header.index('Focal_Length_mm')
            focals = {_number(row[column]) for row in rows[1:] if row and any(cell.strip() for cell in row)}
            if len(focals) != 1:
                raise ValueError('Variable focal length is not supported')
            focal = focals.pop()
        elif focal_index is None or focal_index + 1 >= len(header):
            raise ValueError("missing focal length metadata")
        else:
            focal = _number(header[focal_index + 1])
        sensor = None
        for cell in rows[1][max(indexes) + 1:]:
            match = re.search(r"(\d+(?:[.,]\d+)?)\s*[*x×]\s*(\d+(?:[.,]\d+)?)", cell)
            if match:
                sensor = (_number(match[1]), _number(match[2]))
                break
        matrix = path.parent / 'matrix_size.txt'
        if sensor is None and matrix.exists():
            match = re.search(r'(\d+(?:[.,]\d+)?)\s*(?:mm)?\s*[*x×]\s*(\d+(?:[.,]\d+)?)',
                              matrix.read_text(encoding='utf-8-sig'), re.I)
            if match:
                sensor = (_number(match[1]), _number(match[2]))
        if sensor is None:
            raise ValueError("missing sensor dimensions in first pose row")
        if min(focal, *sensor) <= 0:
            raise ValueError("focal length and sensor dimensions must be positive")
        poses = {}
        for line_number, row in enumerate(rows[1:], start=2):
            if not row or not any(cell.strip() for cell in row):
                continue
            try:
                values = [_number(row[index]) for index in indexes]
                frame = int(values[0])
                if values[0] != frame or frame < 0:
                    raise ValueError("frame must be a nonnegative integer")
                if frame in poses:
                    raise ValueError("duplicate frame")
                poses[frame] = values[1:]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"invalid pose on CSV line {line_number}: {exc}") from exc
        if not poses:
            raise ValueError("no valid pose rows")
        frames = tuple(sorted(poses))
        values = np.asarray([poses[frame] for frame in frames], dtype=float)
        return _Camera(frames, values[:, :3], values[:, 3:],
                       width * focal / sensor[0], height * focal / sensor[1])
    except (ValueError, UnicodeError, IndexError, csv.Error) as exc:
        raise ValueError(f"Invalid camera telemetry {path.name}: {exc}") from exc


def _ue_rotation(angles: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.radians(angles)
    sr, sp, sy = np.sin([roll, pitch, yaw])
    cr, cp, cy = np.cos([roll, pitch, yaw])
    # Columns are world-space forward, right and up (UE FRotator convention).
    return np.array([
        [cp * cy, sr * sp * cy - cr * sy, -cr * sp * cy - sr * sy],
        [cp * sy, sr * sp * sy + cr * cy, sr * cy - cr * sp * sy],
        [sp, -sr * cp, cr * cp],
    ], dtype=float)


def _finite_vector(value, length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain {length} finite numbers") from exc
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


class CameraRig:
    """Camera telemetry with pinhole projection and causal, held-pose lookup."""

    def __init__(self, data_dir: str | Path, width: int = 1920, height: int = 1080):
        if width <= 0 or height <= 0:
            raise ValueError("Image width and height must be positive")
        self.width, self.height = int(width), int(height)
        directory = Path(data_dir)
        paths = sorted(directory.glob("object_transform_CineCameraActor*.csv"))
        if not paths:
            raise FileNotFoundError(f"No camera telemetry CSVs found in {directory}")
        self._cameras = {}
        for path in paths:
            match = re.fullmatch(r"object_transform_CineCameraActor(\d+)\.csv", path.name)
            if match:
                self._cameras[int(match[1])] = _read_camera(path, self.width, self.height)
        if not self._cameras:
            raise FileNotFoundError(f"No numbered camera telemetry CSVs found in {directory}")

    @property
    def camera_ids(self) -> list[int]:
        return sorted(self._cameras)

    def _pose(self, camera_id: int, frame: int):
        if camera_id not in self._cameras:
            raise ValueError(f"Unknown camera {camera_id}")
        if not isinstance(frame, (int, np.integer)):
            raise ValueError("Frame must be an integer")
        camera = self._cameras[camera_id]
        index = bisect_right(camera.frames, frame) - 1
        if index < 0:
            raise ValueError(f"No camera {camera_id} pose available at or before frame {frame}")
        return camera, camera.locations[index], _ue_rotation(camera.angles[index])

    def ray(self, camera_id: int, frame: int,
            pixel: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
        """Return camera origin in UE cm and a unit world-space image ray."""
        pixel = _finite_vector(pixel, 2, "Pixel")
        camera, origin, rotation = self._pose(camera_id, frame)
        local = np.array([1.0, (pixel[0] - self.width / 2) / camera.fx,
                          -(pixel[1] - self.height / 2) / camera.fy])
        direction = rotation @ local
        direction /= np.linalg.norm(direction)
        return origin.copy(), direction

    def project(self, camera_id: int, frame: int,
                world_cm: tuple | np.ndarray) -> tuple[float, float] | None:
        """Project a world point; return None on or behind the camera plane."""
        world = _finite_vector(world_cm, 3, "World point")
        camera, origin, rotation = self._pose(camera_id, frame)
        forward, right, up = rotation.T @ (world - origin)
        if forward <= 1e-9:
            return None
        return (float(self.width / 2 + camera.fx * right / forward),
                float(self.height / 2 - camera.fy * up / forward))


def _observations(rig: CameraRig, frame: int, observations: list[dict], anchor='bottom_center') -> list[dict]:
    selected = {}
    for observation in observations:
        try:
            camera_id = int(observation["camera_id"])
            confidence = float(observation["confidence"])
            bbox = _finite_vector(observation["bbox"], 4, "Bounding box")
            x1, y1, x2, y2 = bbox
            if (not math.isfinite(confidence) or not 0 < confidence <= 1
                    or not 0 <= x1 < x2 <= rig.width
                    or not 0 <= y1 < y2 <= rig.height):
                continue
            pixel = np.array([(x1 + x2) / 2, (y1+y2)/2 if anchor == 'center' else y2])
            origin, direction = rig.ray(camera_id, frame, pixel)
        except (ValueError, TypeError, KeyError, OverflowError):
            continue
        candidate = dict(camera_id=camera_id, confidence=confidence, pixel=pixel,
                         origin=origin, direction=direction)
        if camera_id not in selected or confidence > selected[camera_id]["confidence"]:
            selected[camera_id] = candidate
    return sorted(selected.values(), key=lambda item: (-item["confidence"], item["camera_id"]))


def _pair_position(first: dict, second: dict) -> np.ndarray | None:
    d1, d2 = first["direction"], second["direction"]
    dot = float(d1 @ d2)
    denominator = 1 - dot * dot
    if denominator < _MIN_PAIR_SIN_SQUARED:
        return None
    offset = first["origin"] - second["origin"]
    d, e = float(d1 @ offset), float(d2 @ offset)
    first_depth = (dot * e - d) / denominator
    second_depth = (e - dot * d) / denominator
    if first_depth <= 0 or second_depth <= 0:
        return None
    p1 = first["origin"] + first_depth * d1
    p2 = second["origin"] + second_depth * d2
    if np.linalg.norm(p1 - p2) > _MAX_RAY_SEPARATION_CM:
        return None
    return (p1 + p2) / 2


def _support(rig: CameraRig, frame: int, point: np.ndarray, views: list[dict]) -> list[dict]:
    supported = []
    reprojection_limit = max(8.0, 0.02 * rig.width)
    for view in views:
        projection = rig.project(view["camera_id"], frame, point)
        if projection is None:
            continue
        if np.linalg.norm(np.asarray(projection) - view["pixel"]) <= reprojection_limit:
            supported.append(view)
    return supported


def estimate_position(rig: CameraRig, frame: int, observations: list[dict], *, anchor='bottom_center') -> dict | None:
    """Estimate one current contact position, using multi-view geometry if valid.

    Each observation is {camera_id, bbox:[x1,y1,x2,y2], confidence}, in original
    image pixels. Caller must establish object identity and frame synchronization.
    Bad detections are ignored. Ground-plane results expose their assumption and
    have residual_cm=None because one ray supplies no independent error estimate.
    """
    if anchor not in {'center', 'bottom_center'}:
        raise ValueError('Anchor must be center or bottom_center')
    views = _observations(rig, frame, observations, anchor=anchor)
    candidates = []
    for first, second in combinations(views, 2):
        point = _pair_position(first, second)
        if point is None or not np.isfinite(point).all():
            continue
        supported = _support(rig, frame, point, views)
        if len(supported) >= 2:
            candidates.append((supported, point))
    candidates.sort(key=lambda item: (len(item[0]), sum(view["confidence"] for view in item[0])), reverse=True)
    for supported, point in candidates:
        # Weighted closest point to all supporting rays, with no motion/route prior.
        matrix = np.zeros((3, 3))
        vector = np.zeros(3)
        for view in supported:
            projector = np.eye(3) - np.outer(view["direction"], view["direction"])
            weighted = view["confidence"] * projector
            matrix += weighted
            vector += weighted @ view["origin"]
        if np.linalg.cond(matrix) > 1e6:
            continue
        point = np.linalg.solve(matrix, vector)
        if not np.isfinite(point).all() or len(_support(rig, frame, point, supported)) != len(supported):
            continue
        distances = [float(np.linalg.norm(np.cross(point - view["origin"], view["direction"])))
                     for view in supported]
        if max(distances) > _MAX_RAY_SEPARATION_CM:
            continue
        return {"position_cm": point.tolist(), "method": "triangulated",
                "cameras": sorted(view["camera_id"] for view in supported),
                "residual_cm": math.sqrt(sum(value * value for value in distances) / len(distances))}
    for view in views:
        origin, direction = view["origin"], view["direction"]
        if direction[2] >= -_MIN_GROUND_DIRECTION_Z:
            continue
        distance = -origin[2] / direction[2]
        if not 0 < distance <= _MAX_GROUND_DISTANCE_CM:
            continue
        point = origin + distance * direction
        point[2] = 0.0
        if np.isfinite(point).all():
            return {"position_cm": point.tolist(), "method": "ground_plane",
                    "cameras": [view["camera_id"]], "residual_cm": None,
                    "assumption": "bbox bottom-center touches a flat ground plane z=0 (UE cm)"}
    return None


def estimate_measurement(rig: CameraRig, frame: int, observations: list[dict],
                         prior_cm=None, allow_ground=True) -> dict | None:
    """Quality-aware volume-center geometry; ground mode is a contact proxy.

    Clipped boxes remain useful to the image tracker, but not as contact points.
    Covariance includes a 1.5m object-extent floor: box centers from different
    views are approximate correspondences. It is a model uncertainty, not a
    calibrated confidence interval or a mesh-pivot accuracy guarantee.
    """
    complete = []
    margin = max(2., rig.width*.002)
    for item in observations:
        try:
            x1, y1, x2, y2 = _finite_vector(item['bbox'], 4, 'Bounding box')
            if x1 > margin and y1 > margin and x2 < rig.width-margin and y2 < rig.height-margin:
                complete.append(item)
        except (ValueError, KeyError, TypeError):
            continue
    views = _observations(rig, frame, complete, anchor='center')
    if not views:
        return None
    result = estimate_position(rig, frame, complete, anchor='center')
    if result and result['method'] == 'triangulated':
        point = np.asarray(result['position_cm'])
        supported = [v for v in views if v['camera_id'] in result['cameras']]
        information = np.zeros((3, 3))
        reprojections = []
        for view in supported:
            camera = rig._cameras[view['camera_id']]
            depth = np.linalg.norm(point-view['origin'])
            sigma = max(150., depth*5/min(camera.fx, camera.fy), result['residual_cm'])
            information += (np.eye(3)-np.outer(view['direction'], view['direction']))/sigma**2
            reprojections.append(float(np.linalg.norm(np.asarray(rig.project(view['camera_id'], frame, point))-view['pixel'])))
        covariance = np.linalg.inv(information)
        angle = max(math.degrees(math.acos(np.clip(abs(a['direction'] @ b['direction']), 0, 1)))
                    for a, b in combinations(supported, 2))
        return {**result, 'quality': 'multi_view', 'covariance_cm2': covariance.tolist(),
                'uncertainty_m': math.sqrt(float(np.linalg.eigvalsh(covariance).max()))/100,
                'ray_angle_deg': angle, 'reprojection_max_px': max(reprojections),
                'metric_depth_observed': True, 'reference_point': 'bbox_visual_center'}
    if prior_cm is not None:
        prior = _finite_vector(prior_cm, 3, 'Prior point')
        choices = []
        for view in views:
            camera = rig._cameras[view['camera_id']]
            depth = float((prior-view['origin']) @ view['direction'])
            if depth <= 0:
                continue
            sigma = max(150., depth*5/min(camera.fx, camera.fy))
            closest = view['origin'] + depth*view['direction']
            choices.append((sigma, view, closest))
        if choices:
            sigma, view, point = min(choices, key=lambda v: v[0])
            along = np.outer(view['direction'], view['direction'])
            covariance = sigma**2*(np.eye(3)-along)+1e12*along
            return {'position_cm': point.tolist(), 'method': 'bearing_update',
                    'cameras': [view['camera_id']], 'origin_cm': view['origin'].tolist(),
                    'direction': view['direction'].tolist(), 'transverse_sigma_cm': sigma,
                    'covariance_cm2': covariance.tolist(), 'quality': 'depth_from_motion_prior',
                    'reference_point': 'bbox_visual_center',
                    'metric_depth_observed': False,
                    'assumption': 'Only bearing is observed; depth comes from past metric state and motion model'}
    if not allow_ground:
        return None
    # Explicit ground-assumed fallback. Terrain error is a correlated bias;
    # keep it separately from pixel noise so repeated samples cannot erase it.
    choices = []
    for view in _observations(rig, frame, complete):
        hit = estimate_position(rig, frame, [o for o in complete if o['camera_id'] == view['camera_id']])
        if not hit:
            continue
        point = np.asarray(hit['position_cm'])
        camera = rig._cameras[view['camera_id']]
        depth = np.linalg.norm(point-view['origin'])
        transverse = max(150., depth*5/min(camera.fx, camera.fy))
        radial = max(1000., 1000./max(abs(view['direction'][2]), .001))
        along = np.outer(view['direction'], view['direction'])
        jacobian = np.zeros((3, 2))
        for axis in range(2):
            shifted = view['pixel'].copy()
            shifted[axis] += 1.
            shifted_origin, shifted_direction = rig.ray(view['camera_id'], frame, shifted)
            if shifted_direction[2] >= -_MIN_GROUND_DIRECTION_Z:
                jacobian[:, axis] = view['direction'] * radial
            else:
                shifted_point = shifted_origin - shifted_origin[2]/shifted_direction[2]*shifted_direction
                jacobian[:, axis] = shifted_point-point
        covariance = 25.*jacobian @ jacobian.T + np.eye(3)*150.**2
        systematic = radial**2*along
        choices.append((np.trace(covariance+systematic), hit, covariance, systematic))
    if not choices:
        return None
    _, result, covariance, systematic = min(choices, key=lambda x: x[0])
    return {**result, 'quality': 'assumed_ground', 'metric_depth_observed': False,
            'covariance_cm2': covariance.tolist(),
            'systematic_covariance_cm2': systematic.tolist(),
            'uncertainty_m': math.sqrt(float(np.linalg.eigvalsh(covariance+systematic).max()))/100,
            'reference_point': 'approximate_bbox_ground_contact'}
