"""Preflight camera inputs without opening the target's reference trajectory."""
from pathlib import Path
import math

import cv2

from .geometry import CameraRig


def camera_video(directory, camera_id):
    directory = Path(directory)
    matches = sorted(directory.glob(f'*.FinalImage.CineCameraActor{camera_id}.mp4'))
    if len(matches) > 1:
        raise ValueError(f'Ambiguous camera video {camera_id}: {[p.name for p in matches]}')
    return matches[0] if matches else directory/f'Way1.FinalImage.CineCameraActor{camera_id}.mp4'


def inspect_dataset(directory):
    directory = Path(directory)
    rig = CameraRig(directory)
    streams = []
    warnings = []
    for camera_id in rig.camera_ids:
        path = camera_video(directory, camera_id)
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise ValueError(f'Cannot open camera video: {path.name}')
            fps = capture.get(cv2.CAP_PROP_FPS)
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if not math.isfinite(fps) or fps <= 0 or min(count, width, height) <= 0:
                raise ValueError(f'Invalid video metadata: {path.name}')
            streams.append({'camera_id': camera_id, 'fps': fps, 'frames': count,
                            'width': width, 'height': height})
            poses = rig._cameras[camera_id].frames
            if poses[0] != 0:
                raise ValueError(f'Camera {camera_id}: no pose at frame 0')
            if poses[-1] < count-1:
                warnings.append(f'Camera {camera_id}: last pose held after frame {poses[-1]}')
        finally:
            capture.release()
    first = streams[0]
    for stream in streams[1:]:
        if abs(stream['fps']-first['fps']) > .01:
            raise ValueError('Camera recordings must have equal FPS')
        if stream['frames'] != first['frames']:
            raise ValueError('Camera recordings must have equal frame counts')
        if (stream['width'], stream['height']) != (first['width'], first['height']):
            raise ValueError('Camera recordings must have equal image dimensions')
    return {'camera_count': len(streams), 'camera_ids': rig.camera_ids,
            'frames_per_camera': first['frames'], 'fps': first['fps'],
            'width': first['width'], 'height': first['height'],
            'duration_s': first['frames']/first['fps'], 'warnings': warnings,
            'synchronization_assumption': 'shared frame zero and equal constant FPS; no measured clock offset',
            'coordinate_units_assumed': 'UE cm', 'ground_truth_read': False}
