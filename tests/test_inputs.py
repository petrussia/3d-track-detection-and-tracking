from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2

from test_geometry import write_camera


class InputsTests(unittest.TestCase):
    def test_preflight_needs_only_camera_data_and_rejects_unequal_streams(self):
        try:
            from peleng.inputs import inspect_dataset
        except ModuleNotFoundError:
            self.fail('input preflight is missing')
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            write_camera(folder, 1)
            write_camera(folder, 2)
            counts = {1: 10, 2: 10}
            released = []

            class Capture:
                def __init__(self, path):
                    self.camera = int(Path(path).stem[-1])
                def isOpened(self): return True
                def get(self, key):
                    return {cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: counts[self.camera],
                            cv2.CAP_PROP_FRAME_WIDTH: 1920, cv2.CAP_PROP_FRAME_HEIGHT: 1080}[key]
                def release(self): released.append(self.camera)

            with patch('peleng.inputs.cv2.VideoCapture', Capture):
                result = inspect_dataset(folder)
                self.assertEqual(result['camera_count'], 2)
                self.assertEqual(result['frames_per_camera'], 10)
                self.assertFalse(result['ground_truth_read'])
                counts[2] = 9
                with self.assertRaisesRegex(ValueError, 'frame|count|length'):
                    inspect_dataset(folder)
            self.assertEqual(released, [1, 2, 1, 2])
