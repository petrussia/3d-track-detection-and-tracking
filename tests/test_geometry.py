"""Synthetic camera tests; no object telemetry or learned route is used."""

from pathlib import Path
import math
import tempfile
import unittest

try:
    from peleng.geometry import CameraRig, estimate_position
except ModuleNotFoundError as exc:
    if exc.name not in {"peleng", "peleng.geometry"}:
        raise
    CameraRig = estimate_position = None


def write_camera(directory, camera_id=1, poses=None, focal=10, sensor="40*20мм"):
    """Write the same CP1251 semicolon format as the supplied camera export."""
    poses = poses or [(0, 0, -100, 100, 0, 0, 0)]
    lines = [
        "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;"
        + str(focal)
    ]
    for index, pose in enumerate(poses):
        metadata = ";;матрица;" + sensor if index == 0 else ";;;;"
        lines.append(";".join(str(value) for value in pose) + ";" + metadata)
    path = Path(directory) / f"object_transform_CineCameraActor{camera_id}.csv"
    path.write_text("\n".join(lines) + "\n", encoding="cp1251")
    return path


class GeometryTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(CameraRig, "CameraRig geometry feature has not been implemented")
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def make_rig(self, poses=None, **kwargs):
        write_camera(self.directory, poses=poses, **kwargs)
        return CameraRig(self.directory, width=400, height=200)

    def assert_vector_close(self, actual, expected, tolerance=1e-9):
        self.assertEqual(len(actual), len(expected))
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(float(actual_value), float(expected_value), delta=tolerance)

    def test_zero_rotation_maps_forward_right_and_up_to_image_axes(self):
        # Catches swapping UE X-forward/Y-right/Z-up or the vertical pixel sign.
        rig = self.make_rig(poses=[(0, 0, 0, 0, 0, 0, 0)])
        self.assertEqual(rig.camera_ids, [1])
        self.assertEqual(rig.project(1, 0, (100, 0, 0)), (200.0, 100.0))
        self.assertEqual(rig.project(1, 0, (100, 100, 0)), (300.0, 100.0))
        self.assertEqual(rig.project(1, 0, (100, 0, 100)), (200.0, 0.0))
        origin, direction = rig.ray(1, 0, (300, 0))
        self.assert_vector_close(origin, [0, 0, 0])
        self.assert_vector_close(direction, [1 / math.sqrt(3)] * 3)

    def test_ue_positive_yaw_pitch_and_roll_conventions(self):
        # Catches conventional right-handed Euler code applied to UE export angles.
        write_camera(self.directory, 1, [(0, 0, 0, 0, 0, 0, 90)])
        write_camera(self.directory, 2, [(0, 0, 0, 0, 0, 30, 0)])
        write_camera(self.directory, 3, [(0, 0, 0, 0, 90, 0, 0)])
        rig = CameraRig(self.directory, width=400, height=200)
        self.assert_vector_close(rig.ray(1, 0, (200, 100))[1], [0, 1, 0])
        self.assert_vector_close(rig.ray(2, 0, (200, 100))[1], [math.sqrt(3) / 2, 0, 0.5])
        self.assert_vector_close(rig.ray(3, 0, (300, 100))[1], [1 / math.sqrt(2), 0, -1 / math.sqrt(2)])

    def test_decimal_comma_sensor_sets_intrinsics_and_roundtrips_pixel(self):
        # Catches reading commas as delimiters or silently assuming a full-frame sensor.
        write_camera(self.directory, poses=[(0, 0, 0, 0, 0, 0, 0)], focal=35, sensor="23,76*13,365мм")
        rig = CameraRig(self.directory)
        self.assertAlmostEqual(rig.project(1, 0, (100, 10, 0))[0], 1242.8282828282827)
        origin, direction = rig.ray(1, 0, (1234, 345))
        self.assert_vector_close(rig.project(1, 0, origin + 1000 * direction), [1234, 345])

    def test_wrapped_yaw_keeps_nearby_view_directions(self):
        # Catches treating the +-180 degree boundary as a full turn.
        rig = self.make_rig(poses=[(0, 0, 0, 0, 0, 0, 179), (1, 0, 0, 0, 0, 0, -179)])
        first = rig.ray(1, 0, (200, 100))[1]
        second = rig.ray(1, 1, (200, 100))[1]
        self.assertGreater(sum(float(a * b) for a, b in zip(first, second)), 0.999)
        self.assertLess(first[0], -0.999)
        self.assertLess(second[0], -0.999)

    def test_missing_frame_pose_uses_only_past_telemetry(self):
        # Catches leaking a later pan pose into a current estimate.
        rig = self.make_rig(poses=[(0, 0, 0, 0, 0, 0, 0), (10, 0, 0, 0, 0, 0, 90)])
        self.assert_vector_close(rig.ray(1, 5, (200, 100))[1], [1, 0, 0])
        self.assert_vector_close(rig.ray(1, 10, (200, 100))[1], [0, 1, 0])
        with self.assertRaises(ValueError):
            rig.ray(1, -1, (200, 100))

    def test_rejects_nonfinite_rays_and_points_behind_camera(self):
        # Catches NaNs propagating into a published location and negative depth.
        rig = self.make_rig(poses=[(0, 0, 0, 0, 0, 0, 0)])
        self.assertIsNone(rig.project(1, 0, (-100, 0, 0)))
        self.assertIsNone(rig.project(1, 0, (0, 100, 0)))
        with self.assertRaises(ValueError):
            rig.ray(1, 0, (float("nan"), 100))
        with self.assertRaises(ValueError):
            rig.project(1, 0, (100, float("inf"), 0))

    def test_one_camera_ground_contact_uses_bbox_bottom_center(self):
        # Catches using bbox center, which projects above the contact point.
        rig = self.make_rig()
        result = estimate_position(rig, 0, [{"camera_id": 1, "bbox": [200, 90, 220, 110], "confidence": 0.9}])
        self.assertIsNotNone(result)
        self.assertEqual(result["method"], "ground_plane")
        self.assertEqual(result["cameras"], [1])
        self.assertIsNone(result["residual_cm"])
        self.assertIn("z=0", result["assumption"])
        self.assert_vector_close(result["position_cm"], [1000, 0, 0], tolerance=1e-8)

    def test_two_views_triangulate_a_known_contact_point(self):
        # Catches wrong ray least-squares signs and silently ignoring the second view.
        write_camera(self.directory, 1, [(0, 0, -100, 100, 0, 0, 0)])
        write_camera(self.directory, 2, [(0, 0, 100, 100, 0, 0, 0)])
        rig = CameraRig(self.directory, width=400, height=200)
        result = estimate_position(rig, 0, [
            {"camera_id": 1, "bbox": [200, 90, 220, 110], "confidence": 0.9},
            {"camera_id": 2, "bbox": [180, 90, 200, 110], "confidence": 0.8},
        ])
        self.assertEqual(result["method"], "triangulated")
        self.assertEqual(result["cameras"], [1, 2])
        self.assertLess(result["residual_cm"], 1e-7)
        self.assert_vector_close(result["position_cm"], [1000, 0, 0], tolerance=1e-7)

    def test_parallel_and_behind_intersections_do_not_triangulate(self):
        # Catches dividing by a tiny inter-ray angle or accepting negative ray depth.
        write_camera(self.directory, 1, [(0, 0, -100, 100, 0, 0, 0)])
        write_camera(self.directory, 2, [(0, 0, 100, 100, 0, 0, 0)])
        rig = CameraRig(self.directory, width=400, height=200)
        for first_x, second_x in [(210, 210), (190, 210)]:
            result = estimate_position(rig, 0, [
                {"camera_id": 1, "bbox": [first_x - 10, 90, first_x + 10, 110], "confidence": 0.9},
                {"camera_id": 2, "bbox": [second_x - 10, 90, second_x + 10, 110], "confidence": 0.8},
            ])
            self.assertIsNotNone(result)
            self.assertEqual(result["method"], "ground_plane")

    def test_ground_plane_rejects_horizon_upward_and_bad_detections(self):
        # Catches infinite/behind ground hits, zero-width boxes, and invalid confidence.
        rig = self.make_rig()
        for bbox, confidence in [([190, 80, 210, 100], 0.9), ([190, 70, 210, 90], 0.9),
                                 ([200, 90, 200, 110], 0.9), ([190, 90, 210, 110], float("nan")),
                                 ([190, 90, 210, 110], 0), ([float("nan"), 90, 210, 110], 0.9)]:
            self.assertIsNone(estimate_position(rig, 0, [{"camera_id": 1, "bbox": bbox, "confidence": confidence}]))
        self.assertIsNone(estimate_position(rig, 0, []))

    def test_large_ray_separation_rejects_false_multi_view_match(self):
        # Catches claiming triangulation for detections on incompatible 3D lines.
        write_camera(self.directory, 1, [(0, 0, -100, 100, 0, 0, 0)])
        write_camera(self.directory, 2, [(0, 0, 100, 10100, 0, 0, 0)])
        rig = CameraRig(self.directory, width=400, height=200)
        result = estimate_position(rig, 0, [
            {"camera_id": 1, "bbox": [200, 90, 220, 110], "confidence": 0.9},
            {"camera_id": 2, "bbox": [180, 90, 200, 110], "confidence": 0.8},
        ])
        self.assertEqual(result["method"], "ground_plane")

    def test_large_reprojection_rejects_even_short_skew_rays(self):
        # Catches accepting a small metric gap that is large in this image geometry.
        write_camera(self.directory, 1, [(0, 0, -10, 100, 0, 0, 0)])
        write_camera(self.directory, 2, [(0, 0, 10, 120, 0, 0, 0)])
        rig = CameraRig(self.directory, width=400, height=200)
        result = estimate_position(rig, 0, [
            {"camera_id": 1, "bbox": [200, 90, 220, 110], "confidence": 0.9},
            {"camera_id": 2, "bbox": [180, 90, 200, 110], "confidence": 0.8},
        ])
        self.assertEqual(result["method"], "ground_plane")

    def test_bad_headers_missing_calibration_and_nonfinite_pose_are_actionable(self):
        # Catches using silently invented camera intrinsics or malformed telemetry.
        path = write_camera(self.directory)
        for text in [
            "Frame;Loc_X\n0;1\n",
            "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw\n0;0;0;100;0;0;0\n",
            "Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;10\n0;nan;0;100;0;0;0;;;матрица;40*20мм\n",
        ]:
            path.write_text(text, encoding="cp1251")
            with self.assertRaisesRegex(ValueError, "CineCameraActor1"):
                CameraRig(self.directory)

    def test_empty_directory_reports_missing_camera_files(self):
        # Catches constructing an apparently usable rig with zero loaded cameras.
        with self.assertRaises(FileNotFoundError):
            CameraRig(self.directory)

    def test_metric_estimator_rejects_truncated_boxes_but_not_interior_boxes(self):
        from peleng import geometry
        estimator = getattr(geometry, 'estimate_measurement', None)
        self.assertIsNotNone(estimator, 'quality-aware geometry is missing')
        rig = self.make_rig()
        self.assertIsNone(estimator(rig, 0, [{'camera_id': 1, 'bbox': [390, 90, 400, 110], 'confidence': .9}]))
        result = estimator(rig, 0, [{'camera_id': 1, 'bbox': [200, 90, 220, 110], 'confidence': .9}])
        self.assertIsNotNone(result)
        self.assertEqual(result['quality'], 'assumed_ground')
        self.assertGreater(result['uncertainty_m'], 1)

    def test_bearing_update_preserves_prior_depth_without_forcing_ground(self):
        from peleng import geometry
        estimator = getattr(geometry, 'estimate_measurement', None)
        self.assertIsNotNone(estimator, 'quality-aware geometry is missing')
        rig = self.make_rig(poses=[(0, 0, 0, 100, 0, 0, 0)])
        result = estimator(rig, 0, [{'camera_id': 1, 'bbox': [190, 80, 210, 100], 'confidence': .9}],
                           prior_cm=[1000, 0, 200])
        self.assertEqual(result['method'], 'bearing_update')
        self.assertFalse(result['metric_depth_observed'])
        self.assert_vector_close(result['position_cm'], [1000, 0, 200], tolerance=1e-7)
        self.assertGreater(result['covariance_cm2'][0][0], result['covariance_cm2'][1][1] * 100)

    def test_metric_views_triangulate_visual_centers_instead_of_unmatched_bottoms(self):
        from peleng.geometry import estimate_measurement
        write_camera(self.directory, 1, [(0, 0, -100, 100, 0, 0, 0)])
        write_camera(self.directory, 2, [(0, 0, 100, 100, 0, 0, 0)])
        rig = CameraRig(self.directory, width=400, height=200)
        result = estimate_measurement(rig, 0, [
            {'camera_id': 1, 'bbox': [200,70,220,90], 'confidence': .9},
            {'camera_id': 2, 'bbox': [180,70,200,90], 'confidence': .9}])
        self.assertEqual(result['method'], 'triangulated')
        self.assert_vector_close(result['position_cm'], [1000, 0, 300], tolerance=1e-7)
        self.assertEqual(result['reference_point'], 'bbox_visual_center')


if __name__ == "__main__":
    unittest.main()
