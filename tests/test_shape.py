"""Cuboid geometry fixtures are projected independently of the estimator."""
import copy
import itertools
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from peleng.geometry import CameraRig

try:
    from peleng.shape import ShapeEstimator
except ModuleNotFoundError as exc:
    if exc.name != 'peleng.shape':
        raise
    ShapeEstimator = None


class ShapeTests(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(ShapeEstimator, 'Causal shape estimation is not implemented')
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.poses = {
            1: ([-10000., -3000., 2200.], -7., 15.),
            2: ([3000., -10000., 1800.], -5., 105.),
            3: ([-10000., -3000., 2200.], -7., 15.),
            4: ([3000., -10000., 1800.], -5., 105.),
        }
        for camera_id, (origin, pitch, yaw) in self.poses.items():
            text = ('Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;15\n'
                    f'0;{origin[0]};{origin[1]};{origin[2]};0;{pitch};{yaw};;;матрица;24*13,5мм\n')
            (Path(self.temp.name) / f'object_transform_CineCameraActor{camera_id}.csv').write_text(
                text, encoding='cp1251')
        self.rig = CameraRig(self.temp.name)
        self.dimensions = np.array([2000., 600., 700.])
        self.yaw = math.radians(37.)
        self.direction = np.array([math.cos(self.yaw), math.sin(self.yaw), 0.])

    def boxes(self, center, yaw=None, dimensions=None, cameras=(1, 2)):
        # Independent forward/right/up dot products; no estimator projection helper.
        yaw = self.yaw if yaw is None else yaw
        dimensions = self.dimensions if dimensions is None else np.asarray(dimensions)
        c, s = math.cos(yaw), math.sin(yaw)
        axes = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
        corners = np.asarray(center) + np.array(list(itertools.product([-1., 1.], repeat=3))) * dimensions / 2 @ axes.T
        observations = []
        for camera_id in cameras:
            origin, pitch, camera_yaw = self.poses[camera_id]
            p, y = math.radians(pitch), math.radians(camera_yaw)
            forward = np.array([math.cos(p)*math.cos(y), math.cos(p)*math.sin(y), math.sin(p)])
            right = np.array([-math.sin(y), math.cos(y), 0.])
            up = np.cross(forward, right)
            relative = corners - origin
            depth = relative @ forward
            pixels = np.column_stack((960 + 1200 * (relative @ right) / depth,
                                      540 - 1200 * (relative @ up) / depth))
            lo, hi = pixels.min(axis=0), pixels.max(axis=0)
            observations.append({'camera_id': camera_id, 'bbox': [*lo, *hi],
                                 'confidence': .9, 'source': 'neural', 'measured': True})
        return observations

    def learn(self, estimator, start=0, count=6, reverse=False):
        result = None
        velocity = self.direction * (-400 if reverse else 400)
        for index in range(start, start + count):
            center = np.array([1000., 500., 400.]) + self.direction * index * 40
            result = estimator.update(index, index / 10, self.boxes(center), velocity_cm_s=velocity)
            self.assertIsNotNone(result)
        return result

    def test_two_views_recover_known_volume_and_front_without_reference_coordinates(self):
        # Replacing the extent fit with bbox-center rays must fail this projection test.
        estimator = ShapeEstimator(self.rig)
        center = np.array([1000., 500., 400.])
        result = estimator.update(0, 0., self.boxes(center), velocity_cm_s=self.direction * 400)
        self.assertIsNotNone(result)
        np.testing.assert_allclose(result['center_cm'], center, atol=8.)
        np.testing.assert_allclose(result['dimensions_cm'], [2000., 600., 700.], atol=20.)
        np.testing.assert_allclose(result['front_cm'], center + self.direction * 1000, atol=15.)
        self.assertEqual(result['position_cm'], result['center_cm'])
        self.assertTrue(result['front_observable'])
        self.assertFalse(result['dimensions_learned'])
        self.assertLess(result['fit_rms_px'], .1)
        self.assertGreater(np.linalg.eigvalsh(result['covariance_cm2']).min(), 0)

    def test_rejected_proposal_does_not_change_shape_history_or_rewind_time(self):
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        checkpoint = estimator.geometry_checkpoint()
        old_dimensions = estimator._dimensions.copy()
        old_samples = estimator._sample_count
        estimator.update(6, .6, self.boxes([1400., 800., 400.]), velocity_cm_s=self.direction*400)
        self.assertGreater(estimator._sample_count, old_samples)
        estimator.restore_geometry(checkpoint)
        self.assertEqual(estimator._sample_count, old_samples)
        np.testing.assert_array_equal(estimator._dimensions, old_dimensions)
        self.assertEqual(estimator._frame, 6)
        self.assertEqual(estimator._timestamp, .6)
        with self.assertRaises(ValueError):
            estimator.update(6, .6, [])

    def test_known_geometry_preserves_size_and_nonzero_shared_uncertainty(self):
        estimator = ShapeEstimator(self.rig, dimensions_m=self.dimensions/100.)
        first = estimator.update(0, 0., self.boxes([1000.,500.,400.]),
                                 velocity_cm_s=self.direction*400)
        self.assertIsNotNone(first)
        self.assertEqual(first['dimension_source'], 'configured_asset_geometry')
        self.assertEqual(first['shape_samples'], 0)
        np.testing.assert_allclose(first['center_cm'], [1000.,500.,400.], atol=5.)
        for frame in range(1, 5):
            result = estimator.update(frame, frame/10,
                self.boxes([1000.+frame*30.,500.,400.], cameras=(1,)),
                prior_cm=[1000.+frame*30.,500.,400.], velocity_cm_s=self.direction*400)
            self.assertIsNotNone(result)
            np.testing.assert_allclose(result['dimensions_cm'], self.dimensions)
            np.testing.assert_allclose(result['dimension_sigma_cm'], .1*self.dimensions)
            self.assertFalse(result['metric_depth_observed'])

    def test_forward_sign_follows_causal_motion_not_a_fixed_world_axis(self):
        # Choosing +X or a fixed yaw branch silently flips the front on reversing motion.
        forward = self.learn(ShapeEstimator(self.rig))
        backward = self.learn(ShapeEstimator(self.rig), reverse=True)
        a = np.array(forward['front_cm']) - forward['center_cm']
        b = np.array(backward['front_cm']) - backward['center_cm']
        self.assertGreater(a @ self.direction, 950)
        self.assertLess(b @ self.direction, -950)
        np.testing.assert_allclose(a, -b, atol=10.)

    def test_unknown_motion_does_not_invent_a_front_sign(self):
        # Bboxes are invariant to a 180 degree cuboid rotation.
        result = ShapeEstimator(self.rig).update(0, 0., self.boxes([1000., 500., 400.]))
        self.assertIsNotNone(result)
        self.assertFalse(result['front_observable'])
        self.assertIsNone(result['front_cm'])

    def test_rejected_third_view_seed_cannot_reverse_the_front_sign(self):
        # An all-view ray seed is not a validated body-motion observation.
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        center = np.array([1000., 500., 400.])+self.direction*240
        observations = self.boxes(center, cameras=(1, 2, 3))
        observations[2]['bbox'] = (np.array(observations[2]['bbox'])+[-200., 0., -200., 0.]).tolist()
        observations[2]['confidence'] = .3
        result = estimator.update(6, .6, observations)
        self.assertIsNotNone(result)
        self.assertEqual(result['cameras'], [1, 2])
        self.assertGreater(np.dot(np.array(result['front_cm'])-result['center_cm'], self.direction), 900.)
        np.testing.assert_allclose(result['center_cm'], center, atol=15.)

    def test_initial_front_sign_can_be_learned_from_robust_center_motion(self):
        # Holding a past sign is insufficient when the first frame has no velocity hint.
        estimator = ShapeEstimator(self.rig)
        first = estimator.update(0, 0., self.boxes([1000., 500., 400.]))
        self.assertIsNotNone(first)
        self.assertFalse(first['front_observable'])
        for frame in range(1, 9):
            center = np.array([1000., 500., 400.])+self.direction*frame*50
            result = estimator.update(frame, frame/10, self.boxes(center))
            self.assertIsNotNone(result)
        self.assertTrue(result['front_observable'])
        offset = np.array(result['front_cm'])-result['center_cm']
        self.assertGreater(np.dot(offset, self.direction)/np.linalg.norm(offset), .8)

    def test_consensus_prefers_three_supporting_views_over_confident_outlier(self):
        estimator = ShapeEstimator(self.rig)
        center = np.array([1000., 500., 400.])
        observations = self.boxes(center, cameras=(1, 2, 3, 4))
        observations[2]['bbox'] = (np.array(observations[2]['bbox'])+[-200., 0., -200., 0.]).tolist()
        observations[2]['confidence'] = 1.
        result = estimator.update(0, 0., observations, velocity_cm_s=self.direction*400)
        self.assertIsNotNone(result)
        self.assertEqual(result['view_consensus']['supporting_cameras'], [1, 2, 4])
        self.assertEqual(result['view_consensus']['rejected_cameras'], [3])
        self.assertEqual(result['cameras'], [1, 2, 4])
        np.testing.assert_allclose(result['center_cm'], center, atol=15.)

    def test_established_image_dimensions_supply_single_view_range(self):
        # A bearing-only implementation keeps the deliberately wrong prior range.
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        self.assertTrue(learned['dimensions_learned'])
        center = np.array([2100., 1300., 450.])
        result = estimator.update(6, .6, self.boxes(center, cameras=(1,)),
                                  prior_cm=[5500., 2300., 700.], velocity_cm_s=self.direction * 400)
        self.assertIsNotNone(result)
        self.assertTrue(result['range_from_shape'])
        np.testing.assert_allclose(result['center_cm'], center, atol=35.)
        self.assertEqual(result['shape_samples'], learned['shape_samples'])
        np.testing.assert_allclose(result['dimensions_cm'], learned['dimensions_cm'], atol=1e-8)

    def test_bad_full_label_does_not_hide_a_good_known_size_single_view(self):
        # An unflagged foreground fragment must not suppress the intact camera.
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        center = [1300., 700., 420.]
        observations = self.boxes(center)
        small = np.array(observations[1]['bbox'])
        midpoint = (small[:2]+small[2:])/2
        half = (small[2:]-small[:2])*.15
        observations[1]['bbox'] = np.r_[midpoint-half, midpoint+half].tolist()
        result = estimator.update(6, .6, observations, prior_cm=center,
                                  velocity_cm_s=self.direction*400)
        self.assertIsNotNone(result)
        self.assertEqual(result['cameras'], [1])
        self.assertEqual(result['shape_samples'], learned['shape_samples'])
        np.testing.assert_allclose(result['center_cm'], center, atol=35.)

    def test_only_a_far_fragment_cannot_override_the_causal_range_prior(self):
        # Known size applied to a small fragment otherwise manufactures a distant vehicle.
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        center = [1300., 700., 420.]
        observations = self.boxes(center, cameras=(1, 2))
        for observation in observations:
            box = np.array(observation['bbox'])
            midpoint = (box[:2]+box[2:])/2
            half = (box[2:]-box[:2])*.15
            observation['bbox'] = np.r_[midpoint-half, midpoint+half].tolist()
        self.assertIsNone(estimator.update(6, .6, observations, prior_cm=center,
                                          velocity_cm_s=self.direction*400))

    def test_single_view_cannot_initialize_metric_scale_even_with_position_prior(self):
        # A motion/ground prior must not masquerade as image-derived object dimensions.
        result = ShapeEstimator(self.rig).update(0, 0., self.boxes([1000., 500., 400.], cameras=(1,)),
                                                 prior_cm=[1000., 500., 400.], velocity_cm_s=[400, 0, 0])
        self.assertIsNone(result)

    def test_one_good_multiview_height_supports_uncertain_provisional_range(self):
        # Requiring repeated copies of the same geometry discards already observed metric scale.
        estimator = ShapeEstimator(self.rig)
        estimator.update(0, 0., self.boxes([1000., 500., 400.]), velocity_cm_s=self.direction * 400)
        result = estimator.update(1, .1, self.boxes([1300., 700., 420.], cameras=(1,)),
                                  velocity_cm_s=self.direction * 400)
        self.assertIsNotNone(result)
        self.assertTrue(result['provisional_scale'])
        self.assertFalse(result['dimensions_learned'])
        self.assertEqual(result['shape_samples'], 1)
        np.testing.assert_allclose(result['center_cm'], [1300., 700., 420.], atol=35.)

    def test_parallel_baseline_cannot_learn_shape(self):
        # Two camera IDs at the same optical center do not observe metric depth.
        estimator = ShapeEstimator(self.rig)
        for frame in range(4):
            self.assertIsNone(estimator.update(frame, frame / 10,
                                              self.boxes([1000., 500., 400.], cameras=(1, 3))))

    def test_clipped_and_explicitly_partial_boxes_do_not_teach_dimensions(self):
        # Treating a visible fragment as a complete silhouette collapses object width.
        for flag in ('crop_only', 'occluded', 'truncated'):
            estimator = ShapeEstimator(self.rig)
            observations = self.boxes([1000., 500., 400.])
            observations[0][flag] = True
            self.assertIsNone(estimator.update(0, 0., observations))
        observations = self.boxes([1000., 500., 400.])
        observations[0]['bbox'][0] = 0.
        self.assertIsNone(ShapeEstimator(self.rig).update(0, 0., observations))

    def test_collapsed_cross_section_is_rejected_not_learned(self):
        # A numerically perfect sheet fit is not a usable vehicle volume.
        observations = self.boxes([1000., 500., 400.], dimensions=[2000., 2., 700.])
        self.assertIsNone(ShapeEstimator(self.rig).update(0, 0., observations,
                                                        velocity_cm_s=self.direction * 400))

    def test_explicit_fragment_is_not_a_full_shape_measurement(self):
        observations = self.boxes([1000., 500., 400.])
        observations[0].update(source='foreground', quality={'foreground_fill': .8, 'partial': True})
        self.assertIsNone(ShapeEstimator(self.rig).update(0, 0., observations))

    def test_low_fill_alone_does_not_mean_partial_object(self):
        # An intact diagonal truck can occupy little of its axis-aligned box.
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        observations = self.boxes([1300., 700., 420.], cameras=(1,))
        observations[0].update(source='foreground', quality={'foreground_fill': .39})
        result = estimator.update(6, .6, observations, velocity_cm_s=self.direction * 400)
        self.assertIsNotNone(result)
        self.assertFalse(result['partial_silhouette'])
        self.assertEqual(result['shape_samples'], learned['shape_samples'])
        np.testing.assert_allclose(result['center_cm'], [1300., 700., 420.], atol=50.)

    def test_clipped_single_view_does_not_update_depth_or_shape(self):
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        observations = self.boxes([1000., 12000., 400.], cameras=(1,))
        observations[0]['bbox'][2] = 1920.
        result = estimator.update(6, .6, observations, velocity_cm_s=self.direction * 400)
        self.assertIsNone(result)
        self.assertEqual(estimator._sample_count, learned['shape_samples'])

    def test_containment_only_fragment_has_no_metric_range_measurement(self):
        # All four satisfied inequalities carry no local position information.
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        observations = self.boxes([1300., 700., 420.], cameras=(1,))
        observations[0]['quality'] = {'partial': True, 'foreground_fill': .8}
        self.assertIsNone(estimator.update(6, .6, observations,
                                          prior_cm=[1300., 700., 420.], velocity_cm_s=self.direction*400))

    def test_inactive_second_view_does_not_claim_new_stereo_depth(self):
        # A second camera ID is not stereo evidence when every one of its inequalities is inactive.
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        observations = self.boxes([1300., 700., 420.])
        box = np.asarray(observations[1]['bbox'])
        midpoint = (box[:2]+box[2:])/2
        half = (box[2:]-box[:2])*.25
        observations[1]['bbox'] = np.r_[midpoint-half, midpoint+half].tolist()
        observations[1]['quality'] = {'partial': True}
        result = estimator.update(6, .6, observations, velocity_cm_s=self.direction*400)
        self.assertIsNotNone(result)
        self.assertFalse(result['metric_depth_observed'])
        self.assertTrue(result['range_from_shape'])
        self.assertEqual(result['method'], 'cuboid_singleview')
        self.assertEqual(result['cameras'], [1])
        values = np.linalg.eigvalsh(result['systematic_covariance_cm2'])
        self.assertGreater(values.max(), values.min()*10)

    def test_two_true_edges_cannot_determine_three_position_coordinates(self):
        # A yaw prior cannot supply the missing positional rank of two genuine edges.
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        observations = self.boxes([1300., 700., 420.], cameras=(1,))
        observations[0]['bbox'][0], observations[0]['bbox'][2] = 0., 1920.
        self.assertIsNone(estimator.update(6, .6, observations, velocity_cm_s=self.direction*400))

    def test_weak_information_becomes_large_variance_never_zero_variance(self):
        # pinv(J.T@J) with a relative cutoff silently erases weak depth variance.
        covariance_from_jacobian = getattr(ShapeEstimator, '_parameter_covariance', None)
        self.assertIsNotNone(covariance_from_jacobian, 'Stable information inversion is missing')
        covariance = covariance_from_jacobian(np.diag([1., 2., 1e-5, 500.]))
        self.assertIsNotNone(covariance)
        self.assertAlmostEqual(covariance[2, 2]/1e10, 1., places=8)
        self.assertIsNone(covariance_from_jacobian(np.diag([1., 2., 0., 500.])))

    def test_repeated_front_updates_cannot_average_away_shared_length_uncertainty(self):
        # The same learned physical length is reused, not independently remeasured 301 times.
        from peleng.world import WorldTrack
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        systematic = learned.get('front_systematic_covariance_cm2')
        self.assertIsNotNone(systematic, 'A persistent front-reference uncertainty floor is missing')
        measurement = dict(learned, position_cm=learned['front_cm'],
                           covariance_cm2=learned['front_covariance_cm2'],
                           systematic_covariance_cm2=systematic,
                           reference_point='front_face_center')
        track = WorldTrack()
        for frame in range(301):
            result = track.update(measurement, frame/30)
        self.assertEqual(result['state'], 'observed')
        self.assertGreaterEqual(result['uncertainty_m'], learned['dimension_sigma_cm'][0]/200)
        self.assertGreater(np.linalg.eigvalsh(learned['systematic_covariance_cm2']).min(), 0.)

    def test_dense_fragment_does_not_become_full_depth_even_with_known_size(self):
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        observations = self.boxes([1300., 700., 420.], cameras=(1,))
        observations[0].update(quality={'foreground_fill': .95},
            geometry_eligibility={'depth': False, 'shape': False})
        self.assertIsNone(estimator.update(6, .6, observations,
            prior_cm=[1300., 700., 420.], velocity_cm_s=self.direction*400))

    def test_one_inconsistent_view_does_not_destroy_two_good_views(self):
        # A partial box in a third camera must not poison the geometrically consistent pair.
        observations = self.boxes([1000., 500., 400.], cameras=(1, 2, 3))
        observations[2]['bbox'] = (np.array(observations[2]['bbox']) + [100., -80., 100., -80.]).tolist()
        observations[2]['confidence'] = .3
        result = ShapeEstimator(self.rig).update(0, 0., observations, velocity_cm_s=self.direction * 400)
        self.assertIsNotNone(result)
        self.assertEqual(result['cameras'], [1, 2])
        np.testing.assert_allclose(result['center_cm'], [1000., 500., 400.], atol=15.)

    def test_partial_only_view_cannot_turn_occlusion_into_a_new_heading(self):
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        observations = self.boxes([1300., 700., 420.], yaw=self.yaw+.2, cameras=(1,))
        observations[0].update(source='foreground', quality={'partial': True, 'foreground_fill': .8})
        result = estimator.update(6, .6, observations, velocity_cm_s=[0., 400., 0.])
        self.assertIsNone(result)
        self.assertAlmostEqual(estimator._yaw, learned['yaw_rad'], delta=.001)

    def test_current_multiview_pose_is_not_vetoed_by_smoothed_dimensions(self):
        # Both cameras agree on the current extent. Old shared dimensions must
        # not force a monocular fallback; one sample also must not replace history.
        estimator = ShapeEstimator(self.rig)
        learned = self.learn(estimator)
        changed = self.boxes([1300., 700., 400.], dimensions=[2000., 600., 900.])
        pose = estimator.update(6, .6, changed, velocity_cm_s=self.direction * 400)
        self.assertEqual(pose['method'], 'cuboid_multiview')
        np.testing.assert_allclose(pose['center_cm'], [1300.,700.,400.], atol=8.)
        self.assertGreater(pose['dimensions_cm'][2], 880.)
        np.testing.assert_allclose(pose['shared_dimensions_cm'], learned['dimensions_cm'], atol=1.)
        result = estimator.update(7, .7, self.boxes([1300., 700., 400.]),
                                  velocity_cm_s=self.direction * 400)
        self.assertEqual(result['shape_samples'], learned['shape_samples'] + 2)
        np.testing.assert_allclose(result['dimensions_cm'], learned['dimensions_cm'], atol=1.)

    def test_motion_does_not_turn_a_degenerate_two_view_fit_into_learned_scale(self):
        # Real camera/box evidence only: no reference position or fitted pivot offset.
        for camera_id, location, pitch, yaw, focal in [
            (1, [-24701.9592, 6623.5717, 1200.], 0., -70., 35.),
            (2, [-45784.0013, -6500., 7911.8821], -10., 0., 60.),
        ]:
            text = ('Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;'
                    f'{focal}\n0;{location[0]};{location[1]};{location[2]};0;{pitch};{yaw}'
                    ';;;матрица;23,76*13,365мм\n')
            (Path(self.temp.name) / f'object_transform_CineCameraActor{camera_id}.csv').write_text(
                text, encoding='cp1251')
        observations = [
            {'camera_id': 1, 'bbox': [866.9094, 612.4078, 1215.7644, 733.9235], 'confidence': .86},
            {'camera_id': 2, 'bbox': [191.3988, 896.3635, 521.7949, 1067.1533], 'confidence': .206},
        ]
        result = ShapeEstimator(CameraRig(self.temp.name)).update(
            95, 95/30, observations, velocity_cm_s=[1795.4, 1346.4, 0.])
        # A motion hint resolves front sign, but does not magically observe width.
        self.assertIsNone(result)

    def test_output_prefix_is_identical_and_reset_discards_learned_scale(self):
        # Future updates or retained dimensions after replay would contaminate inference.
        first, second = ShapeEstimator(self.rig), ShapeEstimator(self.rig)
        prefix = []
        for frame in range(8):
            center = [1000 + frame * 30., 500 + frame * 20., 400.]
            a = first.update(frame, frame / 10, self.boxes(center), velocity_cm_s=self.direction * 400)
            if frame < 4:
                prefix.append(copy.deepcopy(a))
                self.assertEqual(a, second.update(frame, frame / 10, self.boxes(center),
                                                  velocity_cm_s=self.direction * 400))
        self.assertEqual(prefix[0]['shape_samples'], 1)
        first.reset()
        self.assertIsNone(first.update(0, 0., self.boxes([1000, 500, 400], cameras=(1,))))

    def test_backward_or_nonfinite_time_does_not_mutate_shape(self):
        # Invalid ordering must fail before it can enter a shared-dimension history.
        estimator = ShapeEstimator(self.rig)
        self.learn(estimator)
        for frame, timestamp in [(3, .3), (6, float('nan')), (6, float('inf'))]:
            with self.assertRaises(ValueError):
                estimator.update(frame, timestamp, self.boxes([1000., 500., 400.]))


if __name__ == '__main__':
    unittest.main()
